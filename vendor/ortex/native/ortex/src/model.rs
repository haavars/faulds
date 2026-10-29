//! Abstractions for creating an ONNX Runtime Session and Environment which can be safely
//! passed to and from the BEAM.
//!
//! # Examples
//!
//! ```
//! let model = init("./models/resnet50.onnx", vec![])?;
//! let (inputs, outputs) = show(model)?;
//! ```

use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_opt_level};
use std::convert::TryInto;
use std::iter::zip;

use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, SessionInputValue};
use ort::Error;
use std::sync::{Mutex, OnceLock};

// Since ort 2.0.0-rc.12 every session is built in an explicit environment;
// one for the whole VM.
static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();

fn environment() -> Result<&'static ort::environment::Environment, Error> {
    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("ortex").build()?;
    Ok(ENVIRONMENT.get_or_init(|| env))
}
use rustler::resource::ResourceArc;
use rustler::Atom;

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    // `Session::run` takes `&mut self` since ort 2.0.0-rc.10, so runs on one
    // model are serialised; ONNX Runtime still parallelises within a run.
    pub session: Mutex<Session>,
}

// Since we're only using the session for inference and
// inference is threadsafe, this Sync is safe. Additionally,
// Environment is global and also threadsafe
// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Sync for OrtexModel {}

/// Creates a model given the path to the model and vector of execution providers.
/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
    deterministic: bool,
) -> Result<OrtexModel, Error> {
    // TODO: send tracing logs to erlang/elixir _somehow_
    // tracing_subscriber::fmt::init();

    let session = Session::builder(environment()?)?
        .with_optimization_level(map_opt_level(opt))?
        .with_deterministic_compute(deterministic)?
        .with_execution_providers(eps)?
        .commit_from_file(model_path)?;

    let state = OrtexModel {
        session: Mutex::new(session),
    };
    Ok(state)
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where
/// `Dimension` elements of -1 are dynamic.
pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
) {
    let session = model.session.lock().unwrap();

    let describe = |outlet: &ort::value::Outlet| {
        let name = outlet.name().to_string();
        let repr = format!("{:#?}", outlet.dtype());
        let dims = outlet.dtype().tensor_shape().map(|shape| shape.to_vec());
        (name, repr, dims)
    };

    let inputs = session.inputs().iter().map(describe).collect();
    let outputs = session.outputs().iter().map(describe).collect();

    (inputs, outputs)
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Error> {
    // Grab the session and run a forward pass with it
    let mut session = model.session.lock().unwrap();

    let mut ortified_inputs: Vec<SessionInputValue> = Vec::new();

    for (elixir_input, onnx_input) in zip(inputs, session.inputs()) {
        let derefed_input: &OrtexTensor = &elixir_input;
        if is_bool_input(onnx_input.dtype()) {
            // this assumes that the boolean input isn't huge -- we're cloning it twice;
            // once below, once in the try_into()
            let boolified_input: &OrtexTensor = &derefed_input.clone().to_bool();
            let v: SessionInputValue = boolified_input.try_into()?;
            ortified_inputs.push(v);
        } else {
            let v: SessionInputValue = derefed_input.try_into()?;
            ortified_inputs.push(v);
        }
    }

    // Construct a Vec of ModelOutput enums based on the DynOrtTensor data type
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|outlet| outlet.name().to_string())
        .collect();
    let outputs = session.run(&ortified_inputs[..])?;
    let mut collected_outputs = Vec::new();

    for output_name in &output_names {
        let output_name: &str = output_name;
        let val = outputs.get(output_name).expect(
            &format!(
                "Expected {} to be in the outputs, but didn't find it",
                output_name
            )[..],
        );

        // NOTE: try_into impl here will implicitly map bool outputs to u8 outputs
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();

        let collected_output = (ResourceArc::new(ortextensor), shape, dtype, bits);
        collected_outputs.push(collected_output)
    }

    Ok(collected_outputs)
}
