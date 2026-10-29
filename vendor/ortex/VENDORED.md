# Vendored Ortex

From https://github.com/elixir-nx/ortex at
`55dc3de016041b16ec6d17589f020937b0bb18fc` (2026-02-10), MIT (`LICENSE`).

Changed here:

- `native/ortex`: ported from `ort` 2.0.0-rc.8 to 2.0.0-rc.15, whose
  prebuilt ONNX Runtime (1.31) has a CUDA 13 build (rc.8's needs CUDA 12,
  and this machine has CUDA 13). `Session::run` takes `&mut self` since
  rc.10, so a model's session sits behind a `Mutex` (runs on one model are
  serialised); sessions are built in one global `Environment`;
  execution providers not compiled in fall back to the CPU; `ndarray` 0.17.
  The CUDA provider runs without TF32 (results match the CPU's) and grows
  its memory pool by what's requested, not in powers of two.
- `Ortex.load/4` takes `deterministic: true`: the session's deterministic
  compute, and the CUDA provider on cuDNN's default convolution algorithm
  instead of benchmarking them per process, so a model gives the same
  results from run to run (FlareNx; LEADER and DMD++ were already
  deterministic and load without it).
- `Ortex.Util.copy_ort_libs/0` also looks in the crate's `target/`, where
  Rustler builds a path dependency.
- `mix.exs`: accepts Nx 1.0; the dev-only dependencies are dropped.
- `examples/`, `models/`, `python/`, `test/` and the upstream config are
  left out.

The Elixir API is unchanged.
