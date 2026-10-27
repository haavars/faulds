defmodule Faulds.Matchers.FlareNx do
  @moduledoc """
  Template-mode `Faulds.Matcher` backed by FLARE (Pan,
  Guan, Duan, Feng, Zhou, "Fixed-Length Dense Fingerprint Representation
  With Alignment and Robust Enhancement", IEEE TIFS 2026 —
  github.com/Yu-Yy/FLARE): a native `Ortex`/`Nx` port of FLARE's
  extraction, with no Python at run time. It was checked against FLARE's
  own Python pipeline, template for template, when written.

  Requires `mix faulds.setup_flare` to have been run once: it fetches
  FLARE's checkpoints and writes the two ONNX exports this module loads: `priv/flare/pose.onnx` and
  `fdd.onnx` (both gitignored). The same task also exports the rest of
  FLARE's full configuration, which only `extract_variants/2` uses and
  which is loaded when present: `regression_pose.onnx` (RegressionPose)
  and `unetenh.onnx` (UNetEnh, from FLARE_ENH).

  ## Pipeline (FLARE's two-stage estimate-then-extract, exactly)

  1. Pad/center the raw image onto the pose model's 512×512 input
     canvas (`Flare.Warp`, replicating `FingerPoseEvalDataset.process_img`
     — pure translation, no rotation/scale, white/255 border).
  2. Run `Flare.FastCartoonTexture` on that canvas (replicating
     `GRIDNET4`'s own `preprocess_tv` step — done here in Nx because
     `torch.onnx.export` can't trace its FFT calls, see
     `priv/flare/export_pose_onnx.py`'s header).
  3. Run the exported pose ONNX graph (`Ortex`) on the result, getting a
     `pose_2d = {x, y, theta_deg}` estimate in the *512-canvas*'s
     coordinate space.
  4. Correct that pose back into the *original* image's coordinate
     space via the pose-stage warp's inverse (`Flare.Warp.invert/1` +
     `apply_point/2`, replicating FLARE's inverse-transform step) and
     wrap `theta` back into `(-180, 180]`.
  5. Normalize the *original* (unpadded) image to `[-1, 1]`
     (`(pixel - 127.5) / 127.5`) and warp it onto the descriptor model's
     256×256 input canvas using the corrected pose (`Flare.Warp`,
     replicating `Descdataset.process_img` — scale 0.5, rotation
     `theta`, 0-value border).
  6. Run the exported FDD ONNX graph (`Ortex`) on that crop, getting
     `feature`/`mask`, FLARE's template.

  `score/2` delegates to `Matchers.Flare.score/2`, FLARE's masked cosine.

  ## DPI

  FLARE's checkpoints assume 500 ppi with no rescaling of their own, so
  images come in at 500 ppi (`Resample`).
  """

  use GenServer

  alias Faulds.Flare.{FastCartoonTexture, Warp}
  alias Faulds.Matchers.Flare
  alias Faulds.Minutiae.FddMap
  alias Faulds.Onnx

  @behaviour Faulds.Matcher

  @pose_canvas 512
  @desc_canvas 256
  @pose_border 255.0
  @desc_border 0.0
  @map_bucket 32
  # float32 -inf
  @neg_inf <<0xFF800000::32-native>>

  @doc "Starts the GenServer holding the two loaded Ortex models, registered under this module's name."
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc """
  Whether both ONNX exports this module needs are present locally (i.e.
  whether `mix faulds.setup_flare` has been run). Every mix task
  that uses this matcher assumes that setup already happened and lets
  `init/1` raise loudly if not; a long-running web request instead
  needs to check this *before* trying to start the GenServer, so it can
  show an operator a clear "not set up yet" state instead of crashing.
  """
  @spec available?() :: boolean()
  def available? do
    File.exists?(onnx_path("pose.onnx")) and File.exists?(onnx_path("fdd.onnx"))
  end

  @doc """
  Starts the GenServer under `Faulds.MatcherSupervisor` if
  it isn't already running (idempotent), unlike `start_link/1`. Every mix
  task that uses this matcher owns its own process and calls
  `start_link/1` directly exactly once, so a bare `{:ok, pid} =
  start_link()` match — and that process being the one linked to the
  GenServer — is fine there. A LiveView's `mount/3` runs once per
  connection, with no coordination over who starts the shared,
  name-registered GenServer first, and — unlike a mix task's own
  process — has no business being the process this long-lived, shared
  singleton is linked to: an unrelated crash in whichever LiveView
  happened to start it would otherwise take the shared model server down
  with it. Starting it under a supervisor instead means the *supervisor*
  holds that link, and restarts the GenServer itself if it ever crashes.
  Callers should check `available?/0` first; calling this when the ONNX
  files are missing still raises (surfaced here as a `DynamicSupervisor`
  child-start failure), same as `init/1` always has.
  """
  @spec ensure_started() :: {:ok, pid()}
  def ensure_started do
    case DynamicSupervisor.start_child(Faulds.MatcherSupervisor, __MODULE__) do
      {:ok, pid} -> {:ok, pid}
      {:error, {:already_started, pid}} -> {:ok, pid}
    end
  end

  @impl Faulds.Matcher
  def extract(image), do: GenServer.call(__MODULE__, {:extract, image}, :infinity)

  @impl Faulds.Matcher
  def score(template_a, template_b), do: Flare.score(template_a, template_b)

  @doc """
  Extracts templates for many images concurrently (`Task.async_stream/3`),
  for bulk work like enrolling a gallery, rather than one `extract/1`
  call at a time. Fetches the two loaded `Ortex` models once, then runs
  both stages of the pipeline directly from each task process instead of
  funneling every image through the single GenServer mailbox. That is
  safe: ONNX Runtime sessions support concurrent `Run` calls, and every
  other step here (`Warp`'s `Vix` NIF calls, `FastCartoonTexture`'s pure
  `Nx` arithmetic) carries no process affinity.

  Takes **loader thunks**, not already-decoded images, so only the images
  being worked on are in memory at once.
  """
  @spec extract_many([(-> Nx.Tensor.t())]) :: [{:ok, map()}]
  def extract_many(image_loaders) do
    {pose_model, fdd_model} = GenServer.call(__MODULE__, :get_models, :infinity)

    image_loaders
    |> Task.async_stream(
      fn load -> do_extract(load.(), pose_model, fdd_model) end,
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, result} -> result end)
  end

  @doc """
  Several templates of one image, for searching a latent (or enrolling a
  print) at more than the one pose `extract/1` uses. The
  variants are every combination of:

    * `:input_rotations` — degrees to rotate the image by before pose
      estimation, so the pose network gets several tries at a print it
      misjudges upright (default `[0]`);
    * `:theta_offsets` — degrees added to each estimated pose angle
      (default `[0]`);
    * `:shifts` — `{dx, dy}` offsets of the descriptor window's centre in
      the aligned frame, in original-image pixels (y towards the
      fingertip is negative), so a print can be enrolled with windows on
      its tip and sides as well as its centre (default `[{0, 0}]`).

  and, for FLARE's full configuration (the paper's two pose estimators ×
  two enhancers, each used after alignment):

    * `:poses` — `:voting` (VotingPose, `extract/1`'s) and/or
      `:regression` (RegressionPose, `regression_pose.onnx`); default
      `[:voting]`;
    * `:enhancers` — `:none` (the image itself) and/or `:unet` (UNetEnh,
      `unetenh.onnx`): the descriptor is taken from the enhanced image,
      the pose always from the original, as FLARE does; default
      `[:none]`.

  Variants come in the order rotation, pose, theta offset, shift,
  enhancer (the last varying fastest), so two images extracted with the
  same options have their variants in the same order. With every option
  at its default this is exactly `extract/1`'s template, in a
  one-element list. Runs in the calling process, like `extract_many/1`.
  """
  @spec extract_variants(Nx.Tensor.t(), keyword()) :: [map()]
  def extract_variants(image, opts \\ []) do
    models = GenServer.call(__MODULE__, :get_all_models, :infinity)
    raw_image = Nx.multiply(image, 255.0)
    enhancers = Keyword.get(opts, :enhancers, [:none])

    for rotation <- Keyword.get(opts, :input_rotations, [0]),
        rotated = rotate(raw_image, rotation),
        inputs = Enum.map(enhancers, &descriptor_input(rotated, &1, models)),
        pose <- Keyword.get(opts, :poses, [:voting]),
        {x, y, theta} = estimate_pose_with(rotated, pose, models),
        offset <- Keyword.get(opts, :theta_offsets, [0]),
        shift <- Keyword.get(opts, :shifts, [{0, 0}]),
        input <- inputs do
      angle = theta + offset
      extract_descriptor(input, shift_centre({x, y}, angle, shift), angle, models.fdd_model)
    end
  end

  defp estimate_pose_with(raw_image, :voting, models),
    do: estimate_pose(raw_image, models.pose_model)

  # RegressionPose normalises its own input: no FastCartoonTexture step.
  defp estimate_pose_with(raw_image, :regression, models),
    do: estimate_pose(raw_image, optional_model!(models, :regression_pose_model), & &1)

  defp descriptor_input(raw_image, :none, _models), do: raw_image

  defp descriptor_input(raw_image, :unet, models),
    do: enhance(raw_image, optional_model!(models, :unetenh_model))

  defp optional_model!(models, key) do
    Map.fetch!(models, key) ||
      raise "FlareNx #{key} not found — run `mix faulds.setup_flare`"
  end

  # UNetEnh on the whole image, which is padded with white to a multiple
  # of 16 (`deploy_unetenh.py` stretches it to one instead). Returned in
  # raw [0, 255] like the image, so `extract_descriptor/4` normalises it
  # the same way (`2 * enh - 1`).
  defp enhance(raw_image, unetenh_model) do
    {height, width} = Nx.shape(raw_image)
    {padded_h, padded_w} = {ceil_to(height, 16), ceil_to(width, 16)}

    input =
      raw_image
      |> Nx.subtract(127.5)
      |> Nx.divide(127.5)
      |> Nx.pad(1.0, [{0, padded_h - height, 0}, {0, padded_w - width, 0}])
      |> Nx.reshape({1, 1, padded_h, padded_w})

    {enh} = Ortex.run(unetenh_model, input)

    enh
    |> Nx.backend_transfer()
    |> Nx.reshape({padded_h, padded_w})
    |> Nx.slice([0, 0], [height, width])
    |> Nx.multiply(255.0)
  end

  defp ceil_to(n, multiple), do: div(n + multiple - 1, multiple) * multiple

  @doc """
  The pose `extract/1` aligns `image` to: `{x, y, theta_deg}` in the
  image's own pixel coordinates. With `template_at_pose/2`, lets a
  diagnostic replace the estimate with a known pose.
  """
  @spec pose(Nx.Tensor.t()) :: {float(), float(), float()}
  def pose(image) do
    {pose_model, _fdd_model} = GenServer.call(__MODULE__, :get_models, :infinity)
    estimate_pose(Nx.multiply(image, 255.0), pose_model)
  end

  @doc "The template of `image` aligned to the given pose `{x, y, theta_deg}` instead of an estimated one."
  @spec template_at_pose(Nx.Tensor.t(), {number(), number(), number()}) :: map()
  def template_at_pose(image, {x, y, theta}) do
    {_pose_model, fdd_model} = GenServer.call(__MODULE__, :get_models, :infinity)
    extract_descriptor(Nx.multiply(image, 255.0), {x * 1.0, y * 1.0}, theta * 1.0, fdd_model)
  end

  @doc """
  The minutiae FDD's minutiae head finds in `image` (500 ppi, `[0, 1]`),
  decoded by `Minutiae.FddMap.decode/2` (options pass through), in the
  image's pixels. FDD sees the whole image at half resolution, as its
  descriptor crop does, but unaligned: the head is fully convolutional,
  so no pose (and no 512 px window) limits which minutiae it finds.
  Needs `fdd_minutiae.onnx` (`mix faulds.setup_flare`).
  """
  @spec minutiae(Nx.Tensor.t(), keyword()) :: [FddMap.minutia()]
  def minutiae(image, opts \\ []) do
    models = GenServer.call(__MODULE__, :get_all_models, :infinity)
    {map, cells} = minutiae_map(image, optional_model!(models, :fdd_minutiae_model))
    FddMap.decode(map, Keyword.put(opts, :cells, cells))
  end

  # FDD's input: half resolution, [-1, 1], padded with 0 (the descriptor
  # crop's border) to a multiple of 8. Its map is returned padded on with
  # -inf to a multiple of @map_bucket cells (padding no cell's 5 × 5 window
  # or channels can see as larger), with the cells it covers: the decode
  # then compiles for a few shapes, not one per latent (~0.5 s each).
  defp minutiae_map(image, model) do
    {height, width} = Nx.shape(image)
    {h, w} = {ceil_to(ceil_div(height, 2), 8), ceil_to(ceil_div(width, 2), 8)}
    t = Warp.affine_matrix(@desc_canvas / @pose_canvas, 0.0, {0.0, 0.0}, {0.0, 0.0})

    input =
      image
      |> Warp.warp(t, {h, w},
        background: @desc_border,
        linear: {2.0, -1.0},
        backend: Nx.BinaryBackend
      )
      |> Nx.reshape({1, 1, h, w})

    {map} = Ortex.run(model, input)
    {channels, rows, cols} = map |> Nx.shape() |> Tuple.delete_at(0)
    {bh, bw} = {ceil_to(rows, @map_bucket), ceil_to(cols, @map_bucket)}
    bytes = Nx.to_binary(map)

    padded =
      for c <- 0..(channels - 1) do
        body =
          for r <- 0..(rows - 1),
              do: [
                binary_part(bytes, (c * rows + r) * cols * 4, cols * 4),
                :binary.copy(@neg_inf, bw - cols)
              ]

        [body, :binary.copy(@neg_inf, (bh - rows) * bw)]
      end

    map =
      padded
      |> IO.iodata_to_binary()
      |> Nx.from_binary(:f32, backend: Nx.default_backend())
      |> Nx.reshape({channels, bh, bw})

    {map, {rows, cols}}
  end

  defp ceil_div(n, d), do: div(n + d - 1, d)

  # Rotates about the image centre onto a canvas big enough to keep every
  # corner, white outside (the pose network's own border value).
  defp rotate(raw_image, 0), do: raw_image

  defp rotate(raw_image, degrees) do
    {height, width} = Nx.shape(raw_image)
    side = round(:math.sqrt(width * width + height * height))
    theta = :math.pi() * degrees / 180.0
    t = Warp.affine_matrix(1.0, theta, {-width / 2.0, -height / 2.0}, {side / 2.0, side / 2.0})
    Warp.warp(raw_image, t, {side, side}, background: @pose_border)
  end

  # The descriptor crop maps p to R(theta)(p - c); a shift s in that frame
  # is c + R(-theta) s in the image.
  defp shift_centre(centre, _theta_deg, {0, 0}), do: centre

  defp shift_centre({x, y}, theta_deg, {dx, dy}) do
    theta = :math.pi() * theta_deg / 180.0

    {x + :math.cos(theta) * dx + :math.sin(theta) * dy,
     y - :math.sin(theta) * dx + :math.cos(theta) * dy}
  end

  @impl GenServer
  def init(_opts) do
    pose_path = onnx_path("pose.onnx")
    fdd_path = onnx_path("fdd.onnx")

    unless File.exists?(pose_path) and File.exists?(fdd_path) do
      raise "FlareNx ONNX models not found — run `mix faulds.setup_flare`"
    end

    {:ok,
     %{
       pose_model: Onnx.load(pose_path),
       fdd_model: Onnx.load(fdd_path),
       regression_pose_model: load_optional("regression_pose.onnx"),
       unetenh_model: load_optional("unetenh.onnx"),
       fdd_minutiae_model: load_optional("fdd_minutiae.onnx")
     }}
  end

  defp load_optional(filename) do
    path = onnx_path(filename)
    if File.exists?(path), do: Onnx.load(path)
  end

  @impl GenServer
  def handle_call(
        {:extract, image},
        _from,
        %{pose_model: pose_model, fdd_model: fdd_model} = state
      ) do
    {:reply, do_extract(image, pose_model, fdd_model), state}
  end

  @impl GenServer
  def handle_call(:get_models, _from, %{pose_model: pose_model, fdd_model: fdd_model} = state) do
    {:reply, {pose_model, fdd_model}, state}
  end

  @impl GenServer
  def handle_call(:get_all_models, _from, state), do: {:reply, state, state}

  defp do_extract(image, pose_model, fdd_model) do
    # The harness (and Matchers.Flare's own write_png! step) hands us a
    # [0, 1]-scaled tensor; FLARE's own pipeline (and this port's
    # FastCartoonTexture/normalization arithmetic) assumes raw [0, 255]
    # pixel values, same convention Matchers.Flare's write_png! restores
    # before handing an image to Python.
    raw_image = Nx.multiply(image, 255.0)

    {pose_x, pose_y, theta_deg} = estimate_pose(raw_image, pose_model)
    template = extract_descriptor(raw_image, {pose_x, pose_y}, theta_deg, fdd_model)

    {:ok, template}
  end

  defp estimate_pose(raw_image, pose_model, preprocess \\ &FastCartoonTexture.forward/1) do
    {height, width} = Nx.shape(raw_image)
    img_c = {width / 2.0, height / 2.0}
    center = {@pose_canvas / 2.0, @pose_canvas / 2.0}

    pose_t = Warp.affine_matrix(1.0, 0.0, negate(img_c), center)
    canvas = Warp.warp(raw_image, pose_t, {@pose_canvas, @pose_canvas}, background: @pose_border)
    preprocessed = preprocess.(canvas)

    input = preprocessed |> Nx.reshape({1, 1, @pose_canvas, @pose_canvas})
    {pose_2d} = Ortex.run(pose_model, input)

    [raw_x, raw_y, raw_theta] =
      pose_2d |> Nx.backend_transfer() |> Nx.reshape({3}) |> Nx.to_list()

    {corrected_x, corrected_y} = Warp.apply_point(Warp.invert(pose_t), {raw_x, raw_y})
    {corrected_x, corrected_y, wrap_theta(raw_theta)}
  end

  defp extract_descriptor(raw_image, {pose_x, pose_y}, theta_deg, fdd_model) do
    normalized = raw_image |> Nx.subtract(127.5) |> Nx.divide(127.5)
    theta = :math.pi() * theta_deg / 180.0
    scale = @desc_canvas / @pose_canvas
    center = {@desc_canvas / 2.0, @desc_canvas / 2.0}

    desc_t = Warp.affine_matrix(scale, theta, negate({pose_x, pose_y}), center)
    crop = Warp.warp(normalized, desc_t, {@desc_canvas, @desc_canvas}, background: @desc_border)

    input = Nx.reshape(crop, {1, 1, @desc_canvas, @desc_canvas})
    {feature, mask} = Ortex.run(fdd_model, input)

    %{
      feature: feature |> Nx.backend_transfer() |> Nx.reshape({3072}),
      mask: mask |> Nx.backend_transfer() |> Nx.reshape({256})
    }
  end

  defp negate({x, y}), do: {-x, -y}

  # Ports `(pose_2d[2] + 180) % 360 - 180` (Python's `%` always returns a
  # non-negative result for a positive divisor; Elixir's doesn't for
  # negative operands, hence the explicit floor-based formula).
  defp wrap_theta(theta) do
    shifted = theta + 180.0
    wrapped = shifted - 360.0 * Float.floor(shifted / 360.0)
    wrapped - 180.0
  end

  defp onnx_path(filename) do
    :faulds |> Application.app_dir("priv/flare") |> Path.join(filename)
  end
end
