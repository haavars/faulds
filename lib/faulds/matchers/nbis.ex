defmodule Faulds.Matchers.Nbis do
  @moduledoc """
  Template-mode `Faulds.Matcher` backed by NIST's own NBIS
  reference tools — `mindtct` (minutiae extraction) and `bozorth3`
  (minutiae matching), the public baseline.

  Requires `mix faulds.setup_nbis` to have been run once (builds
  both binaries from source into `priv/nbis/bin/`, gitignored
  — see that task's moduledoc for why source, not a prebuilt binary).

  `mindtct`/`bozorth3` are lightweight, fast-starting native binaries, so
  each call runs them afresh; no process stays resident.

  ## Image format and DPI

  `mindtct` reads ANSI/NIST, WSQ, JPEGB, JPEGL, or its own IHead raw
  format — not arbitrary formats, and this project's NBIS build was
  configured `--STDLIBS` (no PNG/JPEG2000 libraries), so `extract/2`
  writes the tensor to a temp JPEG via `Image`/libvips, with the image's
  resolution (`:dpi`, default 500, what every image here is resampled
  to) in its density header.

  ## Template format

  The "template" is simply the raw text contents of `mindtct`'s `.xyt`
  minutiae file (`x y theta quality` rows) — already exactly the input
  `bozorth3` expects, so `score/2` just writes both templates back out to
  temp files and reads its stdout.

  ## Minutiae type (ridge ending vs. bifurcation)

  `mindtct` also writes a `.min` file alongside `.xyt`, a human-readable
  dump of the same detected minutiae that additionally carries each
  point's classification (`RIG` ridge ending / `BIF` bifurcation) —
  `.xyt`'s own `x y theta quality` rows don't carry this. `minutiae/1`
  reads both and merges them by position, so callers get each point's
  type.

  ## Coordinates and angles

  `extract/1`'s templates are `.xyt` as `mindtct` writes it (without
  `-m1`), which `bozorth3` expects: the origin at the **bottom left**, `y`
  up, and `theta` in degrees counter-clockwise as seen, along the ridge
  but pointing the opposite way to ANSI/INCITS 378's direction. `.min`
  has the origin at the top left: a `.min` row's `y` is `image_height -
  y` of the same point in `.xyt` (every point of a real FVC image's pair
  matched that way, none missed).

  `minutiae/2` returns them in the image's own frame, the convention the
  rest of this library uses: `x`, `y` in pixels from the **top left**, and
  `angle` in degrees in those coordinates (`y` down, so clockwise as
  seen), in ANSI 378's direction: `y = image_height - y_xyt` and `angle
  = 180 - theta`. Checked against the synthetic data's exact ground
  truth (two rolled prints, 413 true minutiae): 366 of 489 detections lie
  within 8 px of a true minutia this way and 11 with `y` as written, and
  their angles are a median 6° from the true direction.
  """

  @behaviour Faulds.Matcher

  @mm_per_inch 25.4

  @doc "Whether `mix faulds.setup_nbis` has been run — both binaries are present."
  @spec available? :: boolean()
  def available? do
    File.exists?(binary_path("mindtct")) and File.exists?(binary_path("bozorth3"))
  end

  @impl Faulds.Matcher
  @doc "The `.xyt` template of `image`. Options: `:dpi` (default 500)."
  def extract(image, opts \\ []) do
    with {:ok, %{xyt: xyt}} <- run_mindtct(image, Keyword.get(opts, :dpi, 500)), do: {:ok, xyt}
  end

  @doc """
  Runs `mindtct` and returns its detected minutiae as a list of maps
  (`%{x:, y:, angle:, quality:, type:}`, in `image`'s own frame as the
  moduledoc's "Coordinates and angles" has it, `quality` mindtct's own ~0-100
  reliability score, `type` one of `"ridge_ending"`/`"bifurcation"` — see
  this module's moduledoc for how `type` is recovered from the `.min`
  file) — for UI display/marking, not matching (`extract/1`+`score/2`
  remain the Matcher-behaviour path for that). Options: `:dpi` (default
  500).
  """
  @spec minutiae(Nx.Tensor.t(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def minutiae(image, opts \\ []) do
    {height, _width} = Nx.shape(image)

    with {:ok, %{xyt: xyt, min: min}} <- run_mindtct(image, Keyword.get(opts, :dpi, 500)),
         {:ok, minutiae} <- parse_minutiae(xyt, min, height) do
      {:ok, minutiae}
    else
      :error -> {:error, :malformed_mindtct_output}
      {:error, reason} -> {:error, reason}
    end
  end

  # `mindtct`'s `.xyt` format is stable across the versions this project
  # builds against, but it's still an external tool's output, not this
  # codec's own — a row that doesn't parse as `x y theta quality`
  # (unexpected `mindtct` version/output drift) returns `:error` rather
  # than crashing the caller.
  defp parse_minutiae(xyt, min, height) do
    types = parse_min_types(min)

    xyt
    |> String.trim()
    |> String.split("\n", trim: true)
    |> Enum.reduce_while({:ok, []}, fn line, {:ok, acc} ->
      case parse_xyt_line(line, types, height) do
        {:ok, minutia} -> {:cont, {:ok, [minutia | acc]}}
        :error -> {:halt, :error}
      end
    end)
    |> case do
      {:ok, acc} -> {:ok, Enum.reverse(acc)}
      :error -> :error
    end
  end

  # A `.xyt` row (bottom-left origin) in the image's frame: see the
  # moduledoc's "Coordinates and angles".
  defp parse_xyt_line(line, types, height) do
    with [x_str, y_str, angle_str, quality_str] <- String.split(line, " "),
         {x, ""} <- Integer.parse(x_str),
         {y_xyt, ""} <- Integer.parse(y_str),
         {theta, ""} <- Integer.parse(angle_str),
         {quality, ""} <- Integer.parse(quality_str) do
      y = height - y_xyt
      type = Map.get(types, {x, y}, "other")
      angle = Integer.mod(180 - theta, 360) * 1.0
      {:ok, %{x: x, y: y, angle: angle, quality: quality, type: type}}
    else
      _ -> :error
    end
  end

  @min_row_regex ~r/^\s*\d+\s*:\s*(\d+),\s*(\d+)\s*:\s*\d+\s*:\s*[\d.]+\s*:(RIG|BIF)\s*:/

  # Builds a `{x, y}` (top-left origin, as `.min` has it) → `type` lookup
  # from the `.min` file's rows.
  defp parse_min_types(min) do
    min
    |> String.split("\n", trim: true)
    |> Enum.reduce(%{}, fn line, acc ->
      case Regex.run(@min_row_regex, line) do
        [_, x_str, y_min_str, type] ->
          type = if type == "RIG", do: "ridge_ending", else: "bifurcation"
          Map.put(acc, {String.to_integer(x_str), String.to_integer(y_min_str)}, type)

        nil ->
          acc
      end
    end)
  end

  defp run_mindtct(image, dpi) do
    work_dir = temp_dir!()

    try do
      jpeg_path = Path.join(work_dir, "in.jpg")
      write_jpeg!(image, dpi, jpeg_path)

      oroot = Path.join(work_dir, "out")

      case System.cmd(binary_path("mindtct"), [jpeg_path, oroot], stderr_to_stdout: true) do
        {_output, 0} ->
          {:ok, %{xyt: File.read!(oroot <> ".xyt"), min: File.read!(oroot <> ".min")}}

        {output, status} ->
          {:error, "mindtct exited #{status}: #{output}"}
      end
    after
      File.rm_rf(work_dir)
    end
  end

  @impl Faulds.Matcher
  def score(template_a, template_b) do
    work_dir = temp_dir!()

    try do
      path_a = Path.join(work_dir, "a.xyt")
      path_b = Path.join(work_dir, "b.xyt")
      File.write!(path_a, template_a)
      File.write!(path_b, template_b)

      case System.cmd(binary_path("bozorth3"), [path_a, path_b], stderr_to_stdout: true) do
        {output, 0} -> parse_score(output)
        {output, status} -> {:error, "bozorth3 exited #{status}: #{output}"}
      end
    after
      File.rm_rf(work_dir)
    end
  end

  @doc """
  `bozorth3`'s score of `probe_xyt` against each of `gallery_xyts`, in
  order, using its one-probe-many-gallery mode (`-p probe -G list`)
  instead of one process per pair: a latent search scores a whole
  gallery this way. The gallery is split into chunks of `:chunk` (2000)
  run in parallel. An empty template scores 0.
  """
  @spec score_many(binary(), [binary()], keyword()) :: [float()]
  def score_many(probe_xyt, gallery_xyts, opts \\ []) do
    work_dir = temp_dir!()

    try do
      probe_path = Path.join(work_dir, "probe.xyt")
      File.write!(probe_path, probe_xyt)

      gallery_xyts
      |> Enum.with_index()
      |> Enum.chunk_every(Keyword.get(opts, :chunk, 2000))
      |> Task.async_stream(&score_chunk(work_dir, probe_path, &1), timeout: :infinity)
      |> Enum.flat_map(fn {:ok, scores} -> scores end)
    after
      File.rm_rf(work_dir)
    end
  end

  defp score_chunk(work_dir, probe_path, [{_, first} | _] = chunk) do
    paths =
      Enum.map(chunk, fn {xyt, index} ->
        path = Path.join(work_dir, "g#{index}.xyt")
        File.write!(path, xyt)
        path
      end)

    list = Path.join(work_dir, "gallery-#{first}.lis")
    File.write!(list, Enum.join(paths, "\n") <> "\n")

    {output, 0} =
      System.cmd(
        binary_path("bozorth3"),
        ["-A", "maxfiles=100000", "-p", probe_path, "-G", list],
        stderr_to_stdout: true
      )

    output |> String.split("\n", trim: true) |> Enum.map(&(String.to_integer(&1) * 1.0))
  end

  defp parse_score(output) do
    case output |> String.trim() |> Integer.parse() do
      {score, ""} -> {:ok, score * 1.0}
      _ -> {:error, "bozorth3 produced unparseable output: #{inspect(output)}"}
    end
  end

  defp write_jpeg!(image, dpi, path) do
    {height, width} = Nx.shape(image)

    pixels =
      image
      |> Nx.multiply(255.0)
      |> Nx.round()
      |> Nx.as_type(:u8)
      |> Nx.reshape({height, width, 1})

    {:ok, vimage} = Image.from_nx(pixels)
    res = dpi / @mm_per_inch
    {:ok, vimage} = Vix.Vips.Operation.copy(vimage, xres: res, yres: res)
    Image.write!(vimage, path)
  end

  defp temp_dir! do
    dir = Path.join(System.tmp_dir!(), "nbis-#{:erlang.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    dir
  end

  defp binary_path(name) do
    :faulds |> Application.app_dir("priv/nbis") |> Path.join("bin/#{name}")
  end
end
