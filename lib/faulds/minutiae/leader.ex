defmodule Faulds.Minutiae.Leader do
  @moduledoc """
  Minutiae by LEADER (Cappelli & Ferrara, arXiv 2602.15493; pyfing,
  MIT), an end-to-end extractor of 0.9 M parameters, the best zero-shot
  latent extractor in an independent comparison (SD302h value latents F1
  0.64 against `mindtct`'s 0.25). Exported by `mix
  faulds.setup_leader` (`priv/leader/leader.onnx`).

  Ports pyfing's `Leader.run`: the 500 ppi image, raw 0..255, padded to
  a multiple of 32 on every side equally with its top-left pixel's value;
  a minutia wherever the network's own non-maximum-suppressed position
  map reaches `:threshold` (pyfing's default 0.6), its quality that value
  and its direction the network's. Directions come out
  counter-clockwise as seen and are returned in this library's
  convention (clockwise, `Matchers.Nbis`, "Coordinates and angles").

  On synthetic training-split latents, at 8 px and 20°: 37.7% of the
  true minutiae found at 80.8% precision (F1 0.51), or 42.8% at 64.5% at
  threshold 0.4; FDD's map 0.50, `mindtct` 0.34.
  """

  alias Vix.Vips.{Image, Operation}

  @multiple 32
  @default_threshold 0.6

  @doc "Whether `leader.onnx` has been exported."
  @spec available?() :: boolean()
  def available?, do: File.exists?(onnx_path())

  @doc """
  The minutiae in `image` (500 ppi, `[0, 1]`): `%{x:, y:, angle:,
  quality:, type:}` (`type` `:ending` or `:bifurcation`, pyfing's 0.5
  rule). Options: `:threshold`; `:view`, `{turns, mirrored}`: the network
  sees the image mirrored left to right (if `mirrored`), then turned
  `turns` quarter turns clockwise, and the minutiae come back in the
  image's own frame (default `{0, false}`, the image as it is).
  """
  @spec extract(Nx.Tensor.t(), keyword()) :: [map()]
  def extract(image, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, @default_threshold)
    view = Keyword.get(opts, :view, {0, false})
    {h, w} = Nx.shape(image)
    {vh, vw} = if rem(elem(view, 0), 2) == 1, do: {w, h}, else: {h, w}
    {ph, pw} = {padded(vh), padded(vw)}
    {top, left} = {div(ph - vh, 2), div(pw - vw, 2)}

    {out} = Ortex.run(model(), input(image, view, {top, left}, {ph, pw}))

    # Read straight from the output's bytes, as every op on the device
    # (EXLA) compiles anew for each image size, ~2 s a latent.
    out
    |> Nx.to_binary()
    |> rows(top, left, vh, vw, pw)
    |> Enum.with_index()
    |> Enum.flat_map(fn {row, y} -> passing(row, 0, threshold, y, []) end)
    |> Enum.map(&from_view(&1, view, {h, w}))
  end

  @doc """
  A latent's minutiae: LEADER's at `:threshold` (default 0.6), topped up
  to `:at_least` (default 12) with its strongest detections down to
  `:floor` (default 0.3) when it finds fewer. On synthetic validation
  runs this gains 1.8 points at rank 1 for `ldmd` over the threshold
  alone; the gallery's minutiae stay at the threshold.
  """
  @spec extract_latent(Nx.Tensor.t(), keyword()) :: [map()]
  def extract_latent(image, opts \\ []) do
    image
    |> extract(threshold: Keyword.get(opts, :floor, 0.3))
    |> top_up(Keyword.get(opts, :at_least, 12), Keyword.get(opts, :threshold, @default_threshold))
  end

  @doc """
  `detections` above `threshold`, or, if fewer than `at_least` are, the
  `at_least` strongest of them all.
  """
  @spec top_up([map()], non_neg_integer(), float()) :: [map()]
  def top_up(detections, at_least, threshold) do
    strong = Enum.filter(detections, &(&1.quality >= round(threshold * 100)))

    if length(strong) >= at_least,
      do: strong,
      else: detections |> Enum.sort_by(&(-&1.quality)) |> Enum.take(at_least)
  end

  @views for turns <- 0..3, mirrored <- [false, true], do: {turns, mirrored}

  @doc """
  LEADER over the eight views of `image` (`extract/2`'s `:view`: four
  quarter turns, each also mirrored), the detections merged: a cluster
  is the strongest detection not yet taken and every other view's within
  `:tolerance` px (default 6) and `:angle_tolerance` degrees (default
  20), one per view; it is kept when at least `:votes` views (default 2)
  found it, at the mean position and direction of its members, with
  their mean quality. `:views` and `:threshold` as for `extract/2`.
  """
  @spec extract_views(Nx.Tensor.t(), keyword()) :: [map()]
  def extract_views(image, opts \\ []) do
    views = Keyword.get(opts, :views, @views)

    views
    |> Enum.with_index()
    |> Enum.flat_map(fn {view, k} ->
      image
      |> extract(Keyword.merge(opts, view: view))
      |> Enum.map(&Map.put(&1, :view, k))
    end)
    |> merge_views(opts)
  end

  @doc """
  `extract_views/2`'s merge of detections tagged with `:view`.
  """
  @spec merge_views([map()], keyword()) :: [map()]
  def merge_views(detections, opts \\ []) do
    tolerance = Keyword.get(opts, :tolerance, 6)
    angle_tolerance = Keyword.get(opts, :angle_tolerance, 20)
    votes = Keyword.get(opts, :votes, 2)

    detections
    |> Enum.sort_by(&(-&1.quality))
    |> cluster(tolerance, angle_tolerance, [])
    |> Enum.filter(&(length(&1) >= votes))
    |> Enum.map(&centre/1)
  end

  defp cluster([], _tolerance, _angle_tolerance, clusters), do: Enum.reverse(clusters)

  defp cluster([seed | rest], tolerance, angle_tolerance, clusters) do
    # The nearest detection of each other view, if close enough.
    {members, rest} =
      rest
      |> Enum.filter(fn d ->
        d.view != seed.view and :math.sqrt((d.x - seed.x) ** 2 + (d.y - seed.y) ** 2) <= tolerance and
          angle_difference(d.angle, seed.angle) <= angle_tolerance
      end)
      |> Enum.group_by(& &1.view)
      |> Enum.map(fn {_, ds} ->
        Enum.min_by(ds, &((&1.x - seed.x) ** 2 + (&1.y - seed.y) ** 2))
      end)
      |> then(fn members -> {members, rest -- members} end)

    cluster(rest, tolerance, angle_tolerance, [[seed | members] | clusters])
  end

  defp centre([seed | _] = members) do
    n = length(members)
    # Directions averaged as unit vectors.
    {c, s} =
      Enum.reduce(members, {0.0, 0.0}, fn m, {c, s} ->
        r = m.angle * :math.pi() / 180
        {c + :math.cos(r), s + :math.sin(r)}
      end)

    %{
      x: Enum.sum(Enum.map(members, & &1.x)) / n,
      y: Enum.sum(Enum.map(members, & &1.y)) / n,
      angle: wrap(:math.atan2(s, c) * 180 / :math.pi()),
      quality: round(Enum.sum(Enum.map(members, & &1.quality)) / n),
      type: seed.type,
      votes: n
    }
  end

  defp angle_difference(a, b), do: abs(wrap(a - b + 180.0) - 180.0)

  # A minutia found in `view` back in the image's frame ({h, w} its size).
  # The view mirrors x first, then turns clockwise: a quarter turn takes
  # (x, y) to (h' − 1 − y, x) for the height h' before it, and a
  # direction θ (clockwise as seen) to θ + 90.
  defp from_view(m, {0, false}, _size), do: m

  defp from_view(m, {turns, mirrored}, {h, w}) do
    {x, y, angle} =
      Enum.reduce(1..turns//1, {m.x, m.y, m.angle}, fn k, {x, y, a} ->
        # Undo the last quarter turn: (x', y') came from (y', H − 1 − x'),
        # H the height before that turn.
        before = if rem(turns - k, 2) == 1, do: w, else: h
        {y, before - 1 - x, a - 90.0}
      end)

    {x, angle} = if mirrored, do: {w - 1 - x, 180.0 - angle}, else: {x, angle}
    %{m | x: x, y: y, angle: wrap(angle)}
  end

  # The image raw (0..255) in `view`, padded to {ph, pw} with its top-left
  # pixel's value, by libvips: the same values as Nx would give, with no
  # compile.
  defp input(image, {turns, mirrored}, {top, left}, {ph, pw}) do
    {h, w} = Nx.shape(image)
    binary = image |> Nx.as_type(:f32) |> Nx.to_binary()
    {:ok, vimage} = Image.new_from_binary(binary, w, h, 1, :VIPS_FORMAT_FLOAT)

    {:ok, vimage} =
      if mirrored, do: Operation.flip(vimage, :VIPS_DIRECTION_HORIZONTAL), else: {:ok, vimage}

    {:ok, vimage} =
      Operation.rot(
        vimage,
        Enum.at([:VIPS_ANGLE_D0, :VIPS_ANGLE_D90, :VIPS_ANGLE_D180, :VIPS_ANGLE_D270], turns)
      )

    {:ok, raw} = Operation.linear(vimage, [255.0], [0.0])
    {:ok, [fill]} = Operation.getpoint(raw, 0, 0)

    {:ok, padded} =
      Operation.embed(raw, left, top, pw, ph, extend: :VIPS_EXTEND_BACKGROUND, background: [fill])

    {:ok, bytes} = Image.write_to_binary(padded)
    bytes |> Nx.from_binary(:f32, backend: Nx.BinaryBackend) |> Nx.reshape({1, ph, pw, 1})
  end

  # The image's rows of the output (4 float32 channels a pixel).
  defp rows(bytes, top, left, h, w, pw) do
    for y <- top..(top + h - 1)//1, do: binary_part(bytes, (y * pw + left) * 16, w * 16)
  end

  # Channel 3 is the suppressed position map, 1 the direction, 2 the type.
  defp passing(
         <<_::32, d::float-32-native, t::float-32-native, q::float-32-native, rest::binary>>,
         x,
         threshold,
         y,
         acc
       )
       when q >= threshold do
    minutia = %{
      x: x * 1.0,
      y: y * 1.0,
      angle: wrap(-d * 180 / :math.pi()),
      quality: q |> Kernel.*(100) |> round() |> min(100),
      type: if(t >= 0.5, do: :ending, else: :bifurcation)
    }

    passing(rest, x + 1, threshold, y, [minutia | acc])
  end

  defp passing(<<_::binary-size(16), rest::binary>>, x, threshold, y, acc),
    do: passing(rest, x + 1, threshold, y, acc)

  defp passing(<<>>, _x, _threshold, _y, acc), do: Enum.reverse(acc)

  defp padded(n), do: div(n + @multiple - 1, @multiple) * @multiple

  defp wrap(angle), do: angle - 360.0 * Float.floor(angle / 360.0)

  defp model do
    case :persistent_term.get({__MODULE__, :model}, nil) do
      nil ->
        model = Faulds.Onnx.load(onnx_path())
        :persistent_term.put({__MODULE__, :model}, model)
        model

      model ->
        model
    end
  end

  defp onnx_path, do: Application.app_dir(:faulds, "priv/leader/leader.onnx")
end
