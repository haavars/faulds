defmodule Faulds.Minutiae.FddMap do
  @moduledoc """
  Minutiae from FDD's minutiae map, the head FLARE's descriptor network
  trains beside its descriptor and FlareNx's `fdd.onnx` leaves out
  (`priv/flare/export_fdd_minutiae_onnx.py` exports it as
  `fdd_minutiae.onnx`; `Matchers.FlareNx.minutiae/2` runs it). F-DMD
  (MALD, arXiv 2507.15297) anchors its patches on these; FLARE's repo
  doesn't release that decoder, so this one follows MSU-LatentAFIS's
  (`extraction/prepare_data.py`, `get_minutiae_from_cylinder` and
  `refine_minutiae`), the DeepPrint map format FDD's targets follow.

  ## The map

  6 channels at half FDD's input resolution; channel `k` holds minutiae
  whose direction is near `k · 60°`, as Gaussian blobs. FlareNx feeds FDD
  at half of 500 ppi, so one map cell is 4 px of the 500 ppi image.

  ## Decoding

    1. Each cell's strongest channel; cells under `:threshold` are dropped.
    2. A cell is a minutia if no value in a 5 × 5 window of its own and
       the two neighbouring channels (circularly) is larger.
    3. The direction is refined by a parabola through the three channel
       values: `(k + (a − c) / 2(a − 2b + c)) · 60°`.
    4. Strongest first, a minutia within 10 px of a kept one, or within
       20 px and 45° of one, is dropped.

  Directions come out clockwise as seen (`y` down) from the `x` axis, in
  ANSI/INCITS 378's direction: this library's convention, as
  `Matchers.Nbis.minutiae/2` returns. Checked against the synthetic
  data's exact ground truth on training-split prints: on 24 rolled prints
  79% of the true minutiae found at 75% precision (8 px, 20°), on 71
  latents 40% at 68%, where `mindtct` finds 38% at 30%. The threshold, 0.5 (MSU's), gave the best F1 there.
  """

  @type minutia :: %{x: float(), y: float(), angle: float(), quality: 0..100}

  @channels 6
  @cell_px 4
  @default_threshold 0.5

  @doc """
  The minutiae in `map` (`{6, h, w}`, FDD's minutiae head on an image fed
  at half its resolution), in the image's pixels. Options: `:threshold`
  (default #{@default_threshold}); `:cells`, `{rows, cols}`: only the
  map's top-left `rows × cols` cells are read (the rest is padding that
  shouldn't suppress them; default the whole map).
  """
  @spec decode(Nx.Tensor.t(), keyword()) :: [minutia()]
  def decode(map, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, @default_threshold)
    {@channels, h, w} = Nx.shape(map)
    {rows, cols} = Keyword.get(opts, :cells, {h, w})

    value = Nx.reduce_max(map, axes: [0])
    channel = Nx.argmax(map, axis: 0)

    window = Nx.window_max(map, {1, 5, 5}, padding: :same)
    around = window |> Nx.max(roll(window, 1)) |> Nx.max(roll(window, -1))
    peak = at_channel(around, channel)

    [a, b, c] =
      Enum.map(
        [-1, 0, 1],
        &at_channel(map, Nx.remainder(Nx.add(channel, @channels + &1), @channels))
      )

    curvature = a |> Nx.subtract(Nx.multiply(2, b)) |> Nx.add(c)

    shift =
      Nx.select(
        Nx.equal(curvature, 0),
        0.0,
        Nx.divide(Nx.multiply(0.5, Nx.subtract(a, c)), curvature)
      )

    angle = channel |> Nx.add(shift) |> Nx.multiply(360.0 / @channels)

    keep = Nx.logical_and(Nx.greater_equal(value, threshold), Nx.greater_equal(value, peak))

    [keep, value, angle]
    |> Enum.map(&Nx.to_flat_list/1)
    |> Enum.zip_with(& &1)
    |> Enum.with_index()
    |> Enum.flat_map(fn
      {[1, v, theta], i} when div(i, w) < rows and rem(i, w) < cols ->
        [
          %{
            x: cell_centre(rem(i, w)),
            y: cell_centre(div(i, w)),
            angle: wrap(theta),
            quality: quality(v),
            value: v
          }
        ]

      _ ->
        []
    end)
    |> Enum.sort_by(& &1.value, :desc)
    |> suppress()
    |> Enum.map(&Map.delete(&1, :value))
  end

  # FlareNx feeds FDD every other pixel (its scale-0.5 warp samples 500
  # ppi pixel 2u for input pixel u), and map cell j stands for input
  # pixels 2j and 2j + 1: 500 ppi pixels 4j and 4j + 2.
  defp cell_centre(j), do: @cell_px * j + 1.0

  defp quality(v), do: v |> Kernel.*(100) |> round() |> min(100) |> max(0)

  # Each cell's value in its own channel `index` (`{h, w}`).
  defp at_channel(tensor, index) do
    tensor |> Nx.take_along_axis(Nx.new_axis(index, 0), axis: 0) |> Nx.squeeze(axes: [0])
  end

  # Channels shifted circularly by `n`.
  defp roll(tensor, n) do
    n = rem(n + @channels, @channels)
    Nx.concatenate([tensor[n..-1//1], tensor[0..(n - 1)//1]], axis: 0)
  end

  defp suppress(minutiae) do
    minutiae
    |> Enum.reduce([], fn m, kept ->
      if Enum.any?(kept, &too_close?(m, &1)), do: kept, else: [m | kept]
    end)
    |> Enum.reverse()
  end

  defp too_close?(a, b) do
    d = :math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    d < 10 or (d < 20 and abs(wrap180(a.angle - b.angle)) < 45)
  end

  defp wrap(angle), do: angle - 360.0 * Float.floor(angle / 360.0)
  defp wrap180(angle), do: angle - 360.0 * Float.floor((angle + 180.0) / 360.0)

  @doc """
  `minutiae` as `bozorth3`'s `.xyt` (`Matchers.Nbis`, "Coordinates and
  angles"): origin at the bottom left of an image `height` tall, `theta`
  counter-clockwise as seen and opposite to ANSI's direction; minutiae
  without a `:quality` (ground truth) get 100.
  """
  @spec to_xyt([minutia()], pos_integer()) :: String.t()
  def to_xyt(minutiae, height) do
    Enum.map_join(minutiae, fn m ->
      "#{round(m.x)} #{height - round(m.y)} #{rem(round(540 - m.angle), 360)} #{Map.get(m, :quality, 100)}\n"
    end)
  end
end
