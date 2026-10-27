defmodule Faulds.Minutiae.RidgeCount do
  @moduledoc """
  The number of ridges crossed between every two minutiae of a print, read
  off the image, for ridge-count agreement in LSA-R's relaxation
  (`Matchers.Dmd`'s `ridges: true`).

  Along the segment from minutia k to minutia l, a band of the image (500
  ppi, `[0, 1]`) `2 · :band + 1` px wide (default 5) is sampled every
  pixel and averaged across
  its width, smoothed along its length (`[1, 4, 6, 4, 1] / 16`), and
  compared with its own running mean over 19 px (about two ridge
  periods). Each fall from above that mean to below it (with
  hysteresis: past `:dead`, default a quarter, of the local contrast), between 3 px after
  k and 3 px before l, is a dark band entered: one ridge crossed
  for dark ridges, one valley for light ones, so polarity moves a count
  by at most one, the same on both sides of a comparison. Pairs further
  apart than `:max_distance` (default 120 px) count as unknown (255):
  long segments cross creases, scars and the latent's edge.

  No minutia's own ridge is excluded, as a ridge count by an examiner
  would: the count only has to agree between a latent and its mate.
  """

  import Nx.Defn

  @max_distance 120
  @band 2
  @dead 0.25
  @margin 3
  @half_window 9
  @floor 0.005
  @unknown 255
  @block 32

  @doc """
  `{n, n}` ridge counts (`:u8`, symmetric, 255 on the diagonal and for
  unknown pairs) between `minutiae` (`%{x:, y:}` in pixels from the top
  left) on `image`; nil for no minutiae.
  """
  @spec matrix(Nx.Tensor.t(), [map()], keyword()) :: Nx.Tensor.t() | nil
  def matrix(image, minutiae, opts \\ [])
  def matrix(_image, [], _opts), do: nil

  def matrix(image, minutiae, opts) do
    max_distance = Keyword.get(opts, :max_distance, @max_distance)
    band = Keyword.get(opts, :band, @band)
    dead = Keyword.get(opts, :dead, @dead)
    n = length(minutiae)
    # Padded to a multiple of 32, so EXLA compiles a few shapes.
    padded = max(@block, div(n + @block - 1, @block) * @block)
    xy = Enum.map(minutiae, &[&1.x * 1.0, &1.y * 1.0])
    xy = Nx.tensor(xy ++ List.duplicate(hd(xy), padded - n), type: :f32)
    # The image padded to a multiple of 128 each way, for the same reason;
    # samples are clamped to its true size.
    {h, w} = Nx.shape(image)
    size = Nx.tensor([h, w], type: :f32)

    image =
      image
      |> Nx.as_type(:f32)
      |> Nx.pad(0.0, [{0, round_up(h) - h, 0}, {0, round_up(w) - w, 0}])

    0..(padded - 1)//@block
    |> Enum.map(fn start ->
      Nx.Defn.jit_apply(
        &counts_n(&1, &2, &3, &4, max_distance: max_distance, band: band, dead: dead),
        [image, size, xy, Nx.tensor(start, type: :s32)],
        compiler: EXLA
      )
    end)
    |> Nx.concatenate()
    |> Nx.slice([0, 0], [n, n])
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  defp round_up(x), do: div(x + 127, 128) * 128

  # Rows `start..start + 31` of the count matrix: `{32, n}`; `size` the
  # image's true `[h, w]` within its padding.
  defn counts_n(image, size, xy, start, opts \\ []) do
    max_distance = opts[:max_distance]
    {n, 2} = Nx.shape(xy)
    steps = samples(opts)

    from = Nx.slice_along_axis(xy, start, @block, axis: 0)
    dx = Nx.new_axis(xy[[.., 0]], 0) - Nx.new_axis(from[[.., 0]], 1)
    dy = Nx.new_axis(xy[[.., 1]], 0) - Nx.new_axis(from[[.., 1]], 1)
    d = Nx.sqrt(dx * dx + dy * dy)
    ux = dx / Nx.max(d, 1.0e-6)
    uy = dy / Nx.max(d, 1.0e-6)

    s = Nx.iota({1, 1, steps, 1}, type: :f32)
    o = band_offsets(opts)
    expand = &Nx.reshape(&1, {@block, n, 1, 1})

    x0 = Nx.broadcast(Nx.new_axis(from[[.., 0]], 1), {@block, n})
    y0 = Nx.broadcast(Nx.new_axis(from[[.., 1]], 1), {@block, n})
    # Along the segment by s, across it by o.
    px = expand.(x0) + s * expand.(ux) - o * expand.(uy)
    py = expand.(y0) + s * expand.(uy) + o * expand.(ux)

    profile = image |> bilinear(px, py, size) |> Nx.mean(axes: [3]) |> smooth()
    centred = profile - running_mean(profile)
    # Hysteresis: a sample is above or below only past a quarter of the
    # local contrast (and a floor), so a flat profile, along a ridge,
    # never crosses.
    dead = opts[:dead] * running_mean(Nx.abs(centred)) + @floor
    count = falls(centred, dead, d)

    rows = Nx.iota({@block, n}, axis: 0) + start
    cols = Nx.iota({@block, n}, axis: 1)
    known = d <= max_distance and rows != cols
    Nx.select(known, Nx.min(count, @unknown - 1), @unknown) |> Nx.as_type(:u8)
  end

  # Falls from above to below, sample by sample, from the last side the
  # profile was clearly on; counted when inside the margins. That side is
  # a forward fill, the side of the last decided sample, so it is taken
  # by a cumulative max of their indices, not a loop over the samples
  # (which compiled for 16 s).
  defnp falls(centred, dead, d) do
    {b, n, steps} = Nx.shape(centred)
    side = Nx.select(centred > dead, 1, Nx.select(centred < -dead, -1, 0))
    t = Nx.iota({1, 1, steps}, axis: 2)
    last = Nx.cumulative_max(Nx.select(side != 0, t, -1), axis: 2)
    now = Nx.select(last >= 0, Nx.take_along_axis(side, Nx.max(last, 0), axis: 2), 0)
    before = Nx.concatenate([Nx.broadcast(0, {b, n, 1}), now[[.., .., 0..-2//1]]], axis: 2)
    inside = t >= @margin and t <= Nx.new_axis(d, 2) - @margin
    fell = before == 1 and now == -1 and inside
    fell |> Nx.as_type(:s32) |> Nx.sum(axes: [2])
  end

  deftransformp(samples(opts), do: opts[:max_distance] + @half_window + 1)

  deftransformp band_offsets(opts) do
    offsets = Enum.map(-opts[:band]..opts[:band], &(&1 * 1.0))
    Nx.reshape(Nx.tensor(offsets, type: :f32), {1, 1, 1, length(offsets)})
  end

  # Bilinear samples of `image` at `{x, y}`, clamped to its true edge.
  defnp bilinear(image, px, py, size) do
    {h, w} = Nx.shape(image)
    x = Nx.clip(px, 0, size[1] - 1.001)
    y = Nx.clip(py, 0, size[0] - 1.001)
    x0 = Nx.floor(x)
    y0 = Nx.floor(y)
    fx = x - x0
    fy = y - y0
    x0 = Nx.as_type(x0, :s32)
    y0 = Nx.as_type(y0, :s32)
    flat = Nx.reshape(image, {h * w})
    at = fn yy, xx -> Nx.take(flat, yy * w + xx) end

    (1 - fy) * ((1 - fx) * at.(y0, x0) + fx * at.(y0, x0 + 1)) +
      fy * ((1 - fx) * at.(y0 + 1, x0) + fx * at.(y0 + 1, x0 + 1))
  end

  # `[1, 4, 6, 4, 1] / 16` along the last axis, the ends repeated.
  defnp smooth(p) do
    {_, _, steps} = Nx.shape(p)

    padded =
      Nx.concatenate(
        [p[[.., .., 0..0]], p[[.., .., 0..0]], p, p[[.., .., -1..-1]], p[[.., .., -1..-1]]],
        axis: 2
      )

    at = fn k -> Nx.slice_along_axis(padded, k, steps, axis: 2) end
    (at.(0) + 4 * at.(1) + 6 * at.(2) + 4 * at.(3) + at.(4)) / 16
  end

  # The mean over the 19 samples centred on each (fewer at the ends).
  defnp running_mean(p) do
    {b, n, steps} = Nx.shape(p)
    zero = Nx.broadcast(0.0, {b, n, 1})
    c = Nx.concatenate([zero, Nx.cumulative_sum(p, axis: 2)], axis: 2)
    t = Nx.iota({1, 1, steps})
    lo = Nx.max(t - @half_window, 0)
    hi = Nx.min(t + @half_window + 1, steps)
    take = fn idx -> Nx.take_along_axis(c, Nx.broadcast(idx, {b, n, steps}), axis: 2) end
    (take.(hi) - take.(lo)) / (hi - lo)
  end
end
