defmodule Faulds.Minutiae.Foreground do
  @moduledoc """
  Drops minutiae a latent's background or low-quality areas produce,
  before minutiae matching (`bozorth3`) or alignment
  (`Minutiae.Alignment`): `mindtct` finds about 60 minutiae in a typical
  synthetic latent, of which about a third are real.

  Two tests, either or both:

    * FlareNx's foreground mask: the 16×16 grid (row-major) over FlareNx's
      aligned 256×256 crop, which shows the image at half scale around
      the pose `{x, y, theta}`. A minutia is kept when its cell's mask
      value is above `:mask_threshold`.
    * `mindtct`'s own quality (the `.xyt` fourth column, 0–100), at least
      `:min_quality`.

  Works on `.xyt` text in `mindtct`'s own frame (origin bottom left, see
  `Minutiae.Alignment.from_xyt/3`) and returns it in that frame, so the
  result goes straight back to `bozorth3`.
  """

  @crop 256
  @cells 16
  @scale 0.5

  @doc """
  The rows of `xyt` that pass. `height` is the image's height (to map
  `.xyt`'s y into the image frame); `pose` and `mask` (256 values, as in
  a FlareNx template) are only needed with `:mask_threshold`.
  """
  @spec filter_xyt(String.t(), pos_integer(), keyword()) :: String.t()
  def filter_xyt(xyt, height, opts) do
    min_quality = Keyword.get(opts, :min_quality, 0)
    mask_threshold = Keyword.get(opts, :mask_threshold)
    pose = Keyword.get(opts, :pose)
    mask = opts |> Keyword.get(:mask) |> mask_tuple()

    xyt
    |> String.split("\n", trim: true)
    |> Enum.filter(fn line ->
      [x, y, _theta | rest] = String.split(line)
      quality = rest |> List.first("100") |> String.to_integer()
      point = {String.to_integer(x) * 1.0, height - String.to_integer(y) * 1.0}

      quality >= min_quality and
        (mask_threshold == nil or mask_value(mask, point, pose) > mask_threshold)
    end)
    |> Enum.map_join(&(&1 <> "\n"))
  end

  @doc """
  The FlareNx mask value at image point `{x, y}` for a crop at `pose`
  (`{cx, cy, theta_deg}`): the point's place in the crop is
  `0.5 R(theta)(p - c) + 128`; 0.0 outside the crop.
  """
  @spec mask_value(tuple() | [number()], {number(), number()}, {number(), number(), number()}) ::
          float()
  def mask_value(mask, {px, py}, {cx, cy, theta_deg}) do
    mask = mask_tuple(mask)
    theta = theta_deg * :math.pi() / 180.0
    {dx, dy} = {px - cx, py - cy}
    u = @scale * (:math.cos(theta) * dx - :math.sin(theta) * dy) + @crop / 2
    v = @scale * (:math.sin(theta) * dx + :math.cos(theta) * dy) + @crop / 2
    cell = div(@crop, @cells)
    {col, row} = {floor(u / cell), floor(v / cell)}

    if col in 0..(@cells - 1) and row in 0..(@cells - 1),
      do: elem(mask, row * @cells + col),
      else: 0.0
  end

  defp mask_tuple(nil), do: nil
  defp mask_tuple(mask) when is_tuple(mask), do: mask
  defp mask_tuple(%Nx.Tensor{} = mask), do: mask |> Nx.to_flat_list() |> List.to_tuple()
  defp mask_tuple(mask) when is_list(mask), do: List.to_tuple(mask)
end
