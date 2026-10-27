defmodule Faulds.Minutiae.FddMapTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.FddMap

  # A {6, h, w} map with Gaussian blobs: {row, col, channel, peak}, the
  # neighbouring channels at `side` of the peak (an angle between bins).
  defp map(blobs, {h, w} \\ {32, 32}) do
    for c <- 0..5, i <- 0..(h - 1), j <- 0..(w - 1) do
      blobs
      |> Enum.map(fn {bi, bj, bc, peak, sides} ->
        spatial = :math.exp(-((i - bi) ** 2 + (j - bj) ** 2) / 2)

        weight =
          cond do
            c == bc -> 1.0
            c == rem(bc + 1, 6) -> elem(sides, 1)
            c == rem(bc + 5, 6) -> elem(sides, 0)
            true -> 0.0
          end

        peak * spatial * weight
      end)
      |> Enum.sum()
    end
    |> Nx.tensor(type: :f32)
    |> Nx.reshape({6, h, w})
  end

  test "a blob becomes one minutia at its cell, in the image's pixels" do
    assert [m] = FddMap.decode(map([{10, 20, 2, 0.9, {0.2, 0.2}}]))

    assert {m.x, m.y} == {20 * 4 + 1.0, 10 * 4 + 1.0}
    assert_in_delta m.angle, 120.0, 1.0e-3
    assert m.quality == 90
  end

  test "the angle is refined towards the stronger neighbouring channel, circularly" do
    [toward_next] = FddMap.decode(map([{10, 10, 2, 0.9, {0.1, 0.5}}]))
    [past_zero] = FddMap.decode(map([{10, 10, 0, 0.9, {0.5, 0.1}}]))

    assert toward_next.angle > 120 and toward_next.angle < 150
    assert past_zero.angle > 330 and past_zero.angle < 360
  end

  test "below the threshold, or near a stronger minutia, a peak is dropped" do
    # 2 cells (8 px) from a stronger one: inside 10 px.
    blobs = [
      {10, 10, 1, 0.9, {0.2, 0.2}},
      {10, 12, 4, 0.8, {0.2, 0.2}},
      {25, 25, 1, 0.3, {0.2, 0.2}}
    ]

    assert [%{x: 41.0, y: 41.0}] = FddMap.decode(map(blobs))
    assert length(FddMap.decode(map(blobs), threshold: 0.2)) == 2
  end

  test "within 20 px only a similar direction is suppressed" do
    # 4 cells = 16 px apart.
    similar = [{10, 10, 1, 0.9, {0.2, 0.2}}, {10, 14, 1, 0.8, {0.2, 0.2}}]
    opposite = [{10, 10, 1, 0.9, {0.2, 0.2}}, {10, 14, 4, 0.8, {0.2, 0.2}}]

    assert length(FddMap.decode(map(similar))) == 1
    assert length(FddMap.decode(map(opposite))) == 2
  end

  test "to_xyt/2 writes bozorth3's bottom-left frame and direction" do
    xyt = FddMap.to_xyt([%{x: 41.0, y: 10.0, angle: 30.0, quality: 90}], 100)

    # y = 100 - 10; theta = 180 - 30 (`Matchers.Nbis`, "Coordinates and angles")
    assert xyt == "41 90 150 90\n"
  end
end
