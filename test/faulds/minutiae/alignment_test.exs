defmodule Faulds.Minutiae.AlignmentTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.Alignment

  # `count` minutiae spread over a 300x300 px area, deterministically.
  defp minutiae(count, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    for _ <- 1..count, do: {:rand.uniform() * 300, :rand.uniform() * 300, :rand.uniform() * 360}
  end

  defp transform(points, phi, {tx, ty}, sign) do
    {cos, sin} = {:math.cos(phi * :math.pi() / 180), :math.sin(phi * :math.pi() / 180)}

    for {x, y, t} <- points do
      {cos * x - sin * y + tx, sin * x + cos * y + ty, t + sign * phi}
    end
  end

  test "recovers a rotation and translation from a partial, noisy copy plus spurious minutiae" do
    rolled = minutiae(80, 1)
    # The latent: 15 of the rolled minutiae, moved by the inverse of the
    # transform we want back, jittered, plus 10 spurious ones.
    {phi, t} = {35.0, {120.0, -40.0}}
    shared = Enum.take(rolled, 15)

    {cos, sin} = {:math.cos(-phi * :math.pi() / 180), :math.sin(-phi * :math.pi() / 180)}

    latent =
      for {x, y, th} <- shared do
        {dx, dy} = {x - elem(t, 0), y - elem(t, 1)}
        {cos * dx - sin * dy + 1.5, sin * dx + cos * dy - 1.0, th - phi + 3}
      end ++ minutiae(10, 7)

    assert {:ok, %{phi: got_phi, tx: tx, ty: ty, inliers: inliers}} =
             Alignment.align(latent, rolled)

    assert_in_delta got_phi, phi, 2.0
    assert_in_delta tx, 120.0, 6.0
    assert_in_delta ty, -40.0, 6.0
    assert inliers >= 12
  end

  test "honours the extractor's angle convention" do
    a = minutiae(20, 3)
    b = transform(a, -50.0, {30.0, 60.0}, -1)

    assert {:ok, %{phi: phi}} = Alignment.align(a, b, angle_sign: -1)
    assert_in_delta phi, -50.0, 1.0
  end

  test "unrelated prints give no transform" do
    assert Alignment.align(minutiae(12, 11), minutiae(60, 12), min_inliers: 6) == :none
  end

  test "fit/1 is exact for noise-free pairs" do
    a = [{0.0, 0.0}, {10.0, 0.0}, {0.0, 20.0}]

    [{bx0, by0, _}, {bx1, by1, _}, {bx2, by2, _}] =
      transform(Enum.map(a, fn {x, y} -> {x, y, 0} end), 30.0, {5.0, 7.0}, 1)

    fit = Alignment.fit(Enum.zip(a, [{bx0, by0}, {bx1, by1}, {bx2, by2}]))
    assert_in_delta fit.phi, 30.0, 1.0e-9
    assert_in_delta fit.tx, 5.0, 1.0e-9
    assert_in_delta fit.ty, 7.0, 1.0e-9
  end

  test "from_xyt/1 reads mindtct's x y theta quality rows" do
    assert Alignment.from_xyt("10 20 90 50\n30 40 180 12\n") ==
             [{10.0, 20.0, 90.0}, {30.0, 40.0, 180.0}]
  end
end
