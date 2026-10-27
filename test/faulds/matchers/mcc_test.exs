defmodule Faulds.Matchers.MccTest do
  # Not async: EXLA compilations (see DmdScoreTest).
  use ExUnit.Case, async: false

  alias Faulds.Matchers.Mcc

  defp minutiae(n, seed) do
    :rand.seed(:exsss, {seed, seed, seed})

    for _ <- 1..n,
        do: %{
          x: 100 + :rand.uniform() * 250,
          y: 100 + :rand.uniform() * 300,
          angle: :rand.uniform() * 360
        }
  end

  # Turned by `deg` clockwise as seen (y down) about the origin, then moved.
  defp moved(ms, deg, {tx, ty}) do
    phi = deg * :math.pi() / 180
    {c, s} = {:math.cos(phi), :math.sin(phi)}
    for m <- ms, do: %{x: c * m.x - s * m.y + tx, y: s * m.x + c * m.y + ty, angle: m.angle + deg}
  end

  test "a cylinder's circle holds 208 cells per layer, as the paper's" do
    t = Mcc.cylinders(minutiae(40, 1))
    # Cells valid inside a large hull are exactly the in-circle ones for a
    # central minutia; count the most-valid cylinder's.
    most = t.valid |> Nx.sum(axes: [1]) |> Nx.reduce_max() |> Nx.to_number()
    assert most == 208 * 6
  end

  test "moving the print rigidly leaves its cylinders unchanged" do
    ms = minutiae(40, 2)
    a = Mcc.cylinders(ms)
    b = Mcc.cylinders(moved(ms, 33.0, {40.0, -15.0}))

    assert Nx.shape(a.cells) == Nx.shape(b.cells)
    agree = Nx.equal(a.cells, b.cells) |> Nx.mean() |> Nx.to_number()
    assert agree > 0.99
  end

  test "a moved copy with noise scores far above an unrelated print" do
    probe = minutiae(25, 3)

    copy =
      probe
      |> moved(-20.0, {30.0, 10.0})
      |> Enum.map(&%{&1 | x: &1.x + :rand.normal() * 2, angle: &1.angle + :rand.normal() * 5})

    gallery =
      Enum.map([copy ++ minutiae(15, 4), minutiae(40, 5), minutiae(30, 6)], &Mcc.cylinders/1)

    [mated | others] = Mcc.score_many(Mcc.cylinders(probe), gallery)
    assert mated > 0.4
    assert Enum.all?(others, &(&1 < mated / 2))
  end

  test "too few minutiae give no cylinders and score 0" do
    empty = Mcc.cylinders(minutiae(2, 7))
    assert empty.minutiae == nil
    assert Mcc.score_many(empty, [Mcc.cylinders(minutiae(30, 8))]) == [0.0]
  end
end
