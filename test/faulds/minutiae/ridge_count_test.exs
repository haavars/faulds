defmodule Faulds.Minutiae.RidgeCountTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.RidgeCount

  setup do
    Nx.default_backend({EXLA.Backend, client: :host})
    :ok
  end

  # Vertical ridges, dark, one every `period` px.
  defp ridges(period, size \\ 300) do
    x = Nx.iota({size, size}, axis: 1, type: :f32)
    x |> Nx.multiply(2 * :math.pi() / period) |> Nx.cos() |> Nx.multiply(0.4) |> Nx.add(0.5)
  end

  test "counts the ridges a segment crosses, none along a ridge, unknown when far" do
    minutiae = [%{x: 100, y: 100}, %{x: 150, y: 100}, %{x: 100, y: 160}, %{x: 250, y: 100}]
    rc = ridges(10) |> RidgeCount.matrix(minutiae) |> Nx.to_list()

    # 50 px across ridges 10 px apart, less the 3 px margins: 4 or 5.
    assert Enum.at(Enum.at(rc, 0), 1) in 4..5
    assert Enum.at(Enum.at(rc, 0), 2) == 0
    # 150 px apart: beyond 120.
    assert Enum.at(Enum.at(rc, 0), 3) == 255
    assert Enum.at(Enum.at(rc, 0), 0) == 255
    assert rc == Enum.zip_with(rc, & &1)
  end

  test "twice the ridge frequency, about twice the count" do
    minutiae = [%{x: 60, y: 100}, %{x: 160, y: 130}]
    [[_, coarse], _] = ridges(12) |> RidgeCount.matrix(minutiae) |> Nx.to_list()
    [[_, fine], _] = ridges(6) |> RidgeCount.matrix(minutiae) |> Nx.to_list()

    # 100 px across: 8 periods of 12 px, 16 of 6.
    assert coarse in 7..9
    assert fine in 15..17
  end

  test "more minutiae than one block, and none" do
    minutiae = for i <- 0..39, do: %{x: 40 + 5 * i, y: 50 + rem(i, 7) * 20}
    assert Nx.shape(RidgeCount.matrix(ridges(10), minutiae)) == {40, 40}
    assert RidgeCount.matrix(ridges(10), []) == nil
  end
end
