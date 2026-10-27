defmodule Faulds.ResampleTest do
  use ExUnit.Case, async: true

  alias Faulds.Resample

  defp ramp(height, width) do
    Nx.iota({height, width}, type: :f32) |> Nx.divide(height * width)
  end

  test "leaves a 500 ppi image, or one within 2% of it, untouched" do
    tensor = ramp(30, 40)

    assert Resample.to_500_ppi(tensor, 500) == tensor
    assert Resample.to_500_ppi(tensor, 500.38) == tensor
    assert Resample.output_ppi(500.38) == 500.38
  end

  test "halves a 1000 ppi image (NISTIR 7839) and keeps a flat background flat to the edges" do
    white = Nx.broadcast(Nx.tensor(1.0, type: :f32), {60, 81})

    halved = Resample.to_500_ppi(white, 1000)

    assert Nx.shape(halved) == {30, 40}
    assert Nx.to_number(Nx.reduce_min(halved)) > 0.999
    assert Resample.output_ppi(1000) == 500.0
  end

  test "the 1000 ppi path low-passes: a one-pixel checkerboard averages out to mid-grey" do
    checker =
      Nx.iota({64, 64})
      |> then(&Nx.remainder(Nx.add(Nx.quotient(&1, 64), Nx.remainder(&1, 64)), 2))
      |> Nx.as_type(:f32)

    halved = Resample.to_500_ppi(checker, 1000)

    inner = halved[[4..-5//1, 4..-5//1]]
    assert_in_delta Nx.to_number(Nx.reduce_min(inner)), 0.5, 0.05
    assert_in_delta Nx.to_number(Nx.reduce_max(inner)), 0.5, 0.05
  end

  test "resizes any other resolution by 500 / ppi" do
    resized = Resample.to_500_ppi(ramp(300, 450), 750)

    assert Nx.shape(resized) == {200, 300}
    assert Resample.output_ppi(750) == 500.0
  end
end
