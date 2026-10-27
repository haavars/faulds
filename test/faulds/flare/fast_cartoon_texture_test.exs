defmodule Faulds.Flare.FastCartoonTextureTest do
  use ExUnit.Case, async: true

  alias Faulds.Flare.FastCartoonTexture

  # Fixture generated once from real eager PyTorch
  # (`models.units.FastCartoonTexture(sigma=2.5)`, the exact class this
  # module ports) on a fixed-seed random 32x32 image. Measured max abs diff is
  # ~4.0e-4 on a 0-255 scale (float32 FFT rounding, not a math
  # discrepancy) — 0.01 leaves ample headroom while still catching a
  # real algorithmic divergence.
  @fixture_path "test/fixtures/flare_fast_cartoon_texture.json"

  test "matches real eager-PyTorch FastCartoonTexture(sigma=2.5) output" do
    fixture = @fixture_path |> File.read!() |> Jason.decode!()

    input = Nx.tensor(fixture["input"], type: :f32)
    expected = Nx.tensor(fixture["output"], type: :f32)

    actual = FastCartoonTexture.forward(input)

    assert Nx.shape(actual) == Nx.shape(expected)

    max_abs_diff =
      actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert max_abs_diff < 0.01, "max abs diff #{max_abs_diff} exceeds tolerance"
  end

  test "output is clamped to [0, 255]" do
    rows = for _ <- 1..32, do: for(_ <- 1..32, do: :rand.uniform() * 255)
    image = Nx.tensor(rows, type: :f32)
    output = FastCartoonTexture.forward(image)

    assert Nx.to_number(Nx.reduce_min(output)) >= 0.0
    assert Nx.to_number(Nx.reduce_max(output)) <= 255.0
  end

  test "output shape matches input shape" do
    image = Nx.broadcast(128.0, {64, 48})
    output = FastCartoonTexture.forward(image)

    assert Nx.shape(output) == {64, 48}
  end
end
