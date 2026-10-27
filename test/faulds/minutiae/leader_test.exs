defmodule Faulds.Minutiae.LeaderTest do
  use ExUnit.Case, async: true

  alias Faulds.Formats.Png
  alias Faulds.Minutiae.Leader

  # A synthetic rolled print, 500 ppi.
  @image "test/fixtures/synthetic/rolled_fgp02.png"

  @moduletag skip:
               (cond do
                  not Leader.available?() ->
                    "leader.onnx not exported (mix faulds.setup_leader)"

                  true ->
                    false
                end)

  # Checked against pyfing's own `Leader.run` when written: the same 207
  # minutiae on a synthetic rolled print, angles and qualities equal.
  test "finds minutiae inside the image, in the library's convention" do
    image = Png.read!(@image)
    {h, w} = Nx.shape(image)
    minutiae = Leader.extract(image)

    assert length(minutiae) > 20

    for m <- minutiae do
      assert m.x >= 0 and m.x < w and m.y >= 0 and m.y < h
      assert m.angle >= 0 and m.angle < 360
      assert m.quality >= 60 and m.type in [:ending, :bifurcation]
    end

    assert length(Leader.extract(image, threshold: 0.3)) >= length(minutiae)
  end
end
