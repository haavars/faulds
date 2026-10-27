defmodule Faulds.Matchers.NbisTest do
  use ExUnit.Case, async: true

  alias Faulds.Formats.Png
  alias Faulds.Matchers.Nbis

  # A synthetic rolled print, 500 ppi.
  @image "test/fixtures/synthetic/rolled_fgp02.png"
  @mindtct Application.app_dir(:faulds, "priv/nbis") <> "/bin/mindtct"

  @moduletag skip:
               (cond do
                  not File.exists?(@mindtct) ->
                    "NBIS not built — run `mix faulds.setup_nbis`"

                  true ->
                    false
                end)

  test "extract/1 produces a reusable non-empty template" do
    {:ok, template} = Nbis.extract(Png.read!(@image))

    assert is_binary(template)
    assert byte_size(template) > 0
  end

  test "score/2 clearly separates a genuine pair from an impostor pair" do
    image = Png.read!(@image)
    {h, w} = Nx.shape(image)
    # Genuine: a shifted crop of the same print. Impostor: its mirror image.
    crop = Nx.slice(image, [40, 30], [h - 40, w - 30])
    {:ok, template_a1} = Nbis.extract(image)
    {:ok, template_a2} = Nbis.extract(crop)
    {:ok, template_b1} = Nbis.extract(Nx.reverse(image, axes: [1]))

    {:ok, genuine_score} = Nbis.score(template_a1, template_a2)
    {:ok, impostor_score} = Nbis.score(template_a1, template_b1)

    assert genuine_score > impostor_score
    assert genuine_score > 40.0
  end
end
