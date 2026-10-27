defmodule Faulds.Matchers.FlareNxVariantsTest do
  # async: false -- starts the globally-named FlareNx GenServer.
  use ExUnit.Case, async: false

  alias Faulds.Formats.Png
  alias Faulds.Matchers.{Flare, FlareNx}

  @moduletag timeout: 120_000

  # A synthetic rolled print, 500 ppi.
  @image "test/fixtures/synthetic/rolled_fgp02.png"
  @pose_onnx Application.app_dir(:faulds, "priv/flare") <> "/pose.onnx"
  @fdd_onnx Application.app_dir(:faulds, "priv/flare") <> "/fdd.onnx"
  @full_onnx ~w(regression_pose.onnx unetenh.onnx)
             |> Enum.map(&(Application.app_dir(:faulds, "priv/flare") <> "/" <> &1))

  @moduletag skip:
               (cond do
                  not File.exists?(@pose_onnx) -> "FlareNx ONNX models not present"
                  not File.exists?(@fdd_onnx) -> "FlareNx ONNX models not present"
                  true -> false
                end)

  setup do
    start_supervised!(FlareNx)
    {:ok, image: Png.read!(@image)}
  end

  defp same_template?(a, b) do
    {:ok, score} = Flare.score(a, b)
    score > 0.9999
  end

  test "extract_variants/2 with no options is extract/1's template", %{image: image} do
    {:ok, template} = FlareNx.extract(image)
    assert [variant] = FlareNx.extract_variants(image)
    assert same_template?(variant, template)
  end

  test "template_at_pose/2 at the estimated pose is extract/1's template", %{image: image} do
    {:ok, template} = FlareNx.extract(image)
    assert same_template?(FlareNx.template_at_pose(image, FlareNx.pose(image)), template)
  end

  test "one template per combination, the unshifted unrotated one first", %{image: image} do
    variants =
      FlareNx.extract_variants(image, theta_offsets: [0, 30], shifts: [{0, 0}, {0, -128}])

    assert length(variants) == 4
    {:ok, template} = FlareNx.extract(image)
    assert same_template?(hd(variants), template)
    refute same_template?(Enum.at(variants, 1), template)
  end

  @tag skip:
         if(Enum.all?(@full_onnx, &File.exists?/1),
           do: false,
           else: "FLARE full models not exported"
         )
  test "poses and enhancers: one template per combination, VotingPose on the image first",
       %{image: image} do
    variants =
      FlareNx.extract_variants(image, poses: [:voting, :regression], enhancers: [:none, :unet])

    assert length(variants) == 4
    {:ok, template} = FlareNx.extract(image)
    assert same_template?(hd(variants), template)

    for variant <- tl(variants), do: refute(same_template?(variant, template))

    # A clear rolled print: enhancing it, or the other pose estimator,
    # still gives a template much like the plain one.
    for variant <- tl(variants) do
      {:ok, score} = Flare.score(variant, template)
      assert score > 0.5
    end
  end
end
