defmodule Faulds.Matchers.NbisFrameTest do
  # `Nbis.minutiae/1`'s frame (see its moduledoc's "Coordinates and
  # angles"), checked against exact ground truth: a synthetic rolled print
  # with every minutia's position and direction (`test/fixtures/synthetic/`).
  use ExUnit.Case, async: true

  alias Faulds.Formats.Png
  alias Faulds.Matchers.Nbis

  @fixture "test/fixtures/synthetic/rolled_fgp02"
  @mindtct Application.app_dir(:faulds, "priv/nbis") <> "/bin/mindtct"

  @moduletag skip:
               if(File.exists?(@mindtct),
                 do: false,
                 else: "NBIS not built — run `mix faulds.setup_nbis`"
               )

  test "minutiae/1 puts mindtct's minutiae where they are in the image, pointing along the ridge" do
    truth = "#{@fixture}.json" |> File.read!() |> Jason.decode!()
    {:ok, detected} = "#{@fixture}.png" |> File.read!() |> Png.decode!() |> Nbis.minutiae()

    paired =
      for d <- detected,
          t = nearest(d, truth["minutiae"]),
          :math.sqrt((d.x - t["x"]) ** 2 + (d.y - t["y"]) ** 2) <= 8,
          do: {d, t}

    # About three in four lie on a true minutia (with y as `.xyt` writes
    # it, under one in twenty would).
    assert length(paired) >= 0.6 * length(detected)

    # The truth's angles are counter-clockwise as seen; this library's are in
    # image coordinates (y down), so clockwise as seen.
    errors =
      for {d, t} <- paired, do: angle_difference(d.angle, -t["angle"])

    assert median(errors) <= 20
  end

  defp nearest(d, minutiae),
    do: Enum.min_by(minutiae, fn t -> (d.x - t["x"]) ** 2 + (d.y - t["y"]) ** 2 end)

  defp angle_difference(a, b) do
    diff = :math.fmod(a - b, 360.0)
    diff = if diff < 0, do: diff + 360.0, else: diff
    min(diff, 360.0 - diff)
  end

  defp median(values), do: values |> Enum.sort() |> Enum.at(div(length(values), 2))
end
