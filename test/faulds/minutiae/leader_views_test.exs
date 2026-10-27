defmodule Faulds.Minutiae.LeaderViewsTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.Leader

  defp d(view, x, y, angle, q \\ 90),
    do: %{view: view, x: x * 1.0, y: y * 1.0, angle: angle * 1.0, quality: q, type: :ending}

  test "merge_views/2 keeps minutiae enough views found, at their mean" do
    detections = [
      # Found by three views, one direction across 0°/360°.
      d(0, 100, 100, 358),
      d(1, 102, 100, 4),
      d(2, 101, 103, 0),
      # Found by one view only.
      d(0, 200, 50, 90),
      # Near the first, but the same view again: not a second vote.
      d(0, 103, 101, 2, 80),
      # Near in position, opposite in direction: a different minutia.
      d(3, 100, 101, 180)
    ]

    assert [m] = Leader.merge_views(detections, votes: 3)
    assert m.votes == 3
    assert_in_delta m.x, 101.0, 1.0e-9
    assert_in_delta m.y, 101.0, 1.0e-9
    assert_in_delta min(m.angle, 360 - m.angle), 0.67, 0.1

    assert length(Leader.merge_views(detections, votes: 1)) == 4
  end

  test "top_up/3 keeps the strong detections, or the strongest when too few are" do
    qs = [95, 90, 72, 61, 55, 40, 31]
    detections = Enum.map(qs, &d(0, &1, 0, 0, &1))

    assert detections |> Leader.top_up(3, 0.6) |> Enum.map(& &1.quality) == [95, 90, 72, 61]

    assert detections |> Leader.top_up(6, 0.6) |> Enum.map(& &1.quality) == [
             95,
             90,
             72,
             61,
             55,
             40
           ]

    assert length(Leader.top_up(detections, 12, 0.6)) == 7
  end
end
