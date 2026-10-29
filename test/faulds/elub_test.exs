defmodule Faulds.ElubTest do
  use ExUnit.Case, async: true

  alias Faulds.Elub

  # Expected values from lir 1.8.0's own `elub` and
  # `calculate_expected_utility` (numpy), run on the same lists.
  defp assert_bounds(targets, non_targets, {lower, upper}) do
    {l, u} = Elub.bounds(targets, non_targets)
    assert_in_delta l, lower, 1.0e-9
    assert_in_delta u, upper, 1.0e-9
  end

  test "matches lir on a small mixed set" do
    assert_bounds(
      [-0.5037, 0.2037, 0.8037, 1.1037, 1.5037, 2.0037, 2.4037, 3.0037, 4.5037, 6.0037],
      [-3.0037, -2.5037, -2.0037, -1.6037, -1.2037, -0.9037, -0.4037, 0.1037, 0.3037] ++
        [-5.0037, -8.0037, -1.0037, -0.7037, 0.6037, -2.2037],
      {-0.83, 1.0}
    )
  end

  test "matches lir on overlapping evenly spaced classes" do
    assert_bounds(
      for(i <- 0..49, do: -1.0037 + 0.1 * i),
      for(i <- 0..199, do: -4.0037 + 0.03 * i),
      {-1.35, 0.41}
    )
  end

  test "matches lir with strongly misleading trials on both sides" do
    assert_bounds(
      List.flatten(List.duplicate([-2.5037, 1.0037, 2.0037, 3.0037], 5)),
      [-3.0037, -1.0037, 2.5037] ++ List.duplicate(-2.0037, 30),
      {-0.51, 1.0}
    )
  end

  test "separated classes are bounded by the number of trials" do
    targets = for i <- 0..39, do: 5.0037 + 0.01 * i
    non_targets = for i <- 0..299, do: -5.0037 - 0.01 * i

    assert_bounds(targets, non_targets, {-1.61, 2.46})
    {lower, upper} = Elub.bounds(targets, non_targets)
    assert lower >= -:math.log10(41)
    assert upper <= :math.log10(301)
  end
end
