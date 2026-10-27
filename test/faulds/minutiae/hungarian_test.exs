defmodule Faulds.Minutiae.HungarianTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.Hungarian

  defp random(rows, cols, seed) do
    :rand.seed(:exsss, {seed, seed, seed})
    for _ <- 1..rows, do: for(_ <- 1..cols, do: :rand.uniform())
  end

  defp total(cost, pairs), do: Enum.sum(for {i, j} <- pairs, do: cost |> Enum.at(i) |> Enum.at(j))

  # Every injective pairing of the shorter side into the longer.
  defp brute_force(cost) do
    {rows, cols} = {length(cost), length(hd(cost))}

    if rows <= cols do
      for perm <- permutations(Enum.to_list(0..(cols - 1)), rows),
          do: total(cost, Enum.with_index(perm, fn j, i -> {i, j} end))
    else
      for perm <- permutations(Enum.to_list(0..(rows - 1)), cols),
          do: total(cost, Enum.with_index(perm, fn i, j -> {i, j} end))
    end
    |> Enum.min()
  end

  defp permutations(_, 0), do: [[]]

  defp permutations(items, k),
    do: for(x <- items, rest <- permutations(items -- [x], k - 1), do: [x | rest])

  test "finds the least total cost, wide, tall and square" do
    for {{rows, cols}, seed} <- Enum.with_index([{3, 5}, {5, 3}, {4, 4}, {1, 4}, {4, 1}], 1) do
      cost = random(rows, cols, seed)
      pairs = Hungarian.solve(cost)

      assert length(pairs) == min(rows, cols)
      assert pairs |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> length() == length(pairs)
      assert pairs |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length() == length(pairs)
      assert_in_delta total(cost, pairs), brute_force(cost), 1.0e-9
    end
  end

  test "an empty matrix has no pairs" do
    assert Hungarian.solve([]) == []
    assert Hungarian.solve([[]]) == []
  end
end
