defmodule Faulds.Minutiae.Hungarian do
  @moduledoc """
  The assignment problem, exactly: the one-to-one pairing of rows with
  columns of a cost matrix with the least total cost, `min(rows, cols)`
  pairs (Kuhn–Munkres with potentials, the O(n²m) shortest augmenting
  path form). What DMD's LSA-R pairs latent and rolled minutiae by
  (`Matchers.Dmd`, its `:hungarian` assignment); slow in Elixir, so it
  checks the batched greedy assignment rather than replacing it.
  """

  @inf 1.0e18

  @doc """
  The least-cost pairs of `cost` (a list of rows of numbers) as
  `{row, col}` (0-based), sorted by row.
  """
  @spec solve([[number()]]) :: [{non_neg_integer(), non_neg_integer()}]
  def solve([]), do: []
  def solve([[] | _]), do: []

  def solve(cost) do
    rows = length(cost)
    cols = cost |> hd() |> length()

    if rows <= cols do
      cost |> solve_wide(rows, cols) |> Enum.sort()
    else
      cost
      |> Enum.zip_with(& &1)
      |> solve_wide(cols, rows)
      |> Enum.map(fn {c, r} -> {r, c} end)
      |> Enum.sort()
    end
  end

  # n ≤ m. Arrays are 1-based, index 0 the algorithm's sentinel.
  defp solve_wide(cost, n, m) do
    a = cost |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    at = fn i, j -> a |> elem(i - 1) |> elem(j - 1) end
    zeros = fn k -> :array.new(k + 1, default: 0.0) end

    {_u, _v, p} =
      Enum.reduce(1..n, {zeros.(n), zeros.(m), :array.new(m + 1, default: 0)}, fn i, {u, v, p} ->
        p = :array.set(0, i, p)
        minv = :array.new(m + 1, default: @inf)
        used = :array.new(m + 1, default: false)
        way = :array.new(m + 1, default: 0)
        {u, v, way, free} = augment(at, m, {u, v, p, way, minv, used}, 0)
        {u, v, unwind(p, way, free)}
      end)

    for j <- 1..m, (i = :array.get(j, p)) != 0, do: {i - 1, j - 1}
  end

  # Grows the alternating tree from row p[0], adjusting the potentials,
  # until it reaches a free column: `{u, v, way, that column}`.
  defp augment(at, m, {u, v, p, way, minv, used}, j0) do
    used = :array.set(j0, true, used)
    i0 = :array.get(j0, p)

    {minv, way, delta, j1} =
      Enum.reduce(1..m, {minv, way, @inf, 0}, fn j, {minv, way, delta, j1} = acc ->
        if :array.get(j, used) do
          acc
        else
          cur = at.(i0, j) - :array.get(i0, u) - :array.get(j, v)

          {minv, way} =
            if cur < :array.get(j, minv),
              do: {:array.set(j, cur, minv), :array.set(j, j0, way)},
              else: {minv, way}

          mj = :array.get(j, minv)
          if mj < delta, do: {minv, way, mj, j}, else: {minv, way, delta, j1}
        end
      end)

    {u, v, minv} =
      Enum.reduce(0..m, {u, v, minv}, fn j, {u, v, minv} ->
        if :array.get(j, used) do
          pj = :array.get(j, p)

          {:array.set(pj, :array.get(pj, u) + delta, u),
           :array.set(j, :array.get(j, v) - delta, v), minv}
        else
          {u, v, :array.set(j, :array.get(j, minv) - delta, minv)}
        end
      end)

    if :array.get(j1, p) == 0,
      do: {u, v, way, j1},
      else: augment(at, m, {u, v, p, way, minv, used}, j1)
  end

  # Flips the augmenting path back from the free column: each column on
  # it takes the row of the one before (p[0], the new row, last).
  defp unwind(p, _way, 0), do: p

  defp unwind(p, way, j0) do
    j1 = :array.get(j0, way)
    unwind(:array.set(j0, :array.get(j1, p), p), way, j1)
  end
end
