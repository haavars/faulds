defmodule Faulds.Metrics do
  @moduledoc """
  Pure functions on plain scores and ranks: DET curves, TAR@FAR at fixed
  operating points, recall at k, Wilson intervals, exact McNemar tests
  and Holm's correction. No dependency on any matcher or on Nx, and each
  testable against hand-computed examples.
  """

  @type det_point :: %{threshold: float(), far: float(), frr: float()}

  @doc """
  The DET curve of raw genuine and impostor scores.

  A two-pointer sweep over both score lists, sorted once: `O(n log n)`,
  so it stays fast for hundreds of thousands of impostor pairs.
  """
  @spec det_curve([float()], [float()]) :: [det_point()]
  def det_curve(genuine_scores, impostor_scores)
      when genuine_scores != [] and impostor_scores != [] do
    n_genuine = length(genuine_scores)
    n_impostor = length(impostor_scores)

    sorted_genuine = Enum.sort(genuine_scores)
    sorted_impostor = Enum.sort(impostor_scores)

    thresholds =
      (genuine_scores ++ impostor_scores)
      |> Enum.uniq()
      |> Enum.sort()

    # A threshold strictly above every score, so the curve reaches the
    # "reject everything" endpoint (far = 0, frr = 1).
    thresholds = thresholds ++ [List.last(thresholds) + 1.0e-9]

    initial = {[], sorted_genuine, sorted_impostor, 0, 0}

    {points, _, _, _, _} =
      Enum.reduce(thresholds, initial, fn threshold,
                                          {acc, gen_rest, imp_rest, gen_below, imp_below} ->
        {newly_below_gen, gen_rest} = Enum.split_while(gen_rest, &(&1 < threshold))
        {newly_below_imp, imp_rest} = Enum.split_while(imp_rest, &(&1 < threshold))

        gen_below = gen_below + length(newly_below_gen)
        imp_below = imp_below + length(newly_below_imp)

        far = (n_impostor - imp_below) / n_impostor
        frr = gen_below / n_genuine

        {[%{threshold: threshold, far: far, frr: frr} | acc], gen_rest, imp_rest, gen_below,
         imp_below}
      end)

    Enum.reverse(points)
  end

  @doc """
  TAR (= 1 - FRR) at a target FAR operating point, linearly interpolated
  between the two bracketing points of `det_points` on the FAR axis —
  exact operating points like 0.1% or 0.01% are rarely hit exactly by a
  finite impostor pair list. Where several thresholds share the same FAR,
  the loosest (highest-TAR) one is used.
  """
  @spec tar_at_far([det_point()], float()) :: float()
  def tar_at_far(det_points, target_far) do
    det_points
    |> Enum.sort_by(& &1.threshold)
    |> Enum.map(fn %{far: far, frr: frr} -> {far, 1 - frr} end)
    |> Enum.uniq_by(fn {far, _tar} -> far end)
    |> Enum.sort_by(fn {far, _tar} -> far end)
    |> interpolate(target_far)
  end

  @doc """
  Score threshold at a target FAR, linearly interpolated the same way
  `tar_at_far/2` interpolates TAR — the decision-boundary counterpart,
  for turning an operating point into an actual score cutoff (what
  `Faulds.Calibration` persists) rather than just
  reporting accuracy at that point. Ties on FAR resolve to the loosest
  (lowest) threshold, the same "best achievable" duplicate `tar_at_far/2`
  resolves to.
  """
  @spec threshold_at_far([det_point()], float()) :: float()
  def threshold_at_far(det_points, target_far) do
    det_points
    |> Enum.sort_by(& &1.threshold)
    |> Enum.map(fn %{far: far, threshold: threshold} -> {far, threshold} end)
    |> Enum.uniq_by(fn {far, _threshold} -> far end)
    |> Enum.sort_by(fn {far, _threshold} -> far end)
    |> interpolate(target_far)
  end

  defp interpolate(points, target_far) do
    {below, at_or_above} = Enum.split_while(points, fn {far, _tar} -> far < target_far end)

    case {below, at_or_above} do
      {[], [{_far, tar} | _]} ->
        tar

      {_, []} ->
        below |> List.last() |> elem(1)

      {_, [{far_hi, tar_hi} | _]} ->
        {far_lo, tar_lo} = List.last(below)

        if far_hi == far_lo do
          tar_hi
        else
          ratio = (target_far - far_lo) / (far_hi - far_lo)
          tar_lo + ratio * (tar_hi - tar_lo)
        end
    end
  end

  @doc """
  Recall@K for an approximate-search shortlist: given, for
  each probe, the 1-indexed rank at which its true gallery match appeared
  in the shortlist (or `nil` if it didn't appear at all), returns the
  fraction of probes found within each of `ks`.

  The shortlist may be smaller than the gallery: a `nil` means "not
  found within however large a shortlist was actually searched", not
  "ranked last".
  """
  @spec recall_at_k([pos_integer() | nil], [pos_integer()]) ::
          [%{k: pos_integer(), recall: float()}]
  def recall_at_k(found_at_ranks, ks) when found_at_ranks != [] and ks != [] do
    n = length(found_at_ranks)

    Enum.map(ks, fn k ->
      found = Enum.count(found_at_ranks, &(&1 != nil and &1 <= k))
      %{k: k, recall: found / n}
    end)
  end

  @doc """
  Wilson score interval for a proportion of `successes` out of `n`, at
  the normal quantile `z` (1.96 for 95%) — the interval latent rank-k
  hit rates need at the small probe counts this project tests with,
  where the plain normal approximation goes below 0 or above 1.
  """
  @spec wilson_interval(non_neg_integer(), pos_integer(), float()) :: {float(), float()}
  def wilson_interval(successes, n, z \\ 1.96) when n > 0 and successes <= n do
    p = successes / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * :math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    {max(centre - half, 0.0), min(centre + half, 1.0)}
  end

  @doc """
  Two-sided exact McNemar test for two methods run on the same probes:
  `b` probes only the first method got right, `c` only the second. The
  p-value of a binomial(b + c, 1/2) at least as uneven as observed; 1.0
  when neither ever wins alone.
  """
  @spec mcnemar_exact(non_neg_integer(), non_neg_integer()) :: float()
  def mcnemar_exact(0, 0), do: 1.0

  # Exact up to n = 1000; past that 2^n overflows a float, and the normal
  # approximation (with continuity correction) is accurate anyway.
  def mcnemar_exact(b, c) when b + c <= 1000 do
    n = b + c
    k = min(b, c)
    tail = Enum.reduce(0..k, 0, fn i, acc -> acc + binomial(n, i) end) / :math.pow(2, n)
    min(2 * tail, 1.0)
  end

  def mcnemar_exact(b, c) do
    z = (abs(b - c) - 1) / :math.sqrt(b + c)
    min(:math.erfc(z / :math.sqrt(2)), 1.0)
  end

  @doc """
  Holm's step-down adjustment of `p_values` from a family of tests (say,
  every method against one baseline), in their given order: the i-th
  smallest becomes `max over j <= i of min(1, (m - j + 1) p_(j))`, so
  rejecting where an adjusted value is below α holds the family-wise
  error at α.
  """
  @spec holm([float()]) :: [float()]
  def holm(p_values) do
    m = length(p_values)

    adjusted =
      p_values
      |> Enum.with_index()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.map_reduce(0.0, fn {{p, i}, j}, running ->
        adj = max(running, min(1.0, (m - j) * p))
        {{i, adj}, adj}
      end)
      |> elem(0)
      |> Map.new()

    for i <- 0..(m - 1)//1, do: adjusted[i]
  end

  defp binomial(n, k), do: Enum.reduce(1..k//1, 1, fn i, acc -> div(acc * (n - k + i), i) end)
end
