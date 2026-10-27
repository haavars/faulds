defmodule Faulds.SearchMetrics do
  @moduledoc """
  The metric layer for 1:N search:
  what one score matrix says under a protocol's mates, the way the
  published latent numbers are counted.

  Every function takes a score matrix `{probes, gallery}` (higher is more
  alike) and `mates`, one list of gallery columns per probe: one column
  for a probe with a single listed mate, several when every impression
  of the finger counts (`Faulds.Protocol.mates/2`), none
  for a probe without a mate in the gallery. A probe is credited at the
  rank of its best mate, as Tsinghua's evaluator (`get_eval_metric.py`
  in the DMD repository: a hit at k when any genuine column is in the
  top k) and AFID's "any impression of the mated finger" do.

  NaN scores count as 0, as in Tsinghua's evaluator.
  """

  alias Faulds.Metrics

  @type mates :: [[non_neg_integer()]]

  @doc """
  The rank of each probe's best mate (1 = first), nil for a probe with no
  mates: one more than the non-mates ranked above it. `:ties`:

    * `:against` (default): a non-mate scoring as high as the mate ranks
      above it, so a matcher that gives every print 0 ranks the mate last.
    * `:sort_order`: Tsinghua's rule (`np.argsort` ascending, the last k
      columns are the top k), read as a stable sort: of the non-mates
      tied with the mate, only those later in gallery order rank above it;
      of tied mates, the latest counts. NumPy's default sort isn't
      guaranteed stable, so this reproduces their rule, not every tie of
      their runs. Published N2N numbers use it; report it beside
      `:against`.
  """
  @spec mate_ranks(Nx.Tensor.t(), mates(), keyword()) :: [pos_integer() | nil]
  def mate_ranks(matrix, mates, opts \\ []) do
    matrix = clean(matrix)
    mask = mate_mask(matrix, mates)
    best = best_mate_scores(matrix, mask)
    non_mate = Nx.logical_not(mask)

    above =
      case Keyword.get(opts, :ties, :against) do
        :against ->
          Nx.greater_equal(matrix, best)

        :sort_order ->
          columns = Nx.iota(Nx.shape(matrix), axis: 1)
          at_best = Nx.logical_and(mask, Nx.equal(matrix, best))

          best_column =
            at_best |> Nx.select(columns, -1) |> Nx.reduce_max(axes: [1], keep_axes: true)

          Nx.logical_or(
            Nx.greater(matrix, best),
            Nx.logical_and(Nx.equal(matrix, best), Nx.greater(columns, best_column))
          )
      end

    above
    |> Nx.logical_and(non_mate)
    |> Nx.sum(axes: [1])
    |> Nx.add(1)
    |> Nx.to_flat_list()
    |> Enum.zip_with(mates, fn rank, cols -> if cols == [], do: nil, else: rank end)
  end

  @doc """
  The cumulative match characteristic: the fraction of mated probes
  (`ranks` not nil) with their mate at rank `1..max_rank`.
  """
  @spec cmc([pos_integer() | nil], pos_integer()) :: [float()]
  def cmc(ranks, max_rank \\ 100) do
    mated = Enum.reject(ranks, &is_nil/1)
    n = length(mated)
    counts = Enum.frequencies(mated)

    {curve, _} =
      Enum.map_reduce(1..max_rank, 0, fn k, hits ->
        hits = hits + Map.get(counts, k, 0)
        {hits / n, hits}
      end)

    curve
  end

  @doc """
  Hits at each of `ks` among the mated probes (`ranks` not nil), with a
  95% Wilson interval: `[%{k:, hits:, n:, rate:, ci: {low, high}}]`.
  """
  @spec hit_rates([pos_integer() | nil], [pos_integer()]) :: [map()]
  def hit_rates(ranks, ks) do
    mated = Enum.reject(ranks, &is_nil/1)
    n = length(mated)

    for k <- ks do
      hits = Enum.count(mated, &(&1 <= k))
      %{k: k, hits: hits, n: n, rate: hits / n, ci: Metrics.wilson_interval(hits, n)}
    end
  end

  @doc """
  `hit_rates/2` for each group of probe indices: `[{label, n, rates}]`,
  for per-finger, per-value or per-technique tables.
  """
  @spec strata([pos_integer() | nil], [{String.t(), [non_neg_integer()]}], [pos_integer()]) ::
          [{String.t(), non_neg_integer(), [map()]}]
  def strata(ranks, groups, ks) do
    ranks = List.to_tuple(ranks)

    for {label, members} <- groups,
        subset = members |> Enum.map(&elem(ranks, &1)) |> Enum.reject(&is_nil/1),
        subset != [] do
      {label, length(subset), hit_rates(subset, ks)}
    end
  end

  @doc """
  McNemar's exact test of each of `others` (`[{name, ranks}]`) against
  `baseline` ranks at rank `k`, with Holm-adjusted p-values over the
  family: `[%{name:, only_this:, only_baseline:, p:, p_holm:}]`. Probes
  without a mate are left out.
  """
  @spec compare([pos_integer() | nil], [{String.t(), [pos_integer() | nil]}], pos_integer()) ::
          [map()]
  def compare(baseline, others, k) do
    raw =
      for {name, ranks} <- others do
        pairs = for {a, b} <- Enum.zip(ranks, baseline), a != nil and b != nil, do: {a, b}
        only_this = Enum.count(pairs, fn {a, b} -> a <= k and b > k end)
        only_baseline = Enum.count(pairs, fn {a, b} -> b <= k and a > k end)

        %{
          name: name,
          only_this: only_this,
          only_baseline: only_baseline,
          p: Metrics.mcnemar_exact(only_this, only_baseline)
        }
      end

    raw
    |> Enum.map(& &1.p)
    |> Metrics.holm()
    |> Enum.zip_with(raw, fn p_holm, row -> Map.put(row, :p_holm, p_holm) end)
  end

  @doc """
  TAR at each of `fars` over every probe–gallery pair (genuine pairs are
  the mates, every other pair an impostor), exactly as Tsinghua's
  `TAR_flatten` reads it off scikit-learn's `roc_curve`: the ROC at each
  distinct score (accepting scores at or above it), less the points
  `drop_intermediate` drops (where neither count's second difference is
  non-zero), plus the origin; the TAR of the last point whose FAR is at
  most the target. On continuous scores that is the best TAR at that FAR;
  on tied scores (`bozorth3`'s integers) a dropped point can make it
  lower. `[%{far:, tar:}]`.
  """
  @spec tar_at_far(Nx.Tensor.t(), mates(), [float()]) :: [map()]
  def tar_at_far(matrix, mates, fars \\ [1.0e-3, 1.0e-4]) do
    matrix = clean(matrix)
    labels = matrix |> mate_mask(mates) |> Nx.flatten()
    scores = Nx.flatten(matrix)

    order = Nx.argsort(scores, direction: :desc, stable: true)
    sorted = Nx.take(scores, order)
    tps = labels |> Nx.take(order) |> Nx.as_type(:s64) |> Nx.cumulative_sum()
    fps = Nx.subtract(Nx.add(Nx.iota(Nx.shape(tps), type: :s64), 1), tps)

    # The last pair of each distinct score, in order.
    n = Nx.size(sorted)
    changes = Nx.not_equal(Nx.slice(sorted, [0], [n - 1]), Nx.slice(sorted, [1], [n - 1]))
    last = Nx.concatenate([changes, Nx.tensor([1], type: :u8)])
    distinct = last |> Nx.sum() |> Nx.to_number()
    ends = last |> Nx.logical_not() |> Nx.argsort(stable: true) |> Nx.slice([0], [distinct])
    {tps, fps} = {Nx.take(tps, ends), Nx.take(fps, ends)}

    keep =
      if distinct > 2 do
        bends = Nx.logical_or(second_difference(fps), second_difference(tps))
        Nx.concatenate([Nx.tensor([1], type: :u8), bends, Nx.tensor([1], type: :u8)])
      else
        Nx.broadcast(Nx.tensor(1, type: :u8), {distinct})
      end

    n_genuine = tps |> Nx.reduce_max() |> Nx.to_number()
    n_impostor = fps |> Nx.reduce_max() |> Nx.to_number()
    fpr = Nx.divide(Nx.as_type(fps, :f64), n_impostor)
    tpr = Nx.divide(Nx.as_type(tps, :f64), n_genuine)

    for far <- fars do
      tar =
        keep
        |> Nx.logical_and(Nx.less_equal(fpr, far))
        |> Nx.select(tpr, 0.0)
        |> Nx.reduce_max()
        |> Nx.to_number()

      %{far: far, tar: tar}
    end
  end

  defp second_difference(counts) do
    n = Nx.size(counts)
    a = Nx.slice(counts, [0], [n - 2])
    b = Nx.slice(counts, [1], [n - 2])
    c = Nx.slice(counts, [2], [n - 2])
    a |> Nx.subtract(Nx.multiply(b, 2)) |> Nx.add(c) |> Nx.not_equal(0)
  end

  @doc """
  Open-set identification (NIST's FRVT/ELFT terms): FNIR at each FPIR in
  `:fpirs` (default 0.01, 0.02, 0.1). FPIR(T) is the fraction of
  non-mated searches returning any candidate scoring above T, so a
  non-mated search here is a probe's search with its mates taken out (the
  top non-mate score); FNIR(R, T) the fraction of mated searches whose
  mate isn't at rank R or better (ties against the mate) with a score
  above T. `:rank` sets R (default 1). `:demated`:

    * `:all` (default): every mated probe is also searched mate-removed;
    * `:half`: LFR-Net's variant, every second mated probe (odd index
      among them) is searched only mate-removed, the rest only mated, so
      no probe is on both sides.

  `:mate_removed_top`: each probe's top score in its search with its
  mates taken out, scored as such, for a score that depends on the rest
  of its search (a z-score, a gap to the next candidate): taking the
  mate out moves the others. Without it, the top non-mate score of the
  matrix itself, right for a score of the pair alone.

  Probes without any mate are always non-mated searches.
  `[%{fpir:, fnir:, threshold:, n_mated:, n_nonmated:}]`, the threshold
  being the top non-mated score the operating point must exceed.
  """
  @spec open_set(Nx.Tensor.t(), mates(), keyword()) :: [map()]
  def open_set(matrix, mates, opts \\ []) do
    matrix |> searches(mates, opts) |> open_set_of(opts)
  end

  @doc """
  Each probe's search as `open_set_of/2` counts it, `%{mated?:, best:,
  top_non_mate:, rank:}`: the best mate's score and rank, and the top
  score of its non-mated search (`:mate_removed_top` as in
  `open_set/3`). Concatenate several search sets' lists to threshold
  them as one.
  """
  @spec searches(Nx.Tensor.t(), mates(), keyword()) :: [map()]
  def searches(matrix, mates, opts \\ []) do
    matrix = clean(matrix)
    mask = mate_mask(matrix, mates)
    best = matrix |> best_mate_scores(mask) |> Nx.to_flat_list()

    top_non_mate =
      case Keyword.get(opts, :mate_removed_top) do
        nil ->
          mask
          |> Nx.select(neg_inf(matrix), matrix)
          |> Nx.reduce_max(axes: [1])
          |> Nx.to_flat_list()

        tops ->
          length(tops) == length(mates) ||
            raise ArgumentError, "#{length(tops)} mate-removed scores for #{length(mates)} probes"

          tops
      end

    ranks = mate_ranks(matrix, mates)

    Enum.zip_with([mates, best, top_non_mate, ranks], fn [cols, best, top, rank] ->
      %{mated?: cols != [], best: best, top_non_mate: top, rank: rank}
    end)
  end

  @doc """
  `open_set/3` over `searches/3`'s list, one threshold for all of them.
  Options as `open_set/3`'s `:fpirs`, `:rank` and `:demated`.
  """
  @spec open_set_of([map()], keyword()) :: [map()]
  def open_set_of(searches, opts \\ []) do
    fpirs = Keyword.get(opts, :fpirs, [0.01, 0.02, 0.1])
    rank_limit = Keyword.get(opts, :rank, 1)

    probes = List.to_tuple(searches)
    mated_indices = for {%{mated?: true}, i} <- Enum.with_index(searches), do: i

    {mated, demated} =
      case Keyword.get(opts, :demated, :all) do
        :all ->
          {mated_indices, mated_indices}

        :half ->
          mated_indices
          |> Enum.with_index()
          |> Enum.split_with(fn {_i, j} -> rem(j, 2) == 0 end)
          |> then(fn {a, b} -> {Enum.map(a, &elem(&1, 0)), Enum.map(b, &elem(&1, 0))} end)
      end

    unmated = for {%{mated?: false}, i} <- Enum.with_index(searches), do: i

    nonmated_scores =
      (demated ++ unmated)
      |> Enum.map(&elem(probes, &1).top_non_mate)
      |> Enum.sort(:desc)
      |> List.to_tuple()

    n_nonmated = tuple_size(nonmated_scores)
    mated = Enum.map(mated, &elem(probes, &1))

    for fpir <- fpirs do
      m = allowed_count(fpir, n_nonmated)
      threshold = if m < n_nonmated, do: elem(nonmated_scores, m)

      misses =
        Enum.count(mated, fn %{best: score, rank: rank} ->
          rank > rank_limit or (threshold != nil and score <= threshold)
        end)

      %{
        fpir: fpir,
        fnir: misses / length(mated),
        threshold: threshold,
        n_mated: length(mated),
        n_nonmated: n_nonmated
      }
    end
  end

  @doc """
  Each probe's row with its mates' columns taken out, `{probes, gallery −
  k}`: its search had its mates not been enrolled. Every probe must have
  the same number `k` of mates.
  """
  @spec drop_mates(Nx.Tensor.t(), mates()) :: Nx.Tensor.t()
  def drop_mates(matrix, mates) do
    {probes, gallery} = Nx.shape(matrix)

    length(mates) == probes ||
      raise ArgumentError, "#{length(mates)} mate lists for #{probes} probes"

    k = mates |> hd() |> length()

    Enum.all?(mates, &(length(Enum.uniq(&1)) == k)) ||
      raise ArgumentError, "drop_mates/2 needs the same number of mates for every probe"

    if k == 0 do
      matrix
    else
      keep =
        Enum.map(mates, fn cols ->
          drop = MapSet.new(cols)
          Enum.reject(0..(gallery - 1), &MapSet.member?(drop, &1))
        end)

      Nx.take_along_axis(matrix, Nx.tensor(keep, type: :s32), axis: 1)
    end
  end

  @doc """
  The standing report for one score matrix under a protocol's `mates`:
  ranks, CMC to rank 100 and hit rates at 1, 10, 20 and 100 (with Wilson
  intervals) under both tie rules, TAR at FAR 0.1% and 0.01% over all
  pairs, and FNIR at FPIR 1%, 2% and 10% with every mated probe also
  searched mate-removed (`:open_set`) and LFR-Net's half de-mated split
  (`:open_set_half`).
  """
  @spec report(Nx.Tensor.t(), mates()) :: map()
  def report(matrix, mates) do
    by_ties =
      Map.new([:against, :sort_order], fn ties ->
        ranks = mate_ranks(matrix, mates, ties: ties)
        {ties, %{ranks: ranks, cmc: cmc(ranks), hits: hit_rates(ranks, [1, 10, 20, 100])}}
      end)

    Map.merge(by_ties, %{
      n_probes: length(mates),
      n_mated: Enum.count(mates, &(&1 != [])),
      gallery: elem(Nx.shape(matrix), 1),
      tar: tar_at_far(matrix, mates),
      open_set: open_set(matrix, mates),
      open_set_half: open_set(matrix, mates, demated: :half)
    })
  end

  # The most of `n` a rate may let through: the largest count c with
  # c / n <= rate, as the evaluators compare it (in floats).
  defp allowed_count(rate, n) do
    c = trunc(rate * n)

    cond do
      (c + 1) / n <= rate -> c + 1
      c > 0 and c / n > rate -> c - 1
      true -> c
    end
  end

  defp clean(matrix), do: Nx.select(Nx.is_nan(matrix), 0.0, matrix)

  defp mate_mask(matrix, mates) do
    {probes, _} = Nx.shape(matrix)

    length(mates) == probes ||
      raise ArgumentError, "#{length(mates)} mate lists for #{probes} probes"

    indices = for {cols, p} <- Enum.with_index(mates), c <- cols, do: [p, c]
    empty = Nx.broadcast(Nx.tensor(0, type: :u8), Nx.shape(matrix))

    if indices == [],
      do: empty,
      else:
        Nx.indexed_put(
          empty,
          Nx.tensor(indices),
          Nx.broadcast(1, {length(indices)}) |> Nx.as_type(:u8)
        )
  end

  defp best_mate_scores(matrix, mask),
    do: mask |> Nx.select(matrix, neg_inf(matrix)) |> Nx.reduce_max(axes: [1], keep_axes: true)

  defp neg_inf(matrix), do: Nx.Constants.neg_infinity(Nx.type(matrix))
end
