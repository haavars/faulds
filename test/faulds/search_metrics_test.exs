defmodule Faulds.SearchMetricsTest do
  # Hand-sized matrices on the binary backend: no EXLA compiles. The
  # functions were also checked against Tsinghua's evaluator
  # (`get_eval_metric.py`'s argsort ranks, scikit-learn's `roc_curve`) on
  # random 150–300 × 300–500 matrices with and without ties: equal
  # everywhere, ranks under a stable argsort.
  use ExUnit.Case, async: true

  alias Faulds.SearchMetrics

  setup do
    Nx.default_backend(Nx.BinaryBackend)
    :ok
  end

  describe "mate_ranks/3" do
    test "counts non-mates at or above the best mate, nil without a mate" do
      matrix =
        Nx.tensor([
          [0.9, 0.5, 0.5, 0.1],
          [0.2, 0.8, 0.1, 0.7],
          [0.3, 0.2, 0.1, 0.0]
        ])

      # Probe 0's mate ties column 1; probe 1's best of two mates is 0.7.
      assert SearchMetrics.mate_ranks(matrix, [[2], [0, 3], []]) == [3, 2, nil]
    end

    test "with sort-order ties, only tied non-mates later than the mate rank above it" do
      matrix = Nx.tensor([[0.9, 0.5, 0.5, 0.5]])
      assert SearchMetrics.mate_ranks(matrix, [[2]], ties: :sort_order) == [3]
      assert SearchMetrics.mate_ranks(matrix, [[2]]) == [4]
      # Of tied mates the latest counts.
      assert SearchMetrics.mate_ranks(matrix, [[1, 2]], ties: :sort_order) == [3]
    end

    test "a NaN score counts as 0" do
      # Left as NaN it would compare false with everything and rank first.
      matrix = Nx.tensor([[:nan, 0.5, -1.0]])
      assert SearchMetrics.mate_ranks(matrix, [[0]]) == [2]
    end
  end

  test "cmc/2 and hit_rates/2 leave out probes without a mate" do
    ranks = [1, 3, nil, 2, 1]
    assert SearchMetrics.cmc(ranks, 4) == [0.5, 0.75, 1.0, 1.0]

    assert [%{k: 1, hits: 2, n: 4, rate: 0.5, ci: {low, high}}] =
             SearchMetrics.hit_rates(ranks, [1])

    assert low < 0.5 and high > 0.5
  end

  test "strata/3 gives each group's hit rates, skipping groups without a mated probe" do
    ranks = [1, 5, nil, 2]

    assert [{"a", 2, [%{hits: 1}]}, {"b", 1, [%{hits: 0}]}] =
             SearchMetrics.strata(ranks, [{"a", [0, 1]}, {"b", [1, 2]}, {"c", [2]}], [1])
  end

  test "compare/3 counts discordant probes and Holm-adjusts the family" do
    baseline = [1, 1, 5, 5, 5, 5, 5, 5, 5, 5]
    better = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    same = baseline

    [b, s] = SearchMetrics.compare(baseline, [{"better", better}, {"same", same}], 1)
    assert %{only_this: 8, only_baseline: 0, p: p} = b
    assert_in_delta p, 2 * :math.pow(0.5, 8), 1.0e-12
    assert_in_delta b.p_holm, 2 * p, 1.0e-12
    assert %{only_this: 0, only_baseline: 0, p: 1.0, p_holm: 1.0} = s
  end

  describe "tar_at_far/3" do
    # Genuine at 0.9; a genuine and an impostor at each of 0.8 and 0.7;
    # eight impostors at 0.1. The ROC points by threshold (fp, tp): 0.9
    # (0, 1), 0.8 (1, 2), 0.7 (2, 3), 0.1 (10, 3). The 0.8 point lies on
    # the line between its neighbours, so `roc_curve` drops it.
    setup do
      %{
        matrix: Nx.tensor([[0.9, 0.8, 0.8, 0.7, 0.7] ++ List.duplicate(0.1, 8)]),
        mates: [[0, 1, 3]]
      }
    end

    test "reads the TAR off roc_curve's points", %{matrix: matrix, mates: mates} do
      assert [%{far: +0.0, tar: t0}, %{far: 0.1, tar: t1}, %{far: 0.2, tar: t2}] =
               SearchMetrics.tar_at_far(matrix, mates, [0.0, 0.1, 0.2])

      assert_in_delta t0, 1 / 3, 1.0e-12
      # The best TAR at FAR 0.1 is 2/3, at the dropped point.
      assert_in_delta t1, 1 / 3, 1.0e-12
      assert t2 == 1.0
    end
  end

  describe "open_set/3" do
    setup do
      matrix =
        Nx.tensor([
          [0.9, 0.2, 0.1],
          [0.5, 0.6, 0.3],
          [0.8, 0.1, 0.4],
          [0.3, 0.7, 0.2]
        ])

      # Probe 2's mate is second; probe 3 has none.
      %{matrix: matrix, mates: [[0], [1], [2], []]}
    end

    test "every mated probe also searched mate-removed", %{matrix: matrix, mates: mates} do
      # Non-mated top scores: 0.2, 0.5, 0.8, 0.7.
      results = SearchMetrics.open_set(matrix, mates, fpirs: [0.0, 0.25, 0.5, 1.0])

      assert Enum.map(results, &{&1.threshold, Float.round(&1.fnir, 4)}) ==
               [
                 {0.800000011920929, 0.6667},
                 {0.699999988079071, 0.6667},
                 {0.5, 0.3333},
                 {nil, 0.3333}
               ]

      assert hd(results).n_mated == 3 and hd(results).n_nonmated == 4
    end

    test ":half searches every second mated probe only mate-removed", %{
      matrix: matrix,
      mates: mates
    } do
      # Mated: probes 0 and 2; non-mated: 1 (0.5) and 3 (0.7).
      assert [%{fnir: 0.5, threshold: 0.5, n_mated: 2, n_nonmated: 2}] =
               SearchMetrics.open_set(matrix, mates, fpirs: [0.5], demated: :half)
    end

    test ":rank lets the mate be further down", %{matrix: matrix, mates: mates} do
      assert [%{fnir: +0.0}] = SearchMetrics.open_set(matrix, mates, fpirs: [1.0], rank: 2)
    end

    test ":mate_removed_top replaces the matrix's top non-mate scores", %{
      matrix: matrix,
      mates: mates
    } do
      # Probe 0's mate-removed search now tops everything.
      [%{threshold: threshold, fnir: fnir}] =
        SearchMetrics.open_set(matrix, mates,
          fpirs: [0.25],
          mate_removed_top: [0.95, 0.5, 0.8, 0.7]
        )

      assert {Float.round(threshold, 4), Float.round(fnir, 4)} == {0.8, 0.6667}
    end

    test "open_set_of/2 on two search sets' searches holds one threshold for both", %{
      matrix: matrix,
      mates: mates
    } do
      # The same set twice: the same rates as once, over twice the searches.
      searches = SearchMetrics.searches(matrix, mates)
      fpirs = [0.0, 0.25, 0.5, 1.0]
      once = SearchMetrics.open_set(matrix, mates, fpirs: fpirs)
      twice = SearchMetrics.open_set_of(searches ++ searches, fpirs: fpirs)

      assert Enum.map(twice, &{&1.threshold, &1.fnir}) ==
               Enum.map(once, &{&1.threshold, &1.fnir})

      assert hd(twice).n_mated == 6 and hd(twice).n_nonmated == 8
    end
  end

  describe "drop_mates/2" do
    test "takes each row's mates out, keeping the rest in order" do
      matrix = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      assert Nx.to_list(SearchMetrics.drop_mates(matrix, [[0], [2]])) == [[2, 3], [4, 5]]
      assert SearchMetrics.drop_mates(matrix, [[], []]) == matrix
    end

    test "refuses rows with different numbers of mates" do
      assert_raise ArgumentError, fn ->
        SearchMetrics.drop_mates(Nx.tensor([[1, 2], [3, 4]]), [[0], []])
      end
    end
  end
end
