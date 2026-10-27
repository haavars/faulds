defmodule Faulds.MetricsTest do
  use ExUnit.Case, async: true

  alias Faulds.Metrics

  describe "det_curve/2" do
    test "perfectly separated scores reach far=0, frr=0 at some threshold" do
      # genuine always scores higher than impostor here.
      points = Metrics.det_curve([0.7, 0.8, 0.9], [0.4, 0.5, 0.6])

      assert Enum.any?(points, &(&1.far == 0.0 and &1.frr == 0.0))
      assert Enum.any?(points, &(&1.far == 1.0 and &1.frr == 0.0))
      assert List.last(points) == %{threshold: List.last(points).threshold, far: 0.0, frr: 1.0}
    end

    test "hand-computed FAR/FRR at each threshold for overlapping scores" do
      # genuine: [0.5, 0.6], impostor: [0.4, 0.7]
      [first, second, third, fourth, fifth] = Metrics.det_curve([0.5, 0.6], [0.4, 0.7])

      assert first == %{threshold: 0.4, far: 1.0, frr: 0.0}
      assert second == %{threshold: 0.5, far: 0.5, frr: 0.0}
      assert third == %{threshold: 0.6, far: 0.5, frr: 0.5}
      assert fourth == %{threshold: 0.7, far: 0.5, frr: 1.0}
      assert fifth.far == 0.0
      assert fifth.frr == 1.0
    end

    test "FAR is non-increasing and FRR is non-decreasing as threshold rises" do
      points = Metrics.det_curve([0.1, 0.4, 0.55, 0.6, 0.95], [0.2, 0.3, 0.5, 0.65, 0.8])

      thresholds_ascending = Enum.sort_by(points, & &1.threshold)

      fars = Enum.map(thresholds_ascending, & &1.far)
      frrs = Enum.map(thresholds_ascending, & &1.frr)

      assert fars == Enum.sort(fars, :desc)
      assert frrs == Enum.sort(frrs, :asc)
    end
  end

  describe "tar_at_far/2" do
    # far ascending: (0.0, tar=0.0), (0.5, tar=1.0), (1.0, tar=1.0)
    @points [
      %{threshold: 0.7, far: 0.0, frr: 1.0},
      %{threshold: 0.6, far: 0.5, frr: 0.5},
      %{threshold: 0.5, far: 0.5, frr: 0.0},
      %{threshold: 0.4, far: 1.0, frr: 0.0}
    ]

    test "picks the loosest (highest-TAR) threshold among ties on FAR" do
      # far=0.5 appears at both frr=0.5 (tar=0.5) and frr=0.0 (tar=1.0); the
      # latter is the better operating point and should win.
      assert Metrics.tar_at_far(@points, 0.5) == 1.0
    end

    test "interpolates linearly between bracketing points" do
      assert Metrics.tar_at_far(@points, 0.25) == 0.5
    end

    test "clamps to the lowest achievable point when target is below the curve" do
      assert Metrics.tar_at_far(@points, 0.0) == 0.0
    end

    test "clamps to the highest achievable point when target is above the curve" do
      assert Metrics.tar_at_far(@points, 1.5) == 1.0
    end
  end

  describe "threshold_at_far/2" do
    # Same fixture as tar_at_far/2 above, so results are directly comparable.
    @points [
      %{threshold: 0.7, far: 0.0, frr: 1.0},
      %{threshold: 0.6, far: 0.5, frr: 0.5},
      %{threshold: 0.5, far: 0.5, frr: 0.0},
      %{threshold: 0.4, far: 1.0, frr: 0.0}
    ]

    test "picks the loosest (lowest) threshold among ties on FAR" do
      assert Metrics.threshold_at_far(@points, 0.5) == 0.5
    end

    test "interpolates linearly between bracketing points" do
      assert Metrics.threshold_at_far(@points, 0.25) == 0.6
    end

    test "clamps to the lowest-far achievable point's threshold when target is below the curve" do
      assert Metrics.threshold_at_far(@points, 0.0) == 0.7
    end

    test "clamps to the highest-far achievable point's threshold when target is above the curve" do
      assert Metrics.threshold_at_far(@points, 1.5) == 0.4
    end
  end

  describe "recall_at_k/2" do
    test "fraction of probes whose true match rank is within each K, nils never counted" do
      # found at rank 1, found at rank 3, not found at all.
      found_at_ranks = [1, 3, nil]

      assert Metrics.recall_at_k(found_at_ranks, [1, 2, 3, 10]) == [
               %{k: 1, recall: 1 / 3},
               %{k: 2, recall: 1 / 3},
               %{k: 3, recall: 2 / 3},
               %{k: 10, recall: 2 / 3}
             ]
    end
  end

  describe "wilson_interval/3" do
    test "matches a hand-computed 95% interval: 6 of 108" do
      {low, high} = Metrics.wilson_interval(6, 108)
      assert_in_delta low, 0.0257, 0.0005
      assert_in_delta high, 0.1158, 0.0005
    end

    test "stays inside [0, 1] at the extremes" do
      assert {0.0, high} = Metrics.wilson_interval(0, 20)
      assert high > 0.0
      assert {low, 1.0} = Metrics.wilson_interval(20, 20)
      assert low < 1.0
    end
  end

  describe "mcnemar_exact/2" do
    test "no discordant probes: no evidence either way" do
      assert Metrics.mcnemar_exact(0, 0) == 1.0
    end

    test "matches the binomial tail: 0 vs 6 discordant gives 2 * 0.5^6" do
      assert_in_delta Metrics.mcnemar_exact(0, 6), 2 * :math.pow(0.5, 6), 1.0e-12
      assert_in_delta Metrics.mcnemar_exact(6, 0), 2 * :math.pow(0.5, 6), 1.0e-12
    end

    test "balanced counts are not significant" do
      assert Metrics.mcnemar_exact(5, 5) == 1.0
    end

    test "large counts use the normal approximation and agree with the exact test near the boundary" do
      exact = Metrics.mcnemar_exact(480, 520)
      approx = Metrics.mcnemar_exact(481, 521)
      assert_in_delta exact, approx, 0.05
    end
  end

  describe "holm/1" do
    test "multiplies the i-th smallest by m - i + 1, kept monotone, in the given order" do
      assert Enum.map(Metrics.holm([0.01, 0.04, 0.03]), &Float.round(&1, 10)) == [
               0.03,
               0.06,
               0.06
             ]
    end

    test "caps at 1 and leaves a single test alone" do
      assert Metrics.holm([0.6, 0.5]) == [1.0, 1.0]
      assert Metrics.holm([0.2]) == [0.2]
      assert Metrics.holm([]) == []
    end
  end
end
