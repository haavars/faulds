defmodule Faulds.SearchThresholdTest do
  use ExUnit.Case, async: true

  alias Faulds.SearchThreshold

  test "class_of/1 follows the LEADER count classes" do
    assert Enum.map([0, 1, 2, 3, 5, 6, 11, 12, 40], &SearchThreshold.class_of/1) ==
             [0, 1, 1, 2, 2, 3, 3, 4, 4]
  end

  test "shares/1 is each class's share" do
    assert SearchThreshold.shares([0, 0, 7, 20]) == %{0 => 0.5, 3 => 0.25, 4 => 0.25}
  end

  # Class 0's non-mated tops are all low, class 4's all high: a mix with
  # more class 4 needs a higher threshold for the same FPIR.
  @reference %SearchThreshold{
    tops: Enum.map(1..100, &{&1 / 100, 0}) ++ Enum.map(1..100, &{10 + &1 / 100, 4}),
    info: %{}
  }

  test "threshold/3 without shares holds fpir of the reference's tops above it" do
    t = SearchThreshold.threshold(@reference, 0.1)
    assert Enum.count(@reference.tops, fn {top, _} -> top > t end) == 20
  end

  test "threshold/3 reweights the reference to the deployment's class shares" do
    # All class 0: only its tops count.
    t0 = SearchThreshold.threshold(@reference, 0.1, %{0 => 1.0})
    assert_in_delta t0, 0.9, 1.0e-9

    # All class 4.
    assert_in_delta SearchThreshold.threshold(@reference, 0.1, %{4 => 1.0}), 10.9, 1.0e-9

    # Mostly class 4: 10% FPIR lies inside class 4's tops.
    t = SearchThreshold.threshold(@reference, 0.1, %{0 => 0.2, 4 => 0.8})
    assert t > 10.0 and t < 10.9
  end

  test "threshold/4 grows each class's share with the gallery's size" do
    plain = SearchThreshold.threshold(@reference, 0.1, %{0 => 0.2, 4 => 0.8})

    # No factor, or a gallery of 1,000: the same threshold.
    assert SearchThreshold.threshold(@reference, 0.1, %{0 => 0.2, 4 => 0.8}, size: 37_000) ==
             plain

    assert SearchThreshold.threshold(@reference, 0.1, %{0 => 0.2, 4 => 0.8},
             size: 1_000,
             factors: %{4 => 1.0}
           ) == plain

    # Class 4's share doubles (k = 2 at G = 1,000 e): the threshold rises,
    # to where class 4 alone holds 10%: 0.8 · 2 · F(t) = 0.1.
    t =
      SearchThreshold.threshold(@reference, 0.1, %{0 => 0.2, 4 => 0.8},
        size: round(1_000 * :math.exp(1)),
        factors: %{4 => 1.0}
      )

    assert t > plain
    assert_in_delta t, 10.93, 0.011

    # A factor on a class whose tops are all below the threshold changes
    # nothing.
    assert SearchThreshold.threshold(@reference, 0.1, %{0 => 0.2, 4 => 0.8},
             size: 37_000,
             factors: %{0 => 1.0}
           ) == plain
  end

  test "threshold/4 holds a negative factor at the validated size" do
    shares = %{0 => 0.2, 4 => 0.8}
    shrink = [factors: %{4 => -0.2}, validated_size: 37_000]
    at = &SearchThreshold.threshold(@reference, 0.1, shares, [size: &1] ++ shrink)

    # Shrinking class 4's share lowers the threshold, down to its value
    # at 37,000; k there (0.28) stays above 0.
    assert at.(10_000) < SearchThreshold.threshold(@reference, 0.1, shares)
    assert at.(37_000) < at.(10_000)
    assert at.(10_000_000) == at.(37_000)

    # Without a validated size the share would vanish.
    assert SearchThreshold.threshold(@reference, 0.1, shares,
             size: 10_000_000,
             factors: %{4 => -0.2}
           ) < at.(37_000)
  end

  test "size_options/2 reads the reference's factors" do
    assert SearchThreshold.size_options(@reference, 5_000) == []

    reference = %{
      @reference
      | info: %{"size_factors" => %{"a" => %{"1" => -0.19, "4" => 1}, "validated_to" => 37_000}}
    }

    assert SearchThreshold.size_options(reference, 5_000) ==
             [size: 5_000, factors: %{1 => -0.19, 4 => 1.0}, validated_size: 37_000]
  end

  @tag :tmp_dir
  test "write!/2 and read!/1 round-trip", %{tmp_dir: dir} do
    path = Path.join(dir, "threshold.json")
    reference = %SearchThreshold{tops: [{1.5, 0}, {-2.0, 4}], info: %{"runs" => 2}}
    :ok = SearchThreshold.write!(reference, path)
    assert SearchThreshold.read!(path) == reference
  end
end
