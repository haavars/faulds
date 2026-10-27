defmodule Faulds.LatentBenchmarkTest do
  use ExUnit.Case, async: true

  alias Faulds.LatentBenchmark
  alias Faulds.Matchers.Flare

  defp template(seed) do
    key = Nx.Random.key(seed)
    {feature, key} = Nx.Random.normal(key, shape: {3072})
    {mask, _key} = Nx.Random.uniform(key, shape: {256})
    %{feature: Nx.to_binary(feature), mask: Nx.to_binary(mask)}
  end

  defp tensors(%{feature: feature, mask: mask}) do
    %{feature: Nx.from_binary(feature, :f32), mask: Nx.from_binary(mask, :f32)}
  end

  defp score(a, b) do
    {:ok, s} = Flare.score(tensors(a), tensors(b))
    s
  end

  describe "flare_matrix/2" do
    test "equals Flare.score/2 for every pair" do
      probes = [[template(1)], [template(2)]]
      gallery = [[template(3)], [template(4)], [template(5)]]

      matrix = LatentBenchmark.flare_matrix(probes, gallery)

      for {[p], i} <- Enum.with_index(probes), {[g], j} <- Enum.with_index(gallery) do
        assert_in_delta Nx.to_number(matrix[i][j]), score(p, g), 1.0e-4
      end
    end

    test "takes the best over each side's variants, and -1 for an entry with no template" do
      [a, b, c, d] = Enum.map(10..13, &template/1)

      matrix = LatentBenchmark.flare_matrix([[a, b]], [[c], [d, c], []])

      assert_in_delta Nx.to_number(matrix[0][0]), max(score(a, c), score(b, c)), 1.0e-4

      assert_in_delta Nx.to_number(matrix[0][1]),
                      Enum.max([score(a, d), score(a, c), score(b, d), score(b, c)]),
                      1.0e-4

      assert Nx.to_number(matrix[0][2]) == -1.0
    end
  end

  describe "flare_matrix/3 normalised" do
    defp overlap(a, b) do
      Nx.dot(tensors(a).mask, tensors(b).mask) |> Nx.to_number()
    end

    test ":overlap scales each score by the square root of the masks' overlap" do
      [a, b, c] = Enum.map(20..22, &template/1)

      matrix = LatentBenchmark.flare_matrix([[a]], [[b], [c], []], normalise: :overlap)

      for {g, j} <- [{b, 0}, {c, 1}] do
        assert_in_delta Nx.to_number(matrix[0][j]),
                        score(a, g) * :math.sqrt(overlap(a, g) / 256),
                        1.0e-4
      end

      assert Nx.to_number(matrix[0][2]) < -100
    end

    test "with one view, :lse is the standardised score" do
      probes = Enum.map(30..31, &[template(&1)])
      gallery = Enum.map(40..45, &[template(&1)])

      lse = LatentBenchmark.flare_matrix(probes, gallery, normalise: :lse)
      z = LatentBenchmark.flare_matrix(probes, gallery, normalise: :overlap_z)

      assert Nx.all_close(lse, z, atol: 1.0e-4) |> Nx.to_number() == 1
    end
  end

  describe "robust_z/2" do
    test "centres on the median and scales by 1.4826 MAD, unmoved by an outlier" do
      z = LatentBenchmark.robust_z(Nx.tensor([[1.0, 2.0, 3.0, 4.0, 100.0]]), 1)

      # median 3, deviations [2, 1, 0, 1, 97] → MAD 1
      assert_in_delta Nx.to_number(z[0][0]), -2 / 1.4826, 1.0e-4
      assert_in_delta Nx.to_number(z[0][4]), 97 / 1.4826, 1.0e-3
    end

    test "a row with no spread takes the median scale of the others" do
      z =
        LatentBenchmark.robust_z(
          Nx.tensor([[0.0, 0.0, 0.0, 0.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]),
          1
        )

      # row 2: median 3, MAD 1 → scale 1.4826, the floor for row 1
      assert_in_delta Nx.to_number(z[0][4]), 5 / 1.4826, 1.0e-4
    end
  end

  test "impostor_correlation/2 is 1 for a matrix with itself and -1 with its negation" do
    dataset = %{
      probes: [%{subject: "A", fgp: 1}, %{subject: "B", fgp: 2}],
      gallery: [%{key: "A/1"}, %{key: "B/2"}, %{key: "C/3"}]
    }

    m = Nx.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.5]])
    [[aa, ab], [_ba, _bb]] = LatentBenchmark.impostor_correlation(dataset, [m, Nx.negate(m)])

    assert_in_delta aa, 1.0, 1.0e-5
    assert_in_delta ab, -1.0, 1.0e-5
  end

  describe "mate_ranks/2" do
    test "counts ties against the mate" do
      dataset = %{
        probes: [%{subject: "A", fgp: 1}, %{subject: "B", fgp: 2}],
        gallery: [%{key: "A/1"}, %{key: "B/2"}, %{key: "C/3"}]
      }

      matrix = Nx.tensor([[0.9, 0.1, 0.2], [0.0, 0.0, 0.0]])

      assert LatentBenchmark.mate_ranks(dataset, matrix) == [1, 3]
    end

    test "with sort-order ties, only tied entries later in the gallery rank above the mate" do
      dataset = %{
        probes: [%{subject: "A", fgp: 1}, %{subject: "B", fgp: 2}, %{subject: "C", fgp: 3}],
        gallery: [%{key: "A/1"}, %{key: "B/2"}, %{key: "C/3"}]
      }

      matrix = Nx.tensor([[0.9, 0.1, 0.2], [0.0, 0.0, 0.0], [0.3, 0.1, 0.1]])

      assert LatentBenchmark.mate_ranks(dataset, matrix, ties: :sort_order) == [1, 2, 2]
      assert LatentBenchmark.mate_ranks(dataset, matrix) == [1, 3, 3]
    end
  end

  test "hit_rates/2 counts hits at each k with a Wilson interval" do
    assert [%{k: 1, hits: 1, n: 4}, %{k: 10, hits: 3, n: 4, ci: {low, high}}] =
             LatentBenchmark.hit_rates([1, 5, 10, 11], [1, 10])

    assert low < 0.75 and high > 0.75
  end

  test "oracle_pose/2: the latent pose maps onto the rolled pose through the fitted transform" do
    {phi, {tx, ty}} = {35.0, {120.0, -40.0}}
    {cos, sin} = {:math.cos(phi * :math.pi() / 180), :math.sin(phi * :math.pi() / 180)}
    move = fn {x, y} -> {cos * x - sin * y + tx, sin * x + cos * y + ty} end

    pairs =
      for p <- [{10.0, 20.0}, {200.0, 40.0}, {90.0, 300.0}, {150.0, 150.0}], do: {p, move.(p)}

    {lx, ly, ltheta} = LatentBenchmark.oracle_pose(pairs, {400.0, 380.0, 12.0})

    {mx, my} = move.({lx, ly})
    assert_in_delta mx, 400.0, 1.0e-6
    assert_in_delta my, 380.0, 1.0e-6
    assert_in_delta ltheta, 12.0 + phi, 1.0e-6
  end

  describe "hit_rate_at_size/4" do
    test "at the full size it is the plain hit rate" do
      ranks = [1, 1, 3, 10, 400]
      assert_in_delta LatentBenchmark.hit_rate_at_size(ranks, 1000, 1000, 1), 0.4, 1.0e-9
      assert_in_delta LatentBenchmark.hit_rate_at_size(ranks, 1000, 1000, 10), 0.8, 1.0e-9
    end

    test "with one other print, the mate wins unless that one is from above it" do
      # rank 11 of 101: 10 of the 100 others beat it
      assert_in_delta LatentBenchmark.hit_rate_at_size([11], 101, 2, 1), 0.9, 1.0e-9
    end

    test "a smaller gallery never lowers the hit rate" do
      ranks = [1, 2, 5, 30, 200, 900]

      rates =
        for g <- [10, 100, 500, 1000], do: LatentBenchmark.hit_rate_at_size(ranks, 1000, g, 1)

      assert rates == Enum.sort(rates, :desc)
    end
  end

  test "scaling_fit/3 recovers the exponent of a power-law miss rate" do
    # Ranks drawn so that P(rank > 1) grows with gallery size: a probe
    # whose mate has c non-mates above it in a gallery of N.
    ranks = for c <- 0..999, do: if(rem(c, 4) == 0, do: 1, else: div(c, 2) + 2)

    %{fnir: %{b: b}, hit: %{beta: beta}, points: points} =
      LatentBenchmark.scaling_fit(ranks, 2000)

    assert b > 0 and b < 1
    assert beta > 0
    assert length(points) == 7
  end

  test "shortlist_hit_rate/3 counts mates kept by any of the shortlists" do
    # gallery 100, top 2% = 2: probe 1 kept by a, probe 2 by b, probe 3 by neither
    a = [1, 50, 9]
    b = [70, 2, 3]
    assert LatentBenchmark.shortlist_hit_rate([a], 100, 0.02) == 1 / 3
    assert LatentBenchmark.shortlist_hit_rate([a, b], 100, 0.02) == 2 / 3
  end

  @tag :tmp_dir
  test "load/1 reads the data format: resolutions, shared minutiae, visibility", %{tmp_dir: dir} do
    person = Path.join([dir, "run1", "P-1"])
    File.mkdir_p!(Path.join(person, "rolled"))
    File.mkdir_p!(Path.join(person, "latents"))

    write = fn path, meta, {h, w} ->
      File.write!(path <> ".json", JSON.encode!(meta))
      Image.write!(Image.new!(w, h, color: 255, bands: 1), path <> ".png")
    end

    write.(
      Path.join([person, "rolled", "fgp07"]),
      %{
        ppi: 500,
        finger: 7,
        minutiae: [%{id: "a", x: 10, y: 20, angle: 30}, %{id: "b", x: 40, y: 50, angle: 60}]
      },
      {64, 48}
    )

    write.(
      Path.join([person, "latents", "latent_2"]),
      %{
        ppi: 1000,
        finger: 7,
        region: "centre",
        rotation: 15,
        analysis: "of value",
        minutiae: [
          %{id: "a", x: 100, y: 200, angle: 90, visibility: "visible"},
          %{id: "c", x: 300, y: 400, angle: 180, visibility: "hidden"},
          # A neighbouring finger's: not the latent's own.
          %{id: "z", x: 5, y: 5, angle: 0, finger: 8, visibility: "visible"}
        ]
      },
      {128, 96}
    )

    dataset = LatentBenchmark.load(dir)
    assert length(dataset.gallery) == 10
    assert [probe] = dataset.probes
    assert probe.key == "P-1/L2" and probe.fgp == 7

    # 1000 ppi halves the latent's coordinates; the rolled print is at 500.
    assert probe.pairs == [{{50.0, 100.0}, {10.0, 20.0}}]
    assert probe.minutiae == [{50.0, 100.0, 90.0}, {150.0, 200.0, 180.0}]
    assert probe.visibility == ["visible", "hidden"]
    assert probe.truth.shared_minutiae == 1 and probe.truth.angle == 15

    assert Nx.shape(LatentBenchmark.latent_image(probe)) == {64, 48}
    assert %{7 => rolled} = LatentBenchmark.rolled_images(dataset, "P-1")
    assert Nx.shape(rolled) == {64, 48}
  end

  test "compare/3 counts discordant probes for McNemar" do
    assert %{only_a: 2, only_b: 1} = LatentBenchmark.compare([1, 1, 1, 9], [1, 9, 9, 1], 1)
  end
end
