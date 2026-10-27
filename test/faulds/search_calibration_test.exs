defmodule Faulds.SearchCalibrationTest do
  use ExUnit.Case, async: true

  alias Faulds.{SearchCalibration, SearchMetrics}

  setup do
    Nx.default_backend({EXLA.Backend, client: :host})
    :ok
  end

  # Probe i's mate is column i, 6 above its non-mates. Half the probes
  # are "rich" (covariate 1): every score of theirs, mate or not, sits 4
  # higher, as a latent with many minutiae scores high against
  # everything. A raw threshold set by the rich searches' non-mates then
  # misses the poor searches' mates.
  defp search_set(seed, probes \\ 400, gallery \\ 500) do
    key = Nx.Random.key(seed)
    {base, _} = Nx.Random.normal(key, shape: {probes, gallery})
    rich = Nx.remainder(Nx.iota({probes}), 2)

    mate =
      Nx.concatenate([Nx.eye(probes), Nx.broadcast(0.0, {probes, gallery - probes})], axis: 1)

    scores =
      base
      |> Nx.add(Nx.multiply(mate, 6.0))
      |> Nx.add(Nx.multiply(Nx.new_axis(rich, 1), 4.0))

    covariates = rich |> Nx.as_type(:f32) |> Nx.new_axis(1)
    {scores, covariates, Enum.map(0..(probes - 1), &[&1])}
  end

  defp fnir_at_1(scores, mates, removed_top) do
    [%{fnir: fnir}] =
      SearchMetrics.open_set(scores, mates, fpirs: [0.01], mate_removed_top: removed_top)

    fnir
  end

  test "features/2: the score, its gap, the covariates and their products with the score" do
    scores = Nx.tensor([[0.2, 0.9, 0.5]])
    features = SearchCalibration.features(scores, Nx.tensor([[2.0]]))
    assert Nx.shape(features) == {1, 3, 4}

    assert features[[0, 1]] |> Nx.to_list() |> Enum.map(&Float.round(&1, 4)) ==
             [0.9, 0.4, 2.0, 1.8]

    assert Nx.shape(SearchCalibration.features(scores, nil)) == {1, 3, 2}
  end

  test "rows/4: each search's top k, and its mate-removed top scored without the mate" do
    scores = Nx.tensor([[0.9, 0.2, 0.5], [0.1, 0.6, 0.8]])
    {x, y} = SearchCalibration.rows(scores, nil, [[0], [1]], 1)

    # Probe 0's mate tops its search; probe 1's top is a non-mate, 0.2
    # ahead of the mate. Mate-removed, probe 0's top is 0.5, 0.3 ahead of
    # the next, and probe 1's lead grows to 0.7.
    assert Nx.to_list(y) == [1.0, 0.0, 0.0, 0.0]

    assert x |> Nx.to_list() |> Enum.map(fn r -> Enum.map(r, &Float.round(&1, 4)) end) ==
             [[0.9, 0.4], [0.8, 0.2], [0.5, 0.3], [0.8, 0.7]]
  end

  test "fit/3 with the covariate puts poor and rich searches on one scale" do
    {scores, covariates, mates} = search_set(1)
    model = SearchCalibration.fit("s", [{scores, covariates, mates}], covariates: ["rich"])

    {scores, covariates, mates} = search_set(2)
    removed = SearchMetrics.drop_mates(scores, mates)
    top = fn m -> m |> Nx.reduce_max(axes: [1]) |> Nx.to_flat_list() end

    raw = fnir_at_1(scores, mates, top.(removed))

    calibrated =
      SearchCalibration.score(model, scores, covariates)
      |> fnir_at_1(
        mates,
        removed |> then(&SearchCalibration.score(model, &1, covariates)) |> top.()
      )

    # The raw threshold sits among the rich non-mates' tops (about 4 +
    # 3.5), above nearly every poor mate (about 6); calibrated, it doesn't.
    assert raw > 0.4
    assert calibrated < 0.5 * raw

    # Within a search the order holds.
    ranks = SearchMetrics.mate_ranks(scores, mates)

    assert SearchMetrics.mate_ranks(SearchCalibration.score(model, scores, covariates), mates) ==
             ranks
  end

  @tag :tmp_dir
  test "write!/2 and read!/1 round-trip", %{tmp_dir: dir} do
    {scores, covariates, mates} = search_set(3, 50, 60)
    model = SearchCalibration.fit("s", [{scores, covariates, mates}], covariates: ["rich"], k: 2)
    path = Path.join(dir, "calibration.json")
    :ok = SearchCalibration.write!(%{"s" => model}, path)
    assert SearchCalibration.read!(path) == %{"s" => model}
  end
end
