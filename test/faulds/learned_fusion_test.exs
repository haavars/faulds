defmodule Faulds.LearnedFusionTest do
  use ExUnit.Case, async: true

  alias Faulds.{LearnedFusion, SearchMetrics}

  # EXLA's CPU client: the fits need a few hundred mates to be stable,
  # too many pairs for the binary backend.
  setup do
    Nx.default_backend({EXLA.Backend, client: :host})
    :ok
  end

  test "gap/1: the top entry's lead over the second, everyone else's distance to the top" do
    scores = Nx.tensor([[0.2, 0.9, 0.5], [0.4, 0.4, 0.1]])

    gaps =
      scores
      |> LearnedFusion.gap()
      |> Nx.to_list()
      |> Enum.map(fn r -> Enum.map(r, &Float.round(&1, 4)) end)

    assert gaps == [[-0.7, 0.4, -0.4], [0.0, 0.0, -0.3]]
  end

  # Probe i's mate is column i. `signal` scores mates higher (noisily),
  # `noise` doesn't.
  defp matrices(seed, probes \\ 500, gallery \\ 700) do
    key = Nx.Random.key(seed)
    {base, key} = Nx.Random.normal(key, shape: {probes, gallery})
    {noise, _} = Nx.Random.normal(key, shape: {probes, gallery})

    boost =
      Nx.concatenate([Nx.eye(probes), Nx.broadcast(0.0, {probes, gallery - probes})], axis: 1)

    signal = Nx.add(base, Nx.multiply(boost, 2.0))
    {[signal, noise], Enum.map(0..(probes - 1), &[&1])}
  end

  defp fitted(seed) do
    {ms, mates} = matrices(seed)
    LearnedFusion.fit(["signal", "noise"], [{LearnedFusion.features(ms), mates}])
  end

  test "fit/3 leans on the informative matcher and ranks about as well as it alone" do
    model = fitted(1)

    # How much each matcher's features move the score (its raw score and
    # z-score are nearly collinear, so single weights say little).
    {ms, _} = matrices(1)
    features = LearnedFusion.features(ms)

    spread = fn range ->
      w = model.weights |> Enum.zip(model.std) |> Enum.map(fn {w, s} -> w / s end)
      w = Enum.with_index(w, fn x, i -> if i in range, do: x, else: 0.0 end)
      features |> Nx.dot([2], Nx.tensor(w), [0]) |> Nx.standard_deviation() |> Nx.to_number()
    end

    assert spread.(3..5) < 0.25 * spread.(0..2)

    # On data it wasn't fitted on.
    {ms, mates} = matrices(2)
    fused = LearnedFusion.score(model, ms)
    [%{hits: fused_hits}] = SearchMetrics.hit_rates(SearchMetrics.mate_ranks(fused, mates), [1])
    [%{hits: alone}] = SearchMetrics.hit_rates(SearchMetrics.mate_ranks(hd(ms), mates), [1])
    # Here the signal's raw score is already the best ranking possible (a
    # Gaussian shift), so the fusion can only match it, give or take chance.
    assert fused_hits >= 0.95 * alone
  end

  test "score/2 is the standardised linear score of the features" do
    model = fitted(3)
    {ms, _} = matrices(4, 5, 7)
    fused = LearnedFusion.score(model, ms)
    features = LearnedFusion.features(ms)

    x = features[[2, 3]] |> Nx.to_flat_list()

    expected =
      [x, model.mean, model.std, model.weights]
      |> Enum.zip_with(fn [v, m, s, w] -> w * (v - m) / s end)
      |> Enum.sum()
      |> Kernel.+(model.bias)

    assert_in_delta Nx.to_number(fused[[2, 3]]), expected, 1.0e-4
  end

  @tag :tmp_dir
  test "write!/2 and read!/1 round-trip", %{tmp_dir: dir} do
    model = fitted(5)
    path = Path.join(dir, "fusion.json")
    :ok = LearnedFusion.write!(model, path)
    assert LearnedFusion.read!(path) == model
  end
end
