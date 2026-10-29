defmodule Mix.Tasks.Faulds.LatentBenchmark do
  @shortdoc "Latent search benchmark: rank-k with CIs, breakdowns, McNemar, open-set FNIR"

  @moduledoc """
  Runs `Faulds.LatentBenchmark` over a dataset directory (layout in
  `docs/data_format.md`) and prints, for each
  method: rank-1/10/50 hit rates with 95% Wilson intervals (ties counted
  against the mate; rank 1 also with Tsinghua's sort-order ties,
  `LatentBenchmark.mate_ranks/3`), the same by
  latent region, rotation, analysis and minutiae count, and McNemar's
  exact test against the baseline method on the same probes, Holm-adjusted
  over the methods; then TAR at FAR 0.1% and 0.01% over all pairs and
  open-set FNIR at FPIR 1/2/10% (`Faulds.SearchMetrics`).

      $ mix faulds.latent_benchmark data/benchmark
      $ mix faulds.latent_benchmark DIR --methods flare,nbis,fuse_z --baseline flare
      $ mix faulds.latent_benchmark DIR --json results.json

  `--methods` defaults to all of `LatentBenchmark.methods/0`; `--baseline`
  to the first method given; `--json` writes every probe's rank per
  method. `--min-shared N` reports only latents that share at least N
  true minutiae with their rolled print (so every one is findable: a tip
  latent can come from skin the rolled print never captured), and
  `--probes REGEX` only latents whose file matches, e.g. one run.
  `nbis_truth_visible` and `mcc_truth_visible` need ground truth that
  labels each latent minutia's visibility.
  `--shortlist` adds how often each method (and each pair's union) keeps
  the mate in the top 0.5–5% of the gallery.
  `--scaling` adds rank 1 at smaller gallery sizes and its power-law
  projection to 10,000 and 100,000 prints
  (`LatentBenchmark.scaling_fit/3`). `--concurrency extract=8,nbis=16`
  sets how many tasks each stage group runs at once
  (`LatentBenchmark.concurrency/1`). Templates and score matrices are cached in `DIR/.cache/`
  (delete it after changing an extractor).

  EXLA starts without preallocating GPU memory and capped at half of it,
  so this can run next to another GPU job.
  """

  use Mix.Task

  alias Faulds.LatentBenchmark
  alias Faulds.SearchMetrics
  alias Faulds.Matchers.FlareNx

  @ks [1, 10, 50]

  @impl Mix.Task
  def run(args) do
    {opts, argv} =
      OptionParser.parse!(args,
        strict: [
          methods: :string,
          baseline: :string,
          json: :string,
          min_shared: :integer,
          probes: :string,
          scaling: :boolean,
          shortlist: :boolean,
          concurrency: :string
        ]
      )

    root =
      case argv do
        [root] ->
          root

        _ ->
          Mix.raise("Usage: mix faulds.latent_benchmark DIR [--methods a,b] [--baseline a]")
      end

    methods = String.split(opts[:methods] || Enum.join(LatentBenchmark.methods(), ","), ",")

    Enum.each(methods, fn method ->
      method in LatentBenchmark.methods() || Mix.raise("Unknown method #{method}")
    end)

    baseline = opts[:baseline] || hd(methods)

    Mix.Task.run("app.config")

    if opts[:concurrency], do: put_concurrency(opts[:concurrency])

    Application.put_env(:exla, :clients,
      cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.5],
      host: [platform: :host]
    )

    Mix.Task.run("app.start")
    {:ok, _pid} = FlareNx.ensure_started()

    full = LatentBenchmark.load(root)

    keep =
      LatentBenchmark.probe_indices(full,
        min_shared: opts[:min_shared] || 0,
        probes: opts[:probes] && Regex.compile!(opts[:probes])
      )

    dataset = %{full | probes: Enum.map(keep, &Enum.at(full.probes, &1))}

    Mix.shell().info(
      "#{length(dataset.probes)} of #{length(full.probes)} latents vs " <>
        "#{length(dataset.gallery)} rolled prints (#{map_size(dataset.subjects)} people)\n"
    )

    # Scored on every probe (one cache), reported on the kept ones.
    LatentBenchmark.extract_for(full, methods)

    ranks =
      Map.new(methods, fn method ->
        {micros, matrix} = :timer.tc(fn -> LatentBenchmark.scores(full, method) end)
        Mix.shell().info("#{method}: scored in #{Float.round(micros / 1.0e6, 1)} s")
        all = LatentBenchmark.mate_ranks(full, matrix)
        sort_order = LatentBenchmark.mate_ranks(full, matrix, ties: :sort_order)
        kept = Nx.take(matrix, Nx.tensor(keep))

        {method,
         {Enum.map(keep, &Enum.at(all, &1)), Enum.map(keep, &Enum.at(sort_order, &1)), kept}}
      end)

    sort_ranks = Map.new(ranks, fn {method, {_, sort_order, _}} -> {method, sort_order} end)
    matrices = Map.new(ranks, fn {method, {_, _, matrix}} -> {method, matrix} end)
    ranks = Map.new(ranks, fn {method, {against, _, _}} -> {method, against} end)

    print_overall(methods, ranks, sort_ranks)
    print_groups(dataset, methods, ranks)
    print_comparisons(methods, baseline, ranks)
    print_operating_points(dataset, methods, matrices)
    if opts[:scaling], do: print_scaling(methods, ranks, length(dataset.gallery))
    if opts[:shortlist], do: print_shortlist(methods, ranks, length(dataset.gallery))

    if path = opts[:json], do: write_json(path, dataset, ranks)
  end

  defp print_overall(methods, ranks, sort_ranks) do
    Mix.shell().info(
      "\n| Method | " <>
        Enum.map_join(@ks, " | ", &"Rank #{&1}") <> " | Median rank | Rank 1, sort-order ties |"
    )

    Mix.shell().info("|---|" <> String.duplicate("---|", length(@ks) + 2))

    for method <- methods do
      cells =
        ranks[method]
        |> LatentBenchmark.hit_rates(@ks)
        |> Enum.map_join(" | ", fn %{hits: hits, rate: rate, ci: {low, high}} ->
          "#{pct(rate)} (#{hits}) [#{pct(low)}–#{pct(high)}]"
        end)

      [%{hits: sort_hits, rate: sort_rate}] = LatentBenchmark.hit_rates(sort_ranks[method], [1])

      Mix.shell().info(
        "| #{method} | #{cells} | #{median(ranks[method])} | #{pct(sort_rate)} (#{sort_hits}) |"
      )
    end
  end

  defp print_groups(dataset, methods, ranks) do
    Mix.shell().info("\nRank 1 / rank 10 by latent:\n")
    Mix.shell().info("| Group | n | " <> Enum.join(methods, " | ") <> " |")
    Mix.shell().info("|---|---|" <> String.duplicate("---|", length(methods)))

    for {label, members} <- LatentBenchmark.groups(dataset) do
      cells =
        Enum.map_join(methods, " | ", fn method ->
          subset = Enum.map(members, &Enum.at(ranks[method], &1))
          [r1, r10] = LatentBenchmark.hit_rates(subset, [1, 10])
          "#{r1.hits} / #{r10.hits}"
        end)

      Mix.shell().info("| #{label} | #{length(members)} | #{cells} |")
    end
  end

  defp print_comparisons(methods, baseline, ranks) do
    others = Enum.reject(methods, &(&1 == baseline))

    if others != [] do
      Mix.shell().info(
        "\nAgainst #{baseline} (McNemar, exact; Holm over the methods at each rank): " <>
          "only-this / only-#{baseline} / p / p Holm\n"
      )

      Mix.shell().info("| Method | Rank 1 | Rank 10 |")
      Mix.shell().info("|---|---|---|")

      [1, 10]
      |> Enum.map(
        &SearchMetrics.compare(ranks[baseline], Enum.map(others, fn m -> {m, ranks[m]} end), &1)
      )
      |> Enum.zip_with(& &1)
      |> Enum.each(fn [row | _] = rows ->
        cells =
          Enum.map_join(rows, " | ", fn r ->
            "#{r.only_this} / #{r.only_baseline} / p=#{Float.round(r.p, 3)} / #{Float.round(r.p_holm, 3)}"
          end)

        Mix.shell().info("| #{row.name} | #{cells} |")
      end)
    end
  end

  # Verification over all pairs (Tsinghua's TAR_flatten) and open-set
  # search, each probe searched mate-removed as a non-mated search
  # (`SearchMetrics.open_set/3`).
  defp print_operating_points(dataset, methods, matrices) do
    index = dataset.gallery |> Enum.with_index() |> Map.new(fn {g, i} -> {g.key, i} end)
    mates = Enum.map(dataset.probes, &[Map.fetch!(index, "#{&1.subject}/#{&1.fgp}")])

    Mix.shell().info(
      "\nTAR at FAR over all pairs; FNIR (rank 1) at FPIR with mate-removed searches as " <>
        "the non-mated ones (all of them, and LFR-Net's half de-mated split):\n"
    )

    Mix.shell().info(
      "| Method | TAR@0.1% | TAR@0.01% | FNIR@FPIR 1% | 2% | 10% | Half: FNIR@FPIR 2% |"
    )

    Mix.shell().info("|---|---|---|---|---|---|---|")

    for method <- methods do
      matrix = matrices[method]
      tars = SearchMetrics.tar_at_far(matrix, mates)
      open = SearchMetrics.open_set(matrix, mates)
      [half] = SearchMetrics.open_set(matrix, mates, fpirs: [0.02], demated: :half)

      Mix.shell().info(
        "| #{method} | #{Enum.map_join(tars, " | ", &pct(&1.tar))} | " <>
          "#{Enum.map_join(open, " | ", &pct(&1.fnir))} | #{pct(half.fnir)} |"
      )
    end
  end

  # Rank 1 at smaller galleries (exact, hypergeometric),
  # power laws fitted to them, and the hit-rate law projected to larger
  # galleries (`LatentBenchmark.scaling_fit/3`).
  defp print_scaling(methods, ranks, full) do
    sizes = Enum.filter([250, 500, 1000, 2000], &(&1 < full))

    Mix.shell().info(
      "\nRank 1 by gallery size (expected over random galleries); TPIR ∝ G^-β and FNIR ∝ G^b fitted; projected by the TPIR law:\n"
    )

    Mix.shell().info(
      "| Method | " <>
        Enum.map_join(sizes, " | ", &"G #{&1}") <> " | G #{full} | β | b | G 10,000 | G 100,000 |"
    )

    Mix.shell().info("|---|" <> String.duplicate("---|", length(sizes) + 5))

    for method <- methods do
      r = ranks[method]
      at = Enum.map_join(sizes, " | ", &pct(LatentBenchmark.hit_rate_at_size(r, full, &1, 1)))
      %{hit: %{a: a, beta: beta}, fnir: %{b: b}} = LatentBenchmark.scaling_fit(r, full)
      project = fn g -> pct(min(1.0, a * :math.pow(g, -beta))) end
      full_rate = pct(LatentBenchmark.hit_rate_at_size(r, full, full, 1))

      Mix.shell().info(
        "| #{method} | #{at} | #{full_rate} | #{Float.round(beta, 3)} | #{Float.round(b, 3)} | #{project.(10_000)} | #{project.(100_000)} |"
      )
    end
  end

  # Each method as a cascade's first stage, alone and with
  # every other in a union shortlist.
  defp print_shortlist(methods, ranks, full) do
    fractions = [0.005, 0.01, 0.02, 0.05]

    Mix.shell().info(
      "\nMate kept by a shortlist of the top fraction of the gallery (one stage, or the union of two):\n"
    )

    Mix.shell().info(
      "| First stage | " <>
        Enum.map_join(fractions, " | ", &"#{&1 * 100}% (#{round(&1 * full)})") <> " |"
    )

    Mix.shell().info("|---|" <> String.duplicate("---|", length(fractions)))

    rows =
      Enum.map(methods, &{&1, [ranks[&1]]}) ++
        for a <- methods, b <- methods, a < b, do: {"#{a} ∪ #{b}", [ranks[a], ranks[b]]}

    for {label, lists} <- rows do
      cells =
        Enum.map_join(fractions, " | ", &pct(LatentBenchmark.shortlist_hit_rate(lists, full, &1)))

      Mix.shell().info("| #{label} | #{cells} |")
    end
  end

  defp write_json(path, dataset, ranks) do
    rows =
      dataset.probes
      |> Enum.with_index()
      |> Enum.map(fn {probe, i} ->
        probe
        |> Map.take([:key, :subject, :fgp, :truth])
        |> Map.put(:ranks, Map.new(ranks, fn {method, list} -> {method, Enum.at(list, i)} end))
      end)

    File.write!(path, JSON.encode!(rows))
    Mix.shell().info("\nPer-probe ranks written to #{path}")
  end

  # "extract=8,nbis=16" into the application env `LatentBenchmark.concurrency/1` reads.
  defp put_concurrency(spec) do
    groups = Map.new(LatentBenchmark.concurrency_groups(), &{Atom.to_string(&1), &1})

    concurrency =
      for pair <- String.split(spec, ","), pair != "" do
        with [name, n] <- String.split(pair, "="),
             {:ok, group} <- Map.fetch(groups, name),
             {n, ""} when n > 0 <- Integer.parse(n) do
          {group, n}
        else
          _ ->
            Mix.raise(
              "--concurrency takes GROUP=N pairs, GROUP one of " <>
                Enum.join(Map.keys(groups), ", ")
            )
        end
      end

    config = Application.get_env(:faulds, LatentBenchmark, [])
    Application.put_env(:faulds, LatentBenchmark, Keyword.put(config, :concurrency, concurrency))
    Mix.shell().info("Concurrency: #{inspect(concurrency)}")
  end

  defp pct(rate), do: "#{Float.round(rate * 100, 1)}%"

  defp median(list) do
    sorted = Enum.sort(list)
    Enum.at(sorted, div(length(sorted), 2))
  end
end
