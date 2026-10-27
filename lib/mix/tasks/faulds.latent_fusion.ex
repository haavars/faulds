defmodule Mix.Tasks.Faulds.LatentFusion do
  @shortdoc "Fits the learnt latent fusion on synthetic training runs; reports on validation runs or the benchmark"

  @moduledoc """
  Fits `Faulds.LearnedFusion` on synthetic training runs and
  reports it beside each matcher alone and their z-score sum.

      $ mix faulds.latent_fusion --train data/train --val-runs run-09,run-10
      $ mix faulds.latent_fusion --test data/benchmark --probes REGEX

  `--train DIR`: every run directory under DIR (one run of synthetic
  people each, never one of the benchmark's) is a search set of its own:
  its latents against its own people's rolled prints; `--runs a,b,…`
  keeps only those. The runs in `--val-runs a,b,…` (required) are held
  out, and the model fitted on the rest is reported on them and written
  to `--out` (default `data/calibration/latent_fusion.json`).
  `--matchers` (default `ldmd,fdmd,mcc_leader,nbis,flare`) are
  `LatentBenchmark` methods; `--l2` the penalty (default 1.0).

  Then `Faulds.SearchCalibration` on the same training
  runs, for each score in `--calibrate` (default `ldmd,fused`: a matcher
  or the fused score), written to `--calibration-out` (default
  `data/calibration/latent_search_calibration.json`) and
  reported as `<score>+cal`. `--k` (default 1) is the top candidates per
  search it is fitted on; `--covariates` a comma list of
  `SearchCalibration.covariate_names/0`, or `none`.

  `--test DIR` scores the written models (`--model`, `--calibration` to
  read others) once on a benchmark directory (`--probes REGEX` to keep
  some latents, e.g. the held-out runs), with McNemar against the first
  matcher, Holm-adjusted. Run it once, after the models are settled on
  the validation runs.

  Every method's mate-removed searches are scored with the mate taken
  out of the search (`SearchMetrics.drop_mates/2`): a z-score or a gap
  to the next candidate changes without it. One threshold holds for all
  the searches reported together, the validation runs pooled.

  Score matrices are cached in each run's `.cache/` as the benchmark's.
  EXLA starts without preallocating, at half the GPU.
  """

  use Mix.Task

  alias Faulds.{LatentBenchmark, LearnedFusion, SearchCalibration, SearchMetrics}
  alias Faulds.Matchers.FlareNx

  @default_out "data/calibration/latent_fusion.json"
  @default_calibration_out "data/calibration/latent_search_calibration.json"
  @default_matchers "ldmd,fdmd,mcc_leader,nbis,flare"

  @impl Mix.Task
  def run(args) do
    {opts, _} =
      OptionParser.parse!(args,
        strict: [
          train: :string,
          runs: :string,
          val_runs: :string,
          matchers: :string,
          l2: :float,
          out: :string,
          test: :string,
          probes: :string,
          model: :string,
          calibrate: :string,
          calibration_out: :string,
          calibration: :string,
          k: :integer,
          covariates: :string
        ]
      )

    Mix.Task.run("app.config")

    Application.put_env(:exla, :clients,
      cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.5],
      host: [platform: :host]
    )

    Mix.Task.run("app.start")
    {:ok, _pid} = FlareNx.ensure_started()

    cond do
      dir = opts[:train] ->
        train(dir, opts)

      dir = opts[:test] ->
        test(dir, opts)

      true ->
        Mix.raise("Give --train DIR or --test DIR: see `mix help faulds.latent_fusion`")
    end
  end

  defp train(dir, opts) do
    matchers = String.split(opts[:matchers] || @default_matchers, ",")
    runs = dir |> File.ls!() |> Enum.filter(&File.dir?(Path.join(dir, &1))) |> Enum.sort()

    runs =
      case opts[:runs] do
        nil ->
          runs

        list ->
          wanted = String.split(list, ",")
          Enum.each(wanted, &(&1 in runs || Mix.raise("No run #{&1} under #{dir}")))
          Enum.filter(runs, &(&1 in wanted))
      end

    runs != [] || Mix.raise("No run directories under #{dir}")

    val_runs =
      case opts[:val_runs] do
        nil -> Mix.raise("Give the held-out runs: --val-runs a,b,…")
        list -> String.split(list, ",")
      end

    Enum.each(val_runs, &(&1 in runs || Mix.raise("No run #{&1} under #{dir}")))

    Mix.shell().info(
      "#{length(runs) - length(val_runs)} training runs, validation: #{Enum.join(val_runs, ", ")}"
    )

    sets =
      Map.new(runs, fn run ->
        dataset = LatentBenchmark.load(Path.join(dir, run))

        {micros, matrices} =
          :timer.tc(fn ->
            LatentBenchmark.extract_for(dataset, matchers)
            Enum.map(matchers, &LatentBenchmark.scores(dataset, &1))
          end)

        Mix.shell().info(
          "  #{run}: #{length(dataset.probes)} latents vs #{length(dataset.gallery)} prints, " <>
            "scored in #{Float.round(micros / 6.0e7, 1)} min"
        )

        {run,
         %{
           matrices: matrices,
           mates: mates(dataset),
           covariates: SearchCalibration.covariates(dataset)
         }}
      end)

    training = for {run, set} <- sets, run not in val_runs, do: set

    model =
      LearnedFusion.fit(
        matchers,
        Enum.map(training, &{LearnedFusion.features(&1.matrices), &1.mates}),
        l2: opts[:l2] || 1.0
      )

    out = opts[:out] || @default_out
    LearnedFusion.write!(model, out)
    Mix.shell().info("\nWritten to #{out}; weights (standardised features):")

    for {name, w} <- Enum.zip(LearnedFusion.feature_names(matchers), model.weights),
        do: Mix.shell().info("  #{name}: #{Float.round(w, 3)}")

    # On the fused scores of the runs the fusion was fitted on: about 15
    # weights over millions of pairs, so they are barely in-sample.
    names = covariate_names(opts)

    calibrations =
      Map.new(calibrated(opts), fn base ->
        sets =
          for set <- training do
            {base_scores(base, model, matchers, set.matrices),
             covariate_columns(set.covariates, names), set.mates}
          end

        {base, SearchCalibration.fit(base, sets, k: opts[:k] || 1, covariates: names)}
      end)

    calibration_out = opts[:calibration_out] || @default_calibration_out
    SearchCalibration.write!(calibrations, calibration_out)
    Mix.shell().info("\nCalibrations written to #{calibration_out}; weights (standardised):")

    for {base, cal} <- Enum.sort(calibrations) do
      weights =
        cal.covariates
        |> SearchCalibration.feature_names()
        |> Enum.zip(cal.weights)
        |> Enum.map_join(", ", fn {name, w} -> "#{name} #{Float.round(w, 3)}" end)

      Mix.shell().info("  #{base}+cal (#{cal.info["rows"]} rows): #{weights}")
    end

    # The validation runs pooled: each latent ranked within its own run,
    # one threshold for all.
    results =
      val_runs
      |> Enum.map(&evaluate(model, calibrations, matchers, sets[&1]))
      |> Enum.reduce(&merge_results(&2, &1))

    report("Validation runs", matchers, results)
  end

  defp test(dir, opts) do
    model = LearnedFusion.read!(opts[:model] || @default_out)
    calibration_path = opts[:calibration] || @default_calibration_out

    calibrations =
      if File.exists?(calibration_path), do: SearchCalibration.read!(calibration_path), else: %{}

    dataset = LatentBenchmark.load(dir)

    keep =
      LatentBenchmark.probe_indices(dataset,
        min_shared: 0,
        probes: opts[:probes] && Regex.compile!(opts[:probes])
      )

    index = Nx.tensor(keep)

    LatentBenchmark.extract_for(dataset, model.matchers)

    matrices =
      Enum.map(model.matchers, &(dataset |> LatentBenchmark.scores(&1) |> Nx.take(index)))

    mates = dataset |> mates() |> then(fn all -> Enum.map(keep, &Enum.at(all, &1)) end)

    Mix.shell().info(
      "#{length(keep)} latents vs #{length(dataset.gallery)} prints; model #{inspect(model.info)}; " <>
        "calibrated: #{calibrations |> Map.keys() |> Enum.sort() |> Enum.join(", ")}"
    )

    set = %{
      matrices: matrices,
      mates: mates,
      covariates: dataset |> SearchCalibration.covariates() |> Nx.take(index)
    }

    report("Test", model.matchers, evaluate(model, calibrations, model.matchers, set))
  end

  # Each matcher alone, the z-score sum of all, the fused score and each
  # calibration, as `{name, fun(matrices, covariates)}`, in report order.
  defp candidates(model, calibrations, matchers) do
    alone =
      matchers
      |> Enum.with_index()
      |> Enum.map(fn {name, i} -> {name, fn ms, _ -> Enum.at(ms, i) end} end)

    zsum = fn ms, _ ->
      ms |> Enum.map(&LatentBenchmark.robust_z(&1, 1)) |> Enum.reduce(&Nx.add/2)
    end

    calibrated =
      for {base, cal} <- Enum.sort(calibrations) do
        {"#{base}+cal",
         fn ms, covariates ->
           SearchCalibration.score(
             cal,
             base_scores(base, model, matchers, ms),
             covariate_columns(covariates, cal.covariates)
           )
         end}
      end

    alone ++
      [{"z-sum", zsum}, {"fused", fn ms, _ -> LearnedFusion.score(model, ms) end}] ++ calibrated
  end

  # Every candidate's ranks and searches on one search set, its
  # mate-removed searches scored without the mate; `:order` the names.
  defp evaluate(model, calibrations, matchers, set) do
    removed = Enum.map(set.matrices, &SearchMetrics.drop_mates(&1, set.mates))
    candidates = candidates(model, calibrations, matchers)

    for {name, fun} <- candidates, into: %{order: Enum.map(candidates, &elem(&1, 0))} do
      matrix = fun.(set.matrices, set.covariates)
      tops = removed |> fun.(set.covariates) |> Nx.reduce_max(axes: [1]) |> Nx.to_flat_list()

      {name,
       %{
         ranks: SearchMetrics.mate_ranks(matrix, set.mates),
         searches: SearchMetrics.searches(matrix, set.mates, mate_removed_top: tops)
       }}
    end
  end

  defp merge_results(a, b) do
    Map.merge(a, b, fn
      :order, order, _ -> order
      _, x, y -> %{ranks: x.ranks ++ y.ranks, searches: x.searches ++ y.searches}
    end)
  end

  defp base_scores("fused", model, _matchers, matrices), do: LearnedFusion.score(model, matrices)

  defp base_scores(base, _model, matchers, matrices) do
    case Enum.find_index(matchers, &(&1 == base)) do
      nil ->
        Mix.raise("Can't calibrate #{base}: neither fused nor one of #{Enum.join(matchers, ",")}")

      i ->
        Enum.at(matrices, i)
    end
  end

  defp calibrated(opts), do: String.split(opts[:calibrate] || "ldmd,fused", ",", trim: true)

  defp covariate_names(opts) do
    known = SearchCalibration.covariate_names()

    case opts[:covariates] do
      nil ->
        known

      "none" ->
        []

      list ->
        names = String.split(list, ",")
        Enum.each(names, &(&1 in known || Mix.raise("Unknown covariate #{&1}")))
        names
    end
  end

  # The named columns of `SearchCalibration.covariates/1`'s tensor.
  defp covariate_columns(_covariates, []), do: nil

  defp covariate_columns(covariates, names) do
    all = SearchCalibration.covariate_names()
    columns = Enum.map(names, fn name -> Enum.find_index(all, &(&1 == name)) end)
    Nx.take(covariates, Nx.tensor(columns), axis: 1)
  end

  defp report(title, matchers, results) do
    names = results.order
    baseline = hd(matchers)

    Mix.shell().info(
      "\n#{title}: rank 1 / rank 10 (95% Wilson), FNIR at FPIR 1% / 10% (every mated search also searched mate-removed, one threshold for all), McNemar at rank 1 against #{baseline} (Holm over the rest)\n"
    )

    Mix.shell().info(
      "| Method | Rank 1 | Rank 10 | FNIR@1% | FNIR@10% | only-this / only-#{baseline} / p Holm |"
    )

    Mix.shell().info("|---|---|---|---|---|---|")

    others = Enum.reject(names, &(&1 == baseline))

    tests =
      results[baseline].ranks
      |> SearchMetrics.compare(Enum.map(others, &{&1, results[&1].ranks}), 1)
      |> Map.new(&{&1.name, &1})

    for name <- names do
      %{ranks: ranks, searches: searches} = results[name]
      [r1, r10] = SearchMetrics.hit_rates(ranks, [1, 10])
      [at1, at10] = SearchMetrics.open_set_of(searches, fpirs: [0.01, 0.1])

      test =
        case tests[name] do
          nil -> "—"
          t -> "#{t.only_this} / #{t.only_baseline} / #{Float.round(t.p_holm, 4)}"
        end

      Mix.shell().info(
        "| #{name} | #{pct(r1.rate)} (#{r1.hits}) [#{pct(elem(r1.ci, 0))}–#{pct(elem(r1.ci, 1))}] | " <>
          "#{pct(r10.rate)} | #{pct(at1.fnir)} | #{pct(at10.fnir)} | #{test} |"
      )
    end
  end

  defp mates(dataset) do
    index = dataset.gallery |> Enum.with_index() |> Map.new(fn {g, i} -> {g.key, i} end)
    Enum.map(dataset.probes, &[Map.fetch!(index, "#{&1.subject}/#{&1.fgp}")])
  end

  defp pct(rate), do: "#{Float.round(rate * 100, 1)}%"
end
