defmodule Mix.Tasks.Faulds.LatentMinutiae do
  @shortdoc "How well each minutiae extractor finds the latent benchmark's true minutiae"

  @moduledoc """
  Recall and precision of latent minutiae extractors against the exact
  ground truth of a dataset
  (`Faulds.LatentBenchmark.minutiae_accuracy/3`): `nbis`
  (`mindtct`), `fdd` (FDD's minutiae map, `Minutiae.FddMap`) and
  `leader` (`Minutiae.Leader`), by
  default at 8 px and 20°, overall and by latent region and analysis.

      $ mix faulds.latent_minutiae data/train/run-01 \\
          --extractors nbis,fdd,leader

  `--visible` scores against `visible` minutiae only, for latents whose
  ground truth labels each minutia's visibility.
  `--tolerance PX`, `--angle-tolerance DEG`; `--probes REGEX` as in
  `mix faulds.latent_benchmark`. Templates are cached in
  `DIR/.cache/` like the benchmark's.
  """

  use Mix.Task

  alias Faulds.LatentBenchmark
  alias Faulds.Matchers.FlareNx

  @impl Mix.Task
  def run(args) do
    {opts, argv} =
      OptionParser.parse!(args,
        strict: [
          extractors: :string,
          probes: :string,
          visible: :boolean,
          tolerance: :integer,
          angle_tolerance: :integer
        ]
      )

    root =
      case argv do
        [root] -> root
        _ -> Mix.raise("Usage: mix faulds.latent_minutiae DIR [--extractors nbis,fdd]")
      end

    Mix.Task.run("app.config")

    Application.put_env(:exla, :clients,
      cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.5],
      host: [platform: :host]
    )

    Mix.Task.run("app.start")
    {:ok, _pid} = FlareNx.ensure_started()

    full = LatentBenchmark.load(root)

    keep =
      LatentBenchmark.probe_indices(full, probes: opts[:probes] && Regex.compile!(opts[:probes]))

    dataset = %{full | probes: Enum.map(keep, &Enum.at(full.probes, &1))}
    extractors = String.split(opts[:extractors] || "nbis,fdd,leader", ",")
    tolerances = [tolerance: opts[:tolerance] || 8, angle_tolerance: opts[:angle_tolerance] || 20]
    tolerances = if opts[:visible], do: tolerances ++ [only: :visible], else: tolerances

    groups = [
      {"all", fn _ -> true end},
      {"region: centre", &(&1.region == "centre")},
      {"region: tip", &(&1.region == "tip")},
      {"region: side", &(&1.region == "side")},
      {"of value", &(&1.analysis == "value")},
      {"limited value", &(&1.analysis == "limited")}
    ]

    Mix.shell().info(
      "#{length(dataset.probes)} latents, #{tolerances[:tolerance]} px / #{tolerances[:angle_tolerance]}°: recall / precision (found of true, detected)\n"
    )

    Mix.shell().info("| Latents | n | " <> Enum.join(extractors, " | ") <> " |")
    Mix.shell().info("|---|---|" <> String.duplicate("---|", length(extractors)))

    for {label, pred} <- groups do
      subset = %{dataset | probes: Enum.filter(dataset.probes, &pred.(&1.truth))}

      if subset.probes != [] do
        cells =
          Enum.map_join(extractors, " | ", fn extractor ->
            a = LatentBenchmark.minutiae_accuracy(subset, extractor, tolerances)
            "#{pct(a.recall)} / #{pct(a.precision)} (#{a.found} of #{a.truth}, #{a.detected})"
          end)

        Mix.shell().info("| #{label} | #{length(subset.probes)} | #{cells} |")
      end
    end
  end

  defp pct(rate), do: "#{Float.round(rate * 100, 1)}%"
end
