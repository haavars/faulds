defmodule Mix.Tasks.Faulds.LatentImpostors do
  @shortdoc "FlareNx impostor scores against mask overlap, and how correlated its views are"

  @moduledoc """
  Two impostor studies on the latent benchmark's cached templates
  (`Faulds.LatentBenchmark`):

    1. how the spread of a FlareNx method's impostor cosines depends on
       the overlap of the two masks, before and after the overlap factor
       (`LatentBenchmark.overlap_stats/3`). The factor is right if the
       raw spread falls as `overlap^-1/2` and the normalised one is flat;
    2. how correlated the impostor scores of several methods (`--views`),
       or of one method's own probe views, are
       (`LatentBenchmark.impostor_correlation/2`): views that err on the
       same impostors can't gain much in a max.

  Uses no mate rank, so nothing it shows is tuned on the benchmark.

      $ mix faulds.latent_impostors data/benchmark \\
          --method flare --views flare,flare_reg,flare_unet,flare_reg_unet

  `--probes REGEX` keeps the latents whose file matches, as in
  `mix faulds.latent_benchmark`; `--bins` (default 10).
  """

  use Mix.Task

  alias Faulds.LatentBenchmark

  @impl Mix.Task
  def run(args) do
    {opts, argv} =
      OptionParser.parse!(args,
        strict: [method: :string, views: :string, probes: :string, bins: :integer]
      )

    root =
      case argv do
        [root] ->
          root

        _ ->
          Mix.raise("Usage: mix faulds.latent_impostors DIR [--method flare] [--views a,b]")
      end

    Mix.Task.run("app.config")

    Application.put_env(:exla, :clients,
      cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.5],
      host: [platform: :host]
    )

    Mix.Task.run("app.start")

    full = LatentBenchmark.load(root)

    keep =
      LatentBenchmark.probe_indices(full, probes: opts[:probes] && Regex.compile!(opts[:probes]))

    dataset = %{full | probes: Enum.map(keep, &Enum.at(full.probes, &1))}
    method = opts[:method] || "flare"

    Mix.shell().info(
      "#{length(dataset.probes)} latents vs #{length(dataset.gallery)} rolled prints\n"
    )

    print_overlap(LatentBenchmark.overlap_stats(dataset, method, opts[:bins] || 10), method)

    views = LatentBenchmark.view_matrices(dataset, method)

    if length(views) > 1 do
      labels = Enum.map(1..length(views), &"view #{&1}")

      print_correlation(
        "#{method}'s probe views",
        labels,
        LatentBenchmark.impostor_correlation(dataset, views)
      )
    end

    if names = opts[:views] do
      names = String.split(names, ",")
      matrices = Enum.map(names, &rows(full, keep, LatentBenchmark.scores(full, &1)))
      print_correlation("methods", names, LatentBenchmark.impostor_correlation(dataset, matrices))
    end
  end

  # Score matrices are cached for every probe; keep the reported ones.
  defp rows(full, keep, matrix) do
    if length(keep) == length(full.probes), do: matrix, else: Nx.take(matrix, Nx.tensor(keep))
  end

  defp print_overlap(stats, method) do
    Mix.shell().info(
      "#{method}: #{stats.pairs} impostor pairs (every view) in #{length(stats.bins)} bins by mask overlap\n"
    )

    Mix.shell().info("| Overlap (cells) | Mean cosine | Spread | Spread × √(n/256) |")
    Mix.shell().info("|---|---|---|---|")

    for bin <- stats.bins do
      Mix.shell().info(
        "| #{f(bin.overlap, 1)} | #{f(bin.mean, 4)} | #{f(bin.sd, 4)} | #{f(bin.sd_normalised, 4)} |"
      )
    end

    Mix.shell().info("""

    Spread ∝ overlap^#{f(stats.slope, 2)} raw, ^#{f(stats.slope_normalised, 2)} normalised (−0.5 and 0 if the factor fits).
    Median overlap: impostors #{f(stats.impostor_overlap, 1)}, mates #{f(stats.genuine_overlap, 1)} (best view); mates' median cosine #{f(stats.genuine_cosine, 3)}.
    """)
  end

  defp print_correlation(title, labels, matrix) do
    Mix.shell().info("Impostor score correlation, #{title}:\n")
    Mix.shell().info("| | " <> Enum.join(labels, " | ") <> " |")
    Mix.shell().info("|---|" <> String.duplicate("---|", length(labels)))

    for {label, row} <- Enum.zip(labels, matrix) do
      Mix.shell().info("| #{label} | " <> Enum.map_join(row, " | ", &f(&1, 2)) <> " |")
    end

    Mix.shell().info("")
  end

  defp f(nil, _), do: "–"
  defp f(x, digits), do: :erlang.float_to_binary(x * 1.0, decimals: digits)
end
