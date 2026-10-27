defmodule Mix.Tasks.Faulds.FetchProtocols do
  @shortdoc "Downloads the published SD302 protocol files (Tsinghua's N2N list, NIST's errata)"

  @moduledoc """
  Downloads into `priv/protocols/` (gitignored) the files
  `Faulds.Protocol.Sd302` builds protocols from, and checks
  each against the SHA-256 it was first read with, so a protocol can't
  change under a result:

    * `N2NLatent_genuine_pairs.txt`, Tsinghua's N2N mate list, from the
      DMD repository (Apache-2.0) at commit 982c1b1;
    * `erratasd302.txt`, NIST's SD302 errata (updated by NIST from time
      to time: a new version fails the check; read it, then pass
      `--accept-new` to keep it and update `@files` here).

      $ mix faulds.fetch_protocols
      $ mix faulds.fetch_protocols --summary

  `--summary` prints what the protocols hold: probes, fingers, the
  latents the errata flags and the probes whose hand disagrees with their
  finger.
  """

  use Mix.Task

  alias Faulds.Protocol
  alias Faulds.Protocol.Sd302

  @dir "priv/protocols"
  @files [
    {"N2NLatent_genuine_pairs.txt",
     "https://raw.githubusercontent.com/Yu-Yy/DMD/982c1b18c235fd2f27e1f73bab7ace32c2b1174b/datasets/N2NLatent_genuine_pairs.txt",
     "472f01f90d4871a1297bdd63427bb21e19bc93eb84fe67b506714babdc81fe09"},
    {"erratasd302.txt", "https://www.nist.gov/document/erratasd302txt",
     "2d3012f31a2c0f5c7830b75a9aac295a5a2275fe73cdba4b7ae0bb148370ab20"}
  ]

  @impl Mix.Task
  def run(args) do
    {opts, _} = OptionParser.parse!(args, strict: [accept_new: :boolean, summary: :boolean])
    Application.ensure_all_started(:req)
    File.mkdir_p!(@dir)

    for {name, url, sha} <- @files do
      path = Path.join(@dir, name)

      unless File.exists?(path) do
        Mix.shell().info("Downloading #{name}")
        File.write!(path, Req.get!(url, decode_body: false).body)
      end

      actual = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)

      cond do
        actual == sha ->
          Mix.shell().info("#{name}: ok")

        opts[:accept_new] ->
          Mix.shell().info("#{name}: kept a new version, SHA-256 #{actual}")

        true ->
          Mix.raise(
            "#{path} has SHA-256 #{actual}, not #{sha}: see `mix help faulds.fetch_protocols`"
          )
      end
    end

    if opts[:summary], do: summary()
  end

  defp summary do
    protocol = Sd302.tsinghua(Path.join(@dir, "N2NLatent_genuine_pairs.txt"))
    errata = Sd302.errata(Path.join(@dir, "erratasd302.txt"))
    corrected = Sd302.corrected(protocol, errata)
    fingers = protocol.probes |> Enum.uniq_by(&{&1.subject, &1.finger}) |> length()
    subjects = protocol.probes |> Enum.uniq_by(& &1.subject) |> length()

    flagged =
      for probe <- protocol.probes,
          reason = errata[Sd302.canonical_key(probe.id)],
          do: reason

    with_hand = Enum.count(protocol.probes, &(Sd302.parse(&1.id).hand in ["L", "R"]))

    Mix.shell().info("""

    #{protocol.name}: #{length(protocol.probes)} latents of #{fingers} fingers, #{subjects} subjects; gallery #{length(protocol.gallery)}
      unlisted mates in the gallery: #{length(Protocol.unlisted_mates(protocol))}
      flagged by the errata: #{length(flagged)} #{inspect(Enum.frequencies(flagged))}
      hand in the name disagrees with the finger: #{length(Sd302.hand_mismatches(protocol))} of #{with_hand}
    #{corrected.name}: #{length(corrected.probes)} latents
    """)
  end
end
