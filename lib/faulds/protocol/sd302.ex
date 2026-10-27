defmodule Faulds.Protocol.Sd302 do
  @moduledoc """
  NIST SD302's latent names and the protocols published on them, from their
  public definition files, before the images themselves arrive:

    * Tsinghua's N2N list (`datasets/N2NLatent_genuine_pairs.txt` in the
      DMD repository): 3,383 SD302e latents, each with its finger in a
      2,000-print gallery of device U rolled prints, one per finger of the
      200 subjects. `tsinghua/1`.
    * NIST's errata (`erratasd302.txt`): latents flagged as palm or joint
      prints, moved to another subject, mislabelled, or not friction
      ridge. `errata/1`; `corrected/2` drops them from a protocol, the
      "corrected list" to report beside the published one.

  `mix faulds.fetch_protocols` downloads both into
  `priv/protocols/`.

  A latent name is eleven fields: subject, activity (1A–7D, "3" for some
  sets), hand (L, R, X), encounter, development technique (BP, IN, WT,
  BT, CA), digitizer, resolution, bit depth, channels, LP number and
  source, e.g. `00002302_1E_L_L01_BP_S04_1200PPI_8BPC_1CH_LP05_1`.
  SD302e names carry the native resolution and depth, SD302h names
  `1000PPI_8BPC`, so `canonical_key/1` leaves those out to join them.
  """

  alias Faulds.Protocol

  @name ~r/(\d{8})_([0-9A-Z]+)_([LRX])_([0-9A-Z]+)_([A-Z]{2})_([0-9A-Z]+)_(\d+)PPI_(\d+)BPC_(\d+)CH_LP(\d+)_(\d+)/

  @type latent :: %{
          subject: String.t(),
          activity: String.t(),
          hand: String.t(),
          encounter: String.t(),
          technique: String.t(),
          digitizer: String.t(),
          ppi: pos_integer(),
          bits: pos_integer(),
          channels: pos_integer(),
          lp: pos_integer(),
          source: String.t()
        }

  @doc "The fields of a latent name (a path or extension is ignored); nil if it isn't one."
  @spec parse(String.t()) :: latent() | nil
  def parse(name) do
    case Regex.run(@name, Path.basename(name)) do
      [_, subject, activity, hand, encounter, technique, digitizer, ppi, bits, ch, lp, source] ->
        %{
          subject: subject,
          activity: activity,
          hand: hand,
          encounter: encounter,
          technique: technique,
          digitizer: digitizer,
          ppi: String.to_integer(ppi),
          bits: String.to_integer(bits),
          channels: String.to_integer(ch),
          lp: String.to_integer(lp),
          source: source
        }

      nil ->
        nil
    end
  end

  @doc """
  The latent's identity across SD302's distributions: every field but
  resolution, bit depth and channels.
  """
  @spec canonical_key(String.t() | latent()) :: String.t()
  def canonical_key(name) when is_binary(name), do: name |> parse() |> canonical_key()

  def canonical_key(%{} = l) do
    lp = l.lp |> Integer.to_string() |> String.pad_leading(2, "0")

    Enum.join(
      [
        l.subject,
        l.activity,
        l.hand,
        l.encounter,
        l.technique,
        l.digitizer,
        "LP" <> lp,
        l.source
      ],
      "_"
    )
  end

  @doc """
  Tsinghua's N2N protocol from its mate list (`latent,subject_finger` per
  line): the listed latents as probes, and the gallery of every finger
  (01–10) of every subject in the list, as `subject_finger` ids in that
  order (device U rolled prints; the order Tsinghua's columns would have
  if sorted by name).
  """
  @spec tsinghua(Path.t()) :: Protocol.t()
  def tsinghua(path) do
    pairs =
      path
      |> File.read!()
      |> String.split(["\r\n", "\n"], trim: true)
      |> Enum.map(fn line ->
        [latent, print] = line |> String.split(",") |> Enum.map(&String.trim/1)
        {latent, print}
      end)

    probes =
      Enum.map(pairs, fn {latent, print} ->
        %{id: latent, subject: parse!(latent).subject, finger: print_finger(print)}
      end)

    subjects = probes |> Enum.map(& &1.subject) |> Enum.uniq() |> Enum.sort()

    gallery =
      for subject <- subjects, finger <- 1..10 do
        %{id: print_id(subject, finger), subject: subject, finger: finger}
      end

    %Protocol{
      name: "N2N-Tsinghua G-2K (published list)",
      probes: probes,
      gallery: gallery,
      listed: Map.new(pairs, fn {latent, print} -> {latent, [print]} end),
      notes: "SD302e latents vs device U rolled, one per finger; Tsinghua ties are sort order"
    }
  end

  defp parse!(latent), do: parse(latent) || raise(ArgumentError, "not an SD302 latent: #{latent}")

  defp print_finger(print) do
    [_subject, finger] = String.split(print, "_")
    String.to_integer(finger)
  end

  defp print_id(subject, finger),
    do: "#{subject}_#{finger |> Integer.to_string() |> String.pad_leading(2, "0")}"

  @doc """
  The latents NIST's errata flags, `%{canonical key => reason}`: `:palm`,
  `:joint` (or interdigital), `:palm_or_joint`, `:moved` (to another
  subject; the old name is flagged), `:finger` (source finger
  mislabelled), `:hand` (hand maybe mislabelled), `:mislabelled` (to be
  removed), `:not_friction_ridge`, `:dimensions` (image size changed
  slightly, labels fine) or `:other`; for a latent named more than once,
  the newest reason other than `:dimensions`. Bullets naming a device, not a
  latent (swapped exemplar thumbs), are left out: none names device U.
  """
  @spec errata(Path.t()) :: %{String.t() => atom()}
  def errata(path) do
    path
    |> File.read!()
    |> String.split("\n")
    |> Enum.chunk_while(
      [],
      fn line, acc ->
        if String.match?(line, ~r/^\s*\*|^##/),
          do: {:cont, Enum.reverse(acc), [line]},
          else: {:cont, [line | acc]}
      end,
      &{:cont, Enum.reverse(&1), []}
    )
    |> Enum.flat_map(&bullet_flags/1)
    # Newest first; a size change doesn't hide a label problem.
    |> Enum.reduce(%{}, fn {key, reason}, acc ->
      Map.update(acc, key, reason, &if(&1 == :dimensions, do: reason, else: &1))
    end)
  end

  defp bullet_flags([]), do: []

  defp bullet_flags([head | rest]) do
    reason = [head | rest] |> Enum.join(" ") |> String.downcase() |> reason()

    [head | rest]
    |> Enum.flat_map(fn line ->
      names = Regex.scan(@name, line) |> Enum.map(&hd/1)
      # "A changed to B": A is the wrong label.
      if reason == :moved, do: Enum.take(names, 1), else: names
    end)
    |> Enum.map(&{canonical_key(&1), reason})
  end

  defp reason(text) do
    cond do
      text =~ "dimensions" -> :dimensions
      text =~ "joint or palm" or text =~ "palm/joint" -> :palm_or_joint
      text =~ "palm" -> :palm
      text =~ "joint" or text =~ "interdigital" -> :joint
      text =~ "belong to subject" or text =~ "belongs to subject" -> :moved
      text =~ "source is" -> :finger
      text =~ "l/r hand" -> :hand
      text =~ "removed" -> :mislabelled
      text =~ "not friction ridge" -> :not_friction_ridge
      true -> :other
    end
  end

  @doc """
  `protocol` without the probes `errata` flags for anything but
  `:dimensions`: the corrected list. (Re-labelling them from SD302i's
  `finger_positions.csv` needs the data.)
  """
  @spec corrected(Protocol.t(), %{String.t() => atom()}) :: Protocol.t()
  def corrected(protocol, errata) do
    Protocol.filter_probes(
      protocol,
      &(Map.get(errata, canonical_key(&1.id), :dimensions) == :dimensions),
      String.replace(protocol.name, "published list", "corrected list")
    )
  end

  @doc """
  Probes whose name gives a hand (L or R) that disagrees with their
  finger (1–5 right, 6–10 left): suspect labels, for a stratum.
  """
  @spec hand_mismatches(Protocol.t()) :: [String.t()]
  def hand_mismatches(protocol) do
    for probe <- protocol.probes,
        hand = parse!(probe.id).hand,
        hand in ["L", "R"],
        probe.finger != nil,
        hand == "R" != probe.finger <= 5,
        do: probe.id
  end
end
