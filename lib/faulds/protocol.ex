defmodule Faulds.Protocol do
  @moduledoc """
  A 1:N benchmark protocol: the probes, the gallery in its order (so
  `:sort_order` ties are reproducible) and who is whose mate. Every
  published latent number hides one, so a result is only reported with its protocol.

  Probes and gallery entries are `%{id:, subject:, finger:}` (finger
  1–10, ANSI/NIST positions; nil when a probe's finger isn't known).
  `listed` is the protocol's own mate list, `%{probe id => [gallery
  id]}`, when it has one (Tsinghua's N2N list); `mates/2` gives the mate
  columns for `Faulds.SearchMetrics` at a credit level.
  """

  @enforce_keys [:name, :probes, :gallery]
  defstruct [:name, :probes, :gallery, listed: nil, notes: nil]

  @type entry :: %{id: String.t(), subject: String.t(), finger: 1..10 | nil}
  @type t :: %__MODULE__{
          name: String.t(),
          probes: [entry()],
          gallery: [entry()],
          listed: %{String.t() => [String.t()]} | nil,
          notes: String.t() | nil
        }

  @doc """
  Each probe's mate columns in the gallery at `credit`:

    * `:listed` (default when the protocol has a list): its listed mates;
    * `:finger` (default otherwise): every impression of the probe's
      finger (Region credit, AFID's "any impression of the mated finger");
    * `:subject`: any print of the probe's subject.
  """
  @spec mates(t(), :listed | :finger | :subject | nil) :: [[non_neg_integer()]]
  def mates(protocol, credit \\ nil)

  def mates(%__MODULE__{listed: listed} = protocol, nil),
    do: mates(protocol, if(listed, do: :listed, else: :finger))

  def mates(%__MODULE__{listed: nil}, :listed),
    do: raise(ArgumentError, "this protocol has no mate list")

  def mates(%__MODULE__{probes: probes, gallery: gallery, listed: listed}, :listed) do
    column = gallery |> Enum.with_index() |> Map.new(fn {g, i} -> {g.id, i} end)

    Enum.map(probes, fn probe ->
      listed |> Map.get(probe.id, []) |> Enum.flat_map(&List.wrap(column[&1])) |> Enum.sort()
    end)
  end

  def mates(%__MODULE__{probes: probes, gallery: gallery}, credit)
      when credit in [:finger, :subject] do
    key = if credit == :finger, do: &{&1.subject, &1.finger}, else: & &1.subject
    columns = gallery |> Enum.with_index() |> Enum.group_by(&key.(elem(&1, 0)), &elem(&1, 1))

    Enum.map(probes, fn probe ->
      if credit == :finger and probe.finger == nil,
        do: [],
        else: Map.get(columns, key.(probe), [])
    end)
  end

  @doc """
  Gallery entries that are another impression of a probe's finger but not
  listed as its mate, as `[{probe id, gallery id}]`. Under a list such an
  entry counts as an impostor that is really a mate (the Tsinghua
  evaluator would score a hit on it as a miss), so a distractor tier must
  have none. Empty for a protocol without a list.
  """
  @spec unlisted_mates(t()) :: [{String.t(), String.t()}]
  def unlisted_mates(%__MODULE__{listed: nil}), do: []

  def unlisted_mates(%__MODULE__{probes: probes, gallery: gallery, listed: listed}) do
    by_finger = Enum.group_by(gallery, &{&1.subject, &1.finger}, & &1.id)

    for probe <- probes,
        probe.finger != nil,
        id <- Map.get(by_finger, {probe.subject, probe.finger}, []),
        id not in Map.get(listed, probe.id, []),
        do: {probe.id, id}
  end

  @doc """
  The protocol with `distractors` appended to its gallery (after the
  existing entries, so their columns keep their numbers), named `name`.
  """
  @spec with_distractors(t(), [entry()], String.t()) :: t()
  def with_distractors(protocol, distractors, name) do
    ids = MapSet.new(protocol.gallery, & &1.id)

    if dup = Enum.find(distractors, &MapSet.member?(ids, &1.id)),
      do: raise(ArgumentError, "distractor #{dup.id} is already in the gallery")

    %{protocol | name: name, gallery: protocol.gallery ++ distractors}
  end

  @doc "The protocol with only the probes `keep?` accepts, named `name`."
  @spec filter_probes(t(), (entry() -> boolean()), String.t()) :: t()
  def filter_probes(protocol, keep?, name) do
    probes = Enum.filter(protocol.probes, keep?)
    ids = MapSet.new(probes, & &1.id)
    listed = protocol.listed && Map.filter(protocol.listed, fn {id, _} -> id in ids end)
    %{protocol | name: name, probes: probes, listed: listed}
  end
end
