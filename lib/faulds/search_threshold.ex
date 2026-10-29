defmodule Faulds.SearchThreshold do
  @moduledoc """
  The open-set threshold on a calibrated search score
  (`Faulds.SearchCalibration`), carried to a deployment's mix of latents
  by covariate-shift reweighting.

  Within a class of LEADER minutiae count, the calibrated score of the
  top non-mated candidate behaves the same from one mix of latents to
  another; what moves a threshold is the share of each class. So the
  reference is the training searches' mate-removed top scores, each with
  its latent's class, and a threshold for a target FPIR is set on them
  reweighted to the deployment's class shares (the counts of the latents
  it searches; no labels needed). Without shares the reference's own mix
  is used.

  For galleries larger than the reference's, each class's share of
  non-mated tops above a threshold can be grown by a per-class size
  factor (`threshold/4`'s `:size` and `:factors`).
  """

  @enforce_keys [:tops, :info]
  defstruct [:tops, :info]

  @typedoc """
  `tops`: the reference searches as `{top, class}`, `top` the calibrated
  score of the mate-removed search's top candidate, `class` its latent's
  `class_of/1`.
  """
  @type t :: %__MODULE__{tops: [{float(), class()}], info: map()}
  @type class :: 0..4
  @type shares :: %{class() => float()}

  # LEADER minutiae counts (at its 0.6 threshold) per class.
  @classes [{0, 0}, {1, 2}, {3, 5}, {6, 11}, {12, :infinity}]

  @doc "The classes' count ranges, `{low, high}`, in class order."
  @spec classes() :: [{non_neg_integer(), non_neg_integer() | :infinity}]
  def classes, do: @classes

  @doc "The class of a latent with `n` LEADER minutiae."
  @spec class_of(non_neg_integer()) :: class()
  def class_of(n) when n >= 12, do: 4
  def class_of(n) when n >= 6, do: 3
  def class_of(n) when n >= 3, do: 2
  def class_of(n) when n >= 1, do: 1
  def class_of(0), do: 0

  @doc "Each class's share of latents with LEADER counts `counts`."
  @spec shares([non_neg_integer()]) :: shares()
  def shares(counts) do
    total = length(counts)
    counts |> Enum.frequencies_by(&class_of/1) |> Map.new(fn {k, c} -> {k, c / total} end)
  end

  @doc """
  The threshold on the calibrated score at which a share `fpir` of
  non-mated searches would still return a candidate (`top > threshold`),
  with the reference reweighted to `shares` (`shares/1` of the
  deployment's latents), or as it is when `shares` is nil. A class the
  reference doesn't hold gets no weight.

  Options, for a gallery larger than the reference's 1,000 prints:
  `:size`, the gallery's size `G`, and `:factors`, `%{class => a}`: each
  class's share of non-mated searches above a threshold is taken to grow
  by `k = 1 + a · ln(G / 1,000)`. Without them, or at `G` ≤ 1,000, `k`
  is 1.
  `:validated_size`, the largest gallery the factors were confirmed on:
  beyond it a negative factor shrinks a share no further, so `k` keeps
  its value there and never turns negative. `size_options/2` gives all
  three from the reference.
  """
  @spec threshold(t(), float(), shares() | nil, keyword()) :: float()
  def threshold(%__MODULE__{tops: tops}, fpir, shares \\ nil, opts \\ [])
      when fpir > 0 and fpir < 1 do
    weight =
      case shares do
        nil ->
          fn _ -> 1.0 end

        shares ->
          source = tops |> Enum.map(&elem(&1, 1)) |> Enum.frequencies()
          total = length(tops)
          fn class -> Map.get(shares, class, 0.0) / (source[class] / total) end
      end

    size = Keyword.get(opts, :size, 1_000)
    factors = Keyword.get(opts, :factors, %{})
    validated = Keyword.get(opts, :validated_size, :infinity)

    growth = fn class ->
      a = Map.get(factors, class, 0.0)
      g = if a < 0 and validated != :infinity, do: min(size, validated), else: size
      if g > 1_000, do: 1 + a * :math.log(g / 1_000), else: 1.0
    end

    tops
    |> Enum.map(fn {top, class} -> {top, weight.(class), growth.(class)} end)
    |> Enum.filter(fn {_, w, _} -> w > 0 end)
    |> weighted_threshold(fpir)
  end

  @doc """
  `threshold/4`'s size options for a gallery of `size` fingers, from the
  per-class factors the reference holds (`info["size_factors"]`); none if
  it holds none.
  """
  @spec size_options(t(), non_neg_integer()) :: keyword()
  def size_options(%__MODULE__{info: info}, size) do
    case info do
      %{"size_factors" => %{"a" => a, "validated_to" => validated}} ->
        [
          size: size,
          factors: Map.new(a, fn {class, f} -> {String.to_integer(class), f * 1.0} end),
          validated_size: validated
        ]

      _ ->
        []
    end
  end

  # The first top, from the highest down, whose cumulative weight (each
  # grown by its class's `k`) would pass `fpir` of the total weight: tops
  # above it hold at most `fpir`.
  defp weighted_threshold(weighted, fpir) do
    total = weighted |> Enum.map(&elem(&1, 1)) |> Enum.sum()

    {_, t} =
      weighted
      |> Enum.sort_by(&elem(&1, 0), :desc)
      |> Enum.reduce_while({0.0, nil}, fn {top, w, k}, {acc, _} ->
        if (acc + w * k) / total > fpir,
          do: {:halt, {acc, top}},
          else: {:cont, {acc + w * k, top}}
      end)

    t
  end

  @doc "Writes `reference` as JSON."
  @spec write!(t(), Path.t()) :: :ok
  def write!(%__MODULE__{tops: tops, info: info}, path) do
    File.mkdir_p!(Path.dirname(path))

    File.write!(
      path,
      JSON.encode!(%{info: info, tops: Enum.map(tops, fn {top, class} -> [top, class] end)})
    )
  end

  @doc "Reads a reference `write!/2` wrote."
  @spec read!(Path.t()) :: t()
  def read!(path) do
    %{"info" => info, "tops" => tops} = path |> File.read!() |> JSON.decode!()
    %__MODULE__{info: info, tops: Enum.map(tops, fn [top, class] -> {top * 1.0, class} end)}
  end
end
