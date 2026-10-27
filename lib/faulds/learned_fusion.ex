defmodule Faulds.LearnedFusion do
  @moduledoc """
  A learnt fusion: one score
  per latent–print pair from several matchers' score matrices, by
  logistic regression, so it ranks better than any one of them and sits
  on one scale from search to search (a threshold can act on it:
  open-set search, FPIR).

  Per matcher, three features of each candidate, all from its own
  search's row of scores:

    * the raw score;
    * its robust z-score over the row (`LatentBenchmark.robust_z/2`);
    * its gap to the best other candidate in the row: positive only for
      the top one, so a clear winner stands out from a crowded top.

  Fitted on every pair, mates against all non-mates with the two classes
  weighted equally, features standardised, an L2 penalty, by Newton's
  method (`Logistic`). Train it on synthetic training runs, never
  the benchmark (`mix faulds.latent_fusion`).

  `score/2` gives the fitted linear score (log-odds up to a constant) for
  every pair; it ranks, and `SearchMetrics` thresholds it.
  """

  alias Faulds.{LatentBenchmark, Logistic}

  @enforce_keys [:matchers, :mean, :std, :weights, :bias]
  defstruct [:matchers, :mean, :std, :weights, :bias, info: %{}]

  @type t :: %__MODULE__{
          matchers: [String.t()],
          mean: [float()],
          std: [float()],
          weights: [float()],
          bias: float(),
          info: map()
        }

  @per_matcher ~w(raw z gap)

  @doc "The feature names, in order, for `matchers`."
  @spec feature_names([String.t()]) :: [String.t()]
  def feature_names(matchers), do: for(m <- matchers, f <- @per_matcher, do: "#{m}.#{f}")

  @doc """
  The features of every pair, `{probes, gallery, 3 × matchers}`, from
  each matcher's `{probes, gallery}` score matrix (in `matchers` order).
  """
  @spec features([Nx.Tensor.t()]) :: Nx.Tensor.t()
  def features(matrices) do
    matrices
    |> Enum.flat_map(fn s ->
      s = Nx.as_type(s, :f32)
      [s, LatentBenchmark.robust_z(s, 1), gap(s)]
    end)
    |> Nx.stack(axis: 2)
  end

  @doc """
  Each score less the best other score in its row: for the row's top
  entry, top − second (0 when tied); for the rest, score − top.
  """
  @spec gap(Nx.Tensor.t()) :: Nx.Tensor.t()
  def gap(scores) do
    {top2, _} = Nx.top_k(scores, k: 2)
    first = Nx.slice_along_axis(top2, 0, 1, axis: 1)
    second = Nx.slice_along_axis(top2, 1, 1, axis: 1)
    best_other = Nx.select(Nx.greater_equal(scores, first), second, first)
    Nx.subtract(scores, best_other)
  end

  @doc """
  Fits the model on `datasets`, each `{features, mates}`: `features/1`'s
  tensor of one search set and its mates (as in `SearchMetrics`). Every
  pair is a training row, mates against all non-mates, the two classes
  weighted equally; sampling only the close non-mates would teach it
  that a high score from a weak matcher means a non-mate. Options: `:l2`
  (default 1.0, on standardised features), `:iterations` (Newton steps,
  default 20).
  """
  @spec fit([String.t()], [{Nx.Tensor.t(), [[non_neg_integer()]]}], keyword()) :: t()
  def fit(matchers, datasets, opts \\ []) do
    l2 = Keyword.get(opts, :l2, 1.0)
    f = 3 * length(matchers)

    chunks =
      Enum.map(datasets, fn {features, mates} ->
        {probes, gallery, ^f} = Nx.shape(features)
        x = features |> Nx.reshape({probes * gallery, f}) |> Nx.as_type(:f64)
        {x, mate_labels(probes, gallery, mates)}
      end)

    fit = Logistic.fit(chunks, l2: l2, iterations: Keyword.get(opts, :iterations, 20))

    %__MODULE__{
      matchers: matchers,
      mean: fit.mean,
      std: fit.std,
      weights: fit.weights,
      bias: fit.bias,
      info: %{
        "l2" => l2,
        "pairs" => fit.rows,
        "mates" => fit.positives,
        "features" => feature_names(matchers)
      }
    }
  end

  defp mate_labels(probes, gallery, mates) do
    indices = for {cols, p} <- Enum.with_index(mates), c <- cols, do: p * gallery + c
    empty = Nx.broadcast(Nx.tensor(0.0, type: :f64), {probes * gallery})

    if indices == [],
      do: empty,
      else:
        Nx.indexed_put(
          empty,
          Nx.tensor(Enum.map(indices, &[&1])),
          Nx.broadcast(Nx.tensor(1.0, type: :f64), {length(indices)})
        )
  end

  @doc "The fused score of every pair, `{probes, gallery}`, from the matchers' matrices (in the model's order)."
  @spec score(t(), [Nx.Tensor.t()]) :: Nx.Tensor.t()
  def score(%__MODULE__{} = model, matrices) do
    length(matrices) == length(model.matchers) ||
      raise ArgumentError, "#{length(matrices)} matrices for #{length(model.matchers)} matchers"

    {scale, offset} = Logistic.folded(Map.from_struct(model))

    matrices
    |> features()
    |> Nx.dot([2], Nx.tensor(scale, type: :f32), [0])
    |> Nx.add(offset)
  end

  @doc "Writes the model as JSON."
  @spec write!(t(), Path.t()) :: :ok
  def write!(%__MODULE__{} = model, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, JSON.encode!(Map.from_struct(model)))
  end

  @doc "Reads a model `write!/2` wrote."
  @spec read!(Path.t()) :: t()
  def read!(path) do
    map = path |> File.read!() |> JSON.decode!()

    %__MODULE__{
      matchers: map["matchers"],
      mean: map["mean"],
      std: map["std"],
      weights: map["weights"],
      bias: map["bias"],
      info: map["info"] || %{}
    }
  end
end
