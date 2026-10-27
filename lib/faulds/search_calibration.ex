defmodule Faulds.SearchCalibration do
  @moduledoc """
  Search-level calibration: maps one score of a search (`ldmd`, or
  `LearnedFusion`'s fused score) to the log-odds that a candidate is the
  mate, so that one threshold means the same from search to search and
  FNIR at a fixed FPIR moves toward the rank-1 miss rate.

  `LearnedFusion` is fitted on every pair, nearly all of them easy
  non-mates, which a threshold never has to beat. This is fitted on the
  candidates a threshold does decide: each training search's top `k`
  (labelled mate or not), and the top candidate of the same search with
  its mate taken out (the non-mated search open-set metrics count). Per
  candidate:

    * the score `s` and its gap to the best other candidate in the search
      (`LearnedFusion.gap/1`);
    * the probe's covariates (`covariates/1`): LEADER's minutiae count
      (as `log(1 + n)`) and their mean confidence, and the FlareNx
      mask's share of the 256 cells (the latent's foreground);
    * `s` times each covariate, so the slope can change with the latent,
      not only the offset.

  The covariates are the same for every candidate of a search, so they
  move a search's scores, not their order; the order changes only if the
  fitted slope in `s` turns negative for some latent (`score/3` doesn't
  prevent it; compare the ranks). The rows are not reweighted, so
  `sigmoid(score)` is the probability that a top candidate is the mate
  when as many searches are non-mated as mated. Fit it on synthetic
  training runs, never the benchmark (`mix faulds.latent_fusion`).
  """

  alias Faulds.{LatentBenchmark, LearnedFusion, Logistic, SearchMetrics}

  @enforce_keys [:base, :covariates, :mean, :std, :weights, :bias]
  defstruct [:base, :covariates, :mean, :std, :weights, :bias, k: 1, info: %{}]

  @type t :: %__MODULE__{
          base: String.t(),
          covariates: [String.t()],
          mean: [float()],
          std: [float()],
          weights: [float()],
          bias: float(),
          k: pos_integer(),
          info: map()
        }

  @covariates ~w(leader_minutiae leader_confidence flare_area)

  @doc "The covariates `covariates/1` gives, in order."
  @spec covariate_names() :: [String.t()]
  def covariate_names, do: @covariates

  @doc """
  The probe covariates of `dataset`, `{probes, 3}` in `covariate_names/0`
  order, from its cached LEADER and FlareNx probe templates.
  """
  @spec covariates(LatentBenchmark.dataset()) :: Nx.Tensor.t()
  def covariates(dataset) do
    leader = LatentBenchmark.templates(dataset, "leader", :probe)
    flare = LatentBenchmark.templates(dataset, "flare", :probe)

    Enum.zip_with(leader, flare, fn xyt, views ->
      confidences =
        for line <- String.split(xyt, "\n", trim: true),
            [_x, _y, _t, q] = String.split(line),
            do: String.to_integer(q) / 100

      n = length(confidences)
      confidence = if n == 0, do: 0.0, else: Enum.sum(confidences) / n
      [:math.log(1 + n), confidence, area(views)]
    end)
    |> Nx.tensor(type: :f32)
  end

  # The share of the mask's cells in the first view (the estimated pose).
  defp area([%{mask: mask} | _]), do: mask |> Nx.from_binary(:f32) |> Nx.mean() |> Nx.to_number()
  defp area(_), do: 0.0

  @doc "The feature names, in order, for covariates `names`."
  @spec feature_names([String.t()]) :: [String.t()]
  def feature_names(names), do: ["s", "gap"] ++ names ++ Enum.map(names, &"s*#{&1}")

  @doc """
  The features of every candidate, `{probes, gallery, 2 + 2c}`, from a
  `{probes, gallery}` score matrix and `{probes, c}` covariates, or nil
  for none (`{probes, gallery, 2}`).
  """
  @spec features(Nx.Tensor.t(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def features(scores, covariates) do
    s = Nx.as_type(scores, :f32)
    own = Nx.stack([s, LearnedFusion.gap(s)], axis: 2)

    case covariates do
      nil ->
        own

      covariates ->
        {probes, gallery} = Nx.shape(s)
        {^probes, c} = Nx.shape(covariates)

        per_search =
          covariates
          |> Nx.as_type(:f32)
          |> Nx.new_axis(1)
          |> Nx.broadcast({probes, gallery, c})

        Nx.concatenate([own, per_search, Nx.multiply(per_search, Nx.new_axis(s, 2))], axis: 2)
    end
  end

  @doc """
  The training rows of one search set, `{x, y}`: each mated search's top
  `k` candidates (1 for the mate) and its mate-removed search's top one
  (0), its features computed without the mate.
  """
  @spec rows(Nx.Tensor.t(), Nx.Tensor.t() | nil, SearchMetrics.mates(), pos_integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def rows(scores, covariates, mates, k) do
    scores = Nx.as_type(scores, :f32)
    features = features(scores, covariates)
    {probes, _gallery, f} = Nx.shape(features)
    {_, top} = Nx.top_k(scores, k: k)

    mated_x =
      features
      |> Nx.take_along_axis(top |> Nx.new_axis(2) |> Nx.broadcast({probes, k, f}), axis: 1)
      |> Nx.reshape({probes * k, f})

    mated_y =
      top
      |> Nx.to_list()
      |> Enum.zip_with(mates, fn cols, mate_cols ->
        Enum.map(cols, &if(&1 in mate_cols, do: 1.0, else: 0.0))
      end)
      |> List.flatten()
      |> Nx.tensor(type: :f32)

    removed = SearchMetrics.drop_mates(scores, mates)
    {_, best} = Nx.top_k(removed, k: 1)

    removed_x =
      removed
      |> features(covariates)
      |> Nx.take_along_axis(best |> Nx.new_axis(2) |> Nx.broadcast({probes, 1, f}), axis: 1)
      |> Nx.reshape({probes, f})

    {Nx.concatenate([mated_x, removed_x]),
     Nx.concatenate([mated_y, Nx.broadcast(Nx.tensor(0.0, type: :f32), {probes})])}
  end

  @doc """
  Fits on `sets`, each `{scores, covariates, mates}`: one search set's
  `{probes, gallery}` matrix of `base`'s scores, `covariates/1` of its
  probes (nil for none, with `:covariates` `[]`) and its mates. Options: `:k` (default 1), `:covariates` (the names of
  the covariate columns, default `covariate_names/0`), `:l2` (default
  1.0).
  """
  @spec fit(String.t(), [{Nx.Tensor.t(), Nx.Tensor.t() | nil, SearchMetrics.mates()}], keyword()) ::
          t()
  def fit(base, sets, opts \\ []) do
    k = Keyword.get(opts, :k, 1)
    names = Keyword.get(opts, :covariates, @covariates)
    l2 = Keyword.get(opts, :l2, 1.0)

    chunks =
      Enum.map(sets, fn {scores, covariates, mates} ->
        c = if covariates, do: elem(Nx.shape(covariates), 1), else: 0

        c == length(names) ||
          raise ArgumentError, "#{c} covariate columns for #{length(names)} names"

        rows(scores, covariates, mates, k)
      end)

    fit = Logistic.fit(chunks, l2: l2, balance: false)

    %__MODULE__{
      base: base,
      covariates: names,
      mean: fit.mean,
      std: fit.std,
      weights: fit.weights,
      bias: fit.bias,
      k: k,
      info: %{
        "l2" => l2,
        "rows" => fit.rows,
        "mates" => fit.positives,
        "features" => feature_names(names)
      }
    }
  end

  @doc """
  The calibrated score (log-odds) of every candidate, `{probes, gallery}`,
  from the base scores and the probes' covariates (the model's columns,
  nil if it has none).
  """
  @spec score(t(), Nx.Tensor.t(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def score(%__MODULE__{} = model, scores, covariates) do
    {scale, offset} = Logistic.folded(Map.from_struct(model))

    scores
    |> features(covariates)
    |> Nx.dot([2], Nx.tensor(scale, type: :f32), [0])
    |> Nx.add(offset)
  end

  @doc "Writes the models, keyed by name, as JSON."
  @spec write!(%{String.t() => t()}, Path.t()) :: :ok
  def write!(models, path) do
    File.mkdir_p!(Path.dirname(path))

    File.write!(
      path,
      JSON.encode!(Map.new(models, fn {name, m} -> {name, Map.from_struct(m)} end))
    )
  end

  @doc "Reads the models `write!/2` wrote."
  @spec read!(Path.t()) :: %{String.t() => t()}
  def read!(path) do
    path
    |> File.read!()
    |> JSON.decode!()
    |> Map.new(fn {name, map} ->
      {name,
       %__MODULE__{
         base: map["base"],
         covariates: map["covariates"],
         mean: map["mean"],
         std: map["std"],
         weights: map["weights"],
         bias: map["bias"],
         k: map["k"] || 1,
         info: map["info"] || %{}
       }}
    end)
  end
end
