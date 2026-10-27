defmodule Faulds.Matchers.Flare do
  @moduledoc """
  FLARE's template score (Pan, Guan, Duan, Feng, Zhou, "Fixed-Length
  Dense Fingerprint Representation With Alignment and Robust
  Enhancement", IEEE TIFS 2026; github.com/Yu-Yy/FLARE, Academic Research
  License, non-commercial): a 16 × 16 grid of 12-channel feature vectors
  and a foreground mask, compared by masked cosine. The templates come
  from `Matchers.FlareNx`, the native port of FLARE's extraction.

  Also the Python that `mix faulds.setup_flare` exports FLARE's networks
  with.
  """

  @channels 12
  # The denominator's floor, as FLARE's own scoring has it.
  @mask_clip 1.0e-3

  @doc """
  The masked cosine of two templates (`%{feature: {3072}, mask: {256}}`):
  the features compared over the cells both masks keep.
  """
  @spec score(map(), map()) :: {:ok, float()}
  def score(%{feature: f1, mask: m1}, %{feature: f2, mask: m2}) do
    combined_mask = Nx.multiply(Nx.tile(m1, [@channels]), Nx.tile(m2, [@channels]))

    numerator = combined_mask |> Nx.multiply(f1) |> Nx.multiply(f2) |> Nx.sum()

    denom1 = combined_mask |> Nx.multiply(Nx.pow(f1, 2)) |> Nx.sum() |> Nx.sqrt()
    denom2 = combined_mask |> Nx.multiply(Nx.pow(f2, 2)) |> Nx.sum() |> Nx.sqrt()
    denominator = denom1 |> Nx.multiply(denom2) |> Nx.max(@mask_clip)

    {:ok, numerator |> Nx.divide(denominator) |> Nx.to_number()}
  end

  @doc """
  A template collapsed into one L2-normalised `{3072}` vector, for an
  approximate first stage: the cells its own mask marks invalid zeroed,
  then normalised. Its cosine approximates `score/2` where two masks
  mostly agree; `score/2` is pairwise (each pair's own combined mask), so
  no single vector reproduces it exactly.
  """
  @spec ann_vector(%{feature: Nx.Tensor.t(), mask: Nx.Tensor.t()}) :: Nx.Tensor.t()
  def ann_vector(%{feature: feature, mask: mask}) do
    masked = Nx.multiply(feature, Nx.tile(mask, [@channels]))
    norm = masked |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.max(@mask_clip)
    Nx.divide(masked, norm)
  end

  @doc """
  The Python FLARE's export scripts run with: its venv (`priv/flare/.venv`,
  gitignored) when it exists, else the first `python3` on `PATH`; nil if
  neither exists.
  """
  @spec python() :: String.t() | nil
  def python do
    venv_python = Path.join(Application.app_dir(:faulds, "priv/flare"), ".venv/bin/python3")
    if File.exists?(venv_python), do: venv_python, else: System.find_executable("python3")
  end
end
