defmodule Faulds.Onnx do
  @moduledoc """
  Loads the ONNX models (FlareNx, LEADER, DMD++) through Ortex
  with the configured execution providers, in order of preference:
  `config :faulds, :onnx_providers` (default `[:cuda, :cpu]`). ONNX
  Runtime falls back to the next one when a provider can't start, so
  `[:cuda, :cpu]` runs on the CPU without a GPU.
  """

  @doc """
  The Ortex model at `path`, on the configured providers. `deterministic:
  true` makes its results the same from run to run (ONNX Runtime's
  deterministic compute and cuDNN's default convolution algorithm,
  `vendor/ortex/VENDORED.md`), which FlareNx needs and LEADER and DMD++
  already are.
  """
  @spec load(Path.t(), keyword()) :: %Ortex.Model{}
  def load(path, opts \\ []), do: Ortex.load(path, providers(), 3, opts)

  @doc "The configured execution providers."
  @spec providers() :: [atom()]
  def providers, do: Application.get_env(:faulds, :onnx_providers, [:cuda, :cpu])
end
