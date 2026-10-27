defmodule Faulds.Onnx do
  @moduledoc """
  Loads the ONNX models (FlareNx, LEADER, DMD++) through Ortex
  with the configured execution providers, in order of preference:
  `config :faulds, :onnx_providers` (default `[:cuda, :cpu]`). ONNX
  Runtime falls back to the next one when a provider can't start, so
  `[:cuda, :cpu]` runs on the CPU without a GPU.
  """

  @doc "The Ortex model at `path`, on the configured providers."
  @spec load(Path.t()) :: %Ortex.Model{}
  def load(path), do: Ortex.load(path, providers())

  @doc "The configured execution providers."
  @spec providers() :: [atom()]
  def providers, do: Application.get_env(:faulds, :onnx_providers, [:cuda, :cpu])
end
