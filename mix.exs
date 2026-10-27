defmodule Faulds.MixProject do
  use Mix.Project

  def project do
    [
      app: :faulds,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      aliases: aliases(),
      description: "Research on latent fingerprint search in Elixir/Nx",
      source_url: "https://github.com/haavars/faulds"
    ]
  end

  def cli do
    [preferred_envs: [precommit: :test]]
  end

  def application do
    [extra_applications: [:logger], mod: {Faulds.Application, []}]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 1.0", override: true},
      {:exla, "~> 1.0", override: true},
      # Ortex ported to ort 2.0.0-rc.15 (ONNX Runtime 1.31, CUDA 13):
      # upstream pins rc.8, whose CUDA build needs CUDA 12. See
      # vendor/ortex/VENDORED.md.
      {:ortex, path: "vendor/ortex"},
      {:image, "~> 0.54"},
      {:req, "~> 0.5"}
    ]
  end

  defp aliases do
    [
      precommit: ["compile --warnings-as-errors", "deps.unlock --unused", "format", "test"]
    ]
  end
end
