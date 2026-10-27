import Config

# Compiled Nx backend: the default BinaryBackend is a pure-Elixir
# interpreter, far too slow at fingerprint image sizes.
config :nx, default_backend: EXLA.Backend

# Without preallocation XLA's allocator grows to what the work needs, so
# ONNX Runtime (Ortex on CUDA) and other jobs on the same GPU keep room;
# `memory_fraction` caps that growth.
config :exla, :clients,
  cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.75],
  host: [platform: :host]

# Ortex's CUDA execution provider (vendor/ortex, ONNX Runtime built for
# CUDA 13); `Ortex.load(path, [:cuda, :cpu])` falls back to the CPU.
config :ortex, Ortex.Native, features: ["cuda"]

# LSA-R's compiled scoring kernels, kept across runs (`Matchers.Dmd`);
# the tests compile their own.
if config_env() != :test,
  do: config(:faulds, :exla_cache, Path.expand("../_build/exla_cache", __DIR__))

if File.exists?(Path.join(__DIR__, "local.exs")), do: import_config("local.exs")
