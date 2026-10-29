defmodule Faulds.LatentBenchmarkConcurrencyTest do
  # Changes the application env, so not async.
  use ExUnit.Case, async: false

  alias Faulds.LatentBenchmark

  setup do
    before = Application.get_env(:faulds, LatentBenchmark)

    on_exit(fn ->
      if before,
        do: Application.put_env(:faulds, LatentBenchmark, before),
        else: Application.delete_env(:faulds, LatentBenchmark)
    end)
  end

  test "the defaults are the benchmark's, capped at the schedulers" do
    Application.delete_env(:faulds, LatentBenchmark)
    cores = System.schedulers_online()

    expected = [extract: 12, decode: 16, nbis: 16, mcc: 8, dmd: 4]

    for {group, n} <- expected,
        do: assert(LatentBenchmark.concurrency(group) == min(n, cores))

    assert LatentBenchmark.concurrency_groups() == Keyword.keys(expected)
  end

  test "the application env sets a group, uncapped, the rest keep their defaults" do
    Application.put_env(:faulds, LatentBenchmark, concurrency: [extract: 64])

    assert LatentBenchmark.concurrency(:extract) == 64
    assert LatentBenchmark.concurrency(:dmd) == min(4, System.schedulers_online())
  end
end
