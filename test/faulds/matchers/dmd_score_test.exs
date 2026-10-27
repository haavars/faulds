defmodule Faulds.Matchers.DmdScoreTest do
  # The scoring half of DMD (no ONNX model): similarity and LSA-R. Checked
  # against the reference `evaluate_mnt.py` on random templates when
  # written (mates equal to 5 digits; see `Matchers.Dmd`). Not async: its
  # EXLA compilations would hold up the async tests' (one compile lock).
  use ExUnit.Case, async: false

  alias Faulds.Matchers.Dmd

  # The test's own tensors on the host: on EXLA every eager op (the
  # random templates' many) compiles anew for each minutia count, 3–5 s
  # each. Galleries stay within one 32-minutia bucket, so each scoring
  # variant compiles once.
  setup do
    Nx.default_backend(Nx.BinaryBackend)
    :ok
  end

  defp template(n, seed) do
    key = Nx.Random.key(seed)
    {xy, key} = Nx.Random.uniform(key, 0.0, 300.0, shape: {n, 2})
    {theta, key} = Nx.Random.uniform(key, 0.0, 360.0, shape: {n, 1})
    {feature, key} = Nx.Random.normal(key, shape: {n, 768})
    {mask, _key} = Nx.Random.uniform(key, shape: {n, 64})
    %{minutiae: Nx.concatenate([xy, theta], axis: 1), feature: feature, mask: mask}
  end

  # The same minutiae turned by 20° and moved, with noisy descriptors,
  # plus `extra` unrelated ones.
  defp mate(t, extra, seed) do
    phi = 20 * :math.pi() / 180
    r = Nx.tensor([[:math.cos(phi), -:math.sin(phi)], [:math.sin(phi), :math.cos(phi)]])
    xy = t.minutiae[[.., 0..1]] |> Nx.dot([1], r, [1]) |> Nx.add(Nx.tensor([30.0, -20.0]))
    theta = Nx.add(t.minutiae[[.., 2..2]], 20.0)
    {noise, _} = Nx.Random.normal(Nx.Random.key(seed), shape: Nx.shape(t.feature))
    other = template(extra, seed + 1)

    %{
      minutiae: Nx.concatenate([Nx.concatenate([xy, theta], axis: 1), other.minutiae]),
      feature: Nx.concatenate([Nx.add(t.feature, Nx.multiply(noise, 0.8)), other.feature]),
      mask: Nx.concatenate([t.mask, other.mask])
    }
  end

  test "similarity/3 is the cosine over both masks" do
    a = template(2, 1)
    b = template(3, 2)
    s = Dmd.similarity(a, b)

    for i <- 0..1, j <- 0..2 do
      m = Nx.multiply(Nx.tile(a.mask[i], [12]), Nx.tile(b.mask[j], [12]))
      x12 = m |> Nx.multiply(a.feature[i]) |> Nx.multiply(b.feature[j]) |> Nx.sum()
      x1 = m |> Nx.multiply(Nx.pow(a.feature[i], 2)) |> Nx.sum() |> Nx.sqrt()
      x2 = m |> Nx.multiply(Nx.pow(b.feature[j], 2)) |> Nx.sum() |> Nx.sqrt()
      expected = x12 |> Nx.divide(Nx.multiply(x1, x2)) |> Nx.to_number()
      assert_in_delta Nx.to_number(s[i][j]), expected, 1.0e-4
    end
  end

  test "a mate scores far above unrelated prints, greedy and exact alike" do
    probe = template(18, 10)
    gallery = [mate(probe, 10, 20), template(30, 30), template(6, 40)]

    for assignment <- [:greedy, :hungarian] do
      [mated | others] = Dmd.score_many(probe, gallery, assignment: assignment)
      assert mated > 0.4
      assert Enum.all?(others, &(&1 < 0.1))
    end
  end

  test "pack/1 keeps the minutiae, the features' signs and the masks to a byte" do
    t = template(7, 3)
    packed = Dmd.pack(t)
    assert byte_size(packed) == 8 + 7 * (12 + 96 + 64)

    u = Dmd.unpack(packed)
    assert Nx.to_list(u.minutiae) == Nx.to_list(t.minutiae)

    assert Nx.to_list(u.feature) ==
             Nx.to_list(Nx.select(Nx.greater_equal(t.feature, 0), 1.0, -1.0))

    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(u.mask, t.mask)))) <= 0.5 / 255 + 1.0e-6

    empty = %{minutiae: nil, feature: nil, mask: nil}
    assert Dmd.unpack(Dmd.pack(empty)) == empty
  end

  test "a packed gallery scores as its unpacked templates do" do
    probe = template(18, 10) |> Dmd.pack() |> Dmd.unpack()
    gallery = [mate(template(18, 10), 10, 20), template(30, 30), nil, template(6, 40)]
    packed = Enum.map(gallery, &(&1 && Dmd.pack(&1)))

    unpacked =
      Enum.map(packed, fn
        nil -> %{minutiae: nil, feature: nil, mask: nil}
        p -> Dmd.unpack(p)
      end)

    expected = Dmd.score_many(probe, unpacked)
    actual = Dmd.score_many(probe, Dmd.prepare_packed(packed))

    # Near-zero impostor scores move by ~1e-4 between compilations (LSA-R's
    # λ / max(λ₀, 1e-6) amplifies rounding); the mate's doesn't.
    assert_in_delta hd(actual), hd(expected), 1.0e-5
    for {a, e} <- Enum.zip(actual, expected), do: assert_in_delta(a, e, 1.0e-3)
    assert hd(actual) > 0.3 and Enum.at(actual, 2) == 0.0
  end

  test "minutia weights of 1 change nothing; a uniform weight c scales the score by c" do
    probe = template(18, 10)
    gallery = [mate(probe, 10, 20), template(30, 30)]
    ones = fn t -> Map.put(t, :weight, Nx.broadcast(1.0, {Nx.axis_size(t.minutiae, 0)})) end
    halves = Map.put(probe, :weight, Nx.broadcast(0.5, {18}))

    for assignment <- [:greedy, :hungarian] do
      plain = Dmd.score_many(probe, gallery, assignment: assignment)

      assert Dmd.score_many(ones.(probe), Enum.map(gallery, ones), assignment: assignment) ==
               plain

      # LSA-R is linear in the similarities: the same pairs, λ halved.
      for {h, p} <- Enum.zip(Dmd.score_many(halves, gallery, assignment: assignment), plain),
          do: assert_in_delta(h, p / 2, 1.0e-6)
    end
  end

  test "a down-weighted spurious minutia pairs less" do
    # The probe's last 6 minutiae are unrelated to the mate: weighted
    # down, the mate's pairs come from its true minutiae.
    true_part = template(12, 50)
    spurious = template(6, 51)

    probe = %{
      minutiae: Nx.concatenate([true_part.minutiae, spurious.minutiae]),
      feature: Nx.concatenate([true_part.feature, spurious.feature]),
      mask: Nx.concatenate([true_part.mask, spurious.mask])
    }

    weighted =
      Map.put(probe, :weight, Nx.concatenate([Nx.broadcast(1.0, {12}), Nx.broadcast(0.1, {6})]))

    gallery = [mate(true_part, 10, 52)] ++ for(s <- 60..69, do: template(30, s))

    [mated | others] = Dmd.score_many(weighted, gallery)
    [plain_mated | plain_others] = Dmd.score_many(probe, gallery)

    # The impostors' best pairs lean on the spurious minutiae too; the
    # mate's lead over the best impostor doesn't shrink.
    assert mated - Enum.max(others) >= plain_mated - Enum.max(plain_others) - 1.0e-6
  end

  # Symmetric random ridge counts 0..12, 255 on the diagonal.
  defp ridge_counts(n, seed) do
    {r, _} = Nx.Random.randint(Nx.Random.key(seed), 0, 13, shape: {n, n})
    r = Nx.max(r, Nx.transpose(r))
    Nx.select(Nx.eye(n), 255, r) |> Nx.as_type(:u8)
  end

  test "ridge agreement: unknown counts change nothing; disagreeing counts cost a mate-like print" do
    probe = template(18, 10)
    twin = mate(probe, 10, 20)
    gallery = [twin, twin, template(30, 30)]
    plain = Dmd.score_many(probe, gallery)

    # No counts on either side: every factor is exactly 1.
    assert Dmd.score_many(probe, gallery, ridges: true) == plain

    counts = ridge_counts(18, 1)
    probe = Map.put(probe, :ridges, counts)

    # Both copies have the probe's geometry and descriptors; only the
    # first has its ridge counts (its first 18 minutiae are the probe's).
    gallery = [
      Map.put(twin, :ridges, Nx.put_slice(ridge_counts(28, 2), [0, 0], counts)),
      Map.put(twin, :ridges, ridge_counts(28, 3)),
      List.last(gallery)
    ]

    assert Dmd.score_many(probe, gallery) == plain
    [agreeing, disagreeing, _] = Dmd.score_many(probe, gallery, ridges: true)

    # Agreeing counts keep the score (each factor 0.99); random ones cut it.
    assert agreeing > 0.95 * hd(plain)
    assert disagreeing < 0.6 * hd(plain)
  end

  @tag :tmp_dir
  test "a kernel compiled once is kept on disk and loaded from it", %{tmp_dir: dir} do
    probe = template(18, 10)
    gallery = [mate(probe, 10, 20), template(30, 30)]
    plain = Dmd.score_many(probe, gallery)

    Application.put_env(:faulds, :exla_cache, dir)
    on_exit(fn -> Application.delete_env(:faulds, :exla_cache) end)
    # Forget the compiled kernels, as a new VM would.
    forget = fn ->
      for {{Dmd, _, _, _} = key, _} <- :persistent_term.get(), do: :persistent_term.erase(key)
    end

    forget.()
    assert Dmd.score_many(probe, gallery) == plain
    assert [file] = File.ls!(dir)

    forget.()
    assert Dmd.score_many(probe, gallery) == plain
    assert File.ls!(dir) == [file]
  end

  test "no minutiae on either side scores 0" do
    empty = %{minutiae: nil, feature: nil, mask: nil}

    assert Dmd.score_many(empty, [template(5, 1)]) == [0.0]
    assert Dmd.score_many(template(5, 1), [empty]) == [0.0]
  end
end
