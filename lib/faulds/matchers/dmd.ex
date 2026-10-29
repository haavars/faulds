defmodule Faulds.Matchers.Dmd do
  @moduledoc """
  DMD++, the dense minutia descriptor (Pan, Guan, Duan, Feng, Zhou,
  "Minutiae-Anchored Local Dense Representation for Fingerprint
  Matching", arXiv 2507.15297, extending IJCB 2024's DMD —
  github.com/Yu-Yy/DMD, Apache-2.0, academic research only), the matcher
  behind the best published open latent results (F-DMD: 53.15% rank 1 on
  Tsinghua's N2N protocol). A port of `evaluate_mnt.py`'s pipeline.

  Requires `mix faulds.setup_flare`, which also clones DMD (pinned)
  and exports DMD++ to `priv/dmd/dmd_plus.onnx`
  (`export_dmd_onnx.py`).

  ## Template

  One descriptor per minutia: a 128 × 128 patch of the 500 ppi image,
  centred on the minutia and turned so it points along +x
  (`MntDataset`'s affine, 127.5 outside the image, then `(x − 127.5) /
  127.5`), through DMD++: `feature` 12 × 8 × 8 (768, channel-major) and
  `mask` 8 × 8 (64). Minutiae come in this library's convention (`x`, `y`
  from the top left, `angle` clockwise as seen), which is DMD's (its
  README: "clockwise rotation taken as the positive direction"); F-DMD
  takes them from FDD's map (`Minutiae.FddMap`).

  ## Score

    1. `similarity/3`: every latent descriptor against every rolled one,
       masked cosine over both masks (`calculate_score_torchB`); with
       `normalise: true` times `√(n12 / 5)` (the code's `-sn`; its README
       says FDD minutiae don't need it). Then times both minutiae's
       weights, for templates that carry them (`:weight`, e.g. from
       LEADER's confidence; not the reference's), so an unsure minutia
       pairs, supports and scores less; 1 without.
    2. One-to-one pairs (`:assignment`): `:greedy` (default; the best
       remaining pair first, batched on the GPU) or `:hungarian` (exact,
       as the reference's `batch_linear_assignment`).
    3. Relaxation over the pairs (`lsar_score_torchB`): 5 rounds of
       `λ ← λ/2 + Σ ρ λ / 2(n − 1)`, `ρ` the product of sigmoids of the
       pairs' differences in distance (µ 5 px), relative direction and
       radial angle (π/12 each).
    4. The mean `λ` of the top `n_pair` pairs by `λ / λ₀`, `n_pair =
       n_min + round(σ(min(n₁, n₂); µ, τ) (n_max − n_min))`: MALD's FDD
       values n_min 6, n_max 14, τ 0.3, µ 20 by default.
  """

  import Nx.Defn

  require Logger

  alias Faulds.Flare.Warp
  alias Faulds.Minutiae.Hungarian
  alias Faulds.Onnx

  @patch 128
  @border 127.5
  @channels 12
  @batch 256
  # Each byte of packed signs as its 8 values, ±1 as float32 (bit k is value k).
  @signs List.to_tuple(
           for byte <- 0..255,
               do:
                 for(
                   k <- 0..7,
                   into: <<>>,
                   do:
                     <<if(Bitwise.band(Bitwise.bsr(byte, k), 1) == 1, do: 1.0, else: -1.0)::float-32-native>>
                 )
         )

  @type template :: %{
          required(:minutiae) => Nx.Tensor.t() | nil,
          required(:feature) => Nx.Tensor.t() | nil,
          required(:mask) => Nx.Tensor.t() | nil,
          optional(:weight) => Nx.Tensor.t() | nil
        }

  @doc "Whether `dmd_plus.onnx` has been exported."
  @spec available?() :: boolean()
  def available?, do: File.exists?(onnx_path())

  @doc """
  The DMD++ template of `image` (500 ppi, `[0, 1]`) at `minutiae`
  (`%{x:, y:, angle:}`, this library's convention): `%{minutiae: {n, 3},
  feature: {n, 768}, mask: {n, 64}}`; with no minutiae every field is
  `nil` (Nx has no empty tensors).
  """
  @spec extract(Nx.Tensor.t(), [map()]) :: template()
  def extract(_image, []), do: empty()

  def extract(image, minutiae) do
    n = length(minutiae)
    # Padded to bucket(n) patches with copies of the first, dropped after,
    # so the network and EXLA see a few batch shapes: each new one costs
    # ONNX Runtime's cuDNN search and EXLA's compiles.
    transforms = Enum.map(minutiae, &patch_transform/1)
    transforms = transforms ++ List.duplicate(hd(transforms), bucket(n) - n)
    total = length(transforms)

    patches =
      image
      |> Warp.warp_many(transforms, {@patch, @patch}, background: @border, linear: {255.0, 0.0})
      |> Nx.subtract(@border)
      |> Nx.divide(@border)

    {features, masks} =
      0..(total - 1)//@batch
      |> Enum.map(fn start ->
        batch = Nx.slice_along_axis(patches, start, min(@batch, total - start), axis: 0)
        {feature, mask} = Ortex.run(model(), Nx.new_axis(batch, 1))
        {Nx.backend_transfer(feature), Nx.backend_transfer(mask)}
      end)
      |> Enum.unzip()

    %{
      minutiae:
        minutiae |> Enum.map(&[&1.x * 1.0, &1.y * 1.0, &1.angle * 1.0]) |> Nx.tensor(type: :f32),
      feature: features |> Nx.concatenate() |> Nx.slice_along_axis(0, n, axis: 0),
      mask: masks |> Nx.concatenate() |> Nx.slice_along_axis(0, n, axis: 0)
    }
  end

  defp empty, do: %{minutiae: nil, feature: nil, mask: nil}

  defp count(%{minutiae: nil}), do: 0
  defp count(%{minutiae: m}), do: Nx.axis_size(m, 0)

  # Patch pixel u shows image point R(θ)(u − c) + m: the minutia at the
  # centre, its direction along +x. All of a print's patches are cut from
  # one copy of the image (`Warp.warp_many/4`).
  defp patch_transform(%{x: x, y: y, angle: angle}) do
    theta = -:math.pi() * angle / 180.0
    Warp.affine_matrix(1.0, theta, {-x * 1.0, -y * 1.0}, {@patch / 2.0, @patch / 2.0})
  end

  @doc """
  The descriptor similarity of every minutia of `a` with every one of `b`
  (`{n_a, n_b}`): the cosine of their features over both masks.
  `normalise: true` multiplies by `√(n12 / 5)`, `n12` the masks' overlap
  over all 768 values (the reference's `N_mean = 5`).
  """
  @spec similarity(template(), template(), keyword()) :: Nx.Tensor.t()
  def similarity(a, b, opts \\ []) do
    fb = Nx.new_axis(b.feature, 0)
    mb = Nx.new_axis(b.mask, 0)
    s = exla(&similarity_n/4, [a.feature, a.mask, fb, mb])
    s = if opts[:normalise], do: Nx.multiply(s, exla(&overlap_factor_n/2, [a.mask, mb])), else: s
    Nx.squeeze(s, axes: [0])
  end

  # `{b, n1, n2}`: probe `{n1, 768}` against gallery `{b, n2, 768}`.
  defn similarity_n(fa, ma, fb, mb) do
    ma = Nx.tile(ma, [1, @channels])
    mb = Nx.tile(mb, [1, 1, @channels])
    x1 = Nx.sqrt(Nx.dot(ma * fa ** 2, [1], mb, [2]))
    x2 = Nx.sqrt(Nx.dot(ma, [1], fb ** 2 * mb, [2]))
    x12 = Nx.dot(ma * fa, [1], mb * fb, [2])
    Nx.transpose(x12 / Nx.max(x1 * x2, 1.0e-3), axes: [1, 0, 2])
  end

  defn overlap_factor_n(ma, mb) do
    n12 = Nx.dot(Nx.tile(ma, [1, @channels]), [1], Nx.tile(mb, [1, 1, @channels]), [2])
    Nx.transpose(Nx.sqrt(n12 / 5), axes: [1, 0, 2])
  end

  @lsa [min_pair: 6, max_pair: 14, mu: 20.0, tau: 0.3]

  # What the compiled scoring reads of each side (a map of tensors), and
  # of its options.
  @side_keys [:feature, :mask, :minutiae, :valid, :weight, :ridges]
  @packed_keys [:signs, :mask8, :minutiae, :valid, :weight, :ridges]
  @ridge [ridge_mu: 2.5, ridge_tau: 2.0]

  @doc """
  The LSA-R score of latent template `probe` against each of `gallery`
  (a list of templates, or `prepare/2`'s result), as a list of floats in
  gallery order (0 when either side has no minutiae). Options:
  `:assignment` (`:greedy` or `:hungarian`), `:normalise`, and
  `:min_pair`, `:max_pair`, `:mu`, `:tau` (MALD's FDD values by
  default).
  """
  @spec score_many(template(), [template()] | map(), keyword()) :: [float()]
  def score_many(probe, gallery, opts \\ [])

  def score_many(probe, gallery, opts) when is_list(gallery),
    do: score_many(probe, prepare(gallery, opts), opts)

  def score_many(probe, %{chunks: chunks, size: size}, opts) do
    opts = Keyword.merge(@lsa, opts)

    if count(probe) == 0 do
      List.duplicate(0.0, size)
    else
      probe = probe |> pad(bucket(count(probe))) |> to_device()

      chunks
      |> Enum.flat_map(fn {indices, chunk} ->
        Enum.zip(indices, score_chunk(probe, chunk, opts))
      end)
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))
    end
  end

  @doc "`score_many/3` for one pair."
  @spec score(template(), template(), keyword()) :: float()
  def score(probe, rolled, opts \\ []), do: probe |> score_many([rolled], opts) |> hd()

  @doc """
  `gallery` padded and stacked once (with each template's minutia
  weights, 1 where it has none), for scoring many probes against it:
  grouped by size, in chunks of `:chunk` (default 64) templates, on the
  default backend (the GPU).
  """
  @spec prepare([template()], keyword()) :: map()
  def prepare(gallery, opts \\ []) do
    stack(gallery, opts, &count/1, &pad/2, empty(), @side_keys)
  end

  @doc """
  `prepare/2` for templates stored by `pack/1` (`nil` for a print never
  extracted): kept packed on the GPU, 160 bytes a minutia, and unpacked
  chunk by chunk as it's scored, so a large gallery fits.
  """
  @spec prepare_packed([binary() | nil], keyword()) :: map()
  def prepare_packed(gallery, opts \\ []) do
    gallery
    |> Enum.map(&packed_tensors/1)
    |> stack(opts, &packed_count/1, &pad_packed/2, packed_tensors(nil), @packed_keys)
  end

  defp stack(gallery, opts, count, pad, empty, keys) do
    size = Keyword.get(opts, :chunk, 64)

    chunks =
      gallery
      |> Enum.with_index()
      |> Enum.group_by(fn {t, _} -> bucket(count.(t)) end)
      |> Enum.flat_map(fn {n2, members} ->
        members
        |> Enum.chunk_every(size)
        |> Enum.map(fn part ->
          padded = Enum.map(part, fn {t, _} -> pad.(t, n2) end)
          padded = padded ++ List.duplicate(pad.(empty, n2), size - length(part))

          stacked = Map.new(keys, &{&1, stack_host(Enum.map(padded, fn t -> t[&1] end))})

          {Enum.map(part, &elem(&1, 1)), stacked}
        end)
      end)

    %{chunks: chunks, size: length(gallery)}
  end

  @doc """
  `template` as stored: `"DMD1"`, the minutia count (32-bit little-endian),
  then per minutia its `x`, `y`, `angle` (float32), the signs of its 768
  feature values (96 bytes, bit k of byte i the sign of value 8i + k, 1
  for ≥ 0) and its 64 mask values as bytes (`round(255 m)`). Signs lose
  nothing on the synthetic benchmark (`ldmd_bin`: 639 against 641 at rank 1).
  A template without minutiae packs to the 8-byte header.
  """
  @spec pack(template()) :: binary()
  def pack(%{minutiae: nil}), do: <<"DMD1", 0::32-little>>

  # Padded to bucket(n) rows, so EXLA compiles a few shapes rather than one
  # per minutiae count (about 130 ms each, most of a print's packing
  # time), then cut back to n. The bytes are the same.
  def pack(%{minutiae: minutiae, feature: feature, mask: mask}) do
    n = Nx.axis_size(minutiae, 0)
    rows = bucket(n)
    {signs, mask8} = exla(&pack_n/2, [pad_rows(feature, rows), pad_rows(mask, rows)])

    <<"DMD1", n::32-little>> <>
      Nx.to_binary(Nx.as_type(minutiae, :f32)) <>
      binary_part(Nx.to_binary(signs), 0, n * 96) <> binary_part(Nx.to_binary(mask8), 0, n * 64)
  end

  @doc "The template `pack/1` stored: features as ±1, masks as stored."
  @spec unpack(binary()) :: template()
  def unpack(binary) do
    case packed_tensors(binary) do
      %{minutiae: nil} ->
        empty()

      %{minutiae: minutiae, signs: signs, mask8: mask8} ->
        n = Nx.axis_size(minutiae, 0)

        %{
          minutiae: minutiae,
          feature:
            host(
              for(<<byte <- Nx.to_binary(signs)>>, into: <<>>, do: elem(@signs, byte)),
              :f32,
              {n, 768}
            ),
          mask:
            host(
              for(<<b <- Nx.to_binary(mask8)>>, into: <<>>, do: <<b / 255::float-32-native>>),
              :f32,
              {n, 64}
            )
        }
    end
  end

  defp packed_tensors(nil), do: %{minutiae: nil, signs: nil, mask8: nil}
  defp packed_tensors(<<"DMD1", 0::32-little>>), do: packed_tensors(nil)

  defp packed_tensors(<<"DMD1", n::32-little, rest::binary>>) do
    {a, b, c} = {n * 12, n * 96, n * 64}
    <<minutiae::binary-size(^a), signs::binary-size(^b), mask8::binary-size(^c)>> = rest

    %{
      minutiae: host(minutiae, :f32, {n, 3}),
      signs: host(signs, :u8, {n, 96}),
      mask8: host(mask8, :u8, {n, 64})
    }
  end

  defp packed_count(%{minutiae: nil}), do: 0
  defp packed_count(%{minutiae: m}), do: Nx.axis_size(m, 0)

  defp pad_packed(%{minutiae: nil}, n) do
    %{
      signs: zeros(:u8, {n, 96}),
      mask8: zeros(:u8, {n, 64}),
      minutiae: zeros(:f32, {n, 3}),
      valid: zeros(:u8, {n}),
      weight: zeros(:f32, {n}),
      ridges: unknown_ridges(n)
    }
  end

  defp pad_packed(t, n) do
    k = packed_count(t)

    %{
      signs: pad_rows(t.signs, n),
      mask8: pad_rows(t.mask8, n),
      minutiae: pad_rows(t.minutiae, n),
      valid: flags(k, n, :u8),
      weight: flags(k, n, :f32),
      ridges: unknown_ridges(n)
    }
  end

  defn pack_n(feature, mask) do
    mask8 = mask |> Nx.clip(0.0, 1.0) |> Nx.multiply(255) |> Nx.round() |> Nx.as_type(:u8)
    {pack_signs_n(feature), mask8}
  end

  defn pack_signs_n(feature) do
    {n, _} = Nx.shape(feature)
    bits = feature |> Nx.greater_equal(0) |> Nx.as_type(:u8) |> Nx.reshape({n, 96, 8})
    Nx.sum(Nx.left_shift(bits, Nx.iota({1, 1, 8}, type: :u8)), axes: [2]) |> Nx.as_type(:u8)
  end

  # `{b, n, 96}` packed signs to `{b, n, 768}` ±1.
  defn unpack_signs_n(signs) do
    {b, n, _} = Nx.shape(signs)

    signs
    |> Nx.new_axis(3)
    |> Nx.right_shift(Nx.iota({1, 1, 1, 8}, type: :u8))
    |> Nx.bitwise_and(1)
    |> Nx.reshape({b, n, 768})
    |> Nx.as_type(:f32)
    |> Nx.multiply(2)
    |> Nx.subtract(1)
  end

  # Padded sizes, so EXLA compiles a few shapes, not one per print.
  defp bucket(n), do: max(32, div(n + 31, 32) * 32)

  # A chunk's scores, in its order (padding rows dropped by the caller's zip).
  defp score_chunk(probe, %{signs: _} = chunk, opts) do
    if Keyword.get(opts, :assignment, :greedy) != :greedy,
      do: raise(ArgumentError, "a packed gallery is scored with greedy assignment only")

    copts = compiled_opts(opts)

    :packed
    |> kernel(&packed_chunk_scores_n(&1, &2, copts), copts, [
      Map.take(probe, @side_keys),
      Map.take(chunk, @packed_keys)
    ])
    |> Nx.to_flat_list()
  end

  defp score_chunk(probe, chunk, opts) do
    case Keyword.get(opts, :assignment, :greedy) do
      :greedy ->
        copts = compiled_opts(opts)

        :greedy
        |> kernel(&chunk_scores_n(&1, &2, copts), copts, [
          Map.take(probe, @side_keys),
          Map.take(chunk, @side_keys)
        ])
        |> Nx.to_flat_list()

      :hungarian ->
        s = exla(&similarity_n/4, [probe.feature, probe.mask, chunk.feature, chunk.mask])

        s =
          if opts[:normalise],
            do: Nx.multiply(s, exla(&overlap_factor_n/2, [probe.mask, chunk.mask])),
            else: s

        s = exla(&weigh_n/3, [s, probe.weight, chunk.weight])
        {i, j, active} = hungarian_pairs(s, probe.valid, chunk.valid)

        (&relax_sides_n(&1, &2, &3, &4, &5, &6, compiled_opts(opts)))
        |> exla([s, Map.take(probe, @side_keys), Map.take(chunk, @side_keys), i, j, active])
        |> Nx.to_flat_list()
    end
  end

  # A scoring kernel compiled once per VM for each shape and option set,
  # and kept on disk across runs when `:faulds, :exla_cache` names a
  # directory: compiling takes about 2 s a shape, so a first search of a
  # 1,000-print gallery spent about 35 s on it. EXLA's own disk cache
  # knows the shapes but not the function, so the file is named by
  # everything the program depends on, this module's code included,
  # encoded deterministically (a map's keys otherwise come in an order
  # that differs from one VM to the next).
  defp kernel(name, fun, copts, args) do
    templates = Enum.map(args, &Nx.to_template/1)
    key = {__MODULE__, name, copts, templates}

    compiled =
      with nil <- :persistent_term.get(key, nil) do
        # One compile per key, so two searches never write one file.
        :global.trans({key, self()}, fn ->
          with nil <- :persistent_term.get(key, nil) do
            compiled = compile_kernel(fun, templates, disk_cache(key))
            :persistent_term.put(key, compiled)
            compiled
          end
        end)
      end

    apply(compiled, args)
  end

  defp compile_kernel(fun, templates, []), do: Nx.Defn.compile(fun, templates, compiler: EXLA)

  defp compile_kernel(fun, templates, [cache: path] = cache) do
    Nx.Defn.compile(fun, templates, [compiler: EXLA] ++ cache)
  rescue
    # A file another EXLA or driver can't load: compiled again, anew.
    error ->
      Logger.warning("EXLA cache #{path} unusable (#{Exception.message(error)}); recompiling")
      File.rm(path)
      Nx.Defn.compile(fun, templates, [compiler: EXLA] ++ cache)
  end

  defp disk_cache(key) do
    case Application.get_env(:faulds, :exla_cache) do
      nil ->
        []

      dir ->
        depends_on =
          {key, Application.spec(:exla, :vsn), System.get_env("XLA_FLAGS"),
           __MODULE__.module_info(:md5)}

        hash = :crypto.hash(:sha256, :erlang.term_to_binary(depends_on, [:deterministic]))
        [cache: Path.join(dir, "dmd-#{Base.url_encode64(hash, padding: false)}.exla")]
    end
  end

  defp compiled_opts(opts) do
    Keyword.take(
      Keyword.merge(@ridge, opts),
      [:normalise, :ridges] ++ Keyword.keys(@lsa ++ @ridge)
    )
  end

  # Similarity, greedy pairs and relaxation in one compiled call: `a` the
  # probe's tensors, `b` the chunk's (with a batch axis).
  defn chunk_scores_n(a, b, opts \\ []) do
    s =
      similarity_n(a.feature, a.mask, b.feature, b.mask)
      |> maybe_normalise(a.mask, b.mask, opts)
      |> weigh_n(a.weight, b.weight)

    {i, j, active} = greedy_pairs(s, a.valid, b.valid)
    relax_sides_n(s, a, b, i, j, active, opts)
  end

  defn packed_chunk_scores_n(a, b, opts \\ []) do
    s =
      a.feature
      |> packed_similarity_n(a.mask, b.signs, b.mask8, opts)
      |> weigh_n(a.weight, b.weight)

    {i, j, active} = greedy_pairs(s, a.valid, b.valid)
    relax_sides_n(s, a, b, i, j, active, opts)
  end

  # `similarity_n/4` (and the normalisation) against packed templates,
  # whose features are ±1: `fb² = 1`, and the masks are the same for all
  # 12 channels, so the two norms are sums over the 64 cells, not 768
  # values; only `x12` needs the gallery's features.
  defn packed_similarity_n(fa, ma, signs, mask8, opts \\ []) do
    {n1, _} = Nx.shape(fa)
    mb = Nx.as_type(mask8, :f32) / 255
    fb = unpack_signs_n(signs) * Nx.tile(mb, [1, 1, @channels])
    ma_fa2 = (Nx.tile(ma, [1, @channels]) * fa ** 2) |> Nx.reshape({n1, @channels, 64})

    x1 = Nx.sqrt(Nx.dot(Nx.sum(ma_fa2, axes: [1]), [1], mb, [2]))
    n12 = @channels * Nx.dot(ma, [1], mb, [2])
    x12 = Nx.dot(Nx.tile(ma, [1, @channels]) * fa, [1], fb, [2])
    s = x12 / Nx.max(x1 * Nx.sqrt(n12), 1.0e-3)
    s |> normalise_packed(n12, opts) |> Nx.transpose(axes: [1, 0, 2])
  end

  deftransformp normalise_packed(s, n12, opts) do
    if opts[:normalise], do: Nx.multiply(s, Nx.sqrt(Nx.divide(n12, 5))), else: s
  end

  # The relaxation, with ridge-count agreement when `opts[:ridges]`.
  defn relax_sides_n(s, a, b, i, j, active, opts \\ []) do
    compat = ridge_compat(a.ridges, b.ridges, i, j, opts)
    relax_n(s, a.minutiae, b.minutiae, i, j, active, a.valid, b.valid, compat, opts)
  end

  deftransformp ridge_compat(ra, rb, i, j, opts) do
    if opts[:ridges], do: ridge_agreement(ra, rb, i, j, opts), else: Nx.tensor(1.0, type: :f32)
  end

  @doc """
  How well two pairs' ridge counts agree, `{b, p, p}`: for pairs k and l
  (latent minutiae `i_k`, `i_l`; rolled `j_k`, `j_l`), a falling sigmoid
  of `|ra[i_k, i_l] − rb[j_k, j_l]|` (half at `:ridge_mu`, steepness
  `:ridge_tau`); 1 where either count is unknown (255).
  """
  defn ridge_agreement(ra, rb, i, j, opts \\ []) do
    {b, p} = Nx.shape(i)
    {_, n2, _} = Nx.shape(rb)

    r1 = ra |> Nx.take(i, axis: 0) |> Nx.take_along_axis(broadcast_pairs(i, p), axis: 2)

    r2 =
      rb
      |> Nx.take_along_axis(Nx.broadcast(Nx.new_axis(j, 2), {b, p, n2}), axis: 1)
      |> Nx.take_along_axis(broadcast_pairs(j, p), axis: 2)

    known = r1 != 255 and r2 != 255
    diff = Nx.abs(Nx.as_type(r1, :f32) - Nx.as_type(r2, :f32))
    Nx.select(known, sigmoid(diff, opts[:ridge_mu], -opts[:ridge_tau]), 1.0)
  end

  defnp broadcast_pairs(idx, p) do
    {b, _} = Nx.shape(idx)
    Nx.broadcast(Nx.new_axis(idx, 1), {b, p, p})
  end

  # `{b, n1, n2}` similarities times the probe's `{n1}` and each gallery
  # template's `{b, n2}` minutia weights (1 × 1 leaves them exact).
  defn weigh_n(s, wa, wb) do
    {_b, n1, _n2} = Nx.shape(s)
    s * Nx.reshape(wa, {1, n1, 1}) * Nx.new_axis(wb, 1)
  end

  deftransformp maybe_normalise(s, ma, mb, opts) do
    if opts[:normalise], do: Nx.multiply(s, overlap_factor_n(ma, mb)), else: s
  end

  defp exla(fun, args), do: Nx.Defn.jit_apply(fun, args, compiler: EXLA)

  defp pad(%{minutiae: nil}, n) do
    %{
      feature: zeros(:f32, {n, 768}),
      mask: zeros(:f32, {n, 64}),
      minutiae: zeros(:f32, {n, 3}),
      valid: zeros(:u8, {n}),
      weight: zeros(:f32, {n}),
      ridges: unknown_ridges(n)
    }
  end

  defp pad(t, n) do
    k = count(t)

    %{
      feature: pad_rows(t.feature, n),
      mask: pad_rows(t.mask, n),
      minutiae: pad_rows(t.minutiae, n),
      valid: flags(k, n, :u8),
      weight: weight(t, n),
      ridges: ridges(t, n)
    }
  end

  defp weight(%{weight: %Nx.Tensor{} = w}, n), do: w |> Nx.as_type(:f32) |> pad_rows(n)
  defp weight(t, n), do: flags(count(t), n, :f32)

  # The template's ridge counts padded to `{n, n}` with unknowns (255).
  defp ridges(%{ridges: %Nx.Tensor{} = r}, n) do
    k = Nx.axis_size(r, 0)
    fill = :binary.copy(<<255>>, n - k)

    rows =
      for <<row::binary-size(^k) <- Nx.to_binary(Nx.as_type(r, :u8))>>,
        into: <<>>,
        do: row <> fill

    host(rows <> :binary.copy(<<255>>, (n - k) * n), :u8, {n, n})
  end

  defp ridges(_t, n), do: unknown_ridges(n)

  defp unknown_ridges(n), do: host(:binary.copy(<<255>>, n * n), :u8, {n, n})

  # Padding and stacking are done on the host, and each chunk's fields
  # moved to the GPU once: an op on an EXLA tensor compiles a program for
  # every new shape, about a quarter of a second for each minutia count
  # not seen before, on the probe's side and the gallery's.
  defp host(binary, type, shape),
    do: binary |> Nx.from_binary(type, backend: Nx.BinaryBackend) |> Nx.reshape(shape)

  defp zeros(type, shape),
    do:
      host(
        :binary.copy(<<0>>, Tuple.product(shape) * div(elem(Nx.Type.normalize!(type), 1), 8)),
        type,
        shape
      )

  # `t` (`{k, ...}`) with zero rows to `n`.
  defp pad_rows(t, n) do
    shape = Nx.shape(t)
    binary = Nx.to_binary(t)
    row = div(byte_size(binary), elem(shape, 0))

    host(
      binary <> :binary.copy(<<0>>, (n - elem(shape, 0)) * row),
      Nx.type(t),
      put_elem(shape, 0, n)
    )
  end

  # `{n}`: 1 for the first `k`, 0 after.
  defp flags(k, n, :u8), do: host(:binary.copy(<<1>>, k) <> :binary.copy(<<0>>, n - k), :u8, {n})

  defp flags(k, n, :f32),
    do:
      host(
        :binary.copy(<<1.0::float-32-native>>, k) <> :binary.copy(<<0>>, 4 * (n - k)),
        :f32,
        {n}
      )

  defp stack_host([first | _] = tensors) do
    tensors
    |> Enum.map(&Nx.to_binary/1)
    |> IO.iodata_to_binary()
    |> host(Nx.type(first), Tuple.insert_at(Nx.shape(first), 0, length(tensors)))
    |> Nx.backend_transfer(Nx.default_backend())
  end

  defp to_device(side),
    do: Map.new(side, fn {k, t} -> {k, Nx.backend_transfer(t, Nx.default_backend())} end)

  # The best remaining pair, then its row and column are used up, every
  # batch row at once. `{i, j, active}`, each `{b, p}`, p = min(n1, n2).
  defn greedy_pairs(s, valid1, valid2) do
    {b, n1, n2} = Nx.shape(s)
    p = pair_count(s)
    neg = Nx.Constants.neg_infinity(:f32)
    ok_cell = Nx.logical_and(Nx.reshape(valid1, {1, n1, 1}), Nx.new_axis(valid2, 1))
    s0 = Nx.select(Nx.broadcast(ok_cell, {b, n1, n2}), s, neg)

    zeros = Nx.broadcast(Nx.s32(0), {b, p})
    active = Nx.broadcast(Nx.u8(0), {b, p})

    # No row has a valid cell left after min(probe's, largest template's)
    # pairs, so the rest would stay inactive (as `zeros`) anyway.
    steps =
      Nx.min(
        Nx.sum(Nx.as_type(valid1, :s32)),
        Nx.reduce_max(Nx.sum(Nx.as_type(valid2, :s32), axes: [1]))
      )

    {_, i, j, active, _, _} =
      while {s0, i = zeros, j = zeros, active, k = 0, steps}, k < steps do
        flat = Nx.reshape(s0, {b, n1 * n2})
        best = flat |> Nx.argmax(axis: 1) |> Nx.as_type(:s32)
        value = flat |> Nx.take_along_axis(Nx.new_axis(best, 1), axis: 1) |> Nx.squeeze(axes: [1])
        bi = Nx.quotient(best, n2)
        bj = Nx.remainder(best, n2)
        ok = value > neg

        used =
          Nx.logical_or(
            Nx.iota({b, n1, n2}, axis: 1) == Nx.reshape(bi, {b, 1, 1}),
            Nx.iota({b, n1, n2}, axis: 2) == Nx.reshape(bj, {b, 1, 1})
          )

        at = Nx.iota({b, p}, axis: 1) == k
        i = Nx.select(at, Nx.new_axis(bi, 1), i)
        j = Nx.select(at, Nx.new_axis(bj, 1), j)
        active = Nx.select(at, Nx.new_axis(ok, 1), active)
        {Nx.select(used, neg, s0), i, j, active, k + 1, steps}
      end

    {i, j, active}
  end

  deftransformp pair_count(s) do
    {_b, n1, n2} = Nx.shape(s)
    Kernel.min(n1, n2)
  end

  # The exact assignment (largest total similarity) per batch row, by
  # `Minutiae.Hungarian.solve/1` on 1 − S, as the reference's
  # `batch_linear_assignment`.
  defp hungarian_pairs(s, valid1, valid2) do
    {_b, n1, n2} = Nx.shape(s)
    p = Kernel.min(n1, n2)
    k1 = valid1 |> Nx.as_type(:s64) |> Nx.sum() |> Nx.to_number()
    counts = valid2 |> Nx.as_type(:s64) |> Nx.sum(axes: [1]) |> Nx.to_flat_list()

    rows =
      for {matrix, k2} <- Enum.zip(Nx.to_list(s), counts) do
        cost =
          matrix
          |> Enum.take(k1)
          |> Enum.map(fn r -> r |> Enum.take(k2) |> Enum.map(&(1.0 - &1)) end)

        pairs = if k2 == 0, do: [], else: Hungarian.solve(cost)

        Enum.map(pairs, fn {a, c} -> {a, c, 1} end) ++
          List.duplicate({0, 0, 0}, p - length(pairs))
      end

    [i, j, active] =
      for k <- 0..2 do
        rows |> Enum.map(fn r -> Enum.map(r, &elem(&1, k)) end) |> Nx.tensor(type: :s32)
      end

    {i, j, Nx.as_type(active, :u8)}
  end

  defnp(sigmoid(z, mu, tau), do: 1 / (1 + Nx.exp(-tau * Nx.min(z - mu, 100))))

  defnp(wrap180(d), do: Nx.remainder(Nx.remainder(d + 180, 360) + 360, 360) - 180)

  # `lsar_score_torchB`'s relaxation and top-n mean over the pairs.
  defn relax(s, mnt1, mnt2, i, j, active, valid1, valid2, opts \\ []) do
    relax_n(s, mnt1, mnt2, i, j, active, valid1, valid2, Nx.tensor(1.0, type: :f32), opts)
  end

  # `relax/9` with each pair-of-pairs compatibility also times `compat`
  # (`{b, p, p}`, or 1).
  defn relax_n(s, mnt1, mnt2, i, j, active, valid1, valid2, compat, opts \\ []) do
    {b, _n1, n2} = Nx.shape(s)
    {_, p} = Nx.shape(i)
    activef = Nx.as_type(active, :f32)

    lambda0 = Nx.take_along_axis(Nx.reshape(s, {b, :auto}), i * n2 + j, axis: 1) * activef

    m1 = Nx.take(mnt1, i, axis: 0)
    m2 = Nx.take_along_axis(mnt2, Nx.broadcast(Nx.new_axis(j, 2), {b, p, 3}), axis: 1)

    deg = Nx.Constants.pi() / 180
    d1 = Nx.abs(distances(m1) - distances(m2))
    d2 = Nx.abs(wrap180(relative_theta(m1) - relative_theta(m2))) * deg
    d3 = Nx.abs(wrap180(radial(m1) - radial(m2))) * deg

    pi12 = Nx.Constants.pi() / 12
    rp = sigmoid(d1, 5.0, -8.0 / 5) * sigmoid(d2, pi12, -30.0) * sigmoid(d3, pi12, -30.0)
    rp = rp * compat * Nx.new_axis(activef, 2) * Nx.new_axis(activef, 1) * (1 - Nx.eye(p))

    n1_valid = valid1 |> Nx.as_type(:f32) |> Nx.sum()
    n2_valid = valid2 |> Nx.as_type(:f32) |> Nx.sum(axes: [1])
    min_number = Nx.min(n1_valid, n2_valid)
    denom = Nx.new_axis(Nx.max(min_number - 1, 1), 1)

    {lambda, _, _, _} =
      while {lambda = lambda0, rp, denom, k = 0}, k < 5 do
        spread = Nx.sum(rp * Nx.new_axis(lambda, 1), axes: [2]) / denom
        {0.5 * lambda + 0.5 * spread, rp, denom, k + 1}
      end

    efficiency =
      Nx.select(active, lambda / Nx.max(lambda0, 1.0e-6), Nx.Constants.neg_infinity(:f32))

    sorted =
      Nx.take_along_axis(lambda, Nx.argsort(efficiency, axis: 1, direction: :desc), axis: 1)

    n_pair =
      opts[:min_pair] +
        Nx.round(sigmoid(min_number, opts[:mu], opts[:tau]) * (opts[:max_pair] - opts[:min_pair]))

    top = Nx.iota({b, p}, axis: 1) < Nx.new_axis(n_pair, 1)
    score = Nx.sum(sorted * top, axes: [1]) / n_pair
    Nx.select(min_number > 0, score, 0.0)
  end

  defnp distances(m) do
    dx = Nx.new_axis(m[[.., .., 0]], 2) - Nx.new_axis(m[[.., .., 0]], 1)
    dy = Nx.new_axis(m[[.., .., 1]], 2) - Nx.new_axis(m[[.., .., 1]], 1)
    Nx.sqrt(dx ** 2 + dy ** 2)
  end

  defnp(relative_theta(m),
    do: wrap180(Nx.new_axis(m[[.., .., 2]], 2) - Nx.new_axis(m[[.., .., 2]], 1))
  )

  # The direction from minutia k to l relative to k's own: atan2(y_k −
  # y_l, x_l − x_k) + θ_k, as the reference's `distance_R`.
  defnp radial(m) do
    x = m[[.., .., 0]]
    y = m[[.., .., 1]]
    d = Nx.atan2(Nx.new_axis(y, 2) - Nx.new_axis(y, 1), Nx.new_axis(x, 1) - Nx.new_axis(x, 2))
    wrap180(Nx.new_axis(m[[.., .., 2]], 2) + d * (180 / Nx.Constants.pi()))
  end

  defp model do
    case :persistent_term.get({__MODULE__, :model}, nil) do
      nil ->
        model = Onnx.load(onnx_path())
        :persistent_term.put({__MODULE__, :model}, model)
        model

      model ->
        model
    end
  end

  defp onnx_path, do: Application.app_dir(:faulds, "priv/dmd/dmd_plus.onnx")
end
