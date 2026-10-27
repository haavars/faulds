defmodule Faulds.Matchers.Mcc do
  @moduledoc """
  Minutia Cylinder-Code (Cappelli, Ferrara, Maltoni, "Minutia
  Cylinder-Code: A New Representation and Matching Technique for
  Fingerprint Recognition", TPAMI 2010), written from the paper with
  its published parameters (the same in the open `mvisat/mcc-cuda`
  constants and the Apache-2.0 `dperaltac/mpi-afis`). With the same
  minutiae MCC beats `bozorth3` by 15–21 rank-1 points on NIST SD27
  (AFID's supplementary material).

  ## Cylinders

  Each minutia's neighbourhood as a 16 × 16 × 6 cylinder (radius R 70
  px, at 500 ppi): cell (i, j) sits at `m + ΔS R(θ) (i − 7.5, j − 7.5)`,
  turned with the minutia, and layer k collects neighbours whose
  direction differs from the minutia's by about `−π + (k + ½) ΔD`. A
  cell's value is `Σ G_σS(d) · ∫ G_σD` over the neighbours within 3σS
  of it, one bit (> μψ = 0.01). A cell is valid inside the circle and
  inside the minutiae's convex hull grown by Ω 50 px; a cylinder is kept
  if 75% of its in-circle cells are valid and it has at least 2
  neighbours.

  ## Score

  Two cylinders compare as `1 − ‖a − b‖ / (‖a‖ + ‖b‖)` over the cells
  valid in both, 0 if their minutiae's directions differ by more than
  π/2 or they share fewer than 60% valid cells. The latent's cylinders
  are paired one-to-one with the rolled print's and consolidated as
  LSA-R, exactly as `Matchers.Dmd` does (same assignment and relaxation,
  the same constants), with MCC's n_P: 4 + round(σ(min(n_A, n_B); 20,
  0.4) · 8).

  Minutiae in this library's convention (`x`, `y` from the top left, `angle`
  in degrees clockwise as seen); any consistent one gives the same
  scores, since only differences of positions and angles enter.
  """

  import Nx.Defn

  alias Faulds.Matchers.Dmd

  @r 70.0
  @ns 16
  @nd 6
  @sigma_s 28.0 / 3
  @sigma_d 2 * :math.pi() / 9
  @mu_psi 0.01
  @omega 50.0
  @min_vc 0.75
  @min_m 2
  @min_me 0.6
  @delta_theta :math.pi() / 2
  @cells @ns * @ns * @nd

  @lsa [min_pair: 4, max_pair: 12, mu: 20.0, tau: 0.4]

  @type template :: %{
          minutiae: Nx.Tensor.t() | nil,
          cells: Nx.Tensor.t() | nil,
          valid: Nx.Tensor.t() | nil
        }

  @doc """
  The cylinders of `minutiae` (`%{x:, y:, angle:}`): `%{minutiae: {n, 3},
  cells: {n, 1536}, valid: {n, 1536}}` (`u8`) for the n kept, every field
  `nil` if none is.
  """
  @spec cylinders([map()]) :: template()
  def cylinders(minutiae) when length(minutiae) < 3, do: empty()

  def cylinders(minutiae) when is_list(minutiae), do: build(minutiae)

  defp build(minutiae) do
    n = length(minutiae)
    edges = minutiae |> Enum.map(&{&1.x * 1.0, &1.y * 1.0}) |> hull_edges()
    if edges == [], do: empty(), else: build(minutiae, n, edges)
  end

  defp build(minutiae, n, edges) do
    rows = Enum.map(minutiae, &[&1.x * 1.0, &1.y * 1.0, &1.angle * :math.pi() / 180])

    # Padded to a few sizes, so EXLA compiles a few shapes, not one per
    # print: minutiae far off the print (no neighbours, never kept), hull
    # edges repeated (the same inside test).
    far = List.duplicate([1.0e6, 1.0e6, 0.0], bucket(n) - n)
    m = Nx.tensor(rows ++ far, type: :f32)

    hull =
      Nx.tensor(edges ++ List.duplicate(hd(edges), bucket16(length(edges)) - length(edges)),
        type: :f32
      )

    {cells, valid, keep} = Nx.Defn.jit_apply(&cylinders_n/2, [m, hull], compiler: EXLA)

    kept =
      keep
      |> Nx.to_flat_list()
      |> Enum.take(n)
      |> Enum.with_index()
      |> Enum.flat_map(fn {k, i} -> if k == 1, do: [i], else: [] end)

    if kept == [] do
      empty()
    else
      index = Nx.tensor(kept)

      %{
        minutiae: m |> Nx.take(index) |> to_degrees(),
        cells: Nx.take(cells, index),
        valid: Nx.take(valid, index)
      }
    end
  end

  defp empty, do: %{minutiae: nil, cells: nil, valid: nil}

  defp to_degrees(m),
    do: Nx.concatenate([m[[.., 0..1]], Nx.multiply(m[[.., 2..2]], 180 / :math.pi())], axis: 1)

  # The convex hull's edges as {nx, ny, c}, with n·p + c ≥ 0 inside
  # (Andrew's monotone chain); several points on a line give its two
  # sides.
  defp hull_edges(points) do
    sorted = points |> Enum.uniq() |> Enum.sort()
    lower = chain(sorted)
    upper = chain(Enum.reverse(sorted))
    hull = Enum.drop(lower, -1) ++ Enum.drop(upper, -1)

    hull
    |> Enum.zip(tl(hull) ++ [hd(hull)])
    |> Enum.flat_map(fn {{x1, y1}, {x2, y2}} ->
      {dx, dy} = {x2 - x1, y2 - y1}
      len = :math.sqrt(dx * dx + dy * dy)
      if len == 0, do: [], else: [[-dy / len, dx / len, (dy * x1 - dx * y1) / len]]
    end)
  end

  defp bucket16(n), do: max(16, div(n + 15, 16) * 16)

  defp chain(points) do
    Enum.reduce(points, [], fn p, acc -> [p | pop_right_turns(acc, p)] end) |> Enum.reverse()
  end

  defp pop_right_turns([b, a | rest], p) do
    if cross(a, b, p) <= 0, do: pop_right_turns([a | rest], p), else: [b, a | rest]
  end

  defp pop_right_turns(acc, _p), do: acc

  defp cross({ax, ay}, {bx, by}, {px, py}), do: (bx - ax) * (py - ay) - (by - ay) * (px - ax)

  defn cylinders_n(m, hull) do
    n = Nx.axis_size(m, 0)
    x = m[[.., 0]]
    y = m[[.., 1]]
    theta = m[[.., 2]]
    delta_s = 2 * @r / @ns
    delta_d = 2 * Nx.Constants.pi() / @nd

    # Cell centres, {n, 256, 2}: the grid turned with each minutia.
    offsets = (Nx.iota({@ns}) - (@ns - 1) / 2) * delta_s
    gi = Nx.reshape(Nx.broadcast(Nx.new_axis(offsets, 1), {@ns, @ns}), {@ns * @ns})
    gj = Nx.reshape(Nx.broadcast(Nx.new_axis(offsets, 0), {@ns, @ns}), {@ns * @ns})
    c = Nx.cos(theta) |> Nx.new_axis(1)
    s = Nx.sin(theta) |> Nx.new_axis(1)
    cx = Nx.new_axis(x, 1) + c * gi - s * gj
    cy = Nx.new_axis(y, 1) + s * gi + c * gj

    # Spatial contribution of each neighbour t to each cell, {n, 256, n}.
    dx = Nx.new_axis(cx, 2) - Nx.reshape(x, {1, 1, n})
    dy = Nx.new_axis(cy, 2) - Nx.reshape(y, {1, 1, n})
    d2 = dx ** 2 + dy ** 2
    not_self = (1 - Nx.eye(n)) |> Nx.new_axis(1)
    near = Nx.as_type(d2 <= (3 * @sigma_s) ** 2, :f32) * not_self

    spatial =
      Nx.exp(-d2 / (2 * @sigma_s ** 2)) / (@sigma_s * Nx.sqrt(2 * Nx.Constants.pi())) * near

    # Directional contribution, {n, 6, n}: the Gaussian's mass over each
    # layer's angle range, centred on the direction difference.
    dtheta = wrap_pi(Nx.new_axis(theta, 1) - Nx.new_axis(theta, 0))
    layer = -Nx.Constants.pi() + (Nx.iota({@nd}) + 0.5) * delta_d
    alpha = Nx.reshape(layer, {1, @nd, 1}) - Nx.new_axis(dtheta, 1)
    alpha = wrap_pi(alpha)
    k = @sigma_d * Nx.sqrt(2.0)
    directional = 0.5 * (Nx.erf((alpha + delta_d / 2) / k) - Nx.erf((alpha - delta_d / 2) / k))

    value = Nx.dot(spatial, [2], [0], directional, [2], [0])
    cells = Nx.as_type(value > @mu_psi, :u8) |> Nx.reshape({n, @ns * @ns * @nd})

    # Valid: in the circle and in the hull grown by Ω.
    in_circle = (cx - Nx.new_axis(x, 1)) ** 2 + (cy - Nx.new_axis(y, 1)) ** 2 <= @r ** 2

    signed =
      Nx.new_axis(cx, 2) * hull[[.., 0]] + Nx.new_axis(cy, 2) * hull[[.., 1]] + hull[[.., 2]]

    in_hull = Nx.all(signed >= -@omega, axes: [2])
    valid_2d = Nx.logical_and(in_circle, in_hull)

    valid =
      valid_2d
      |> Nx.new_axis(2)
      |> Nx.broadcast({n, @ns * @ns, @nd})
      |> Nx.reshape({n, @cells})
      |> Nx.as_type(:u8)

    circle_cells = Nx.sum(Nx.as_type(in_circle, :f32), axes: [1])
    valid_cells = Nx.sum(Nx.as_type(valid_2d, :f32), axes: [1])

    neighbours =
      Nx.sum(Nx.as_type(d2_to(x, y) <= (@r + 3 * @sigma_s) ** 2, :f32) * (1 - Nx.eye(n)),
        axes: [1]
      )

    keep = Nx.logical_and(valid_cells >= @min_vc * circle_cells, neighbours >= @min_m)

    {cells, valid, Nx.as_type(keep, :u8)}
  end

  defnp(d2_to(x, y),
    do:
      (Nx.new_axis(x, 1) - Nx.new_axis(x, 0)) ** 2 + (Nx.new_axis(y, 1) - Nx.new_axis(y, 0)) ** 2
  )

  defnp(wrap_pi(a),
    do: a - 2 * Nx.Constants.pi() * Nx.floor((a + Nx.Constants.pi()) / (2 * Nx.Constants.pi()))
  )

  @doc """
  The LSA-R score of latent template `probe` against each of `gallery`
  (templates, or `prepare/2`'s result), in gallery order; 0 when either
  side has no cylinders.
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
      probe = pad(probe, bucket(count(probe)))

      fused =
        &chunk_scores_n(&1, &2, &3, &4, &5, &6, &7, &8, Keyword.take(opts, Keyword.keys(@lsa)))

      chunks
      |> Enum.flat_map(fn {indices, chunk} ->
        scores =
          fused
          |> Nx.Defn.jit_apply(
            [
              probe.cells,
              probe.valid,
              probe.minutiae,
              probe.ok,
              chunk.cells,
              chunk.valid,
              chunk.minutiae,
              chunk.ok
            ],
            compiler: EXLA
          )
          |> Nx.to_flat_list()

        Enum.zip(indices, scores)
      end)
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))
    end
  end

  @doc "`gallery` padded and stacked once (as `Matchers.Dmd.prepare/2`)."
  @spec prepare([template()], keyword()) :: map()
  def prepare(gallery, opts \\ []) do
    size = Keyword.get(opts, :chunk, 64)

    chunks =
      gallery
      |> Enum.with_index()
      |> Enum.group_by(fn {t, _} -> bucket(count(t)) end)
      |> Enum.flat_map(fn {n2, members} ->
        members
        |> Enum.chunk_every(size)
        |> Enum.map(fn part ->
          padded = Enum.map(part, fn {t, _} -> pad(t, n2) end)
          padded = padded ++ List.duplicate(pad(empty(), n2), size - length(part))

          stacked =
            Map.new([:cells, :valid, :minutiae, :ok], fn key ->
              {key,
               padded
               |> Enum.map(&Map.fetch!(&1, key))
               |> Nx.stack()
               |> Nx.backend_transfer(Nx.default_backend())}
            end)

          {Enum.map(part, &elem(&1, 1)), stacked}
        end)
      end)

    %{chunks: chunks, size: length(gallery)}
  end

  defp count(%{minutiae: nil}), do: 0
  defp count(%{minutiae: m}), do: Nx.axis_size(m, 0)

  defp bucket(n), do: max(32, div(n + 31, 32) * 32)

  defp pad(%{minutiae: nil}, n) do
    %{
      cells: Nx.broadcast(Nx.u8(0), {n, @cells}),
      valid: Nx.broadcast(Nx.u8(0), {n, @cells}),
      minutiae: Nx.broadcast(0.0, {n, 3}),
      ok: Nx.broadcast(Nx.u8(0), {n})
    }
  end

  defp pad(t, n) do
    k = count(t)
    rows = n - k

    %{
      cells: Nx.pad(t.cells, Nx.u8(0), [{0, rows, 0}, {0, 0, 0}]),
      valid: Nx.pad(t.valid, Nx.u8(0), [{0, rows, 0}, {0, 0, 0}]),
      minutiae: Nx.pad(t.minutiae, 0.0, [{0, rows, 0}, {0, 0, 0}]),
      ok: Nx.as_type(Nx.less(Nx.iota({n}), k), :u8)
    }
  end

  # Cylinder similarity, the gates, then `Matchers.Dmd`'s pairing and
  # relaxation, in one compiled call.
  defn chunk_scores_n(ca, va, ma, oka, cb, vb, mb, okb, opts \\ []) do
    s = similarity_n(ca, va, ma, cb, vb, mb)
    {i, j, active} = Dmd.greedy_pairs(s, oka, okb)
    Dmd.relax(s, ma, mb, i, j, active, oka, okb, opts)
  end

  @doc false
  # {b, n1, n2}: probe cylinders {n1, 1536} against gallery {b, n2, 1536}.
  defn similarity_n(ca, va, ma, cb, vb, mb) do
    a = Nx.as_type(ca, :f32) * Nx.as_type(va, :f32)
    va = Nx.as_type(va, :f32)
    b = Nx.as_type(cb, :f32) * Nx.as_type(vb, :f32)
    vb = Nx.as_type(vb, :f32)

    # Over the cells valid in both: ‖a‖², ‖b‖², a·b and their count.
    aa = Nx.dot(a, [1], vb, [2])
    bb = Nx.dot(va, [1], b, [2])
    ab = Nx.dot(a, [1], b, [2])
    common = Nx.dot(va, [1], vb, [2])
    diff = Nx.sqrt(Nx.max(aa + bb - 2 * ab, 0))
    norms = Nx.sqrt(aa) + Nx.sqrt(bb)
    gamma = Nx.select(norms > 0, 1 - diff / Nx.max(norms, 1.0e-6), 0.0)

    dtheta =
      Nx.abs(
        wrap_pi(
          (Nx.reshape(ma[[.., 2]], {:auto, 1, 1}) - Nx.new_axis(mb[[.., .., 2]], 0)) *
            (Nx.Constants.pi() / 180)
        )
      )

    ok = Nx.logical_and(common >= Nx.floor(@min_me * @cells * 208 / 256), dtheta <= @delta_theta)
    Nx.transpose(Nx.select(ok, gamma, 0.0), axes: [1, 0, 2])
  end
end
