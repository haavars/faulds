defmodule Faulds.Minutiae.Correspondence do
  @moduledoc """
  Finds point correspondences between two minutiae sets (as produced by
  `Faulds.Matchers.Nbis.minutiae/1`) for visualization in a
  side-by-side comparison — which minutia on print A most plausibly
  corresponds to which minutia on print B.

  This is a from-scratch implementation of the same family of algorithm
  NIST's `bozorth3` uses internally, in the spirit of Jiang & Yau's (2000)
  local-structure matching: assume the two prints are related by a single
  rigid transform (rotation + translation — appropriate for full/plain
  prints, this project's scope; not distortion-tolerant), find that
  transform by testing candidate minutia pairs as alignment "roots", then
  count how many other minutiae agree with each candidate transform. The
  transform with the most agreeing ("inlier") minutiae wins, and its
  inliers become the matched pairs.

  Deliberately independent from `Matchers.Nbis.score/2` (`bozorth3`) and
  every other matcher in this project — it exists purely to explain a
  match visually. Its output is a visualization aid and must never be used
  as an accept/reject signal in place of the already-validated
  matcher scores.
  """

  @type minutia :: %{x: number(), y: number(), angle: float(), quality: number()}
  @type transform :: %{rotation: float(), dx: float(), dy: float()}
  @type result :: %{pairs: [{non_neg_integer(), non_neg_integer()}], transform: transform() | nil}

  @default_root_pool 30
  @default_distance_tolerance 12.0
  @default_angle_tolerance 20.0
  @default_min_inlier_fraction 0.25
  @default_min_inlier_count 8

  @doc """
  Finds the best single rigid-transform correspondence between minutiae
  list `a` and minutiae list `b`. Returns matched `{index_in_a,
  index_in_b}` pairs (0-based, positional index into the input lists as
  given) and the winning transform (rotating/translating `a`'s
  coordinate space onto `b`'s). Empty `pairs` and a `nil` transform if
  either list is empty, no candidate transform found any agreement, or
  the best candidate found isn't clearly better than chance (see
  `:min_inlier_fraction`/`:min_inlier_count` below).

  ## Options

    * `:root_pool` - only the top-N minutiae by quality from each list are
      tried as the seed ("root") pair that defines a candidate transform
      (default #{@default_root_pool}) — bounds the search to
      `root_pool * root_pool` candidate transforms regardless of how many
      minutiae are in the full sets, since only one genuinely-corresponding
      root pair is needed to recover the right transform.
    * `:distance_tolerance` - max pixel distance, after alignment, for two
      minutiae to be considered the same point (default
      #{@default_distance_tolerance}, tuned for ~500 PPI captures).
    * `:angle_tolerance` - max degrees difference, after alignment, between
      two minutiae's directions to still count as agreeing (default
      #{@default_angle_tolerance}).
    * `:min_inlier_fraction` - the winning transform's inlier count must be
      at least this fraction of `max(length(a), length(b))` (deliberately
      the larger set, not the smaller — see below), or the match is
      rejected entirely (default #{@default_min_inlier_fraction}). Guards
      against exactly the failure mode this algorithm has no other defense
      against: with `root_pool * root_pool` candidate transforms tested,
      *some* transform will always agree with a handful of minutiae by pure
      coincidence — ridge endings/bifurcations look alike across unrelated
      fingers — so an impostor pair reliably produces a plausible-looking
      but meaningless "best" transform without this check. Empirically,
      genuine same-finger pairs cleared 50-80% agreement (relative to the
      larger set) while impostor pairs, especially against denser
      synthetic distractors (2-3x the minutiae, more chances for
      coincidental agreement), topped out around 25-30%, often at
      implausible rotations.
    * `:min_inlier_count` - an absolute floor alongside
      `:min_inlier_fraction` (default #{@default_min_inlier_count}), so the
      fraction check alone can't be trivially satisfied when one print has
      very few detected minutiae.
  """
  @spec match([minutia()], [minutia()], keyword()) :: result()
  def match(a, b, opts \\ [])
  def match([], _b, _opts), do: %{pairs: [], transform: nil}
  def match(_a, [], _opts), do: %{pairs: [], transform: nil}

  def match(a, b, opts) do
    root_pool = Keyword.get(opts, :root_pool, @default_root_pool)
    distance_tolerance = Keyword.get(opts, :distance_tolerance, @default_distance_tolerance)
    angle_tolerance = Keyword.get(opts, :angle_tolerance, @default_angle_tolerance)
    min_inlier_fraction = Keyword.get(opts, :min_inlier_fraction, @default_min_inlier_fraction)
    min_inlier_count = Keyword.get(opts, :min_inlier_count, @default_min_inlier_count)

    roots_a = top_by_quality(a, root_pool)
    roots_b = top_by_quality(b, root_pool)

    {best_transform, best_score} =
      for ma <- roots_a, mb <- roots_b, reduce: {nil, 0} do
        {best_transform, best_score} ->
          transform = transform_for(ma, mb)
          score = count_inliers(a, b, transform, distance_tolerance, angle_tolerance)

          if score > best_score do
            {transform, score}
          else
            {best_transform, best_score}
          end
      end

    # Fraction is against the *larger* set, not the smaller — chance
    # agreement scales with how many candidate points are available to
    # coincidentally satisfy the tolerance, so a denser larger set (e.g.
    # synthetic distractors with 2-3x the minutiae) inflates a
    # smaller-set-normalized fraction without a real correspondence
    # behind it. Normalizing against the smaller set let several such
    # impostor pairs clear a 25% bar (up to 29.5%) at implausible
    # rotations, while genuine same-finger pairs stayed comfortably clear
    # of a max()-normalized 25% bar in every case tried.
    required = max(min_inlier_count, round(min_inlier_fraction * max(length(a), length(b))))

    case best_transform do
      nil ->
        %{pairs: [], transform: nil}

      _transform when best_score < required ->
        %{pairs: [], transform: nil}

      transform ->
        pairs = resolve_pairs(a, b, transform, distance_tolerance, angle_tolerance)
        %{pairs: pairs, transform: transform}
    end
  end

  @doc """
  Applies a transform (as returned by `match/3`) to a single minutia's
  position, returning the resulting `{x, y}`, for callers that want the
  aligned distance/angle between one specific pair, not the full match.
  """
  @spec apply_transform(minutia(), transform()) :: {float(), float()}
  def apply_transform(minutia, transform), do: transform_point(minutia, transform)

  defp top_by_quality(minutiae, n) do
    minutiae |> Enum.sort_by(& &1.quality, :desc) |> Enum.take(n)
  end

  # The rigid transform (rotation about the origin, then translation) that
  # maps root_a exactly onto root_b, assuming their directions differ by
  # the same rotation that relates the two prints.
  defp transform_for(root_a, root_b) do
    rotation = normalize_angle(root_b.angle - root_a.angle)
    {rx, ry} = rotate_point({root_a.x, root_a.y}, rotation)
    %{rotation: rotation, dx: root_b.x - rx, dy: root_b.y - ry}
  end

  defp transform_point(m, transform) do
    {rx, ry} = rotate_point({m.x, m.y}, transform.rotation)
    {rx + transform.dx, ry + transform.dy}
  end

  # Cheap, allocation-light count used only to pick the winning transform
  # among candidates — allows a minutia in `b` to satisfy more than one
  # transformed point in `a`, which can slightly overcount, but that's an
  # acceptable trade for speed since only the *winning* transform's exact
  # one-to-one pairs (via resolve_pairs/5) are actually returned.
  defp count_inliers(a, b, transform, distance_tolerance, angle_tolerance) do
    Enum.count(a, fn ma ->
      {tx, ty} = transform_point(ma, transform)
      t_angle = normalize_angle(ma.angle + transform.rotation)
      Enum.any?(b, &agrees?(&1, tx, ty, t_angle, distance_tolerance, angle_tolerance))
    end)
  end

  defp agrees?(mb, tx, ty, t_angle, distance_tolerance, angle_tolerance) do
    distance({mb.x, mb.y}, {tx, ty}) <= distance_tolerance and
      angle_diff(mb.angle, t_angle) <= angle_tolerance
  end

  # Exact one-to-one assignment for the winning transform: every candidate
  # agreeing (a, b) pair, greedily claimed nearest-distance-first so no
  # minutia in either set is used in more than one pair.
  defp resolve_pairs(a, b, transform, distance_tolerance, angle_tolerance) do
    indexed_a = Enum.with_index(a)
    indexed_b = Enum.with_index(b)

    candidates =
      for {ma, i} <- indexed_a, {mb, j} <- indexed_b do
        {tx, ty} = transform_point(ma, transform)
        t_angle = normalize_angle(ma.angle + transform.rotation)

        if agrees?(mb, tx, ty, t_angle, distance_tolerance, angle_tolerance) do
          {distance({mb.x, mb.y}, {tx, ty}), i, j}
        end
      end
      |> Enum.reject(&is_nil/1)
      |> Enum.sort()

    greedy_assign(candidates, MapSet.new(), MapSet.new(), [])
  end

  defp greedy_assign([], _used_a, _used_b, acc), do: Enum.reverse(acc)

  defp greedy_assign([{_distance, i, j} | rest], used_a, used_b, acc) do
    if MapSet.member?(used_a, i) or MapSet.member?(used_b, j) do
      greedy_assign(rest, used_a, used_b, acc)
    else
      greedy_assign(rest, MapSet.put(used_a, i), MapSet.put(used_b, j), [{i, j} | acc])
    end
  end

  defp rotate_point({x, y}, degrees) do
    radians = degrees * :math.pi() / 180.0
    cos = :math.cos(radians)
    sin = :math.sin(radians)
    {x * cos - y * sin, x * sin + y * cos}
  end

  defp normalize_angle(degrees) do
    normalized = :math.fmod(degrees, 360.0)
    if normalized < 0, do: normalized + 360.0, else: normalized
  end

  defp angle_diff(a, b) do
    diff = abs(normalize_angle(a) - normalize_angle(b))
    min(diff, 360.0 - diff)
  end

  defp distance({x1, y1}, {x2, y2}) do
    :math.sqrt(:math.pow(x2 - x1, 2) + :math.pow(y2 - y1, 2))
  end
end
