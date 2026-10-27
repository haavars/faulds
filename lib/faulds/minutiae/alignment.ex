defmodule Faulds.Minutiae.Alignment do
  @moduledoc """
  Estimates the rigid transform that carries one print's minutiae onto
  another's (`p_b = R(phi) p_a + t`), for re-aligning a latent to a
  candidate before re-scoring it with a pose-dependent matcher like
  FlareNx.

  A Hough vote: every pair of minutiae (one from each print) whose
  directions differ by `phi` proposes the translation that would carry
  one onto the other at that rotation; the most-voted (phi, t) cell
  wins, and a least-squares fit (2D Procrustes, no scale) to the pairs
  that agree with it gives the transform. Minutiae are `{x, y, theta}`,
  `theta` in degrees; `:angle_sign` says whether a rotation of the image
  by `phi` adds `phi` (1) or `-phi` (-1) to a minutia's direction, which
  depends on the extractor's convention.
  """

  @type minutia :: {number(), number(), number()}
  @type transform :: %{phi: float(), tx: float(), ty: float(), inliers: non_neg_integer()}

  @doc """
  The transform from `a` to `b`, or `:none` with fewer than
  `:min_inliers` (default 4) agreeing pairs. Options: `:angle_bin` (10°),
  `:xy_bin` (16 px), `:max_angle_diff` between a pair's directions once
  rotated (20°), `:inlier_px` (12), `:angle_sign` (1), `:candidates`
  (1): how many of the most-voted cells to refine, keeping the one with
  the most inliers, and `:smooth` (false): rank cells by votes summed
  over their neighbours. On synthetic latents (60-odd `mindtct` minutiae,
  9-ish real) both smoothing and more candidates found the true
  transform less often: the noise wins more cells than the signal.
  """
  @spec align([minutia()], [minutia()], keyword()) :: {:ok, transform()} | :none
  def align(a, b, opts \\ []) do
    angle_bin = Keyword.get(opts, :angle_bin, 10)
    xy_bin = Keyword.get(opts, :xy_bin, 16)
    sign = Keyword.get(opts, :angle_sign, 1)
    min_inliers = Keyword.get(opts, :min_inliers, 4)

    votes =
      for {ax, ay, at} <- a, {bx, by, bt} <- b, reduce: %{} do
        acc ->
          phi = wrap(sign * (bt - at))
          {tx, ty} = translation({ax, ay}, {bx, by}, phi)

          cell =
            {wrap_cell(round(phi / angle_bin), round(360 / angle_bin)), round(tx / xy_bin),
             round(ty / xy_bin)}

          Map.update(acc, cell, 1, &(&1 + 1))
      end

    phi_cells = round(360 / angle_bin)

    votes
    |> maybe_smooth(votes, phi_cells, Keyword.get(opts, :smooth, false))
    |> Enum.sort_by(fn {_cell, count} -> -count end)
    |> Enum.take(Keyword.get(opts, :candidates, 1))
    |> Enum.map(fn {{phi_cell, tx_cell, ty_cell}, _count} ->
      guess = %{phi: phi_cell * angle_bin, tx: tx_cell * xy_bin, ty: ty_cell * xy_bin}
      refine(a, b, guess, opts, min_inliers)
    end)
    |> Enum.filter(&match?({:ok, _}, &1))
    |> Enum.max_by(fn {:ok, t} -> t.inliers end, fn -> :none end)
  end

  defp maybe_smooth(cells, _votes, _phi_cells, false), do: cells

  defp maybe_smooth(cells, votes, phi_cells, true) do
    Enum.map(cells, fn {{p, x, y} = cell, _count} ->
      smoothed =
        for dp <- -1..1, dx <- -1..1, dy <- -1..1, reduce: 0 do
          acc -> acc + Map.get(votes, {wrap_cell(p + dp, phi_cells), x + dx, y + dy}, 0)
        end

      {cell, smoothed}
    end)
  end

  # Angle cells wrap around: with 36 cells, cell 18 is cell -18.
  defp wrap_cell(cell, count), do: Integer.mod(cell + div(count, 2), count) - div(count, 2)

  # Pairs that agree with the guess (each `a` minutia to its nearest `b`
  # within `:inlier_px`, directions within `:max_angle_diff`), then a
  # least-squares fit to them; twice, so the second pass uses the fit.
  defp refine(a, b, guess, opts, min_inliers) do
    Enum.reduce_while(1..2, guess, fn _round, current ->
      pairs = inlier_pairs(a, b, current, opts)

      if length(pairs) >= min_inliers do
        {:cont, pairs |> fit() |> Map.put(:inliers, length(pairs))}
      else
        {:halt, :none}
      end
    end)
    |> case do
      :none -> :none
      transform -> {:ok, transform}
    end
  end

  defp inlier_pairs(a, b, %{phi: phi, tx: tx, ty: ty}, opts) do
    max_px = Keyword.get(opts, :inlier_px, 12)
    max_angle = Keyword.get(opts, :max_angle_diff, 20)
    sign = Keyword.get(opts, :angle_sign, 1)
    {cos, sin} = {cos_deg(phi), sin_deg(phi)}

    for {ax, ay, at} <- a,
        mapped = {cos * ax - sin * ay + tx, sin * ax + cos * ay + ty},
        {d, {bx, by, _bt}} <- [nearest(mapped, at + sign * phi, b, max_angle, sign)],
        d <= max_px do
      {{ax, ay}, {bx, by}}
    end
  end

  defp nearest({mx, my}, direction, b, max_angle, _sign) do
    b
    |> Enum.filter(fn {_x, _y, bt} -> abs(wrap(bt - direction)) <= max_angle end)
    |> Enum.map(fn {bx, by, _bt} = m -> {:math.sqrt((bx - mx) ** 2 + (by - my) ** 2), m} end)
    |> Enum.min_by(&elem(&1, 0), fn -> {:infinity, nil} end)
  end

  @doc "Least-squares rigid fit (no scale) to `{a_point, b_point}` pairs: `%{phi:, tx:, ty:}`, `phi` in degrees."
  @spec fit([{{number(), number()}, {number(), number()}}]) :: %{
          phi: float(),
          tx: float(),
          ty: float()
        }
  def fit(pairs) do
    n = length(pairs)

    {sax, say, sbx, sby} =
      Enum.reduce(pairs, {0.0, 0.0, 0.0, 0.0}, fn {{ax, ay}, {bx, by}}, {s1, s2, s3, s4} ->
        {s1 + ax, s2 + ay, s3 + bx, s4 + by}
      end)

    {max, may, mbx, mby} = {sax / n, say / n, sbx / n, sby / n}

    {num, den} =
      Enum.reduce(pairs, {0.0, 0.0}, fn {{ax, ay}, {bx, by}}, {num, den} ->
        {xa, ya, xb, yb} = {ax - max, ay - may, bx - mbx, by - mby}
        {num + (xa * yb - ya * xb), den + (xa * xb + ya * yb)}
      end)

    phi = :math.atan2(num, den) * 180.0 / :math.pi()
    {tx, ty} = translation({max, may}, {mbx, mby}, phi)
    %{phi: phi, tx: tx, ty: ty}
  end

  # t such that R(phi) a + t = b.
  defp translation({ax, ay}, {bx, by}, phi) do
    {cos, sin} = {cos_deg(phi), sin_deg(phi)}
    {bx - (cos * ax - sin * ay), by - (sin * ax + cos * ay)}
  end

  defp cos_deg(d), do: :math.cos(d * :math.pi() / 180.0)
  defp sin_deg(d), do: :math.sin(d * :math.pi() / 180.0)

  defp wrap(angle), do: angle - 360.0 * Float.floor((angle + 180.0) / 360.0)

  @doc """
  Parses NBIS `.xyt` text (`x y theta quality` rows) into `{x, y, theta}`
  minutiae in the image's own frame (y down), given its `height`:
  `mindtct`'s `.xyt` from this project's JPEG input has its origin at the
  bottom left (checked against the synthetic data's ground truth: 28 of 32 true
  minutiae within 8 px after flipping, 4 before), and flipping y reverses
  the direction angles turn. Without `height`, rows are returned as is.
  """
  @spec from_xyt(String.t(), pos_integer() | nil, non_neg_integer()) :: [minutia()]
  def from_xyt(xyt, height \\ nil, min_quality \\ 0) do
    for line <- String.split(xyt, "\n", trim: true),
        [x, y, theta | rest] = String.split(line),
        quality(rest) >= min_quality do
      {x, y, theta} =
        {String.to_integer(x) * 1.0, String.to_integer(y) * 1.0, String.to_integer(theta) * 1.0}

      if height, do: {x, height - y, -theta}, else: {x, y, theta}
    end
  end

  defp quality([q | _]), do: String.to_integer(q)
  defp quality([]), do: 100

  @doc """
  The latent's pose implied by the transform latent → candidate
  (`p_c = R(phi) p_l + t`, `phi` in degrees, `align/3`'s) and the
  candidate's pose `{x, y, theta}`: centre `R(-phi)(c - t)`, angle
  `theta + phi`, so both FlareNx crops show the same region the same way
  up.
  """
  @spec pose_from_transform(map(), {number(), number(), number()}) :: {float(), float(), float()}
  def pose_from_transform(%{phi: phi_deg, tx: tx, ty: ty}, {cx, cy, theta}) do
    phi = phi_deg * :math.pi() / 180.0
    {cos, sin} = {:math.cos(phi), :math.sin(phi)}
    {dx, dy} = {cx - tx, cy - ty}
    {cos * dx + sin * dy, -sin * dx + cos * dy, theta + phi_deg}
  end
end
