defmodule Faulds.Elub do
  @moduledoc """
  Empirical lower and upper bounds on log10 likelihood ratios (ELUB;
  Vergeer, van Es, de Jongh, Alberink and Stoel, "Numerical likelihood
  ratios outputted by LR systems are often based on extrapolation: when
  to stop extrapolating?", Science & Justice 56, 2016, 482–491), as the
  Netherlands Forensic Institute's `lir` 1.8.0 computes them
  (`lir.algorithms.bayeserror.elub`, Apache 2.0).

  For each threshold LR `θ` on a grid of log10 steps, the expected cost
  of deciding "same source" when `LR > θ` is
  `1 − P(LR > θ | targets) + θ · P(LR > θ | non-targets)`, with one
  misleading trial added to each side (a target below every threshold, a
  non-target above). A system that always answers LR 1 costs `θ` below
  0 and 1 above. The bounds are the grid points nearest 0 where the
  system still costs no more than that neutral one; past them the data
  can't tell its LRs from extrapolation. They never pass
  `−log10(targets + 1)` or `log10(non-targets + 1)`.
  """

  @step 0.01
  @extreme 9.0

  @doc """
  The ELUB `{lower, upper}` of log10 LRs `targets` (same source) and
  `non_targets`, each a non-empty list. LRs beyond ±#{@extreme} count as
  ±#{@extreme}, as `lir` substitutes infinite ones.
  """
  @spec bounds([number()], [number()]) :: {float(), float()}
  def bounds([_ | _] = targets, [_ | _] = non_targets) do
    clip = fn llrs -> Enum.map(llrs, &min(max(&1 * 1.0, -@extreme), @extreme)) end
    targets = clip.(targets)
    non_targets = clip.(non_targets)
    n_t = length(targets)
    n_n = length(non_targets)
    all = targets ++ non_targets

    low = max(Enum.min(all), -:math.log10(n_t + 1))
    high = min(Enum.max(all), :math.log10(n_n + 1))
    steps_low = min(0, floor(low / @step))
    steps_high = max(0, floor(high / @step) + 1)
    grid = linspace(steps_low * @step, steps_high * @step, steps_high - steps_low + 1)

    lr = fn x -> :math.pow(10, x) end
    t_lrs = Enum.map(targets, lr)
    n_lrs = Enum.map(non_targets, lr)

    worse =
      Enum.filter(grid, fn t ->
        theta = lr.(t)
        neutral = if 1.0 > theta, do: theta, else: 1.0
        accepted_t = Enum.count(t_lrs, &(&1 > theta))
        accepted_n = Enum.count(n_lrs, &(&1 > theta)) + 1
        system = 1 - accepted_t / (n_t + 1) + theta * accepted_n / (n_n + 1)
        neutral / system < 1
      end)

    lower = Enum.max([hd(grid) | for(t <- worse, t <= 0, do: t + @step)])
    upper = Enum.min([List.last(grid) | for(t <- worse, t >= 0, do: t - @step)])
    {min(lower, 0.0) * 1.0, max(upper, 0.0) * 1.0}
  end

  # numpy's linspace: start + i · step, the last point exactly `stop`.
  defp linspace(start, _stop, 1), do: [start * 1.0]

  defp linspace(start, stop, n) do
    step = (stop - start) / (n - 1)
    for(i <- 0..(n - 2), do: i * step + start) ++ [stop * 1.0]
  end
end
