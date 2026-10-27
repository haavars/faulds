# Baselines, bootstrap and gallery-size sensitivity for search-level
# calibration, from the cached score matrices of six scored synthetic
# training runs (the training data under FAULDS_TRAIN, default
# data/train). CPU only (EXLA host client). Run from the repo root:
#
#   mix run --no-start scripts/research/search_calibration_baselines.exs \
#     --fit RUN,RUN,... --val RUN,RUN,... [--out OUT_DIR]
#
# Fit on the --fit runs; report on the --val runs pooled under one
# threshold, each latent against its own run's rolled prints, every mated
# search also searched mate-removed (scored without the mate), ties
# against the mate. Nothing touches the benchmark directory.

alias Faulds.{
  LatentBenchmark,
  LearnedFusion,
  Logistic,
  SearchCalibration,
  SearchMetrics
}

{:ok, _} = Application.ensure_all_started(:exla)
Nx.global_default_backend({EXLA.Backend, client: :host})

{args, _} = OptionParser.parse!(System.argv(), strict: [fit: :string, val: :string, out: :string])

(args[:fit] && args[:val]) ||
  raise "give the runs: --fit RUN,RUN,... --val RUN,RUN,... [--out DIR]"

out_dir = args[:out] || "."
File.mkdir_p!(out_dir)

root = System.get_env("FAULDS_TRAIN", "data/train")
matchers = ~w(ldmd fdmd mcc_leader nbis flare)
train_runs = String.split(args[:fit], ",")
val_runs = String.split(args[:val], ",")
boot_n = String.to_integer(System.get_env("BOOT") || "2000")
gallery_sizes = [100, 250, 500, 1000]
draws = 5

IO.puts("Loading cached matrices (#{Enum.join(matchers, ", ")})")

load = fn run ->
  ds = LatentBenchmark.load(Path.join(root, run))

  Enum.each(matchers, fn m ->
    LatentBenchmark.scored?(ds, m) || raise "#{run}: #{m} not cached; refusing to score"
  end)

  matrices = Enum.map(matchers, &LatentBenchmark.scores(ds, &1))
  index = ds.gallery |> Enum.with_index() |> Map.new(fn {g, i} -> {g.key, i} end)
  mates = Enum.map(ds.probes, &[Map.fetch!(index, "#{&1.subject}/#{&1.fgp}")])

  %{
    run: run,
    matrices: matrices,
    mates: mates,
    cov: SearchCalibration.covariates(ds),
    subjects: Enum.map(ds.probes, & &1.subject)
  }
end

sets = Map.new(train_runs ++ val_runs, &{&1, load.(&1)})

for {run, s} <- sets do
  {p, g} = Nx.shape(hd(s.matrices))

  IO.puts(
    "  #{run}: #{p} latents vs #{g} prints, #{s.subjects |> Enum.uniq() |> length()} people"
  )
end

fusion = LearnedFusion.read!("priv/calibration/latent_fusion.json")
IO.puts("Fusion model: #{inspect(fusion.info)}")

ldmd_all = Nx.concatenate(Enum.map(sets, fn {_, s} -> hd(s.matrices) end))

IO.puts(
  "ldmd score range: #{Nx.to_number(Nx.reduce_min(ldmd_all))} .. #{Nx.to_number(Nx.reduce_max(ldmd_all))}"
)

# ---------------------------------------------------------------------------
# Base scores: ldmd (a pair's own score) and fused (depends on the row), each
# as {full, mate_removed}.

drop = &SearchMetrics.drop_mates/2

bases = fn set ->
  removed = Enum.map(set.matrices, &drop.(&1, set.mates))

  %{
    "ldmd" => {hd(set.matrices), hd(removed)},
    "fused" => {LearnedFusion.score(fusion, set.matrices), LearnedFusion.score(fusion, removed)}
  }
end

# Row-wise transforms (the same function on the full and the mate-removed
# matrix, so a mate-removed search is normalised without its mate).
zrow = fn s ->
  mean = Nx.mean(s, axes: [1], keep_axes: true)
  std = s |> Nx.subtract(mean) |> Nx.pow(2) |> Nx.mean(axes: [1], keep_axes: true) |> Nx.sqrt()
  Nx.divide(Nx.subtract(s, mean), Nx.max(std, 1.0e-6))
end

robust = &LatentBenchmark.robust_z(&1, 1)
gap = &LearnedFusion.gap/1

# s / best other (top-1 / top-2 for the top candidate), for positive scores.
ratio = fn s ->
  {top2, _} = Nx.top_k(s, k: 2)
  first = Nx.slice_along_axis(top2, 0, 1, axis: 1)
  second = Nx.slice_along_axis(top2, 1, 1, axis: 1)
  best_other = Nx.select(Nx.greater_equal(s, first), second, first)
  Nx.divide(s, Nx.max(best_other, 1.0e-6))
end

# T-norm: each gallery print's score standardised by its own impostor
# scores against the other probes of the same run (leave-one-probe-out,
# mates excluded from the statistics). Computed on the full matrix, then
# the mate dropped: a column's statistics don't depend on the mate.
tnorm = fn s, mates ->
  {p, g} = Nx.shape(s)
  idx = for {cols, i} <- Enum.with_index(mates), c <- cols, do: [i, c]

  mask =
    Nx.indexed_put(Nx.broadcast(0.0, {p, g}), Nx.tensor(idx), Nx.broadcast(1.0, {length(idx)}))

  valid = Nx.subtract(1.0, mask)
  sv = Nx.multiply(s, valid)
  n = valid |> Nx.sum(axes: [0], keep_axes: true) |> Nx.subtract(valid)
  sum = sv |> Nx.sum(axes: [0], keep_axes: true) |> Nx.subtract(sv)
  sq = Nx.multiply(sv, sv)
  sumsq = sq |> Nx.sum(axes: [0], keep_axes: true) |> Nx.subtract(sq)
  mean = Nx.divide(sum, n)
  var = Nx.subtract(Nx.divide(sumsq, n), Nx.multiply(mean, mean))
  Nx.divide(Nx.subtract(s, mean), Nx.max(Nx.sqrt(Nx.max(var, 0.0)), 1.0e-6))
end

# s times log(1 + LEADER minutiae): a fixed quality weight, no fit.
qweight = fn s, cov ->
  Nx.multiply(s, cov |> Nx.slice_along_axis(0, 1, axis: 1))
end

# ---------------------------------------------------------------------------
# Fits on the training runs.

training = Enum.map(train_runs, &sets[&1])
cov_names = SearchCalibration.covariate_names()

# The named columns of `SearchCalibration.covariates/1`'s tensor, nil for none.
columns = fn
  _cov, [] ->
    nil

  cov, names ->
    idx = Enum.map(names, fn n -> Enum.find_index(cov_names, &(&1 == n)) end)
    Nx.take(cov, Nx.tensor(idx), axis: 1)
end

fit_top = fn base, names ->
  cols = fn cov -> columns.(cov, names) end

  SearchCalibration.fit(
    base,
    Enum.map(training, fn set -> {elem(bases.(set)[base], 0), cols.(set.cov), set.mates} end),
    k: 1,
    covariates: names
  )
end

# The same features fitted on every pair, mates and non-mates weighted
# equally (as LearnedFusion), instead of on each search's top candidates.
fit_all = fn base, names ->
  cols = fn cov -> columns.(cov, names) end

  chunks =
    Enum.map(training, fn set ->
      scores = elem(bases.(set)[base], 0)
      {p, g} = Nx.shape(scores)
      f = 2 + 2 * length(names)
      x = scores |> SearchCalibration.features(cols.(set.cov)) |> Nx.reshape({p * g, f})
      idx = for {cols, i} <- Enum.with_index(set.mates), c <- cols, do: [i * g + c]

      y =
        Nx.indexed_put(
          Nx.broadcast(0.0, {p * g}),
          Nx.tensor(idx),
          Nx.broadcast(1.0, {length(idx)})
        )

      {x, y}
    end)

  fit = Logistic.fit(chunks, l2: 1.0, balance: true)

  %SearchCalibration{
    base: base,
    covariates: names,
    mean: fit.mean,
    std: fit.std,
    weights: fit.weights,
    bias: fit.bias,
    k: 1,
    info: %{
      "rows" => fit.rows,
      "mates" => fit.positives,
      "features" => SearchCalibration.feature_names(names)
    }
  }
end

IO.puts("\nFitting calibrations")

models =
  for base <- ["ldmd", "fused"], into: %{} do
    {base,
     %{
       top_cov: fit_top.(base, cov_names),
       top_nocov: fit_top.(base, []),
       top_count: fit_top.(base, ["leader_minutiae"]),
       all_cov: fit_all.(base, cov_names),
       all_nocov: fit_all.(base, [])
     }}
  end

for {base, ms} <- models, {name, m} <- ms do
  w = m.covariates |> SearchCalibration.feature_names() |> Enum.zip(m.weights)

  IO.puts(
    "  #{base} #{name} (#{m.info["rows"]} rows): " <>
      Enum.map_join(w, ", ", fn {n, v} -> "#{n} #{Float.round(v, 2)}" end)
  )
end

apply_model = fn m, {full, removed}, cov ->
  c = columns.(cov, m.covariates)
  {SearchCalibration.score(m, full, c), SearchCalibration.score(m, removed, c)}
end

# ---------------------------------------------------------------------------
# Methods: name -> fn(set) -> {full, removed}

both = fn f, {full, removed} -> {f.(full), f.(removed)} end

methods =
  [
    {"ldmd", fn set -> bases.(set)["ldmd"] end},
    {"ldmd z-norm", fn set -> both.(zrow, bases.(set)["ldmd"]) end},
    {"ldmd robust z", fn set -> both.(robust, bases.(set)["ldmd"]) end},
    {"ldmd gap (top1 − top2)", fn set -> both.(gap, bases.(set)["ldmd"]) end},
    {"ldmd ratio (top1 / top2)", fn set -> both.(ratio, bases.(set)["ldmd"]) end},
    {"ldmd T-norm",
     fn set ->
       {full, _} = bases.(set)["ldmd"]
       t = tnorm.(full, set.mates)
       {t, drop.(t, set.mates)}
     end},
    {"ldmd ZT-norm",
     fn set ->
       {full, _} = bases.(set)["ldmd"]
       t = tnorm.(full, set.mates)
       {zrow.(t), zrow.(drop.(t, set.mates))}
     end},
    {"ldmd × log(1+n)", fn set -> both.(&qweight.(&1, set.cov), bases.(set)["ldmd"]) end},
    {"ldmd gap × log(1+n)",
     fn set -> both.(&qweight.(gap.(&1), set.cov), bases.(set)["ldmd"]) end},
    {"ldmd cal: all pairs, s+gap",
     fn set -> apply_model.(models["ldmd"].all_nocov, bases.(set)["ldmd"], set.cov) end},
    {"ldmd cal: all pairs, +cov",
     fn set -> apply_model.(models["ldmd"].all_cov, bases.(set)["ldmd"], set.cov) end},
    {"ldmd cal: top rows, s+gap",
     fn set -> apply_model.(models["ldmd"].top_nocov, bases.(set)["ldmd"], set.cov) end},
    {"ldmd cal: top rows, +count",
     fn set -> apply_model.(models["ldmd"].top_count, bases.(set)["ldmd"], set.cov) end},
    {"ldmd+cal (method)",
     fn set -> apply_model.(models["ldmd"].top_cov, bases.(set)["ldmd"], set.cov) end},
    {"fused", fn set -> bases.(set)["fused"] end},
    {"fused z-norm", fn set -> both.(zrow, bases.(set)["fused"]) end},
    {"fused gap", fn set -> both.(gap, bases.(set)["fused"]) end},
    {"fused T-norm",
     fn set ->
       {full, _} = bases.(set)["fused"]
       t = tnorm.(full, set.mates)
       {t, drop.(t, set.mates)}
     end},
    {"fused cal: all pairs, +cov",
     fn set -> apply_model.(models["fused"].all_cov, bases.(set)["fused"], set.cov) end},
    {"fused cal: top rows, s+gap",
     fn set -> apply_model.(models["fused"].top_nocov, bases.(set)["fused"], set.cov) end},
    {"fused+cal (method)",
     fn set -> apply_model.(models["fused"].top_cov, bases.(set)["fused"], set.cov) end}
  ]

evaluate = fn fun, set ->
  {full, removed} = fun.(set)
  tops = removed |> Nx.reduce_max(axes: [1]) |> Nx.to_flat_list()
  ranks = SearchMetrics.mate_ranks(full, set.mates)

  searches =
    full
    |> SearchMetrics.searches(set.mates, mate_removed_top: tops)
    |> Enum.zip_with(set.subjects, &Map.put(&1, :subject, &2))

  %{ranks: ranks, searches: searches}
end

IO.puts("\nEvaluating on #{Enum.join(val_runs, " + ")} (pooled)")

results =
  for {name, fun} <- methods, into: %{} do
    r =
      val_runs
      |> Enum.map(&evaluate.(fun, sets[&1]))
      |> Enum.reduce(fn b, a ->
        %{ranks: a.ranks ++ b.ranks, searches: a.searches ++ b.searches}
      end)

    {name, r}
  end

pct = fn x -> :erlang.float_to_binary(x * 100, decimals: 1) <> "%" end

discordant = fn a, b ->
  Enum.zip(a, b) |> Enum.count(fn {x, y} -> x == 1 != (y == 1) end)
end

IO.puts(
  "\n| Method | Rank 1 | Rank 10 | FNIR@1% | FNIR@2% | FNIR@10% | rank-1 discordant vs base |"
)

IO.puts("|---|---|---|---|---|---|---|")

summary =
  for {name, _} <- methods, into: %{} do
    %{ranks: ranks, searches: searches} = results[name]
    [r1, r10] = SearchMetrics.hit_rates(ranks, [1, 10])
    [a1, a2, a10] = SearchMetrics.open_set_of(searches, fpirs: [0.01, 0.02, 0.1])
    base = if String.starts_with?(name, "fused"), do: "fused", else: "ldmd"
    d = discordant.(ranks, results[base].ranks)

    IO.puts(
      "| #{name} | #{pct.(r1.rate)} (#{r1.hits}) | #{pct.(r10.rate)} | #{pct.(a1.fnir)} | #{pct.(a2.fnir)} | #{pct.(a10.fnir)} | #{d} |"
    )

    {name,
     %{
       rank1: r1.rate,
       hits1: r1.hits,
       rank10: r10.rate,
       fnir1: a1.fnir,
       fnir2: a2.fnir,
       fnir10: a10.fnir,
       n_mated: a1.n_mated,
       n_nonmated: a1.n_nonmated,
       discordant: d
     }}
  end

# ---------------------------------------------------------------------------
# Bootstrap by person: resample the people of the pooled validation runs
# with replacement, keep every search of each drawn person (mated and its
# mate-removed twin), recompute FNIR at FPIR 1% and 10% with the
# threshold set in the resample, paired across methods.

IO.puts("\nBootstrap by person (#{boot_n} resamples)")
:rand.seed(:exsss, {2026, 10, 27})

names = Enum.map(methods, &elem(&1, 0))
first = results[hd(names)].searches
subjects = first |> Enum.map(& &1.subject) |> Enum.uniq()

by_subject =
  first |> Enum.with_index() |> Enum.group_by(fn {s, _} -> s.subject end, fn {_, i} -> i end)

n_subj = length(subjects)
subject_tuple = List.to_tuple(subjects)
tuples = Map.new(names, fn n -> {n, List.to_tuple(results[n].searches)} end)

boot =
  for _ <- 1..boot_n do
    idx =
      for _ <- 1..n_subj, i <- by_subject[elem(subject_tuple, :rand.uniform(n_subj) - 1)], do: i

    Map.new(names, fn n ->
      t = tuples[n]
      [a1, a10] = SearchMetrics.open_set_of(Enum.map(idx, &elem(t, &1)), fpirs: [0.01, 0.1])
      {n, {a1.fnir, a10.fnir}}
    end)
  end

ci = fn values ->
  sorted = Enum.sort(values)
  n = length(sorted)
  {Enum.at(sorted, round(0.025 * (n - 1))), Enum.at(sorted, round(0.975 * (n - 1)))}
end

boot_summary =
  for name <- names, into: %{} do
    f1 = Enum.map(boot, &elem(&1[name], 0))
    f10 = Enum.map(boot, &elem(&1[name], 1))

    diffs =
      for ref <- [
            "ldmd",
            "fused",
            "ldmd+cal (method)",
            "fused+cal (method)",
            "ldmd gap (top1 − top2)",
            "ldmd × log(1+n)",
            "ldmd gap × log(1+n)",
            "ldmd cal: all pairs, +cov",
            "fused cal: all pairs, +cov",
            "ldmd z-norm"
          ],
          ref != name,
          into: %{} do
        d1 = Enum.zip_with(f1, Enum.map(boot, &elem(&1[ref], 0)), &(&1 - &2))
        d10 = Enum.zip_with(f10, Enum.map(boot, &elem(&1[ref], 1)), &(&1 - &2))

        {ref,
         %{
           fnir1: %{
             mean: Enum.sum(d1) / boot_n,
             ci: ci.(d1),
             p_le: Enum.count(d1, &(&1 <= 0)) / boot_n
           },
           fnir10: %{
             mean: Enum.sum(d10) / boot_n,
             ci: ci.(d10),
             p_le: Enum.count(d10, &(&1 <= 0)) / boot_n
           }
         }}
      end

    {name, %{fnir1_ci: ci.(f1), fnir10_ci: ci.(f10), diffs: diffs}}
  end

IO.puts(
  "\n| Method | FNIR@1% [95% CI] | Δ vs ldmd [CI] | Δ vs ldmd+cal [CI] | FNIR@10% [CI] | Δ10 vs ldmd+cal [CI] |"
)

IO.puts("|---|---|---|---|---|---|")

fmt_ci = fn {lo, hi} -> "[#{pct.(lo)}, #{pct.(hi)}]" end

fmt_d = fn
  nil -> "—"
  %{mean: m, ci: c} -> "#{if m >= 0, do: "+", else: ""}#{pct.(m)} #{fmt_ci.(c)}"
end

for name <- names do
  s = summary[name]
  b = boot_summary[name]
  d_ldmd = b.diffs["ldmd"] && b.diffs["ldmd"].fnir1
  d_cal = b.diffs["ldmd+cal (method)"] && b.diffs["ldmd+cal (method)"].fnir1
  d_cal10 = b.diffs["ldmd+cal (method)"] && b.diffs["ldmd+cal (method)"].fnir10

  IO.puts(
    "| #{name} | #{pct.(s.fnir1)} #{fmt_ci.(b.fnir1_ci)} | #{fmt_d.(d_ldmd)} | #{fmt_d.(d_cal)} | #{pct.(s.fnir10)} #{fmt_ci.(b.fnir10_ci)} | #{fmt_d.(d_cal10)} |"
  )
end

# ---------------------------------------------------------------------------
# Gallery size: each probe keeps its mate and G − 1 non-mates drawn at
# random from its run's 1,000 prints (so a mated search of size G and a
# mate-removed one of G − 1); the calibration is refitted at each G on
# the training runs, and the G = 1,000 model is also applied as is.

IO.puts("\nGallery size (#{draws} draws per size, mean [min–max] over draws)")

subsample = fn set, g, seed ->
  :rand.seed(:exsss, seed)
  {p, full} = Nx.shape(hd(set.matrices))

  # Kept columns stay in gallery order, so tied scores (a latent with no
  # minutiae scores every print 0) break exactly as in the full matrix;
  # at G = 1,000 this reproduces it.
  keep =
    for {[mate], _} <- Enum.zip(set.mates, 1..p) do
      others = 0..(full - 1) |> Enum.reject(&(&1 == mate)) |> Enum.take_random(g - 1)
      Enum.sort([mate | others])
    end

  mates =
    Enum.zip_with(keep, set.mates, fn cols, [mate] -> [Enum.find_index(cols, &(&1 == mate))] end)

  idx = Nx.tensor(keep, type: :s32)
  take = fn m -> Nx.take_along_axis(m, idx, axis: 1) end
  %{set | matrices: Enum.map(set.matrices, take), mates: mates}
end

size_rows =
  for g <- gallery_sizes, d <- 1..if(g == 1000, do: 1, else: draws) do
    seed = {g, d, 7}
    sub = Map.new(sets, fn {run, set} -> {run, subsample.(set, g, seed)} end)
    sub_training = Enum.map(train_runs, &sub[&1])

    fit_g = fn base ->
      SearchCalibration.fit(
        base,
        Enum.map(sub_training, fn set -> {elem(bases.(set)[base], 0), set.cov, set.mates} end),
        k: 1,
        covariates: cov_names
      )
    end

    cal_g = %{"ldmd" => fit_g.("ldmd"), "fused" => fit_g.("fused")}

    ms = [
      {"ldmd", fn set -> bases.(set)["ldmd"] end},
      {"ldmd gap", fn set -> both.(gap, bases.(set)["ldmd"]) end},
      {"ldmd × log(1+n)", fn set -> both.(&qweight.(&1, set.cov), bases.(set)["ldmd"]) end},
      {"ldmd+cal fitted at G",
       fn set -> apply_model.(cal_g["ldmd"], bases.(set)["ldmd"], set.cov) end},
      {"ldmd+cal fitted at 1,000",
       fn set -> apply_model.(models["ldmd"].top_cov, bases.(set)["ldmd"], set.cov) end},
      {"fused", fn set -> bases.(set)["fused"] end},
      {"fused+cal fitted at G",
       fn set -> apply_model.(cal_g["fused"], bases.(set)["fused"], set.cov) end},
      {"fused+cal fitted at 1,000",
       fn set -> apply_model.(models["fused"].top_cov, bases.(set)["fused"], set.cov) end}
    ]

    for {name, fun} <- ms do
      r =
        val_runs
        |> Enum.map(&evaluate.(fun, sub[&1]))
        |> Enum.reduce(fn b, a ->
          %{ranks: a.ranks ++ b.ranks, searches: a.searches ++ b.searches}
        end)

      [r1] = SearchMetrics.hit_rates(r.ranks, [1])
      [a1, a10] = SearchMetrics.open_set_of(r.searches, fpirs: [0.01, 0.1])
      %{g: g, draw: d, method: name, rank1: r1.rate, fnir1: a1.fnir, fnir10: a10.fnir}
    end
  end
  |> List.flatten()

size_names = size_rows |> Enum.map(& &1.method) |> Enum.uniq()

IO.puts("\n| Method | G | Rank 1 | FNIR@1% | FNIR@10% |")
IO.puts("|---|---|---|---|---|")

agg = fn rows, key ->
  v = Enum.map(rows, &Map.fetch!(&1, key))
  m = Enum.sum(v) / length(v)
  if length(v) > 1, do: "#{pct.(m)} [#{pct.(Enum.min(v))}–#{pct.(Enum.max(v))}]", else: pct.(m)
end

size_summary =
  for g <- gallery_sizes, name <- size_names do
    rows = Enum.filter(size_rows, &(&1.g == g and &1.method == name))

    IO.puts(
      "| #{name} | #{g} | #{agg.(rows, :rank1)} | #{agg.(rows, :fnir1)} | #{agg.(rows, :fnir10)} |"
    )

    %{
      g: g,
      method: name,
      rank1: agg.(rows, :rank1),
      fnir1: agg.(rows, :fnir1),
      fnir10: agg.(rows, :fnir10)
    }
  end

File.write!(
  Path.join(out_dir, "calib_baselines.json"),
  JSON.encode!(%{
    protocol: %{
      train: train_runs,
      val: val_runs,
      boot: boot_n,
      gallery_sizes: gallery_sizes,
      draws: draws
    },
    summary: summary,
    bootstrap:
      Map.new(boot_summary, fn {k, v} ->
        {k,
         %{
           fnir1_ci: Tuple.to_list(v.fnir1_ci),
           fnir10_ci: Tuple.to_list(v.fnir10_ci),
           diffs:
             Map.new(v.diffs, fn {r, d} ->
               {r,
                %{
                  fnir1: %{d.fnir1 | ci: Tuple.to_list(d.fnir1.ci)},
                  fnir10: %{d.fnir10 | ci: Tuple.to_list(d.fnir10.ci)}
                }}
             end)
         }}
      end),
    gallery: size_summary,
    gallery_rows: size_rows,
    models:
      Map.new(models, fn {b, ms} -> {b, Map.new(ms, fn {n, m} -> {n, Map.from_struct(m)} end)} end)
  })
)

IO.puts("\nWritten #{Path.join(out_dir, "calib_baselines.json")}")
