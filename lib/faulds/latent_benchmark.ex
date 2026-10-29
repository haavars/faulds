defmodule Faulds.LatentBenchmark do
  @moduledoc """
  Closed-set latent search benchmark: every latent is searched against
  every rolled finger of every person, and the measure is the rank of the
  latent's own finger, so that each change can be measured against the
  same probes with paired statistics.

  ## Dataset layout

  A directory per person, anywhere under the benchmark root
  (`docs/data_format.md`): its rolled prints and latents as images, each
  with a JSON file giving its resolution, finger and ground truth.

      <root>/<run>/<person>/rolled/fgpNN.png
      <root>/<run>/<person>/rolled/fgpNN.json
      <root>/<run>/<person>/latents/latent_<n>.png
      <root>/<run>/<person>/latents/latent_<n>.json

  Every image is resampled to 500 ppi (`Resample`), and the ground
  truth's coordinates with it. The ground truth gives each latent's
  finger, region (`centre`, `tip`, `side`), rotation, analysis and true
  minutiae, which ids tie to the rolled print's.

  ## Methods

  A method turns the dataset into a score matrix (probes × gallery,
  higher is better). Templates and matrices are cached under
  `<root>/.cache/`, keyed by entry, so adding people or methods only
  computes what's new.

    * `"flare"`: FlareNx, one template per print, the exact masked score
      (`Matchers.Flare.score/2`, batched here as three matrix products).
    * `"flare_ann"`: the cosine of `Flare.ann_vector/1`, as an
      approximate first stage would rank.
    * `"nbis"`: `mindtct` + `bozorth3` in its one-probe-many-gallery mode.
    * `"fuse_minrank"`, `"fuse_z"`: FlareNx and NBIS combined, by best rank
      per gallery print (ties broken by the other rank) or by the sum of
      per-probe z-scores. Neither has fitted weights.
    * `"flare_reg"`, `"flare_unet"`, `"flare_reg_unet"`: FlareNx with
      FLARE's other pose estimator (RegressionPose), its UNetEnh enhancer,
      or both, on both sides. `"flare_2pose"` is the best of `"flare"` and
      `"flare_reg"`; `"flare_full"` the best of all four combinations,
      FLARE's full configuration less PriorEnh. Each combination is
      compared with the same one on the other side, as FLARE does.
    * A FlareNx method with `_ov`: each view's score times √(mask
      overlap) (DMD's and LDRF's normalisation); with `_ovz` (several
      views only) also each view's robust z over the gallery before the
      max, `_lse` with a log-sum-exp in place of the max
      (`flare_matrix/3`).
    * `"fuse_rrf"`: reciprocal rank fusion of FlareNx and NBIS;
      `"fuse_rz"`, `"fuse_rz_*"`: robust z-score sums (`robust_z/2`) of a
      FlareNx method and NBIS, `"fuse_rz_log"` with `log(1 + s)` on NBIS.

  DMD++ at LEADER's minutiae (`"ldmd"`) with minutia weights from
  LEADER's confidence `q` (0.6–1, `Matchers.Dmd`'s `:weight`):
  `"ldmd_q<α>"` weights both sides' minutiae by `q^α`, `"ldmd_qp<α>"`
  only the latent's (α 1, 4 or 8). `"ldmd_r"`: `"ldmd"` with ridge-count
  agreement in LSA-R's relaxation (`Minutiae.RidgeCount`, the
  `ridges_leader` templates). Richer latent minutiae, the gallery's
  unchanged (`"dmd_leader"`): `"ldmd_t40"`, `"ldmd_t50"` LEADER's
  threshold 0.4 or 0.5 on the latent; `"ldmd_min8"`, `"ldmd_min12"` its
  0.6 minutiae topped up to 8 or 12 with its strongest down to 0.3;
  `"ldmd_views"` LEADER over eight views, kept at 4 votes of 8 at 0.5
  (`Leader.extract_views/2`).

  A rank counts ties against the mate (pessimistic): a matcher that
  gives every print 0 ranks the mate last, not first.
  """

  alias Faulds.Matchers.{Dmd, Flare, FlareNx, Mcc, Nbis}
  alias Faulds.{Metrics, Resample, SearchMetrics}
  alias Faulds.Formats.Png
  alias Faulds.Minutiae.{Alignment, FddMap, Foreground, Leader, RidgeCount}

  # FlareNx extractors: `FlareNx.extract_variants/2` options by name.
  @shifts_128 [{0, 0}, {0, -128}, {0, 128}, {-128, 0}, {128, 0}]
  @extractors %{
    "flare" => [],
    "flare_theta12" => [theta_offsets: Enum.to_list(0..330//30)],
    "flare_theta5" => [theta_offsets: [-60, -30, 0, 30, 60]],
    "flare_rot4" => [input_rotations: [0, 90, 180, 270]],
    "flare_rot12" => [input_rotations: Enum.to_list(0..330//30)],
    "flare_win5" => [shifts: @shifts_128],
    "flare_win9" => [shifts: for(dx <- [-128, 0, 128], dy <- [-128, 0, 128], do: {dx, dy})],
    "flare_reg" => [poses: [:regression]],
    "flare_unet" => [enhancers: [:unet]],
    "flare_reg_unet" => [poses: [:regression], enhancers: [:unet]]
  }

  # FlareNx methods: {probe extractor, gallery extractor}.
  @flare_methods %{
    "flare" => {"flare", "flare"},
    "flare_theta12" => {"flare_theta12", "flare"},
    "flare_theta5" => {"flare_theta5", "flare"},
    "flare_rot4" => {"flare_rot4", "flare"},
    "flare_rot12" => {"flare_rot12", "flare"},
    "flare_win5" => {"flare", "flare_win5"},
    "flare_win9" => {"flare", "flare_win9"},
    "flare_reg" => {"flare_reg", "flare_reg"},
    "flare_unet" => {"flare_unet", "flare_unet"},
    "flare_reg_unet" => {"flare_reg_unet", "flare_reg_unet"}
  }

  # FLARE's full configuration, as the best of single-combination methods.
  @flare_max_methods %{
    "flare_2pose" => ~w(flare flare_reg),
    "flare_full" => ~w(flare flare_reg flare_unet flare_reg_unet)
  }

  # NBIS probe extractors with background/low-quality minutiae dropped
  # (`Minutiae.Foreground`): FlareNx's mask alone, or with mindtct quality.
  @nbis_filters %{
    "nbis_mask" => [mask_threshold: 0.5],
    "nbis_q20m" => [mask_threshold: 0.5, min_quality: 20]
  }

  # Normalised FlareNx scores (`flare_matrix/3`'s `:normalise`), by
  # suffix: `_ov` each view by its overlap, then the max; `_ovz` also each
  # view standardised over the gallery; `_lse` combined by log-sum-exp
  # instead. The last two only differ from `_ov` with several views.
  @normalisations %{"_ov" => :overlap, "_ovz" => :overlap_z, "_lse" => :lse}
  @multi_view ~w(flare_theta12 flare_theta5 flare_rot4 flare_rot12 flare_win5 flare_win9)

  @normalised_flare_methods for {method, _} <- @flare_methods,
                                {suffix, normalise} <- @normalisations,
                                suffix == "_ov" or method in @multi_view,
                                into: %{},
                                do: {method <> suffix, {method, normalise}}

  @normalised_max_methods for {method, _} <- @flare_max_methods,
                              {suffix, normalise} <- @normalisations,
                              into: %{},
                              do: {method <> suffix, {method, normalise}}

  # Fusions with robust z-scores (`robust_z/2`): {the FlareNx method
  # fused with NBIS, NBIS's transform}. `:log` is `log(1 + s)` on
  # `bozorth3`'s counts, which squashes its strongest evidence.
  @robust_fusions %{
    "fuse_rz" => {"flare", :raw},
    "fuse_rz_log" => {"flare", :log},
    "fuse_rz_ov" => {"flare_ov", :raw},
    "fuse_rz_full" => {"flare_full", :raw},
    "fuse_rz_theta5" => {"flare_theta5", :raw},
    "fuse_realign_rz" => {"flare_realign", :raw}
  }

  # Latent-side LEADER minutiae for `"ldmd_<variant>"`.
  @probe_minutiae ~w(t40 t50 min8 min12 views)

  @methods ~w(flare flare_ann nbis nbis_mask nbis_q20m nbis_fdd nbis_leader nbis_truth nbis_truth_visible mcc_nbis mcc_fdd mcc_leader mcc_truth mcc_truth_visible fdmd ldmd ldmd_f16 ldmd_bin ldmd_packed ldmd_q1 ldmd_q4 ldmd_q8 ldmd_qp1 ldmd_qp4 ldmd_qp8 ldmd_r ldmd_t40 ldmd_t50 ldmd_min8 ldmd_min12 ldmd_views fuse_minrank fuse_z fuse_fdd fuse_nbis2 fuse_realign_nbis2 fuse_mcc fuse_mcc2 fuse_mcc_all fuse_mcc_leader fuse_mcc3 fuse_mcc3_all fuse_fdmd fuse_fdmd_mcc fuse_fdmd_all fuse_dmd2 fuse_rrf flare_oracle flare_realign fuse_realign flare_realign_fdd fuse_realign_fdd) ++
             (Map.keys(@flare_methods) -- ["flare"]) ++
             Map.keys(@flare_max_methods) ++
             Map.keys(@normalised_flare_methods) ++
             Map.keys(@normalised_max_methods) ++ Map.keys(@robust_fusions)

  @type probe :: %{
          key: String.t(),
          subject: String.t(),
          fgp: pos_integer(),
          file: Path.t(),
          truth: map()
        }
  @type gallery_entry :: %{key: String.t(), subject: String.t(), fgp: pos_integer()}
  @type dataset :: %{
          root: Path.t(),
          probes: [probe()],
          gallery: [gallery_entry()],
          subjects: %{String.t() => Path.t()}
        }

  @doc "The method names `scores/2` knows."
  @spec methods() :: [String.t()]
  def methods, do: @methods

  @doc """
  Indexes the dataset under `root`: every person
  with a `rolled/` folder is in the gallery (all ten fingers; a missing
  print gets an empty template), every latent with its JSON is a probe.
  """
  @spec load(Path.t()) :: dataset()
  def load(root) do
    subjects =
      root
      |> Path.join("**/rolled")
      |> Path.wildcard()
      |> Enum.filter(&File.dir?/1)
      |> Map.new(&{&1 |> Path.dirname() |> Path.basename(), Path.dirname(&1)})

    gallery =
      for subject <- subjects |> Map.keys() |> Enum.sort(), fgp <- 1..10 do
        %{key: "#{subject}/#{fgp}", subject: subject, fgp: fgp}
      end

    probes =
      subjects
      |> Enum.flat_map(fn {subject, dir} ->
        dir
        |> Path.join("latents/latent_*.json")
        |> Path.wildcard()
        |> Enum.map(&probe(subject, dir, &1))
      end)
      |> Enum.sort_by(& &1.key)

    %{root: root, probes: probes, gallery: gallery, subjects: subjects}
  end

  @doc """
  The indices of the probes to report on: those sharing at least
  `:min_shared` true minutiae with their rolled print (so the mate is
  findable at all, not a tip outside the rolled print's area), and
  whose file matches `:probes` (a regex), e.g. one run.
  """
  @spec probe_indices(dataset(), keyword()) :: [non_neg_integer()]
  def probe_indices(dataset, opts) do
    min_shared = Keyword.get(opts, :min_shared, 0)
    pattern = Keyword.get(opts, :probes)

    for {probe, i} <- Enum.with_index(dataset.probes),
        probe.truth.shared_minutiae >= min_shared,
        pattern == nil or Regex.match?(pattern, probe.file),
        do: i
  end

  defp probe(subject, dir, json_path) do
    [_, n] = Regex.run(~r/latent_(\d+)\.json$/, json_path)
    truth = json_path |> File.read!() |> JSON.decode!()
    scale = to_working(truth["ppi"])
    pairs = shared_minutiae(truth, scale, rolled_truth(dir, truth["finger"]))
    own = own_minutiae(truth)

    %{
      key: "#{subject}/L#{n}",
      subject: subject,
      fgp: truth["finger"],
      file: Path.rootname(json_path) <> ".png",
      pairs: pairs,
      minutiae: Enum.map(own, &{&1["x"] * scale, &1["y"] * scale, &1["angle"] * 1.0}),
      minutiae_ids: Enum.map(own, & &1["id"]),
      visibility: visibility(own),
      truth: truth |> summarize_truth() |> Map.put(:shared_minutiae, length(pairs))
    }
  end

  # Pixels at an image's own resolution to pixels after `Resample`.
  defp to_working(ppi), do: Resample.output_ppi(ppi) / ppi

  # The rolled print's true minutiae on the 500 ppi image, `%{id:, x:, y:,
  # angle:}`; none without a JSON or a print.
  defp rolled_truth(dir, fgp) do
    path = Path.join([dir, "rolled", "fgp#{String.pad_leading("#{fgp}", 2, "0")}.json"])

    if File.exists?(path) do
      meta = path |> File.read!() |> JSON.decode!()
      scale = to_working(meta["ppi"])

      for m <- meta["minutiae"] || [] do
        %{id: m["id"], x: m["x"] * scale, y: m["y"] * scale, angle: m["angle"] * 1.0}
      end
    else
      []
    end
  end

  # The latent's true minutiae that are also in the rolled print (the same
  # id), as {latent point, rolled point} pairs on the 500 ppi images. A
  # mirrored latent's minutiae are in the frame before it was turned back,
  # so it gets none.
  defp shared_minutiae(%{"mirrored" => true}, _scale, _rolled), do: []

  defp shared_minutiae(truth, scale, rolled) do
    by_id = Map.new(rolled, &{&1.id, &1})

    for %{"id" => id} = m <- truth["minutiae"] || [], r = by_id[id], r != nil do
      {{m["x"] * scale, m["y"] * scale}, {r.x, r.y}}
    end
  end

  # The latent's own true minutiae (not a neighbouring finger's or a second
  # touch's); none for a mirrored latent, as for `shared_minutiae/3`.
  defp own_minutiae(%{"mirrored" => true}), do: []

  defp own_minutiae(truth) do
    for m <- truth["minutiae"] || [],
        Map.get(m, "finger", truth["finger"]) == truth["finger"],
        Map.get(m, "touch", 0) == 0,
        do: m
  end

  # Each own minutia's visibility, or nil unless every one has it.
  defp visibility(own) do
    labels = Enum.map(own, & &1["visibility"])
    if own != [] and Enum.all?(labels, &is_binary/1), do: labels
  end

  defp wrap360(angle), do: angle - 360.0 * Float.floor(angle / 360.0)

  defp summarize_truth(truth) do
    %{
      region: truth["region"],
      angle: truth["rotation"],
      analysis: truth["analysis"],
      area_mm2: truth["area_mm2"],
      minutiae_found: truth["minutiae_found"],
      development: truth["development"],
      openlqm: truth["openlqm_vid"]
    }
  end

  @doc "`subject`'s rolled prints at 500 ppi, by finger (those with an image)."
  @spec rolled_images(dataset(), String.t()) :: %{pos_integer() => Nx.Tensor.t()}
  def rolled_images(dataset, subject) do
    dir = Path.join(dataset.subjects[subject], "rolled")

    for fgp <- 1..10,
        png = Path.join(dir, "fgp#{String.pad_leading("#{fgp}", 2, "0")}.png"),
        File.exists?(png),
        into: %{},
        do: {fgp, read_image(png)}
  end

  @doc "`probe`'s latent at 500 ppi."
  @spec latent_image(probe()) :: Nx.Tensor.t()
  def latent_image(probe), do: read_image(probe.file)

  defp read_image(png) do
    meta = (Path.rootname(png) <> ".json") |> File.read!() |> JSON.decode!()
    png |> Png.read!() |> Resample.to_500_ppi(meta["ppi"])
  end

  ## Scores

  @doc """
  The score matrix of `method` (probes × gallery, in `dataset` order), from
  the cache when it was computed for exactly these probes and gallery.
  """
  @spec scores(dataset(), String.t()) :: Nx.Tensor.t()
  def scores(dataset, "fuse_minrank") do
    fuse_minrank(scores(dataset, "flare"), scores(dataset, "nbis"))
  end

  def scores(dataset, "fuse_z") do
    Nx.add(zscore(scores(dataset, "flare")), zscore(scores(dataset, "nbis")))
  end

  def scores(dataset, "fuse_realign") do
    Nx.add(zscore(scores(dataset, "flare_realign")), zscore(scores(dataset, "nbis")))
  end

  def scores(dataset, "fuse_fdd") do
    Nx.add(zscore(scores(dataset, "flare")), zscore(scores(dataset, "nbis_fdd")))
  end

  # z-score sums of the minutiae matchers with FlareNx, by name.
  @z_fusions %{
    "fuse_mcc" => ~w(flare mcc_fdd),
    "fuse_mcc2" => ~w(flare mcc_nbis mcc_fdd),
    "fuse_mcc_all" => ~w(flare mcc_nbis mcc_fdd nbis nbis_fdd),
    "fuse_mcc_leader" => ~w(flare mcc_leader),
    "fuse_mcc3" => ~w(flare mcc_nbis mcc_fdd mcc_leader),
    "fuse_mcc3_all" => ~w(flare mcc_nbis mcc_fdd mcc_leader nbis nbis_fdd),
    "fuse_fdmd" => ~w(flare fdmd),
    "fuse_fdmd_mcc" => ~w(fdmd mcc_leader),
    "fuse_fdmd_all" => ~w(flare fdmd mcc_nbis mcc_fdd mcc_leader nbis nbis_fdd),
    "fuse_dmd2" => ~w(fdmd ldmd)
  }

  def scores(dataset, method) when is_map_key(@z_fusions, method) do
    @z_fusions
    |> Map.fetch!(method)
    |> Enum.map(&zscore(scores(dataset, &1)))
    |> Enum.reduce(&Nx.add/2)
  end

  # NBIS on both minutiae sets: they find different mates (E9's unions).
  def scores(dataset, "fuse_nbis2") do
    ["flare", "nbis", "nbis_fdd"]
    |> Enum.map(&zscore(scores(dataset, &1)))
    |> Enum.reduce(&Nx.add/2)
  end

  def scores(dataset, "fuse_realign_nbis2") do
    ["flare_realign", "nbis", "nbis_fdd"]
    |> Enum.map(&zscore(scores(dataset, &1)))
    |> Enum.reduce(&Nx.add/2)
  end

  def scores(dataset, "fuse_realign_fdd") do
    Nx.add(zscore(scores(dataset, "flare_realign_fdd")), zscore(scores(dataset, "nbis_fdd")))
  end

  # Reciprocal rank fusion, Σ 1/(60 + rank): no scores, so no scale to
  # get wrong; the baseline a score fusion has to beat.
  def scores(dataset, "fuse_rrf") do
    ["flare", "nbis"]
    |> Enum.map(
      &(dataset
        |> scores(&1)
        |> ranks_of_all()
        |> Nx.add(60)
        |> then(fn r -> Nx.divide(1, r) end))
    )
    |> Enum.reduce(&Nx.add/2)
  end

  def scores(dataset, method) when is_map_key(@robust_fusions, method) do
    {flare_method, transform} = Map.fetch!(@robust_fusions, method)
    nbis = scores(dataset, "nbis")
    nbis = if transform == :log, do: nbis |> Nx.max(0) |> Nx.log1p(), else: nbis
    Nx.add(robust_z(scores(dataset, flare_method), 1), robust_z(nbis, 1))
  end

  def scores(dataset, method) when is_map_key(@flare_max_methods, method) do
    @flare_max_methods
    |> Map.fetch!(method)
    |> Enum.map(&scores(dataset, &1))
    |> Enum.reduce(&Nx.max/2)
  end

  # Each single-view method by its overlap, then combined as one method's
  # views are (`flare_matrix/3`).
  def scores(dataset, method) when is_map_key(@normalised_max_methods, method) do
    {base, normalise} = Map.fetch!(@normalised_max_methods, method)

    @flare_max_methods
    |> Map.fetch!(base)
    |> Enum.map(&scores(dataset, &1 <> "_ov"))
    |> Nx.stack(axis: 1)
    |> Nx.new_axis(3)
    |> standardise_hypotheses(normalise)
    |> combine_hypotheses(normalise)
  end

  def scores(dataset, method) do
    cached(dataset, scores_cache(dataset, method), fn ->
      compute_scores(dataset, method)
    end)
    |> then(fn {shape, binary} -> binary |> Nx.from_binary(:f32) |> Nx.reshape(shape) end)
  end

  @doc "Whether `method`'s scores on `dataset` are cached (`scores/2` won't extract)."
  @spec scored?(dataset(), String.t()) :: boolean()
  def scored?(dataset, method),
    do: File.exists?(cache_path(dataset, scores_cache(dataset, method)))

  defp scores_cache(dataset, method), do: "scores-#{method}-#{signature(dataset)}"

  defp compute_scores(dataset, method) do
    matrix =
      case method do
        "flare" <> _ when is_map_key(@flare_methods, method) ->
          {probe_extractor, gallery_extractor} = Map.fetch!(@flare_methods, method)

          flare_matrix(
            templates(dataset, probe_extractor, :probe),
            templates(dataset, gallery_extractor, :gallery)
          )

        _ when is_map_key(@normalised_flare_methods, method) ->
          {base, normalise} = Map.fetch!(@normalised_flare_methods, method)
          {probe_extractor, gallery_extractor} = Map.fetch!(@flare_methods, base)

          flare_matrix(
            templates(dataset, probe_extractor, :probe),
            templates(dataset, gallery_extractor, :gallery),
            normalise: normalise
          )

        "flare_oracle" ->
          flare_matrix(oracle_templates(dataset), templates(dataset, "flare", :gallery))

        "flare_realign" ->
          realign_matrix(dataset, scores(dataset, "fuse_z"), 50)

        "flare_realign_fdd" ->
          realign_matrix(dataset, scores(dataset, "fuse_fdd"), 50, "fdd")

        "nbis_fdd" ->
          nbis_matrix(templates(dataset, "fdd", :probe), templates(dataset, "fdd", :gallery))

        "nbis_leader" ->
          nbis_matrix(
            templates(dataset, "leader", :probe),
            templates(dataset, "leader", :gallery)
          )

        "nbis_truth" <> visible ->
          {probe, gallery} =
            truth_minutiae_sets(dataset, if(visible == "", do: :all, else: :visible))

          heights = probe_heights(dataset)
          poses = gallery_poses(dataset)

          nbis_matrix(
            Enum.zip_with(dataset.probes, probe, &FddMap.to_xyt(&2, heights[&1.key])),
            Enum.zip_with(dataset.gallery, gallery, &FddMap.to_xyt(&2, gallery_height(poses, &1)))
          )

        "mcc_" <> source ->
          {probe, gallery} = minutiae_sets(dataset, source)
          mcc_matrix(probe, gallery)

        "fdmd" ->
          dmd_matrix(templates(dataset, "dmd", :probe), templates(dataset, "dmd", :gallery), [])

        # As a gallery stored packed is scored (`Dmd.pack/1`,
        # `Dmd.prepare_packed/2`).
        "ldmd_packed" ->
          packed = &(&1 |> dmd_tensors() |> Dmd.pack())
          gallery = dataset |> templates("dmd_leader", :gallery) |> Enum.map(packed)
          gallery = Dmd.prepare_packed(gallery, chunk: 128)

          dataset
          |> templates("dmd_leader", :probe)
          |> Task.async_stream(&Dmd.score_many(&1 |> packed.() |> Dmd.unpack(), gallery),
            timeout: :infinity,
            max_concurrency: concurrency(:dmd)
          )
          |> Enum.map(fn {:ok, row} -> row end)
          |> Nx.tensor(type: :f32)

        "ldmd_" <> variant when variant in @probe_minutiae ->
          dmd_matrix(
            templates(dataset, "dmd_leader_" <> variant, :probe),
            templates(dataset, "dmd_leader", :gallery),
            []
          )

        "ldmd_r" ->
          with_ridges = fn side ->
            Enum.zip_with(
              templates(dataset, "dmd_leader", side),
              templates(dataset, "ridges_leader", side),
              fn
                nil, _ -> nil
                template, nil -> template
                template, ridges -> Map.put(template, :ridges, ridges)
              end
            )
          end

          dmd_matrix(with_ridges.(:probe), with_ridges.(:gallery), ridges: true)

        "ldmd_q" <> spec ->
          {sides, alpha} =
            case spec do
              "p" <> alpha -> {[:probe], alpha}
              alpha -> {[:probe, :gallery], alpha}
            end

          alpha = String.to_integer(alpha)

          dmd_matrix(
            confidence_weighted(dataset, :probe, if(:probe in sides, do: alpha)),
            confidence_weighted(dataset, :gallery, if(:gallery in sides, do: alpha)),
            []
          )

        "ldmd" <> precision ->
          quantise = &quantise_dmd(&1, precision)

          dmd_matrix(
            dataset |> templates("dmd_leader", :probe) |> Enum.map(quantise),
            dataset |> templates("dmd_leader", :gallery) |> Enum.map(quantise),
            []
          )

        "flare_ann" ->
          ann_matrix(templates(dataset, "flare", :probe), templates(dataset, "flare", :gallery))

        "nbis" <> _ ->
          nbis_matrix(templates(dataset, method, :probe), templates(dataset, "nbis", :gallery))
      end

    matrix = Nx.as_type(matrix, :f32)
    {Nx.shape(matrix), Nx.to_binary(matrix)}
  end

  ## Templates

  @doc """
  `extractor`'s templates for one side of `dataset`, in its order: a list
  for `:probe`, one per probe; a list for `:gallery`, one per gallery
  entry. Cached per entry key.
  """
  @spec templates(dataset(), String.t(), :probe | :gallery) :: [term()]
  def templates(dataset, extractor, side) do
    cache_name = "templates-#{extractor}-#{side}"
    have = read_cache(dataset, cache_name) || %{}
    keys = side_keys(dataset, side)
    missing = Enum.reject(keys, &Map.has_key?(have, &1))

    have =
      if missing == [] do
        have
      else
        {%{^extractor => fresh}, _heights} =
          extract(dataset, %{extractor => MapSet.new(missing)}, side)

        all = Map.merge(have, fresh)
        write_cache(dataset, cache_name, all)
        all
      end

    Enum.map(keys, &Map.fetch!(have, &1))
  end

  @doc """
  The extractors `method` reads templates of, per side, for
  `extract_all/2`; none for a method this doesn't know (it extracts as
  it goes).
  """
  @spec extractors(String.t()) :: %{probe: [String.t()], gallery: [String.t()]}
  def extractors(method) do
    {probe, gallery} =
      cond do
        is_map_key(@flare_methods, method) ->
          Map.fetch!(@flare_methods, method)

        String.starts_with?(method, "ldmd_") and
            String.replace_prefix(method, "ldmd_", "") in @probe_minutiae ->
          {"dmd_leader_" <> String.replace_prefix(method, "ldmd_", ""), "dmd_leader"}

        method == "ldmd_r" ->
          {~w(dmd_leader ridges_leader), ~w(dmd_leader ridges_leader)}

        String.starts_with?(method, "ldmd_q") ->
          {~w(dmd_leader leader), ~w(dmd_leader leader)}

        String.starts_with?(method, "ldmd") ->
          {"dmd_leader", "dmd_leader"}

        method == "fdmd" ->
          {"dmd", "dmd"}

        method in ~w(mcc_nbis mcc_fdd mcc_leader) ->
          {mcc_source(method), mcc_source(method)}

        method == "nbis" or is_map_key(@nbis_filters, method) ->
          {method, "nbis"}

        true ->
          {nil, nil}
      end

    %{probe: List.wrap(probe), gallery: List.wrap(gallery)}
  end

  defp mcc_source("mcc_" <> source), do: source

  @doc """
  Extracts and caches, as `templates/3` does, every template
  `extractors` (`%{probe: [...], gallery: [...]}`, e.g. `extractors/1`'s,
  merged) still misses, decoding each image once for all of them: a
  latent's JPEG 2000 decode, ~0.2 s, would otherwise repeat for every
  extractor. A first pass over a side also caches its images' heights.
  """
  @spec extract_all(dataset(), %{optional(:probe | :gallery) => [String.t()]}) :: :ok
  def extract_all(dataset, extractors) do
    for {side, names} <- extractors, names != [] do
      names = Enum.uniq(names)
      have = Map.new(names, &{&1, read_cache(dataset, "templates-#{&1}-#{side}") || %{}})
      keys = side_keys(dataset, side)

      wanted =
        Map.new(names, fn name ->
          {name, keys |> Enum.reject(&Map.has_key?(have[name], &1)) |> MapSet.new()}
        end)

      if Enum.any?(wanted, fn {_, w} -> MapSet.size(w) > 0 end) do
        {fresh, heights} = extract(dataset, wanted, side)

        for {name, found} <- fresh,
            map_size(found) > 0,
            do: write_cache(dataset, "templates-#{name}-#{side}", Map.merge(have[name], found))

        # Only a pass over every image knows every height.
        everything = wanted |> Map.values() |> Enum.reduce(&MapSet.union/2)

        if MapSet.size(everything) == length(keys) and
             read_cache(dataset, heights_cache(dataset, side)) == nil,
           do: write_cache(dataset, heights_cache(dataset, side), heights)
      end
    end

    :ok
  end

  @doc """
  `extract_all/2` for the templates `methods` not yet scored on `dataset`
  read (`extractors/1`): call it before scoring several methods.
  """
  @spec extract_for(dataset(), [String.t()]) :: :ok
  def extract_for(dataset, methods) do
    extractors =
      methods
      |> Enum.reject(&scored?(dataset, &1))
      |> Enum.map(&extractors/1)
      |> Enum.reduce(%{probe: [], gallery: []}, &Map.merge(&2, &1, fn _, a, b -> a ++ b end))

    extract_all(dataset, extractors)
  end

  defp heights_cache(dataset, :probe), do: "probe-heights-#{signature(dataset)}"
  defp heights_cache(dataset, :gallery), do: "gallery-heights-#{signature(dataset)}"

  defp side_keys(dataset, :probe), do: Enum.map(dataset.probes, & &1.key)
  defp side_keys(dataset, :gallery), do: Enum.map(dataset.gallery, & &1.key)

  # Each image any extractor in `wanted` (`%{extractor => keys}`) needs,
  # decoded once: `{%{extractor => %{key => template}}, %{key => height}}`.
  defp extract(dataset, wanted, :gallery) do
    any = wanted |> Map.values() |> Enum.reduce(&MapSet.union/2)

    rows =
      dataset.subjects
      |> Enum.filter(fn {subject, _dir} ->
        Enum.any?(1..10, &MapSet.member?(any, "#{subject}/#{&1}"))
      end)
      |> Task.async_stream(
        fn {subject, _dir} ->
          for {fgp, tensor} <- rolled_images(dataset, subject),
              key = "#{subject}/#{fgp}",
              MapSet.member?(any, key) do
            {key, elem(Nx.shape(tensor), 0),
             for(
               {name, keys} <- wanted,
               MapSet.member?(keys, key),
               do: {name, extract_one(name, tensor, :gallery, %{})}
             )}
          end
        end,
        timeout: :infinity,
        max_concurrency: concurrency(:extract)
      )
      |> Enum.flat_map(fn {:ok, rows} -> rows end)

    found =
      Map.new(wanted, fn {name, keys} ->
        {name, rows |> by_extractor(name) |> fill_missing_fingers(keys, name)}
      end)

    {found, Map.new(rows, fn {key, height, _} -> {key, height} end)}
  end

  defp extract(dataset, wanted, :probe) do
    any = wanted |> Map.values() |> Enum.reduce(&MapSet.union/2)

    rows =
      dataset.probes
      |> Enum.filter(&MapSet.member?(any, &1.key))
      |> Task.async_stream(
        fn probe ->
          image = latent_image(probe)

          {probe.key, elem(Nx.shape(image), 0),
           for(
             {name, keys} <- wanted,
             MapSet.member?(keys, probe.key),
             do: {name, extract_one(name, image, :probe, %{})}
           )}
        end,
        timeout: :infinity,
        max_concurrency: concurrency(:extract)
      )
      |> Enum.map(fn {:ok, row} -> row end)

    {Map.new(wanted, fn {name, _} -> {name, by_extractor(rows, name)} end),
     Map.new(rows, fn {key, height, _} -> {key, height} end)}
  end

  defp by_extractor(rows, name) do
    for {key, _height, templates} <- rows, {^name, t} <- templates, into: %{}, do: {key, t}
  end

  # A finger with no image (AMP) still needs a gallery column; it can't match.
  defp fill_missing_fingers(found, wanted, extractor) do
    Enum.reduce(wanted, found, fn key, acc -> Map.put_new(acc, key, empty(extractor)) end)
  end

  defp empty("nbis"), do: ""
  defp empty("fdd"), do: ""
  defp empty("leader"), do: ""
  defp empty("dmd" <> _), do: nil
  defp empty("ridges_leader"), do: nil
  defp empty("flare" <> _), do: []

  # FlareNx: a list of templates (one per pose/window variant), each with
  # binary feature/mask.
  defp extract_one("flare" <> _ = extractor, tensor, _side, _context) do
    tensor
    |> FlareNx.extract_variants(Map.fetch!(@extractors, extractor))
    |> Enum.map(&%{feature: Nx.to_binary(&1.feature), mask: Nx.to_binary(&1.mask)})
  end

  defp extract_one("nbis_" <> _ = extractor, tensor, :probe, _context) do
    xyt = extract_one("nbis", tensor, :probe, %{})
    pose = FlareNx.pose(tensor)
    mask = FlareNx.template_at_pose(tensor, pose).mask

    Foreground.filter_xyt(
      xyt,
      elem(Nx.shape(tensor), 0),
      Map.fetch!(@nbis_filters, extractor) ++ [pose: pose, mask: mask]
    )
  end

  # F-DMD: DMD++ descriptors at FDD's minutiae, stored as binaries (an
  # EXLA tensor can't be cached); nil without minutiae.
  defp extract_one("dmd", tensor, _side, _context),
    do: dmd_template(tensor, FlareNx.minutiae(tensor))

  # The same descriptors at LEADER's minutiae.
  defp extract_one("dmd_leader", tensor, _side, _context),
    do: dmd_template(tensor, Leader.extract(tensor))

  # At richer latent minutiae (`"ldmd_t40"` etc. in the moduledoc).
  defp extract_one("dmd_leader_" <> variant, tensor, _side, _context),
    do: dmd_template(tensor, leader_minutiae(tensor, variant))

  # Ridge counts between LEADER's minutiae (`Minutiae.RidgeCount`), in
  # the `dmd_leader` template's order, as `{{n, n}, u8 binary}`.
  defp extract_one("ridges_leader", tensor, _side, _context) do
    case RidgeCount.matrix(tensor, Leader.extract(tensor)) do
      nil -> nil
      matrix -> {Nx.shape(matrix), Nx.to_binary(matrix)}
    end
  end

  # LEADER's minutiae (`Minutiae.Leader`), as `.xyt` for `bozorth3`.
  defp extract_one("leader", tensor, _side, _context) do
    tensor |> Leader.extract() |> FddMap.to_xyt(elem(Nx.shape(tensor), 0))
  end

  # FDD's minutiae (`FlareNx.minutiae/2`), as `.xyt` for `bozorth3`.
  defp extract_one("fdd", tensor, _side, _context) do
    tensor |> FlareNx.minutiae() |> FddMap.to_xyt(elem(Nx.shape(tensor), 0))
  end

  defp extract_one("nbis", tensor, _side, _context) do
    case Nbis.extract(tensor) do
      {:ok, xyt} -> xyt
      {:error, _} -> ""
    end
  end

  defp leader_minutiae(tensor, "t40"), do: Leader.extract(tensor, threshold: 0.4)
  defp leader_minutiae(tensor, "t50"), do: Leader.extract(tensor, threshold: 0.5)

  defp leader_minutiae(tensor, "views"),
    do: Leader.extract_views(tensor, threshold: 0.5, votes: 4)

  defp leader_minutiae(tensor, "min" <> m),
    do: Leader.extract_latent(tensor, at_least: String.to_integer(m))

  defp dmd_template(_tensor, []), do: nil

  defp dmd_template(tensor, minutiae) do
    tensor
    |> Dmd.extract(minutiae)
    |> Map.new(fn {k, t} -> {k, {Nx.shape(t), Nx.to_binary(t)}} end)
  end

  ## Oracle pose

  @doc """
  Each probe's FlareNx template at its true pose, where the ground truth
  gives one: the rigid transform latent → rolled print, fitted to their
  shared minutiae (≥ 3), carries the rolled print's own FlareNx pose into
  the latent. The rank this gives is the ceiling for every pose fix. Probes
  with fewer shared minutiae keep their estimated pose. Cached.
  """
  @spec oracle_templates(dataset()) :: [[map()]]
  def oracle_templates(dataset) do
    cached(dataset, "templates-flare_oracle-#{signature(dataset)}", fn ->
      dataset.probes
      |> Enum.group_by(& &1.subject)
      |> Task.async_stream(&oracle_subject(dataset, &1), timeout: :infinity, max_concurrency: 4)
      |> Enum.flat_map(fn {:ok, pairs} -> pairs end)
      |> Map.new()
      |> then(fn by_key -> Enum.map(dataset.probes, &Map.fetch!(by_key, &1.key)) end)
    end)
  end

  defp oracle_subject(dataset, {subject, probes}) do
    rolled = rolled_images(dataset, subject)

    for probe <- probes do
      latent = latent_image(probe)

      template =
        with true <- length(probe.pairs) >= 3,
             %Nx.Tensor{} = rolled_image <- rolled[probe.fgp] do
          FlareNx.template_at_pose(latent, oracle_pose(probe.pairs, FlareNx.pose(rolled_image)))
        else
          _ -> latent |> FlareNx.extract_variants() |> hd()
        end

      {probe.key, [%{feature: Nx.to_binary(template.feature), mask: Nx.to_binary(template.mask)}]}
    end
  end

  @doc """
  The latent's pose implied by the rolled print's pose `{x, y, theta}`
  and their shared minutiae: with `p_r = R(phi) p_l + t` fitted by least
  squares (2D Procrustes, no scale), the latent's centre is
  `R(-phi)(c_r - t)` and its angle `theta + phi`, so both crops show the
  same region the same way up.
  """
  @spec oracle_pose([{{float(), float()}, {float(), float()}}], {float(), float(), float()}) ::
          {float(), float(), float()}
  def oracle_pose(pairs, {rx, ry, rtheta}) do
    n = length(pairs)

    {{lx, ly}, {px, py}} =
      Enum.reduce(pairs, {{0, 0}, {0, 0}}, fn {{a, b}, {c, d}}, {{sa, sb}, {sc, sd}} ->
        {{sa + a, sb + b}, {sc + c, sd + d}}
      end)

    {mlx, mly, mrx, mry} = {lx / n, ly / n, px / n, py / n}

    {num, den} =
      Enum.reduce(pairs, {0.0, 0.0}, fn {{a, b}, {c, d}}, {num, den} ->
        {xl, yl, xr, yr} = {a - mlx, b - mly, c - mrx, d - mry}
        {num + (xl * yr - yl * xr), den + (xl * xr + yl * yr)}
      end)

    phi = :math.atan2(num, den)
    {cos, sin} = {:math.cos(phi), :math.sin(phi)}
    # t = mean_r - R(phi) mean_l
    {tx, ty} = {mrx - (cos * mlx - sin * mly), mry - (sin * mlx + cos * mly)}

    Alignment.pose_from_transform(
      %{phi: phi * 180.0 / :math.pi(), tx: tx, ty: ty},
      {rx, ry, rtheta}
    )
  end

  @doc """
  FlareNx scores with each probe's top-`k` candidates (by `shortlist`)
  re-scored after re-aligning the latent to that candidate: the rigid
  transform between their minutiae (`Minutiae.Alignment`; `minutiae` is
  the extractor, `mindtct`'s `"nbis"` or FDD's `"fdd"`)
  carries the candidate's FlareNx pose into the latent, as the oracle
  does with true minutiae. A candidate keeps the better of its original
  and re-aligned score; one with no transform keeps the original.
  """
  @spec realign_matrix(dataset(), Nx.Tensor.t(), pos_integer(), String.t()) :: Nx.Tensor.t()
  def realign_matrix(dataset, shortlist, k, minutiae \\ "nbis") do
    base = scores(dataset, "flare")
    probe_xyts = templates(dataset, minutiae, :probe)
    gallery_xyts = templates(dataset, minutiae, :gallery)
    gallery_templates = templates(dataset, "flare", :gallery)
    poses = gallery_poses(dataset)
    {_, top} = Nx.top_k(shortlist, k: k)

    dataset.probes
    |> Enum.zip(probe_xyts)
    |> Enum.zip(Enum.zip(Nx.to_list(base), Nx.to_list(top)))
    |> Task.async_stream(
      fn {{probe, xyt}, {row, candidates}} ->
        latent = latent_image(probe)
        lat = Alignment.from_xyt(xyt, elem(Nx.shape(latent), 0))

        candidates
        |> Enum.reduce(List.to_tuple(row), fn j, acc ->
          entry = Enum.at(dataset.gallery, j)

          with %{pose: pose, height: height} <- poses[entry.key],
               [template] <- Enum.at(gallery_templates, j),
               rol = Alignment.from_xyt(Enum.at(gallery_xyts, j), height),
               {:ok, transform} <- Alignment.align(lat, rol) do
            realigned =
              FlareNx.template_at_pose(latent, Alignment.pose_from_transform(transform, pose))

            {:ok, score} = Flare.score(realigned, template_tensors(template))
            put_elem(acc, j, max(elem(acc, j), score))
          else
            _ -> acc
          end
        end)
        |> Tuple.to_list()
      end,
      timeout: :infinity,
      max_concurrency: 4
    )
    |> Enum.map(fn {:ok, row} -> row end)
    |> Nx.tensor(type: :f32)
  end

  # Measured on 24 cores, a 1,000 latents × 1,000 prints run: 1,224 s with
  # every stage at 4, 810 s with these, every result identical.
  @concurrency_defaults [extract: 12, decode: 16, nbis: 16, mcc: 8, dmd: 4]
  @concurrency_groups Keyword.keys(@concurrency_defaults)

  @doc """
  How many tasks a stage group runs at once: `:extract` (decoding and
  every extractor, on the CPU and the GPU), `:decode` (image heights and
  FlareNx poses), `:nbis` (`bozorth3` processes), `:mcc` and `:dmd`
  (scoring on EXLA). The defaults were measured on a 24-core machine and
  are capped at the number of schedulers. `config :faulds,
  Faulds.LatentBenchmark, concurrency: [group: n]` overrides them,
  uncapped (`mix faulds.latent_benchmark --concurrency extract=4` beside
  another job). The setting doesn't change the results: `ldmd`, NBIS and
  the templates are equal bit for bit at any setting; `fdmd` and `flare`
  vary from run to run at any setting, the defaults included.
  """
  @spec concurrency(atom()) :: pos_integer()
  def concurrency(group) when group in @concurrency_groups do
    :faulds
    |> Application.get_env(__MODULE__, [])
    |> Keyword.get(:concurrency, [])
    |> Keyword.get_lazy(group, fn ->
      min(Keyword.fetch!(@concurrency_defaults, group), System.schedulers_online())
    end)
  end

  @doc "The groups `concurrency/1` takes."
  def concurrency_groups, do: @concurrency_groups

  @doc "Each rolled print's FlareNx pose and image height, by gallery key. Cached."
  @spec gallery_poses(dataset()) :: %{String.t() => %{pose: tuple(), height: pos_integer()}}
  def gallery_poses(dataset) do
    cached(dataset, "gallery-poses-#{signature(dataset)}", fn ->
      dataset.subjects
      |> Task.async_stream(
        fn {subject, _dir} ->
          for {fgp, tensor} <- rolled_images(dataset, subject) do
            {"#{subject}/#{fgp}",
             %{pose: FlareNx.pose(tensor), height: elem(Nx.shape(tensor), 0)}}
          end
        end,
        timeout: :infinity,
        max_concurrency: concurrency(:decode)
      )
      |> Enum.flat_map(fn {:ok, pairs} -> pairs end)
      |> Map.new()
    end)
  end

  defp signature(dataset),
    do: :erlang.phash2({Enum.map(dataset.probes, & &1.key), dataset.gallery})

  ## FlareNx scoring

  # What a normalised score gives an entry with no template: below any
  # score a template can get (|z| stays far under it).
  @no_template -1.0e3

  @doc """
  The exact masked FlareNx score (`Matchers.Flare.score/2`) of every probe
  against every gallery entry, as three matrix products: with `f` the
  12×256 feature, `m` the 256-cell mask and `e = m · Σ_channels f²`,

      score = (f_p∘m_p)·(f_g∘m_g) / sqrt((m_g·e_p) (e_g·m_p))

  Every entry holds a list of templates (pose or window variants); an
  entry's score is the best over both sides' variants. An entry with no
  template scores -1.
  """
  @spec flare_matrix([[map()]], [[map()]], keyword()) :: Nx.Tensor.t()
  def flare_matrix(probe_lists, gallery_lists, opts \\ []) do
    %{cosine: cosine, overlap: overlap, valid: valid} = flare_parts(probe_lists, gallery_lists)

    case Keyword.get(opts, :normalise, :none) do
      :none ->
        cosine |> Nx.reduce_max(axes: [1, 3]) |> then(&Nx.select(valid, &1, -1.0))

      normalise ->
        cosine
        |> Nx.multiply(overlap_factor(overlap))
        |> standardise_hypotheses(normalise)
        |> combine_hypotheses(normalise)
        |> then(&Nx.select(valid, &1, @no_template))
    end
  end

  # Every score over the same typical overlap: the masks' common cells
  # (soft, out of 256), as DMD's √(h_o/H_o) and LDRF's √(s_o/S_o). Under
  # the impostor hypothesis a masked cosine over n cells has a spread
  # ∝ 1/√n, so this gives small and large overlaps one null distribution.
  # The constant only rescales.
  @typical_overlap 256.0

  defp overlap_factor(overlap), do: overlap |> Nx.divide(@typical_overlap) |> Nx.sqrt()

  # Each hypothesis (probe variant × gallery variant) on one scale over
  # the gallery, so a view that scores everyone higher can't win the max.
  defp standardise_hypotheses(scores, :overlap), do: scores
  defp standardise_hypotheses(scores, _), do: robust_z(scores, 2)

  defp combine_hypotheses(scores, normalise) when normalise in [:overlap, :overlap_z],
    do: Nx.reduce_max(scores, axes: [1, 3])

  # A soft max at temperature 1 (in z units): several agreeing views add
  # evidence, one high one counts less than in a max. Less log K, so it
  # stays on the z scale whatever the number of views.
  defp combine_hypotheses(scores, :lse) do
    {_np, p_k, _ng, g_k} = Nx.shape(scores)
    scores |> Nx.logsumexp(axes: [1, 3]) |> Nx.subtract(:math.log(p_k * g_k))
  end

  @doc """
  The parts of `flare_matrix/3` before variants are combined, each
  `{probes, probe variants, gallery, gallery variants}`: the masked
  cosine, and the overlap of the two masks (`Σ m_p m_g` over the 256
  cells). `valid` (probes × gallery) is 0 where either side has no
  template.
  """
  @spec flare_parts([[map()]], [[map()]]) :: %{
          cosine: Nx.Tensor.t(),
          overlap: Nx.Tensor.t(),
          valid: Nx.Tensor.t()
        }
  def flare_parts(probe_lists, gallery_lists) do
    {p_rows, p_k, p_valid} = stack(probe_lists)
    {g_rows, g_k, g_valid} = stack(gallery_lists)
    shape = {length(probe_lists), p_k, length(gallery_lists), g_k}

    cosine =
      Nx.divide(
        Nx.dot(p_rows.fm, [1], g_rows.fm, [1]),
        Nx.dot(p_rows.e, [1], g_rows.m, [1])
        |> Nx.multiply(Nx.dot(p_rows.m, [1], g_rows.e, [1]))
        |> Nx.sqrt()
        |> Nx.max(1.0e-3)
      )

    %{
      cosine: Nx.reshape(cosine, shape),
      overlap: p_rows.m |> Nx.dot([1], g_rows.m, [1]) |> Nx.reshape(shape),
      valid: Nx.outer(p_valid, g_valid)
    }
  end

  @doc """
  `(x − median) / (1.4826 · MAD)` along `axis`, the robust z-score: a few
  high scores (the mate, a lucky impostor) don't move it. The scale has a
  floor, the median scale of all the other slices: a probe whose scores
  are nearly all equal (`bozorth3`'s zeros for a latent with few
  minutiae) would otherwise blow up its one or two non-zero scores.
  """
  @spec robust_z(Nx.Tensor.t(), non_neg_integer()) :: Nx.Tensor.t()
  def robust_z(scores, axis) do
    median = Nx.median(scores, axis: axis, keep_axis: true)
    deviation = scores |> Nx.subtract(median) |> Nx.abs()
    scale = deviation |> Nx.median(axis: axis, keep_axis: true) |> Nx.multiply(1.4826)
    Nx.divide(Nx.subtract(scores, median), Nx.max(scale, scale_floor(scale)))
  end

  defp scale_floor(scale) do
    positive = scale |> Nx.to_flat_list() |> Enum.filter(&(&1 > 0))
    if positive == [], do: 1.0, else: positive |> Enum.sort() |> Enum.at(div(length(positive), 2))
  end

  ## Minutiae extraction

  @doc """
  How well `extractor`'s latent minutiae (`"nbis"`: `mindtct`; `"fdd"`:
  FDD's map; `"leader"`: LEADER) find each probe's true ones: a detection matches a true
  minutia within `:tolerance` px (default 8, at 500 ppi) and
  `:angle_tolerance` degrees (default 20), each used once, nearest
  first. `%{truth:, detected:, found:, recall:, precision:}`, over the
  probes with ground truth (not mirrored).

  With `only: :visible` (needs `attach_visibility/2`; probes without
  labels are left out), the true minutiae are the `visible` ones, and a
  detection at a `degraded` or `hidden` one counts neither as a hit nor
  as a false one (it isn't in `detected`).
  """
  @spec minutiae_accuracy(dataset(), String.t(), keyword()) :: map()
  def minutiae_accuracy(dataset, extractor, opts \\ []) do
    tolerance = Keyword.get(opts, :tolerance, 8)
    angle_tolerance = Keyword.get(opts, :angle_tolerance, 20)
    heights = probe_heights(dataset)

    only_visible = Keyword.get(opts, :only) == :visible

    {truth, detected, found} =
      dataset.probes
      |> Enum.zip(templates(dataset, extractor, :probe))
      |> Enum.reject(fn {probe, _xyt} -> probe.minutiae == [] end)
      |> Enum.reject(fn {probe, _xyt} -> only_visible and probe.visibility == nil end)
      |> Enum.reduce({0, 0, 0}, fn {probe, xyt}, {t, d, f} ->
        # `.xyt` back to this library's convention (`Matchers.Nbis`).
        detections =
          for line <- String.split(xyt, "\n", trim: true),
              [x, y, theta | _] = line |> String.split() |> Enum.map(&String.to_integer/1) do
            {x * 1.0, heights[probe.key] - y * 1.0, wrap360(180.0 - theta)}
          end

        {targets, others} =
          if only_visible do
            probe.minutiae
            |> Enum.zip(probe.visibility)
            |> Enum.split_with(&(elem(&1, 1) == "visible"))
            |> then(fn {v, o} -> {Enum.map(v, &elem(&1, 0)), Enum.map(o, &elem(&1, 0))} end)
          else
            {probe.minutiae, []}
          end

        {n, left} = count_matches(targets, detections, tolerance, angle_tolerance)
        {ignored, _} = count_matches(others, left, tolerance, angle_tolerance)
        {t + length(targets), d + length(detections) - ignored, f + n}
      end)

    %{
      truth: truth,
      detected: detected,
      found: found,
      recall: found / max(truth, 1),
      precision: found / max(detected, 1)
    }
  end

  defp count_matches(truth, detections, tolerance, angle_tolerance) do
    truth
    |> Enum.reduce({0, detections}, fn {tx, ty, ta}, {n, left} ->
      candidates =
        for {dx, dy, da} = d <- left,
            dist = :math.sqrt((dx - tx) ** 2 + (dy - ty) ** 2),
            dist <= tolerance,
            abs(wrap360(da - ta + 180.0) - 180.0) <= angle_tolerance,
            do: {dist, d}

      case Enum.min_by(candidates, &elem(&1, 0), fn -> nil end) do
        nil -> {n, left}
        {_, d} -> {n + 1, List.delete(left, d)}
      end
    end)
  end

  # Each probe's imported image height (for `.xyt`'s bottom-left origin). Cached.
  defp probe_heights(dataset) do
    cached(dataset, heights_cache(dataset, :probe), fn ->
      dataset.probes
      |> Task.async_stream(
        fn probe ->
          t = latent_image(probe)
          {probe.key, elem(Nx.shape(t), 0)}
        end,
        timeout: :infinity,
        max_concurrency: concurrency(:decode)
      )
      |> Map.new(fn {:ok, pair} -> pair end)
    end)
  end

  ## Impostor statistics (experiment E3)

  @doc """
  How `method`'s impostor scores depend on the overlap of the two masks:
  every impostor pair (every view of it, for a multi-view method) sorted
  by overlap into `bins` equal-sized bins, each with its mean overlap,
  the raw cosine's mean and spread, and the spread after the overlap
  factor `flare_matrix/3` applies. If the spread goes as `n^-1/2`, as for
  a cosine over n independent cells, the factor makes it constant;
  `:slope` is the fitted exponent (log spread on log overlap). Genuine
  pairs (best view) come with their median overlap and cosine.
  """
  @spec overlap_stats(dataset(), String.t(), pos_integer()) :: map()
  def overlap_stats(dataset, method, bins \\ 10) do
    {probe_extractor, gallery_extractor} = Map.fetch!(@flare_methods, method)

    %{cosine: cosine, overlap: overlap, valid: valid} =
      flare_parts(
        templates(dataset, probe_extractor, :probe),
        templates(dataset, gallery_extractor, :gallery)
      )

    mates = mate_mask(dataset)
    shape = Nx.shape(cosine)
    impostor = valid |> Nx.logical_and(Nx.logical_not(mates)) |> as_views(shape)

    # Impostor pairs in overlap order; the rest (key -1) sort before them.
    key = impostor |> Nx.select(overlap, -1.0) |> Nx.flatten()
    order = Nx.argsort(key)
    skip = Nx.size(key) - (impostor |> Nx.as_type(:s64) |> Nx.sum() |> Nx.to_number())
    size = div(Nx.size(key) - skip, bins)

    pick = fn t ->
      t
      |> Nx.flatten()
      |> Nx.take(order)
      |> Nx.slice([skip], [size * bins])
      |> Nx.reshape({bins, size})
    end

    {n, c} = {pick.(overlap), pick.(cosine)}
    normalised = Nx.multiply(c, overlap_factor(n))

    table =
      [
        Nx.mean(n, axes: [1]),
        Nx.mean(c, axes: [1]),
        Nx.standard_deviation(c, axes: [1]),
        Nx.standard_deviation(normalised, axes: [1])
      ]
      |> Enum.map(&Nx.to_flat_list/1)
      |> Enum.zip_with(fn [n, mean, sd, sd_norm] ->
        %{overlap: n, mean: mean, sd: sd, sd_normalised: sd_norm}
      end)

    # Per probe, its mate's best view (and that view's overlap).
    genuine = valid |> Nx.logical_and(mates) |> as_views(shape)

    best_view =
      genuine |> Nx.select(cosine, -1.0e9) |> Nx.flatten(axes: [1, 2, 3]) |> Nx.argmax(axis: 1)

    at_best = fn t ->
      t
      |> Nx.flatten(axes: [1, 2, 3])
      |> Nx.take_along_axis(Nx.new_axis(best_view, 1), axis: 1)
      |> Nx.to_flat_list()
    end

    has_mate = valid |> Nx.logical_and(mates) |> Nx.any(axes: [1]) |> Nx.to_flat_list()
    of_mated = fn list -> for {x, 1} <- Enum.zip(list, has_mate), do: x end

    %{
      bins: table,
      pairs: size * bins,
      slope: slope(Enum.map(table, &{&1.overlap, &1.sd})),
      slope_normalised: slope(Enum.map(table, &{&1.overlap, &1.sd_normalised})),
      genuine_overlap: median(of_mated.(at_best.(overlap))),
      genuine_cosine: median(of_mated.(at_best.(cosine))),
      impostor_overlap: n |> Nx.median() |> Nx.to_number()
    }
  end

  @doc """
  Pearson correlation of the impostor scores of each pair of `matrices`
  (same probes and gallery), over the pairs every matrix has a template
  for: views that fail on the same impostors add little to a max.
  """
  @spec impostor_correlation(dataset(), [Nx.Tensor.t()]) :: [[float()]]
  def impostor_correlation(dataset, matrices) do
    valid =
      Enum.reduce(
        matrices,
        Nx.logical_not(mate_mask(dataset)),
        &Nx.logical_and(&2, Nx.greater(&1, -1.0))
      )

    weight = Nx.as_type(valid, :f32)
    count = Nx.sum(weight)

    centred =
      Enum.map(matrices, fn m ->
        mean = m |> Nx.multiply(weight) |> Nx.sum() |> Nx.divide(count)
        m |> Nx.subtract(mean) |> Nx.multiply(weight)
      end)

    for a <- centred do
      for b <- centred do
        cov = a |> Nx.multiply(b) |> Nx.sum()
        norm = a |> Nx.pow(2) |> Nx.sum() |> Nx.multiply(b |> Nx.pow(2) |> Nx.sum()) |> Nx.sqrt()
        cov |> Nx.divide(norm) |> Nx.to_number()
      end
    end
  end

  @doc "`method`'s cosines for each of its probe views separately, `[probes × gallery]` (best gallery view)."
  @spec view_matrices(dataset(), String.t()) :: [Nx.Tensor.t()]
  def view_matrices(dataset, method) do
    {probe_extractor, gallery_extractor} = Map.fetch!(@flare_methods, method)

    %{cosine: cosine, valid: valid} =
      flare_parts(
        templates(dataset, probe_extractor, :probe),
        templates(dataset, gallery_extractor, :gallery)
      )

    for v <- 0..(elem(Nx.shape(cosine), 1) - 1) do
      cosine[[.., v, .., ..]] |> Nx.reduce_max(axes: [2]) |> then(&Nx.select(valid, &1, -1.0))
    end
  end

  # 1 where the gallery entry is the probe's own finger.
  defp mate_mask(dataset) do
    index = dataset.gallery |> Enum.with_index() |> Map.new(fn {g, i} -> {g.key, i} end)
    mates = Enum.map(dataset.probes, &Map.fetch!(index, "#{&1.subject}/#{&1.fgp}"))

    mates
    |> Nx.tensor()
    |> Nx.new_axis(1)
    |> Nx.equal(Nx.iota({1, length(dataset.gallery)}, axis: 1))
  end

  # A probes × gallery mask over every view pair.
  defp as_views(mask, shape), do: mask |> Nx.new_axis(1) |> Nx.new_axis(3) |> Nx.broadcast(shape)

  # Least-squares slope of log y on log x.
  defp slope(points) do
    logs = for {x, y} <- points, x > 0 and y > 0, do: {:math.log(x), :math.log(y)}
    n = length(logs)
    {sx, sy} = Enum.reduce(logs, {0.0, 0.0}, fn {x, y}, {a, b} -> {a + x, b + y} end)
    {mx, my} = {sx / n, sy / n}

    {num, den} =
      Enum.reduce(logs, {0.0, 0.0}, fn {x, y}, {a, b} ->
        {a + (x - mx) * (y - my), b + (x - mx) ** 2}
      end)

    num / den
  end

  defp median([]), do: nil
  defp median(list), do: list |> Enum.sort() |> Enum.at(div(length(list), 2))

  @doc "The cosine of `Flare.ann_vector/1` (first variant of each entry), what pgvector ranks by."
  @spec ann_matrix([[map()]], [[map()]]) :: Nx.Tensor.t()
  def ann_matrix(probe_lists, gallery_lists) do
    probes = ann_rows(probe_lists)
    gallery = ann_rows(gallery_lists)
    Nx.dot(probes, [1], gallery, [1])
  end

  defp ann_rows(lists) do
    lists
    |> Enum.map(fn
      [first | _] -> first |> template_tensors() |> Flare.ann_vector()
      [] -> Nx.broadcast(0.0, {3072})
    end)
    |> Nx.stack()
  end

  defp template_tensors(%{feature: feature, mask: mask}) do
    %{feature: Nx.from_binary(feature, :f32), mask: Nx.from_binary(mask, :f32)}
  end

  # Exactly `k` rows per entry (k = the most variants any entry has), so
  # the max over variants is a reshape and a reduce: fewer variants are
  # padded by repeating the last (which leaves the max unchanged), an
  # entry with none by zero rows, flagged invalid.
  defp stack(lists) do
    k = lists |> Enum.map(&length/1) |> Enum.max(fn -> 1 end) |> max(1)

    rows =
      Enum.flat_map(lists, fn
        [] -> List.duplicate(row(nil), k)
        variants -> variants |> Enum.map(&row/1) |> pad(k)
      end)

    valid = lists |> Enum.map(&if(&1 == [], do: 0, else: 1)) |> Nx.tensor(type: :u8)

    {%{
       fm: rows |> Enum.map(& &1.fm) |> Nx.stack(),
       m: rows |> Enum.map(& &1.m) |> Nx.stack(),
       e: rows |> Enum.map(& &1.e) |> Nx.stack()
     }, k, valid}
  end

  defp pad(rows, k), do: rows ++ List.duplicate(List.last(rows), k - length(rows))

  defp row(nil) do
    %{fm: Nx.broadcast(0.0, {3072}), m: Nx.broadcast(0.0, {256}), e: Nx.broadcast(0.0, {256})}
  end

  defp row(template) do
    %{feature: feature, mask: mask} = template_tensors(template)
    tiled = Nx.tile(mask, [12])

    %{
      fm: Nx.multiply(feature, tiled),
      m: mask,
      e: feature |> Nx.reshape({12, 256}) |> Nx.pow(2) |> Nx.sum(axes: [0]) |> Nx.multiply(mask)
    }
  end

  ## Minutiae as lists (MCC, ground truth)

  # Each side's minutiae (`%{x:, y:, angle:}`, this library's convention) from
  # `source`: `"nbis"` (`mindtct`) or `"fdd"`, read back from the cached
  # `.xyt`, or `"truth"` (the ground truth's).
  defp minutiae_sets(dataset, "truth"), do: truth_minutiae_sets(dataset)
  defp minutiae_sets(dataset, "truth_visible"), do: truth_minutiae_sets(dataset, :visible)

  defp minutiae_sets(dataset, source) when source in ["nbis", "fdd", "leader"] do
    probe = templates(dataset, source, :probe)
    gallery = templates(dataset, source, :gallery)
    heights = probe_heights(dataset)
    gallery_heights = gallery_heights(dataset)

    {Enum.zip_with(dataset.probes, probe, &xyt_minutiae(&2, heights[&1.key])),
     Enum.zip_with(dataset.gallery, gallery, &xyt_minutiae(&2, gallery_heights[&1.key] || 1))}
  end

  # Each gallery print's height (none for a missing finger), as the first
  # extraction pass caches it; decoded here only if none has run.
  defp gallery_heights(dataset) do
    cached(dataset, heights_cache(dataset, :gallery), fn ->
      dataset.subjects
      |> Task.async_stream(
        fn {subject, _dir} ->
          for {fgp, tensor} <- rolled_images(dataset, subject),
              do: {"#{subject}/#{fgp}", elem(Nx.shape(tensor), 0)}
        end,
        timeout: :infinity,
        max_concurrency: concurrency(:decode)
      )
      |> Enum.flat_map(fn {:ok, pairs} -> pairs end)
      |> Map.new()
    end)
  end

  defp gallery_height(poses, entry) do
    case poses[entry.key] do
      %{height: h} -> h
      nil -> 1
    end
  end

  # `.xyt` back to this library's convention (`Matchers.Nbis`).
  defp xyt_minutiae(xyt, height) do
    for line <- String.split(xyt, "\n", trim: true),
        [x, y, theta | _] = line |> String.split() |> Enum.map(&String.to_integer/1) do
      %{x: x * 1.0, y: height - y * 1.0, angle: wrap360(180.0 - theta)}
    end
  end

  # the synthetic data's true minutiae on both sides, at 500 ppi: the latent's own
  # (`truth_minutiae/1`), and each rolled print's from its ground truth.
  # With `:visible`, only the latent's visible minutiae (the gallery's all):
  # what a perfect extractor could find. Needs `attach_visibility/2`.
  defp truth_minutiae_sets(dataset, only \\ :all) do
    if only == :visible and Enum.all?(dataset.probes, &(&1.visibility == nil)),
      do: raise(ArgumentError, "no visibility labels attached (--labels)")

    probe =
      Enum.map(dataset.probes, fn p ->
        keep =
          if only == :visible,
            do: p.visibility || List.duplicate("hidden", length(p.minutiae)),
            else: List.duplicate("visible", length(p.minutiae))

        for {{x, y, a}, "visible"} <- Enum.zip(p.minutiae, keep), do: %{x: x, y: y, angle: a}
      end)

    gallery =
      cached(dataset, "truth-minutiae-gallery-#{signature(dataset)}", fn ->
        Enum.map(dataset.gallery, fn entry ->
          dataset.subjects[entry.subject]
          |> rolled_truth(entry.fgp)
          |> Enum.map(&Map.take(&1, [:x, :y, :angle]))
        end)
      end)

    {probe, gallery}
  end

  ## MCC scoring

  defp mcc_matrix(probe_minutiae, gallery_minutiae) do
    cylinders =
      &Task.async_stream(&1, fn ms -> Mcc.cylinders(ms) end,
        timeout: :infinity,
        max_concurrency: concurrency(:mcc)
      )

    gallery =
      gallery_minutiae
      |> cylinders.()
      |> Enum.map(fn {:ok, t} -> t end)
      |> Mcc.prepare(chunk: 128)

    probe_minutiae
    |> cylinders.()
    |> Enum.map(fn {:ok, t} -> t end)
    |> Task.async_stream(&Mcc.score_many(&1, gallery),
      timeout: :infinity,
      max_concurrency: concurrency(:mcc)
    )
    |> Enum.map(fn {:ok, row} -> row end)
    |> Nx.tensor(type: :f32)
  end

  ## F-DMD scoring

  # Every probe against the whole gallery (`Dmd.score_many/3`), as the
  # F-DMD papers evaluate it: no shortlist.
  defp dmd_matrix(probe_templates, gallery_templates, opts) do
    gallery = gallery_templates |> Enum.map(&dmd_tensors/1) |> Dmd.prepare(chunk: 128)

    probe_templates
    |> Task.async_stream(&Dmd.score_many(dmd_tensors(&1), gallery, opts),
      timeout: :infinity,
      max_concurrency: Keyword.get_lazy(opts, :concurrency, fn -> concurrency(:dmd) end)
    )
    |> Enum.map(fn {:ok, row} -> row end)
    |> Nx.tensor(type: :f32)
  end

  # `side`'s `dmd_leader` templates with each minutia weighted by LEADER's
  # confidence to the power `alpha` (nil: unweighted). The cached `leader`
  # `.xyt` lists the same minutiae in the same order (both from one
  # `Leader.extract/2`); its fourth column is the confidence in percent.
  defp confidence_weighted(dataset, side, nil), do: templates(dataset, "dmd_leader", side)

  defp confidence_weighted(dataset, side, alpha) do
    Enum.zip_with(
      templates(dataset, "dmd_leader", side),
      templates(dataset, "leader", side),
      fn
        nil, _xyt ->
          nil

        %{minutiae: {{n, 3}, _}} = template, xyt ->
          weights =
            for line <- String.split(xyt, "\n", trim: true) do
              [_x, _y, _t, q] = String.split(line)
              :math.pow(String.to_integer(q) / 100, alpha)
            end

          length(weights) == n ||
            raise "LEADER's .xyt has #{length(weights)} minutiae, the DMD++ template #{n}"

          Map.put(template, :weight, {{n}, weights |> Nx.tensor(type: :f32) |> Nx.to_binary()})
      end
    )
  end

  # What storing the descriptors smaller would cost: `"_f16"` rounds
  # them to float16, `"_bin"` keeps only their signs (768 bits a minutia).
  defp quantise_dmd(nil, _precision), do: nil
  defp quantise_dmd(template, ""), do: template

  defp quantise_dmd(%{feature: {shape, binary}} = template, precision) do
    feature = binary |> Nx.from_binary(:f32) |> Nx.reshape(shape)

    feature =
      case precision do
        "_f16" -> feature |> Nx.as_type(:f16) |> Nx.as_type(:f32)
        "_bin" -> Nx.select(Nx.greater_equal(feature, 0), 1.0, -1.0)
      end

    %{template | feature: {shape, Nx.to_binary(feature)}}
  end

  defp dmd_tensors(nil), do: %{minutiae: nil, feature: nil, mask: nil}

  defp dmd_tensors(template) do
    Map.new(template, fn {k, {shape, binary}} ->
      {k, binary |> Nx.from_binary(if(k == :ridges, do: :u8, else: :f32)) |> Nx.reshape(shape)}
    end)
  end

  ## NBIS scoring

  defp nbis_matrix(probe_xyts, gallery_xyts) do
    probe_xyts
    |> Task.async_stream(&Nbis.score_many(&1, gallery_xyts),
      timeout: :infinity,
      max_concurrency: concurrency(:nbis)
    )
    |> Enum.map(fn {:ok, row} -> row end)
    |> Nx.tensor(type: :f32)
  end

  ## Fusion

  defp fuse_minrank(a, b) do
    {ra, rb} = {ranks_of_all(a), ranks_of_all(b)}
    n = elem(Nx.shape(a), 1)
    best = Nx.min(ra, rb)
    worst = Nx.max(ra, rb)
    # Lower fused rank is better; scores are higher-is-better.
    Nx.negate(Nx.add(best, Nx.divide(worst, n + 1)))
  end

  # Rank of every column within its row (1 = best), ties pessimistic: a
  # score's rank is how many in its row are >= it.
  defp ranks_of_all(matrix) do
    matrix
    |> Nx.to_list()
    |> Enum.map(fn row ->
      ranked =
        row
        |> Enum.with_index()
        |> Enum.sort_by(fn {score, _i} -> -score end)
        |> Enum.chunk_by(fn {score, _i} -> score end)
        |> Enum.map_reduce(0, fn tied, above ->
          rank = above + length(tied)
          {Enum.map(tied, fn {_score, i} -> {i, rank} end), rank}
        end)
        |> elem(0)
        |> List.flatten()
        |> Map.new()

      for i <- 0..(length(row) - 1), do: Map.fetch!(ranked, i)
    end)
    |> Nx.tensor(type: :f32)
  end

  defp zscore(matrix) do
    mean = Nx.mean(matrix, axes: [1], keep_axes: true)
    std = matrix |> Nx.standard_deviation(axes: [1], keep_axes: true) |> Nx.max(1.0e-6)
    Nx.divide(Nx.subtract(matrix, mean), std)
  end

  ## Ranks and report

  @doc """
  The rank of each probe's mate in `matrix` (1 = first), under `:ties`
  `:against` (default) or Tsinghua's `:sort_order`: see
  `SearchMetrics.mate_ranks/3`.
  """
  @spec mate_ranks(dataset(), Nx.Tensor.t(), keyword()) :: [pos_integer()]
  def mate_ranks(dataset, matrix, opts \\ []) do
    index = dataset.gallery |> Enum.with_index() |> Map.new(fn {g, i} -> {g.key, i} end)
    mates = Enum.map(dataset.probes, &[Map.fetch!(index, "#{&1.subject}/#{&1.fgp}")])
    SearchMetrics.mate_ranks(matrix, mates, opts)
  end

  @doc """
  Hits at each of `ks` for the probes `ranks` belong to, with a 95%
  Wilson interval: `[%{k:, hits:, n:, rate:, ci: {low, high}}]`.
  """
  @spec hit_rates([pos_integer()], [pos_integer()]) :: [map()]
  defdelegate hit_rates(ranks, ks), to: SearchMetrics

  @doc """
  The expected rank-`k` hit rate if the gallery held only `size` of its
  `full` prints (the mate and `size − 1` others drawn at random), from
  each probe's rank in the full gallery (ties against the mate): a probe
  ranked `r` has `r − 1` non-mates at or above its mate, and how many of
  them a random gallery keeps is hypergeometric, so this is exact, not
  sampled (experiment E8).
  """
  @spec hit_rate_at_size([pos_integer()], pos_integer(), pos_integer(), pos_integer()) :: float()
  def hit_rate_at_size(ranks, full, size, k) do
    logf = log_factorials(full)
    others = full - 1
    drawn = size - 1

    ranks
    |> Enum.map(fn r ->
      above = r - 1
      # P(fewer than k of the `above` non-mates are drawn)
      0..min(k - 1, above)
      |> Enum.map(fn i ->
        if drawn - i > others - above or drawn - i < 0 do
          0.0
        else
          :math.exp(
            log_choose(logf, above, i) + log_choose(logf, others - above, drawn - i) -
              log_choose(logf, others, drawn)
          )
        end
      end)
      |> Enum.sum()
    end)
    |> Enum.sum()
    |> Kernel./(length(ranks))
  end

  @doc """
  Power laws fitted to the expected rank-`k` results at gallery sizes from
  about a tenth of `full` up to `full` (`hit_rate_at_size/4`, least
  squares in log–log), to project a result to larger galleries:

    * `:fnir` — FNIR(G) = a · G^b, NIST's model for latent search (b ≈
      0.08–0.14 on latent data), right when misses are rare;
    * `:hit` — TPIR(G) = a · G^−β, for weak matchers whose FNIR stays near
      1 (then the FNIR fit's b is near 0 and says little).

  `%{fnir: %{a:, b:}, hit: %{a:, beta:}, points: [{G, TPIR}]}`.
  """
  @spec scaling_fit([pos_integer()], pos_integer(), pos_integer()) :: map()
  def scaling_fit(ranks, full, k \\ 1) do
    sizes = for f <- [0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0], do: max(2, round(f * full))
    points = for g <- sizes, do: {g, hit_rate_at_size(ranks, full, g, k)}
    {a_miss, b} = log_log_fit(for {g, hit} <- points, hit < 1, do: {g, 1.0 - hit})
    {a_hit, slope} = log_log_fit(for {g, hit} <- points, hit > 0, do: {g, hit})
    %{fnir: %{a: a_miss, b: b}, hit: %{a: a_hit, beta: -slope}, points: points}
  end

  # y = a x^b by least squares on logs: {a, b}.
  defp log_log_fit(points) when length(points) < 2, do: {0.0, 0.0}

  defp log_log_fit(points) do
    logs = for {x, y} <- points, do: {:math.log(x), :math.log(y)}
    n = length(logs)
    {sx, sy} = Enum.reduce(logs, {0.0, 0.0}, fn {x, y}, {a, b} -> {a + x, b + y} end)
    {mx, my} = {sx / n, sy / n}

    {num, den} =
      Enum.reduce(logs, {0.0, 0.0}, fn {x, y}, {num, den} ->
        {num + (x - mx) * (y - my), den + (x - mx) ** 2}
      end)

    b = num / den
    {:math.exp(my - b * mx), b}
  end

  defp log_factorials(n) do
    {_, list} =
      Enum.reduce(1..n, {0.0, [0.0]}, fn i, {acc, list} ->
        {acc + :math.log(i), [acc + :math.log(i) | list]}
      end)

    list |> Enum.reverse() |> List.to_tuple()
  end

  defp log_choose(_logf, n, k) when k < 0 or k > n, do: :math.log(1.0e-300)
  defp log_choose(logf, n, k), do: elem(logf, n) - elem(logf, k) - elem(logf, n - k)

  @doc """
  How often a shortlist keeps the mate (experiment E9, a cascade's first
  stage): the fraction of probes whose mate ranks within the top
  `fraction` of the gallery by any of `ranks_lists` (one list of mate
  ranks per first-stage score, so several give their union's shortlist,
  each of that size).
  """
  @spec shortlist_hit_rate([[pos_integer()]], pos_integer(), float()) :: float()
  def shortlist_hit_rate(ranks_lists, gallery_size, fraction) do
    k = max(1, round(fraction * gallery_size))

    ranks_lists
    |> Enum.zip_with(fn ranks -> Enum.any?(ranks, &(&1 <= k)) end)
    |> Enum.count(& &1)
    |> Kernel./(length(hd(ranks_lists)))
  end

  @doc "Probes grouped by what the ground truth says about each latent, for breakdown tables."
  @spec groups(dataset()) :: [{String.t(), [non_neg_integer()]}]
  def groups(dataset) do
    indexed = Enum.with_index(dataset.probes)

    group = fn label, pred ->
      {label, for({probe, i} <- indexed, pred.(probe.truth), do: i)}
    end

    [
      group.("region: centre", &(&1.region == "centre")),
      group.("region: tip", &(&1.region == "tip")),
      group.("region: side", &(&1.region == "side")),
      group.("upright (|angle| <= 30)", &(abs(wrap(&1.angle)) <= 30)),
      group.("rotated (|angle| > 30)", &(abs(wrap(&1.angle)) > 30)),
      group.("of value", &(&1.analysis == "value")),
      group.("limited value", &(&1.analysis == "limited")),
      group.("no value", &(&1.analysis == "no_value")),
      group.("minutiae found < 6", &((&1.minutiae_found || 0) < 6)),
      group.("minutiae found 6-11", &((&1.minutiae_found || 0) in 6..11)),
      group.("minutiae found 12-19", &((&1.minutiae_found || 0) in 12..19)),
      group.("minutiae found >= 20", &((&1.minutiae_found || 0) >= 20)),
      group.("matchable (>= 3 shared minutiae)", &(&1.shared_minutiae >= 3)),
      group.("unmatchable (< 3 shared)", &(&1.shared_minutiae < 3))
    ]
    |> Enum.reject(fn {_label, members} -> members == [] end)
  end

  defp wrap(nil), do: 0.0
  defp wrap(angle), do: angle - 360.0 * Float.floor((angle + 180.0) / 360.0)

  @doc """
  McNemar's exact test of method `a` against `b` at rank `k`:
  `%{only_a:, only_b:, p:}`.
  """
  @spec compare([pos_integer()], [pos_integer()], pos_integer()) :: map()
  def compare(ranks_a, ranks_b, k) do
    pairs = Enum.zip(ranks_a, ranks_b)
    only_a = Enum.count(pairs, fn {a, b} -> a <= k and b > k end)
    only_b = Enum.count(pairs, fn {a, b} -> b <= k and a > k end)
    %{only_a: only_a, only_b: only_b, p: Metrics.mcnemar_exact(only_a, only_b)}
  end

  ## Cache

  defp cached(dataset, name, fun) do
    case read_cache(dataset, name) do
      nil ->
        value = fun.()
        write_cache(dataset, name, value)
        value

      value ->
        value
    end
  end

  defp read_cache(dataset, name) do
    path = cache_path(dataset, name)
    if File.exists?(path), do: path |> File.read!() |> :erlang.binary_to_term()
  end

  defp write_cache(dataset, name, value) do
    path = cache_path(dataset, name)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, :erlang.term_to_binary(value))
  end

  defp cache_path(dataset, name), do: Path.join([dataset.root, ".cache", name <> ".etf"])
end
