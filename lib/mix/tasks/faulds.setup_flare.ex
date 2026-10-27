defmodule Mix.Tasks.Faulds.SetupFlare do
  @shortdoc "Clones FLARE's source and downloads its pretrained pose, descriptor and enhancer checkpoints"

  @moduledoc """
  Clones FLARE (Pan, Guan, Duan, Feng, Zhou, "Fixed-Length Dense
  Fingerprint Representation With Alignment and Robust Enhancement",
  IEEE TIFS 2026 — github.com/Yu-Yy/FLARE, Academic Research License,
  non-commercial) at a pinned commit into `priv/flare/src/`,
  and its enhancers (github.com/Yu-Yy/FLARE_ENH, same terms) into
  `priv/flare/enh_src/`, and downloads the pretrained
  checkpoints into `priv/flare/model_weights/`: the FDD
  descriptor and VotingPose, which `extract/1` uses, and RegressionPose
  and UNetEnh, the rest of FLARE's full configuration
  (`FlareNx.extract_variants/2`'s `:poses` and `:enhancers`).

      $ mix faulds.setup_flare

  It then converts the pretrained checkpoints to ONNX for
  `Matchers.FlareNx`, the native Nx/Ortex port: `export_pose_onnx.py` (with two
  op-compatibility fixes documented in that script's own header) writes
  `priv/flare/pose.onnx`, `export_fdd_onnx.py` writes
  `fdd.onnx`, `export_fdd_minutiae_onnx.py` writes `fdd_minutiae.onnx`
  (FDD's minutiae head, `Minutiae.FddMap`), `export_regression_pose_onnx.py` writes
  `regression_pose.onnx` and `export_unetenh_onnx.py` writes
  `unetenh.onnx` (all gitignored). It also clones DMD (pinned, into
  `priv/dmd/src/`), downloads the DMD++ checkpoint and
  exports it with `priv/dmd/export_dmd_onnx.py` to
  `priv/dmd/dmd_plus.onnx` (`Matchers.Dmd`, F-DMD).

  Requires `git` on PATH, plus a Python with FLARE's own dependencies
  importable (`torch>=1.10`, `numpy`, `opencv-python`, `scipy`,
  `Pillow`, `PyYAML`, `easydict`, `pandas`, `tqdm`, `gdown`, `onnx`,
  `onnxruntime`, and FLARE_ENH's `kornia` and `einops`): the venv
  `priv/flare/.venv` if it exists, else `python3` on `PATH`. The task
  reports what's missing rather than installing a multi-hundred-MB
  toolchain unprompted.

  Run once per machine; not part of `mix precommit`.
  """

  use Mix.Task

  @repo_url "https://github.com/Yu-Yy/FLARE.git"
  @pinned_commit "7d13ca727d55cf43642f9fe1e67df785091fd7c2"

  @enh_repo_url "https://github.com/Yu-Yy/FLARE_ENH.git"
  @enh_pinned_commit "ee735b03669aa0d9e086b50c7d9b1771913a2ba4"

  @fdd_file_id "1zvAI57L0TDC7q6kQgNh5_DwSbicjJ4hs"
  @voting_pose_file_id "1Zg4duNJ8mg-fkTACTzpPPK7DgNb9NRvA"
  @regression_pose_file_id "1AXpN8GBSqhlIXDilqPLfZf9n4Dc0pEpj"
  @unetenh_file_id "1U0uP8XoxWc90IPlEGnt2KIe0ATAA2VKl"

  # DMD (F-DMD's patch descriptor, `Matchers.Dmd`): the DMD++ weights'
  # Google Drive link in its README (the Tsinghua cloud one may not answer).
  @dmd_repo_url "https://github.com/Yu-Yy/DMD.git"
  @dmd_pinned_commit "feb0ad40e5f340caebb457237d7ef21d5d5724e6"
  @dmd_plus_file_id "1QFCPZ7aTHCQeIYxdz1t9l92xI9YH7qhe"
  @dmd_dir "priv/dmd"

  @required_python_packages ~w(gdown torch cv2 numpy scipy PIL yaml easydict pandas tqdm onnx onnxruntime kornia einops)

  @target_dir "priv/flare"

  @impl Mix.Task
  def run(_args) do
    System.find_executable("git") || Mix.raise("`git` not found on PATH")

    python =
      Faulds.Matchers.Flare.python() ||
        Mix.raise("`python3` not found (no #{@target_dir}/.venv, nothing on PATH)")

    check_python_packages!(python)

    src_dir = Path.join(@target_dir, "src")
    enh_src_dir = Path.join(@target_dir, "enh_src")
    weights_dir = Path.join(@target_dir, "model_weights")
    File.mkdir_p!(weights_dir)

    clone!(@repo_url, @pinned_commit, src_dir)
    clone!(@enh_repo_url, @enh_pinned_commit, enh_src_dir)
    fdd_checkpoint = Path.join(weights_dir, "desc_model.pth.tar")
    pose_checkpoint = Path.join(weights_dir, "VotingPose.pth")
    regression_pose_checkpoint = Path.join(weights_dir, "RegressionPose.pth")
    unetenh_checkpoint = Path.join(weights_dir, "unetenh.pth")
    download_checkpoint!(python, @fdd_file_id, fdd_checkpoint)
    download_checkpoint!(python, @voting_pose_file_id, pose_checkpoint)
    download_checkpoint!(python, @regression_pose_file_id, regression_pose_checkpoint)
    download_checkpoint!(python, @unetenh_file_id, unetenh_checkpoint)

    export_onnx!(
      python,
      "export_pose_onnx.py",
      src_dir,
      pose_checkpoint,
      Path.join(@target_dir, "pose.onnx")
    )

    export_onnx!(
      python,
      "export_fdd_onnx.py",
      src_dir,
      fdd_checkpoint,
      Path.join(@target_dir, "fdd.onnx")
    )

    export_onnx!(
      python,
      "export_fdd_minutiae_onnx.py",
      src_dir,
      fdd_checkpoint,
      Path.join(@target_dir, "fdd_minutiae.onnx")
    )

    dmd_src_dir = Path.join(@dmd_dir, "src")
    dmd_checkpoint = Path.join([@dmd_dir, "model_weights", "dmd_plus.pth.tar"])
    File.mkdir_p!(Path.dirname(dmd_checkpoint))
    clone!(@dmd_repo_url, @dmd_pinned_commit, dmd_src_dir)
    download_checkpoint!(python, @dmd_plus_file_id, dmd_checkpoint)

    export_onnx!(
      python,
      "export_dmd_onnx.py",
      dmd_src_dir,
      dmd_checkpoint,
      Path.join(@dmd_dir, "dmd_plus.onnx"),
      @dmd_dir
    )

    export_onnx!(
      python,
      "export_regression_pose_onnx.py",
      src_dir,
      regression_pose_checkpoint,
      Path.join(@target_dir, "regression_pose.onnx")
    )

    export_onnx!(
      python,
      "export_unetenh_onnx.py",
      enh_src_dir,
      unetenh_checkpoint,
      Path.join(@target_dir, "unetenh.onnx")
    )

    Mix.shell().info(
      "FLARE set up at #{@target_dir}/ (src/, enh_src/, model_weights/, pose.onnx, fdd.onnx, " <>
        "fdd_minutiae.onnx, regression_pose.onnx, unetenh.onnx) and DMD++ at #{@dmd_dir}/dmd_plus.onnx"
    )
  end

  defp check_python_packages!(python) do
    missing =
      Enum.filter(@required_python_packages, fn package ->
        match?({_, code} when code != 0, System.cmd(python, ["-c", "import #{package}"]))
      end)

    if missing != [] do
      pip_names =
        Enum.map(missing, fn
          "cv2" -> "opencv-python"
          "PIL" -> "Pillow"
          "yaml" -> "PyYAML"
          other -> other
        end)

      Mix.raise(
        "Missing Python packages: #{Enum.join(missing, ", ")}. Install with:\n\n" <>
          "    pip3 install #{Enum.join(pip_names, " ")}\n\n" <>
          "(torch as a CPU wheel is enough: " <>
          "pip3 install torch --index-url https://download.pytorch.org/whl/cpu)"
      )
    end
  end

  defp clone!(repo_url, commit, dir) do
    if File.dir?(dir) do
      Mix.shell().info("#{dir} already exists, skipping clone")
    else
      Mix.shell().info("Cloning #{repo_url} @ #{commit}...")
      {_, 0} = System.cmd("git", ["clone", "-q", repo_url, dir])
      {_, 0} = System.cmd("git", ["checkout", "-q", commit], cd: dir)
    end
  end

  defp download_checkpoint!(python, file_id, dest_path) do
    if File.exists?(dest_path) do
      Mix.shell().info("#{dest_path} already exists, skipping download")
    else
      Mix.shell().info("Downloading #{Path.basename(dest_path)} (via gdown)...")

      {_, 0} =
        System.cmd(python, [
          "-m",
          "gdown",
          "https://drive.google.com/uc?id=#{file_id}",
          "-O",
          dest_path
        ])
    end
  end

  defp export_onnx!(
         python,
         script,
         src_dir,
         checkpoint_path,
         output_path,
         script_dir \\ @target_dir
       ) do
    if File.exists?(output_path) do
      Mix.shell().info("#{output_path} already exists, skipping export")
    else
      Mix.shell().info("Exporting #{Path.basename(output_path)} (via #{script})...")
      script_path = Path.join(script_dir, script)

      case System.cmd(python, [script_path, src_dir, checkpoint_path, output_path],
             stderr_to_stdout: true
           ) do
        {output, 0} ->
          Mix.shell().info(output)

        {output, status} ->
          Mix.raise("#{script} failed (exit #{status}):\n\n#{output}")
      end
    end
  end
end
