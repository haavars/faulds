defmodule Mix.Tasks.Faulds.SetupNbis do
  @shortdoc "Builds NIST's MINDTCT + BOZORTH3 from source into priv/nbis/"

  @moduledoc """
  Clones and builds NIST Biometric Image Software (NBIS) — `mindtct`
  (minutiae extraction) and `bozorth3` (minutiae matching), the public
  baseline, plus `dwsq` (WSQ decompression, for datasets whose tenprints
  are WSQ-compressed), installing the binaries into `priv/nbis/bin/`
  (gitignored).

      $ mix faulds.setup_nbis

  No prebuilt binary or package exists for NBIS (checked Homebrew,
  conda-forge, and NIST's own site), so this builds from source: clones a mirror of NIST's public-domain source
  (pinned to a fixed commit — NBIS has no releases/tags; this is frozen,
  decades-old software, not under active development) and runs its
  `setup.sh`/`make` build. `--without-X11 --STDLIBS` skips the GUI tools
  and the JASPER/OpenJP2/PNG codec libraries entirely — this project only
  ever feeds `mindtct` pre-decoded pixels via JPEG (still built, since
  `--STDLIBS` doesn't drop the bundled `ijg` JPEG library), never the
  other NBIS tools that would need them.

  Requires `git`, `gcc`/`clang`, and `make` on `PATH` — all standard
  Xcode Command Line Tools on macOS or `build-essential` on Debian/Ubuntu,
  nothing extra to install beyond that. No `sudo`, nothing written
  outside this repo and a temp build directory.

  Run once per machine; not part of `mix precommit`. Only the NBIS and
  MCC-on-`mindtct` baselines need it.
  """

  use Mix.Task

  @repo_url "https://github.com/lessandro/nbis.git"
  @pinned_commit "3d3b05f0144b706bed56407957bc00779baf2fa5"

  @target_dir "priv/nbis"

  @impl Mix.Task
  def run(_args) do
    for tool <- ~w(git make) do
      System.find_executable(tool) || Mix.raise("`#{tool}` not found on PATH")
    end

    work_dir = Path.join(Mix.Project.build_path(), "nbis_setup")
    File.rm_rf!(work_dir)
    File.mkdir_p!(work_dir)

    src_dir = Path.join(work_dir, "src")
    install_dir = Path.join(work_dir, "install")
    File.mkdir_p!(install_dir)

    clone!(src_dir)
    build!(src_dir, install_dir)
    install_binaries!(install_dir)
  end

  defp clone!(src_dir) do
    Mix.shell().info("Cloning NBIS (#{@repo_url} @ #{@pinned_commit})...")
    {_, 0} = System.cmd("git", ["clone", "-q", @repo_url, src_dir])
    {_, 0} = System.cmd("git", ["checkout", "-q", @pinned_commit], cd: src_dir)
  end

  defp build!(src_dir, install_dir) do
    Mix.shell().info("Building mindtct + bozorth3 + dwsq (this takes a minute)...")

    {_, 0} =
      System.cmd("sh", ["setup.sh", install_dir, "--without-X11", "--STDLIBS"], cd: src_dir)

    patch_cflags_for_modern_gcc!(src_dir)

    {_, 0} = System.cmd("make", ["config"], cd: src_dir)
    {_, 0} = System.cmd("make", ["it"], cd: src_dir)
    {_, 0} = System.cmd("make", ["install", "LIBNBIS=no"], cd: src_dir)
  end

  # This 20+ year old codebase relies on tentative-definition merging for
  # a handful of globals declared (not `extern`ed) in multiple .c files —
  # legal under C89/the historical GCC default, but GCC >= 10 changed its
  # default to `-fno-common`, turning that into a real
  # `multiple definition of ...` link error. Not hit on macOS (Apple
  # clang's default differs), but real and reproducible on any current
  # Linux distro. `rules.mak` is regenerated fresh by `setup.sh` above on
  # every run, so this patches it after the fact rather than the repo's
  # (frozen, upstream) `rules.mak.src` template.
  defp patch_cflags_for_modern_gcc!(src_dir) do
    rules_mak = Path.join(src_dir, "rules.mak")

    patched =
      rules_mak
      |> File.read!()
      |> String.replace(~r/^(CFLAGS\s*:=.*)$/m, "\\1\nCFLAGS += -fcommon")

    File.write!(rules_mak, patched)
  end

  defp install_binaries!(install_dir) do
    File.rm_rf!(@target_dir)
    File.mkdir_p!(Path.join(@target_dir, "bin"))

    for binary <- ~w(mindtct bozorth3 dwsq) do
      src = Path.join([install_dir, "bin", binary])
      dest = Path.join([@target_dir, "bin", binary])
      File.cp!(src, dest)
      File.chmod!(dest, 0o755)
    end

    Mix.shell().info("Installed mindtct + bozorth3 + dwsq to #{@target_dir}/bin/")
  end
end
