defmodule Mix.Tasks.Faulds.SetupLeader do
  @shortdoc "Exports LEADER (pyfing's minutiae extractor) to ONNX for Minutiae.Leader"

  @moduledoc """
  Exports LEADER (pyfing, MIT) to `priv/leader/leader.onnx`
  with `priv/leader/export_leader_onnx.py`, which checks the
  export against the Keras model.

  Needs a Python with `pyfing` 0.7.3, Keras 3, PyTorch (CPU is enough),
  `onnx`, `onnxscript`, `onnxruntime`, SciPy and OpenCV: the venv
  `priv/leader/.venv` (gitignored). This task checks for it
  and says how to make it rather than installing a toolchain unprompted:

      uv venv -p 3.11 priv/leader/.venv
      VIRTUAL_ENV=priv/leader/.venv uv pip install torch \\
        --index-url https://download.pytorch.org/whl/cpu
      VIRTUAL_ENV=priv/leader/.venv uv pip install pyfing==0.7.3 \\
        keras numpy scipy opencv-python-headless onnx onnxscript onnxruntime
  """

  use Mix.Task

  @dir "priv/leader"
  @packages ~w(pyfing keras torch onnx onnxscript onnxruntime scipy cv2)

  @impl Mix.Task
  def run(_args) do
    python = Path.expand(Path.join(@dir, ".venv/bin/python"))
    output = Path.join(@dir, "leader.onnx")

    File.exists?(python) || Mix.raise("No #{@dir}/.venv: see `mix help faulds.setup_leader`")

    {missing, _} =
      System.cmd(python, [
        "-c",
        "import importlib.util as u; print(' '.join(p for p in #{inspect(@packages)} if u.find_spec(p) is None))"
      ])

    if String.trim(missing) != "",
      do: Mix.raise("Missing in #{@dir}/.venv: #{String.trim(missing)}")

    if File.exists?(output) do
      Mix.shell().info("#{output} already exists, skipping export")
    else
      case System.cmd(
             python,
             ["-I", Path.join(@dir, "export_leader_onnx.py"), Path.expand(output)],
             stderr_to_stdout: true
           ) do
        {out, 0} ->
          Mix.shell().info(out |> String.split("\n") |> Enum.take(-3) |> Enum.join("\n"))

        {out, status} ->
          Mix.raise("export_leader_onnx.py failed (exit #{status}):\n\n#{out}")
      end
    end
  end
end
