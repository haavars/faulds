"""
Converts FLARE's pretrained VotingPose (GRIDNET4) checkpoint (Pan, Guan,
Duan, Feng, Zhou, "Fixed-Length Dense Fingerprint Representation With
Alignment and Robust Enhancement", IEEE TIFS 2026 --
github.com/Yu-Yy/FLARE) to ONNX, exporting the `pose_2d` output only
(the 512x512-canvas-space pose estimate `Matchers.FlareNx` needs).

Two op-compatibility fixes are required, both applied only in this
export script (never to the vendored FLARE source):

1. `GRIDNET4.forward` runs `FastCartoonTexture` (a cartoon-texture
   decomposition preprocessing step) internally, which uses
   `torch.fft.fft2`/`ifft2` -- unsupported by `torch.onnx.export`
   (`UnsupportedOperatorError: ... 'aten::fft_fft2' ... not supported`).
   Resolved by reimplementing `FastCartoonTexture` in Nx separately
   (`Faulds.Flare.FastCartoonTexture`) and passing
   `with_tv=False` here so the model's own `self.preprocess_tv` is
   `nn.Identity()` -- the exported graph then starts from an
   already-preprocessed input, and the Nx port covers the FFT step.
   `FastCartoonTexture` has no learnable parameters, so nothing is lost.
2. `dense_hough_voting4` (called from `GRIDNET4.forward`) uses
   `torch.rad2deg`, also unsupported by export. Fixed with a one-line,
   mathematically-identical monkeypatch applied only in this process.

Run via `mix faulds.setup_flare`, not directly -- that task clones
FLARE's pinned source commit and downloads the checkpoint first, then
calls this script with the paths it needs.

Usage: python3 export_pose_onnx.py <flare_src_dir> <checkpoint_path> <output_onnx_path>
"""

import math
import sys

flare_src_dir, checkpoint_path, output_path = sys.argv[1:4]
sys.path.insert(0, flare_src_dir)

import numpy as np
import torch
from torch import nn

# Fix 2: torch.onnx.export's TorchScript-based tracer doesn't support
# aten::rad2deg. rad2deg(x) == x * (180 / pi) exactly (both just scale by
# a constant) -- this is a substitution for tracing, not an approximation.
torch.rad2deg = lambda x: x * (180.0 / math.pi)

from models.model_zoo import GRIDNET4  # noqa: E402
from utils.misc import load_model  # noqa: E402

# Exact constructor args FLARE's extract_VotingPose.py uses --
# with_tv=True here so the checkpoint's state_dict (which includes
# preprocess_tv.img_grad's fixed, non-learnable Sobel kernel buffers)
# loads without a key mismatch. self.preprocess_tv is swapped for
# nn.Identity() *after* loading (Fix 1 above), not by constructing with
# with_tv=False -- FastCartoonTexture has no learnable parameters, so
# nothing is lost by loading its buffers and then discarding the module.
POSE_MODEL_KWARGS = dict(
    num_pose_2d=(33, 33, 1),
    num_layers=(64, 128, 256, 512),
    img_ppi=500,
    middle_shape=np.array([512, 512]),
    activate="sigmoid",
    bin_type="invprop",
    with_tv=True,
)
INPUT_SIZE = 512


class Pose2DOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["pose_2d"]


model = GRIDNET4(**POSE_MODEL_KWARGS)
load_model(model, checkpoint_path)
model.preprocess_tv = nn.Identity()
model.eval()

wrapped = Pose2DOnly(model)
wrapped.eval()

dummy_input = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)

with torch.no_grad():
    sanity_out = wrapped(dummy_input)
    assert sanity_out.shape == (1, 3), f"unexpected pose_2d shape: {sanity_out.shape}"

torch.onnx.export(
    wrapped,
    dummy_input,
    output_path,
    input_names=["image"],
    output_names=["pose_2d"],
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {output_path}")

# Verify the exported graph matches eager PyTorch numerically at the real
# inference-time input size (with_tv=False on both sides, so this checks
# graph-export fidelity only -- FastCartoonTexture equivalence is
# verified separately in the Nx unit test against a with_tv=True eager
# reference).
import onnxruntime as ort

session = ort.InferenceSession(output_path)
real_input = torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE)

with torch.no_grad():
    eager_out = wrapped(real_input).numpy()

onnx_out = session.run(None, {"image": real_input.numpy()})[0]

assert onnx_out.shape == (1, 3), f"unexpected output shape: {onnx_out.shape}"
assert np.allclose(eager_out, onnx_out, atol=1e-3), "ONNX output diverges from eager PyTorch!"
print("ONNX Runtime output matches eager PyTorch: OK")
