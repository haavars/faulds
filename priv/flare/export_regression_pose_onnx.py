"""
Converts FLARE's pretrained RegressionPose (FingerPose_2D_Single)
checkpoint (Pan, Guan, Duan, Feng, Zhou, "Fixed-Length Dense Fingerprint
Representation With Alignment and Robust Enhancement", IEEE TIFS 2026 --
github.com/Yu-Yy/FLARE) to ONNX, with `extract_RegressionPose.py`'s
decoding of the class scores folded into the graph, so the output is a
`pose_2d = (x, y, theta_deg)` in the 512x512 canvas's coordinates: the
same shape and frame as `pose.onnx` (VotingPose), so `Matchers.FlareNx`
runs both through the same canvas and inverse warp.

Unlike VotingPose there's no FastCartoonTexture step (the network
normalises its own input), so the graph takes the raw canvas: 0-255,
white border, as `FingerPoseEvalDataset.process_img` builds it.

One op-compatibility fix, applied only in this process: `torch.rad2deg`
(used by `classify2vector_rot`) isn't supported by export, and is
replaced by the identical `x * 180 / pi`, as in `export_pose_onnx.py`.

The checkpoint is read with `torch.load(weights_only=True)` (tensors
only, no pickled code).

Run via `mix faulds.setup_flare`, not directly.

Usage: python3 export_regression_pose_onnx.py <flare_src_dir> <checkpoint_path> <output_onnx_path>
"""

import math
import sys

flare_src_dir, checkpoint_path, output_path = sys.argv[1:4]
sys.path.insert(0, flare_src_dir)

import numpy as np
import torch
from torch import nn

torch.rad2deg = lambda x: x * (180.0 / math.pi)

from models.model_zoo import FingerPose_2D_Single  # noqa: E402
from utils.trans_est import classify2vector_rot, classify2vector_trans  # noqa: E402

# Exact constructor args extract_RegressionPose.py uses.
TRANS_CLASSES = 512
ROT_CLASSES = 180
INPUT_SIZE = 512


class RegressionPose2D(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred_xy, pred_theta = self.model(x)
        vec_xy = classify2vector_trans(pred_xy, out_form="claSum", trans_num_classes=TRANS_CLASSES)
        vec_xy = vec_xy + INPUT_SIZE / 2
        vec_theta = classify2vector_rot(pred_theta, out_form="claSum", rot_num_classes=ROT_CLASSES)
        return torch.cat([vec_xy, vec_theta[:, 2:3]], dim=1)


model = FingerPose_2D_Single(
    inp_mode="fp",
    trans_out_form="claSum",
    trans_num_classes=TRANS_CLASSES,
    rot_out_form="claSum",
    rot_num_classes=ROT_CLASSES,
)
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state.get("model", state))
model.eval()

wrapped = RegressionPose2D(model)
wrapped.eval()

dummy_input = torch.full((1, 1, INPUT_SIZE, INPUT_SIZE), 255.0)

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

import onnxruntime as ort

session = ort.InferenceSession(output_path)
real_input = torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE) * 255.0

with torch.no_grad():
    eager_out = wrapped(real_input).numpy()

onnx_out = session.run(None, {"image": real_input.numpy()})[0]

assert onnx_out.shape == (1, 3), f"unexpected output shape: {onnx_out.shape}"
assert np.allclose(eager_out, onnx_out, atol=1e-2), "ONNX output diverges from eager PyTorch!"
print("ONNX Runtime output matches eager PyTorch: OK")
