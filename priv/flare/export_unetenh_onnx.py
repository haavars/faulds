"""
Converts FLARE's pretrained UNetEnh enhancer (SqueezeUNet, from
github.com/Yu-Yy/FLARE_ENH; Pan, Guan, Duan, Feng, Zhou, IEEE TIFS 2026)
to ONNX, for `Matchers.FlareNx`'s enhanced descriptor variants.

The graph is `deploy_unetenh.py`'s model without the optional CLAHE
pre-enhancement (`pre_enh=False`, the script's default): input the image
normalised to [-1, 1] (`(pixel - 127.5) / 127.5`), height and width any
multiple of 16; output channel 0 of the sigmoid head, the enhanced image
in [0, 1] (`deploy_unetenh.py` saves it as `enh * 255`). The second
channel isn't used by FLARE and isn't exported. Height and width are
dynamic axes (the network is fully convolutional).

The checkpoint is read with `torch.load(weights_only=True)` (tensors
only; its numpy scalar metadata is allowlisted by name).

Run via `mix faulds.setup_flare`, not directly.

Usage: python3 export_unetenh_onnx.py <flare_enh_src_dir> <checkpoint_path> <output_onnx_path>
"""

import sys

enh_src_dir, checkpoint_path, output_path = sys.argv[1:4]
sys.path.insert(0, enh_src_dir)

import numpy as np
import torch
from torch import nn

from model.network import SqueezeUNet  # noqa: E402


class EnhOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, 0:1]


def load_state(path):
    from numpy._core.multiarray import scalar

    safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype] + [
        type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")
    ]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model", checkpoint)


model = SqueezeUNet(input_channels=1, num_classes=2, pre_enh=False)
model.load_state_dict(load_state(checkpoint_path))
model.eval()

wrapped = EnhOnly(model)
wrapped.eval()

dummy_input = torch.zeros(1, 1, 512, 512)

torch.onnx.export(
    wrapped,
    dummy_input,
    output_path,
    input_names=["image"],
    output_names=["enh"],
    dynamic_axes={"image": {2: "height", 3: "width"}, "enh": {2: "height", 3: "width"}},
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {output_path}")

import onnxruntime as ort

session = ort.InferenceSession(output_path)

for height, width in [(512, 512), (752, 800)]:
    real_input = torch.rand(1, 1, height, width) * 2 - 1

    with torch.no_grad():
        eager_out = wrapped(real_input).numpy()

    onnx_out = session.run(None, {"image": real_input.numpy()})[0]

    assert onnx_out.shape == (1, 1, height, width), f"unexpected output shape: {onnx_out.shape}"
    assert np.allclose(eager_out, onnx_out, atol=1e-3), "ONNX output diverges from eager PyTorch!"

print("ONNX Runtime output matches eager PyTorch (512x512, 752x800): OK")
