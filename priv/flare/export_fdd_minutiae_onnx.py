"""
Exports the minutiae map of FLARE's pretrained FDD descriptor (Pan, Guan,
Duan, Feng, Zhou, "Fixed-Length Dense Fingerprint Representation With
Alignment and Robust Enhancement", IEEE TIFS 2026 -- github.com/Yu-Yy/FLARE)
to ONNX: the backbone up to `layer3` and the `minu_map` head, the part of
`FDD.get_embedding` that `export_fdd_onnx.py` leaves out.

The map is a DeepPrint-style heatmap, 6 orientation channels at half the
input's resolution (128 x 128 for the 256 x 256 descriptor crop). Unlike
the descriptor (whose positional encoding fixes a 16 x 16 grid) this part
is fully convolutional, so the input size is dynamic: the aligned crop, or
a whole print at FDD's scale. Height and width must be multiples of 8.

F-DMD (MALD, arXiv 2507.15297) anchors its patches on minutiae read from
this map; FLARE's repo doesn't release that decoder, so
`Faulds.Minutiae.FddMap` decodes it as MSU-LatentAFIS does.

Run via `mix faulds.setup_flare`, not directly.

Usage: python3 export_fdd_minutiae_onnx.py <flare_src_dir> <checkpoint_path> <output_onnx_path>
"""

import sys

flare_src_dir, checkpoint_path, output_path = sys.argv[1:4]
sys.path.insert(0, flare_src_dir)

import numpy as np
import torch
from torch import nn

from models.model_zoo import FDD
from utils.misc import load_model

# The same constants as export_fdd_onnx.py (model_weights/desc_configs.yaml).
DESC_MODEL_KWARGS = dict(ndim_feat=6, input_norm=False, tar_shape=(256, 256))


class MinutiaeMap(nn.Module):
    """`get_embedding`'s path to `minu_map`: layer0..layer3, then the head."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        m = self.model
        if m.input_norm:
            x = m.img_norm(x)
        x3 = m.layer3(m.layer2(m.layer1(m.layer0(x))))
        return m.minu_map(x3)


model = FDD(**DESC_MODEL_KWARGS)
load_model(model, checkpoint_path)
model.eval()

wrapped = MinutiaeMap(model)
wrapped.eval()

torch.onnx.export(
    wrapped,
    torch.zeros(1, 1, 256, 256),
    output_path,
    input_names=["image"],
    output_names=["minu_map"],
    dynamic_axes={"image": {2: "height", 3: "width"}, "minu_map": {2: "map_height", 3: "map_width"}},
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {output_path}")

# Verify against eager PyTorch, and against get_embedding's own minu_map,
# at the descriptor crop's size and at a whole print's.
import onnxruntime as ort

session = ort.InferenceSession(output_path)

with torch.no_grad():
    crop = torch.rand(1, 1, 256, 256) * 2 - 1
    reference = model.get_embedding(crop)["minu_map"].numpy()
    (onnx_crop,) = session.run(None, {"image": crop.numpy()})
    assert onnx_crop.shape == reference.shape == (1, 6, 128, 128), onnx_crop.shape
    assert np.allclose(onnx_crop, reference, atol=1e-4), "minu_map diverges from get_embedding!"

    whole = torch.rand(1, 1, 400, 376) * 2 - 1
    eager = wrapped(whole).numpy()
    (onnx_whole,) = session.run(None, {"image": whole.numpy()})
    assert onnx_whole.shape == eager.shape == (1, 6, 200, 188), onnx_whole.shape
    assert np.allclose(onnx_whole, eager, atol=1e-4), "minu_map diverges from eager PyTorch!"

print("ONNX Runtime output matches eager PyTorch (256 x 256 and 400 x 376): OK")
