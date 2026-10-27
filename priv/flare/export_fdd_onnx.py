"""
Converts FLARE's pretrained FDD descriptor checkpoint (Pan, Guan, Duan,
Feng, Zhou, "Fixed-Length Dense Fingerprint Representation With Alignment
and Robust Enhancement", IEEE TIFS 2026 -- github.com/Yu-Yy/FLARE) to
ONNX, exporting `get_embedding`'s `feature`/`mask` outputs (already
flattened by the model itself -- see `models/model_zoo.py`'s `FDD.
get_embedding`).

Run via `mix faulds.setup_flare`, not directly -- that task clones
FLARE's pinned source commit and downloads the checkpoint first, then
calls this script with the paths it needs.

Usage: python3 export_fdd_onnx.py <flare_src_dir> <checkpoint_path> <output_onnx_path>
"""

import sys

flare_src_dir, checkpoint_path, output_path = sys.argv[1:4]
sys.path.insert(0, flare_src_dir)

import numpy as np
import torch
from torch import nn

from models.model_zoo import FDD
from utils.misc import load_model

# From model_weights/desc_configs.yaml (the config extract_FDD.py reads) --
# same constants FLARE's own extraction uses.
DESC_MODEL_KWARGS = dict(ndim_feat=6, input_norm=False, tar_shape=(256, 256))
INPUT_SIZE = 256


class EmbeddingOnly(nn.Module):
    """Wraps `FDD.get_embedding` (already `@torch.no_grad()`-decorated,
    already returns a plain dict of tensors) to output just the two
    tensors this project's matcher actually uses, as a tuple -- ONNX
    export handles a tuple-of-tensors output, not an arbitrary dict.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.get_embedding(x)
        return out["feature"], out["mask"]


model = FDD(**DESC_MODEL_KWARGS)
load_model(model, checkpoint_path)
model.eval()

wrapped = EmbeddingOnly(model)
wrapped.eval()

dummy_input = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)

torch.onnx.export(
    wrapped,
    dummy_input,
    output_path,
    input_names=["image"],
    output_names=["feature", "mask"],
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {output_path}")

# Verify the exported graph matches eager PyTorch numerically at the real
# inference-time input size.
import onnxruntime as ort

session = ort.InferenceSession(output_path)
real_input = torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE)

with torch.no_grad():
    eager_feature, eager_mask = wrapped(real_input)
    eager_feature = eager_feature.numpy()
    eager_mask = eager_mask.numpy()

onnx_feature, onnx_mask = session.run(None, {"image": real_input.numpy()})

assert eager_feature.shape == onnx_feature.shape == (1, 3072), (
    f"unexpected feature shape: eager={eager_feature.shape} onnx={onnx_feature.shape}"
)
assert eager_mask.shape == onnx_mask.shape == (1, 256), (
    f"unexpected mask shape: eager={eager_mask.shape} onnx={onnx_mask.shape}"
)
assert np.allclose(eager_feature, onnx_feature, atol=1e-4), "feature diverges from eager PyTorch!"
assert np.allclose(eager_mask, onnx_mask, atol=1e-4), "mask diverges from eager PyTorch!"
print("ONNX Runtime output matches eager PyTorch: OK")
