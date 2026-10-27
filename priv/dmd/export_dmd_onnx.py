"""
Exports DMD++ (Pan, Guan, Duan, Feng, Zhou, "Minutiae-Anchored Local Dense
Representation for Fingerprint Matching", arXiv 2507.15297; IJCB 2024's
DMD extended -- github.com/Yu-Yy/DMD, Apache-2.0, academic research only)
to ONNX: `DMD.get_embedding` on a batch of 128 x 128 minutia patches,
giving each a `feature` (12 x 8 x 8 = 768, channel-major) and a `mask`
(8 x 8 = 64), as `evaluate_mnt.py` stores them.

DMD++'s config (`DMD++.yaml`): ndim_feat 6, positional encoding,
tar_shape 128 x 128, input_norm True (each patch standardised to zero
mean, unit variance inside the graph). Patches are cut by
`Faulds.Matchers.Dmd` as `MntDataset` does: centred on the
minutia, turned so it points along +x, 127.5 outside the image, then
(x - 127.5) / 127.5.

The checkpoint is read with torch.load(weights_only=True): tensors only.

Run via `mix faulds.setup_flare`, not directly.

Usage: python3 export_dmd_onnx.py <dmd_src_dir> <checkpoint_path> <output_onnx_path>
"""

import sys

dmd_src_dir, checkpoint_path, output_path = sys.argv[1:4]
sys.path.insert(0, dmd_src_dir)

import numpy as np
import torch
from torch import nn

from models.model_zoo import DMD

MODEL_KWARGS = dict(ndim_feat=6, pos_embed=True, tar_shape=(128, 128), input_norm=True)
PATCH = 128


class EmbeddingOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.get_embedding(x)
        return out["feature"], out["mask"]


model = DMD(**MODEL_KWARGS)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state = checkpoint["model"] if "model" in checkpoint else checkpoint
state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

wrapped = EmbeddingOnly(model)
wrapped.eval()

torch.onnx.export(
    wrapped,
    torch.zeros(2, 1, PATCH, PATCH),
    output_path,
    input_names=["patches"],
    output_names=["feature", "mask"],
    dynamic_axes={"patches": {0: "n"}, "feature": {0: "n"}, "mask": {0: "n"}},
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {output_path}")

import onnxruntime as ort

session = ort.InferenceSession(output_path)

with torch.no_grad():
    for n in (1, 7):
        patches = torch.rand(n, 1, PATCH, PATCH) * 2 - 1
        eager_feature, eager_mask = (t.numpy() for t in wrapped(patches))
        onnx_feature, onnx_mask = session.run(None, {"patches": patches.numpy()})
        assert onnx_feature.shape == eager_feature.shape == (n, 768), onnx_feature.shape
        assert onnx_mask.shape == eager_mask.shape == (n, 64), onnx_mask.shape
        assert np.allclose(onnx_feature, eager_feature, atol=1e-4), "feature diverges from eager PyTorch!"
        assert np.allclose(onnx_mask, eager_mask, atol=1e-4), "mask diverges from eager PyTorch!"

print("ONNX Runtime output matches eager PyTorch (batches of 1 and 7): OK")
