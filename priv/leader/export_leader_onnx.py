"""
Exports LEADER (Cappelli & Ferrara, "LEADER: Lightweight End-to-end
Attention-gated Dual Encoder-decodeR for minutiae extraction", arXiv
2602.15493; pyfing, MIT -- github.com/raffaele-cappelli/pyfing) to ONNX.

pyfing builds the Keras 3 model and loads its own LEADER.weights.h5; it
runs here on Keras's PyTorch backend so torch.onnx's exporter can trace
it (the TorchScript exporter can't, `torch.export` can). The graph takes
raw 0..255 pixels, (1, H, W, 1) at 500 ppi with H and W multiples of 32,
and gives (1, H, W, 4): position, direction (radians, atan2), type, and
the position after its built-in non-maximum suppression, which
`Faulds.Minutiae.Leader` thresholds. Weights are embedded
in the one .onnx file.

Run via `mix faulds.setup_leader`, not directly.

Usage: python3 export_leader_onnx.py <output_onnx_path>
"""

import os
import sys

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from pyfing.minutiae_end_to_end import Leader

torch.set_grad_enabled(False)
output_path = sys.argv[1]

model = Leader().model
model.trainable = False


class Wrapped(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x, training=False)


wrapped = Wrapped(model).eval()
side = lambda name: 32 * torch.export.Dim(name, min=2, max=128)

torch.onnx.export(
    wrapped,
    (torch.rand(1, 384, 352, 1) * 255,),
    output_path,
    input_names=["image"],
    output_names=["out"],
    dynamic_shapes=({1: side("h32"), 2: side("w32")},),
    dynamo=True,
)

# One self-contained file (the exporter writes the weights beside it).
graph = onnx.load(output_path, load_external_data=True)
onnx.save(graph, output_path, save_as_external_data=False)
if os.path.exists(output_path + ".data"):
    os.remove(output_path + ".data")
print(f"Exported to {output_path}")

session = ort.InferenceSession(output_path)
for shape in [(1, 384, 352, 1), (1, 512, 480, 1)]:
    x = (np.random.rand(*shape) * 255).astype(np.float32)
    eager = wrapped(torch.tensor(x)).numpy()
    (exported,) = session.run(None, {"image": x})
    assert exported.shape == eager.shape, exported.shape
    for channel in (0, 2, 3):
        assert np.allclose(exported[..., channel], eager[..., channel], atol=1e-4), f"channel {channel} diverges"
    direction = np.abs(np.angle(np.exp(1j * (exported[..., 1] - eager[..., 1]))))
    assert direction.max() < 1e-2, "direction diverges"

print("ONNX Runtime output matches the Keras model (384 x 352 and 512 x 480): OK")
