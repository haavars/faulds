defmodule Faulds.Flare.FastCartoonTexture do
  @moduledoc """
  Exact `Nx` port of FLARE's `FastCartoonTexture` preprocessing step
  (`models/units.py` in the vendored source — Pan, Guan, Duan, Feng,
  Zhou, "Fixed-Length Dense Fingerprint Representation With Alignment
  and Robust Enhancement", IEEE TIFS 2026 — github.com/Yu-Yy/FLARE), a
  cartoon-texture decomposition that separates a fingerprint's ridge
  texture from broad shading before the pose model sees it.

  Exists because `torch.onnx.export` can't trace this step (it calls
  `torch.fft.fft2`/`ifft2`, unsupported by the TorchScript-based ONNX
  exporter); see `priv/flare/export_pose_onnx.py`'s header. The exported `pose.onnx` graph has
  `preprocess_tv` swapped for the identity, so `Matchers.FlareNx` must
  apply this module's `forward/1` itself before running that graph —
  `FastCartoonTexture` has no learnable parameters, so a from-scratch
  port carries no weight-loading risk, only an arithmetic-fidelity one
  (checked by `test/faulds/flare/fast_cartoon_texture_test.exs` against
  eager-PyTorch reference values).

  Ported math (`FastCartoonTexture.forward`, unchanged):

    1. `grad_img1` = FFT-low-pass-filtered Sobel gradient norm of the
       input.
    2. `img_low` = FFT-low-pass-filtered input.
    3. `grad_img2` = FFT-low-pass-filtered Sobel gradient norm of
       `img_low`.
    4. `diff = grad_img1 - grad_img2`, normalized by `|grad_img1|` where
       that exceeds 1, else zero.
    5. `weight = clamp((diff - 0.3) / (0.7 - 0.3), 0, 1)`.
    6. `cartoon = weight * img_low + (1 - weight) * input`.
    7. `texture = clamp((input - cartoon + 20) * 255 / 40, 0, 255)`.

  The FFT low-pass filter itself (`lowpass_filtering`) is: 2D FFT →
  `fftshift` → multiply by a fixed low-pass mask `L` (a function of
  each pixel's distance from the frequency-domain center and `sigma`)
  → `fftshift` again → inverse 2D FFT → take the real part. Both
  `fftshift` calls use `size |> div(2)`-sized rolls (matching
  `torch.fft.fftshift`'s default); this project only ever calls this
  module on the pose model's fixed 512×512 canvas, always even in both
  dimensions, so the two shifts are exact inverses of each other
  (`ifftshift` and `fftshift` coincide for even sizes).

  `sigma` defaults to `2.5` — `2.5 * img_ppi / 500` in the original,
  and this library always runs at the assumed 500 ppi, so the multiplier
  is always `1.0`.
  """

  import Nx.Defn

  @sigma 2.5
  @eps 1.0e-6
  @cmin 0.3
  @cmax 0.7
  @lim 20.0

  @sobel_x [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
  @sobel_y [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]

  @doc """
  Runs the cartoon-texture decomposition on a single grayscale image.

  Takes and returns an `{height, width}` `f32` tensor with values on the
  original image's scale (matching the PyTorch module's own input/output
  convention — no `/255` normalization on either side); output values
  are clamped to `[0, 255]`.
  """
  @spec forward(Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(image) do
    image
    |> Nx.reshape({1, 1, elem(Nx.shape(image), 0), elem(Nx.shape(image), 1)})
    |> forward_n()
    |> Nx.reshape(Nx.shape(image))
  end

  defnp forward_n(input) do
    {_batch, _channels, height, width} = Nx.shape(input)
    l = lowpass_mask(height, width)

    grad_img1 = input |> gradient_norm() |> lowpass_filtering(l)

    img_low = lowpass_filtering(input, l)
    grad_img2 = img_low |> gradient_norm() |> lowpass_filtering(l)

    diff = grad_img1 - grad_img2
    flag = Nx.abs(grad_img1)
    diff = Nx.select(flag > 1, diff / Nx.max(flag, @eps), 0.0)

    weight = Nx.clip((diff - @cmin) / (@cmax - @cmin), 0.0, 1.0)

    cartoon = weight * img_low + (1 - weight) * input
    texture = Nx.clip((input - cartoon + @lim) * 255 / (2 * @lim), 0.0, 255.0)
    texture
  end

  defnp lowpass_mask(height, width) do
    grid_y = Nx.linspace(-0.5, 0.5, n: height) |> Nx.reshape({height, 1})
    grid_x = Nx.linspace(-0.5, 0.5, n: width) |> Nx.reshape({1, width})
    grid_radius = Nx.sqrt(grid_x ** 2 + grid_y ** 2) + @eps

    (1.0 / (1 + (2 * Nx.Constants.pi() * grid_radius * @sigma) ** 4))
    |> Nx.reshape({1, 1, height, width})
  end

  defnp gradient_norm(x) do
    weight_x = Nx.tensor(@sobel_x) |> Nx.reshape({1, 1, 3, 3})
    weight_y = Nx.tensor(@sobel_y) |> Nx.reshape({1, 1, 3, 3})

    grad_x = Nx.conv(x, weight_x, padding: :same)
    grad_y = Nx.conv(x, weight_y, padding: :same)

    Nx.sqrt(grad_x ** 2 + grad_y ** 2) + @eps
  end

  defnp lowpass_filtering(img, l) do
    img_fft = fftshift2(Nx.fft2(img)) * l
    img_rec = Nx.ifft2(fftshift2(img_fft))
    Nx.real(img_rec)
  end

  # `torch.fft.fftshift`'s default (all axes, shift = size |> div(2)),
  # restricted to the last two (spatial) axes -- the only ones this
  # module ever calls it on.
  defnp fftshift2(t) do
    t |> shift_last_axis(2) |> shift_last_axis(1)
  end

  defnp shift_last_axis(t, axis_from_end) do
    rank = Nx.rank(t)
    axis = rank - axis_from_end
    size = Nx.axis_size(t, axis)
    shift = div(size, 2)

    head = Nx.slice_along_axis(t, 0, shift, axis: axis)
    tail = Nx.slice_along_axis(t, shift, size - shift, axis: axis)
    Nx.concatenate([tail, head], axis: axis)
  end
end
