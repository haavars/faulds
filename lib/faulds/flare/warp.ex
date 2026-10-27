defmodule Faulds.Flare.Warp do
  @moduledoc """
  Exact port of FLARE's affine canvas-warping arithmetic
  (`datasets/FPdataset.py`'s `affine_matrix`/`process_img`, in the
  vendored source — Pan, Guan, Duan, Feng, Zhou, "Fixed-Length Dense
  Fingerprint Representation With Alignment and Robust Enhancement",
  IEEE TIFS 2026 — github.com/Yu-Yy/FLARE), backed by
  `Vix.Vips.Operation.affine/3` in place of `cv2.warpAffine`.

  Used twice by `Matchers.FlareNx`:
  pad/center the raw image onto the pose model's 512×512 input canvas
  (`FingerPoseEvalDataset.process_img`), then, once a pose is estimated,
  crop/align onto the descriptor model's 256×256 input canvas
  (`Descdataset.process_img`).

  ## `matrix`/`odx`/`ody` convention, verified empirically

  `cv2.warpAffine(img, T[:2], ...)` (no `WARP_INVERSE_MAP` flag) treats
  `T` as the *forward* map from source-image coordinates to
  destination-canvas coordinates: `dst = T[:2,:2] @ src + T[:2,2]`, and
  samples the source image via the (internally-computed) inverse.
  `Vix.Vips.Operation.affine/3`'s `matrix`/`odx`/`ody` use that exact
  same forward convention — confirmed by hand against both `cv2` (a
  known-good real pipeline output, matched to floating-point precision
  for scale+rotate+translate compositions) and exact-integer disambiguation
  probes (ruling out a transposed or pre-inverted matrix). No matrix
  inversion is needed on this side; `T`'s own 2×2 block and translation
  feed `matrix`/`odx,ody` directly, with `idx`/`idy` left at `0.0`
  (FLARE's own `T` never uses an input-side shift).

  Border handling matches `cv2.warpAffine`'s `borderMode=BORDER_CONSTANT`
  (the pose stage passes `borderValue=255`; the descriptor stage passes
  no `borderMode` at all, which is `cv2`'s own default — `borderValue=0`)
  via `extend: :VIPS_EXTEND_BACKGROUND` + a matching `background`.
  """

  alias Vix.Vips.{Image, Interpolate, Operation}

  @type affine :: %{a: float(), b: float(), c: float(), d: float(), tx: float(), ty: float()}

  @doc """
  Ports `affine_matrix(scale, theta, trans, trans_2)` — composes a
  scaled rotation with a translate-then-translate (`trans` applied
  first, in the *rotated* frame, then `trans_2`), matching the Python
  `R @ trans + trans_2` composition exactly. `theta` is in radians
  (callers pass `:math.pi() * degrees / 180`, mirroring `np.deg2rad`
  call sites in the ported Python).
  """
  @spec affine_matrix(float(), float(), {float(), float()}, {float(), float()}) :: affine()
  def affine_matrix(scale, theta, {trans_x, trans_y}, {trans2_x, trans2_y}) do
    cos_t = :math.cos(theta)
    sin_t = :math.sin(theta)

    a = cos_t * scale
    b = -sin_t * scale
    c = sin_t * scale
    d = cos_t * scale

    tx = a * trans_x + b * trans_y + trans2_x
    ty = c * trans_x + d * trans_y + trans2_y

    %{a: a, b: b, c: c, d: d, tx: tx, ty: ty}
  end

  @doc """
  Inverts an affine map — ports `np.linalg.inv(T)` applied to FLARE's
  3×3 homogeneous `T` (FLARE's pose estimation step, undoing the
  pose-stage canvas warp so a predicted pose lands back in the
  *original* image's coordinate space). Standard 2×2-block affine
  inversion: `A' = A⁻¹`, `t' = -A⁻¹ @ t`.
  """
  @spec invert(affine()) :: affine()
  def invert(%{a: a, b: b, c: c, d: d, tx: tx, ty: ty}) do
    det = a * d - b * c
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    %{
      a: inv_a,
      b: inv_b,
      c: inv_c,
      d: inv_d,
      tx: -(inv_a * tx + inv_b * ty),
      ty: -(inv_c * tx + inv_d * ty)
    }
  end

  @doc "Applies an affine map to a single `{x, y}` point."
  @spec apply_point(affine(), {float(), float()}) :: {float(), float()}
  def apply_point(%{a: a, b: b, c: c, d: d, tx: tx, ty: ty}, {x, y}) do
    {a * x + b * y + tx, c * x + d * y + ty}
  end

  @doc """
  Warps a single-channel `{height, width}` `f32` image tensor onto a
  `{target_height, target_width}` canvas via `t` (from `affine_matrix/4`),
  bilinear-interpolated, with `background` (default `0.0`) filling
  pixels sampled from outside the source image — matching
  `cv2.warpAffine(..., flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
  borderValue=background)`.
  """
  @spec warp(Nx.Tensor.t(), affine(), {pos_integer(), pos_integer()}, keyword()) :: Nx.Tensor.t()
  def warp(image, t, {target_height, target_width} = size, opts \\ []) do
    image
    |> warp_many([t], size, opts)
    |> Nx.reshape({target_height, target_width})
  end

  @doc """
  `warp/4` for many maps over one image, stacked `{n, target_height,
  target_width}`: the image goes to libvips once, not once per map (it
  may live on the GPU), and each warp is the same libvips call as
  `warp/4`'s, so the results are identical. Options: `:background`;
  `:linear`, `{a, b}`: the image is first mapped to `a · x + b` by libvips
  (the same float32 values as Nx gives, with no device op to compile for
  each image size); `:backend` for the result (default the default one).
  """
  @spec warp_many(Nx.Tensor.t(), [affine()], {pos_integer(), pos_integer()}, keyword()) ::
          Nx.Tensor.t()
  def warp_many(image, transforms, {target_height, target_width}, opts \\ []) do
    background = Keyword.get(opts, :background, 0.0)
    {src_height, src_width} = Nx.shape(image)

    binary = image |> Nx.as_type(:f32) |> Nx.to_binary()
    {:ok, vimage} = Image.new_from_binary(binary, src_width, src_height, 1, :VIPS_FORMAT_FLOAT)

    vimage =
      case Keyword.get(opts, :linear) do
        nil ->
          vimage

        {a, b} ->
          {:ok, mapped} = Operation.linear(vimage, [a * 1.0], [b * 1.0])
          mapped
      end

    interpolate = Interpolate.new!("bilinear")

    transforms
    |> Enum.map(fn %{a: a, b: b, c: c, d: d, tx: tx, ty: ty} ->
      {:ok, warped} =
        Operation.affine(vimage, [a, b, c, d],
          odx: tx,
          ody: ty,
          idx: 0.0,
          idy: 0.0,
          interpolate: interpolate,
          extend: :VIPS_EXTEND_BACKGROUND,
          background: [background * 1.0],
          oarea: [0, 0, target_width, target_height]
        )

      {:ok, out_binary} = Image.write_to_binary(warped)
      out_binary
    end)
    |> IO.iodata_to_binary()
    |> Nx.from_binary(:f32, backend: Keyword.get(opts, :backend, Nx.default_backend()))
    |> Nx.reshape({length(transforms), target_height, target_width})
  end
end
