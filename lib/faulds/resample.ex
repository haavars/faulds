defmodule Faulds.Resample do
  @moduledoc """
  Brings a grayscale image tensor to 500 ppi, the resolution FlareNx,
  LEADER, DMD++, `mindtct` and `bozorth3` all assume. Latents are often
  captured at 1000 ppi, and fed to FlareNx unscaled they drop its rank-1
  rate to chance.

  Exactly 1000 to 500 ppi uses NISTIR 7839's method (NIST SP 500-289
  §4.4): a separable Gaussian low-pass, sigma 0.8475 and radius 4, then
  every other pixel. Any other resolution goes through libvips' `resize`
  (Lanczos 3, which antialiases when shrinking). Edges are extended, not zero-padded, so
  the white background stays white.
  """

  alias Vix.Vips.Operation

  @target_ppi 500.0

  # Resolutions this close to 500 ppi (e.g. SLC 2's 197 px/cm = 500.4
  # ppi) are left alone: resampling by 0.2% would only blur.
  @tolerance 0.02

  @sigma 0.8475
  @radius 4

  @doc "The resolution every matcher in this project assumes."
  @spec target_ppi() :: float()
  def target_ppi, do: @target_ppi

  @doc """
  `tensor` (`{height, width}`, floats in `[0, 1]`) scanned at `ppi`,
  resampled to 500 ppi. Returned unchanged when already within 2% of it.
  """
  @spec to_500_ppi(Nx.Tensor.t(), number()) :: Nx.Tensor.t()
  def to_500_ppi(tensor, ppi) when is_number(ppi) and ppi > 0 do
    scale = @target_ppi / ppi

    cond do
      unchanged?(ppi) -> tensor
      abs(scale - 0.5) <= 0.5 * @tolerance -> halve(tensor)
      true -> resize(tensor, scale)
    end
  end

  @doc """
  The resolution `to_500_ppi/2` leaves an image scanned at `ppi` in:
  `ppi` itself when it was close enough to be left alone, else 500. What
  minutiae given in physical units must be placed at.
  """
  @spec output_ppi(number()) :: float()
  def output_ppi(ppi) when is_number(ppi) and ppi > 0 do
    if unchanged?(ppi), do: ppi * 1.0, else: @target_ppi
  end

  defp unchanged?(ppi), do: abs(@target_ppi / ppi - 1.0) <= @tolerance

  defp halve(tensor) do
    with_vips(tensor, fn image ->
      {:ok, mask} = Vix.Vips.Image.new_matrix_from_array(length(kernel()), 1, [kernel()])
      {:ok, smooth} = Operation.convsep(image, mask, precision: :VIPS_PRECISION_FLOAT)
      {:ok, small} = Operation.subsample(smooth, 2, 2)
      small
    end)
  end

  defp resize(tensor, scale) do
    with_vips(tensor, fn image ->
      {:ok, resized} = Operation.resize(image, scale)
      resized
    end)
  end

  defp kernel do
    taps = for i <- -@radius..@radius, do: :math.exp(-i * i / (2 * @sigma * @sigma))
    total = Enum.sum(taps)
    Enum.map(taps, &(&1 / total))
  end

  defp with_vips(tensor, fun) do
    {height, width} = Nx.shape(tensor)

    {:ok, image} =
      tensor
      |> Nx.as_type(:f32)
      |> Nx.reshape({height, width, 1})
      |> Image.from_nx()

    out = fun.(image)

    out
    |> Image.to_nx!()
    |> Nx.reshape({Image.height(out), Image.width(out)})
    |> Nx.as_type(:f32)
    |> Nx.clip(0.0, 1.0)
  end
end
