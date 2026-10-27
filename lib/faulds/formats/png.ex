defmodule Faulds.Formats.Png do
  @moduledoc """
  Reads a grayscale fingerprint image (PNG, or anything libvips opens)
  into a `{height, width}` tensor of floats in `[0, 1]`: 8-bit images are
  divided by 255, 16-bit ones (SD302e's latents among them) keep their 16
  bits and are divided by 65,535.
  """

  @doc "The image in `binary` as floats in `[0, 1]`. Raises if libvips can't decode it."
  @spec decode!(binary()) :: Nx.Tensor.t()
  def decode!(binary) do
    image = Image.open!(binary)

    {gray, max} =
      case Image.band_format(image) do
        {:u, 16} -> {Image.to_colorspace!(image, :grey16), 65_535.0}
        _ -> {Image.to_colorspace!(image, :bw), 255.0}
      end

    gray
    |> Image.to_nx!()
    |> Nx.reshape({Image.height(gray), Image.width(gray)})
    |> Nx.as_type(:f32)
    |> Nx.divide(max)
  end

  @doc "`decode!/1` of the file at `path`."
  @spec read!(Path.t()) :: Nx.Tensor.t()
  def read!(path), do: path |> File.read!() |> decode!()
end
