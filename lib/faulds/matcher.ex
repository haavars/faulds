defmodule Faulds.Matcher do
  @moduledoc """
  Contract for a fingerprint matcher, treated as a black box by
  `Faulds.Harness`.

  A matcher supports either or both of two modes:

    * 1:1 mode — implement `compare/2` directly (two images in, one score
      out). Simplest to implement; the harness re-extracts nothing.
    * Template mode — implement `extract/1` and `score/2`. The harness
      extracts each distinct image once and caches the template, which is
      how a real 1:N system actually operates (enroll once, compare many
      times against a gallery of stored templates).

  A module implementing template mode does not need to also implement
  `compare/2` — the harness falls back to `extract/1` + `score/2` when
  `compare/2` isn't exported.
  """

  @type image :: Nx.Tensor.t()
  @type template :: term()
  @type score :: float()

  @callback compare(image, image) :: {:ok, score} | {:error, term()}
  @callback extract(image) :: {:ok, template} | {:error, term()}
  @callback score(template, template) :: {:ok, score} | {:error, term()}

  @optional_callbacks compare: 2, extract: 1, score: 2
end
