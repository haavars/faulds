# Faulds

Research on latent fingerprint search: finding the rolled print a
partial, noisy mark came from, in a gallery of many. Faulds implements
published methods and measures them under published protocols:

- **minutiae** by LEADER (Cappelli and Ferrara, 2026);
- **dense minutia descriptors** (DMD++, Pan et al., 2025) matched by
  LSA-R, beside FLARE's fixed-length representation, MCC and NBIS;
- **learnt fusion** of the matchers and **search-level calibration**, so
  that one score threshold means the same in every search (open-set
  identification, NIST's FNIR at FPIR);
- a **benchmark** with paired statistics (McNemar with Holm correction,
  Wilson intervals, person-level bootstrap).

It is written in Elixir, with Nx/EXLA for the scoring kernels and ONNX
Runtime (through Ortex) for the networks, on one GPU.

Named after Henry Faulds, who proposed in 1880 that marks left at a scene
could identify who left them.

## Status

Work in progress. Results so far are on synthetic latents only; they are
not results on real casework.

## Licence

Non-commercial use only: the PolyForm Noncommercial License 1.0.0
(`LICENSE`). Third-party code, models and their terms are listed in
`NOTICE.md`; the Tsinghua models are for academic, non-commercial use.
No fingerprint data is included or redistributed.
