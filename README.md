# Faulds

Research on latent fingerprint search: finding the rolled print a
partial, noisy mark came from, in a gallery of many. Faulds implements
published methods and measures them under published protocols, with
paired statistics, so that each change can be judged on the same probes.

Named after Henry Faulds, who proposed in 1880 that marks left at a scene
could identify who left them.

## What is here

- **Minutiae extraction**: LEADER (Cappelli and Ferrara, 2026), FDD's
  minutiae map (from FLARE), and NIST's `mindtct` as a baseline.
- **Matchers**: DMD++ dense minutia descriptors with LSA-R matching (Pan
  et al., 2025), at LEADER's or FDD's minutiae; FLARE's fixed-length
  representation, ported to Nx and ONNX (`FlareNx`); Minutia
  Cylinder-Code (Cappelli et al., 2010); NBIS `bozorth3`.
- **Fusion and calibration**: a learnt logistic fusion of the matchers,
  and search-level calibration, which fits on the candidates an open-set
  threshold actually decides, so that one threshold means the same in
  every search.
- **Evaluation**: rank-k with Wilson intervals (ties counted against the
  mate), exact McNemar tests with Holm correction, open-set FNIR at fixed
  FPIR with mate-removed searches, TAR at FAR, gallery-size scaling, and
  protocol definitions for NIST SD302 (Tsinghua's N2N list).

It is written in Elixir, with Nx/EXLA for the scoring kernels and ONNX
Runtime (through a vendored Ortex) for the networks, on one GPU.

## Status

Work in progress. Everything so far has been measured on synthetic
latents only. They are not real casework, so the results say how the
methods compare on synthetic data, not how they perform on real
latents. The evaluation code also follows published protocols for real
latent datasets, such as NIST SD302's, for anyone who has access to them.

## Requirements

- Elixir 1.18 or later and Erlang/OTP 27 or later.
- An NVIDIA GPU with CUDA 13 for EXLA and ONNX Runtime (both fall back
  to the CPU, slowly).
- Rust (to build Ortex's native library), libvips (through `image`),
  `git`, a C compiler and `make` (for NBIS).
- Python 3.11 with PyTorch, only to export the third-party models once.

## Setup

```
mix deps.get
mix faulds.setup_leader      # LEADER → priv/leader/leader.onnx
mix faulds.setup_flare       # FLARE, its enhancers and DMD++ → ONNX
mix faulds.setup_nbis        # NIST's mindtct and bozorth3 (baselines)
mix faulds.fetch_protocols   # SD302 protocol files, checked by hash
mix test
```

Each setup task fetches third-party code or weights under its own terms
(`NOTICE.md`) onto your machine; none of it is in this repository. The
tasks say what Python packages they need rather than installing them.

## Data

No fingerprint data is included. The tools read a dataset as images plus
JSON ground truth, in the layout in [`docs/data_format.md`](docs/data_format.md);
any dataset can be converted to it. Put datasets under `data/`
(gitignored) or anywhere else.

## Running

```
# Benchmark: every latent against every rolled finger, by method
mix faulds.latent_benchmark data/benchmark --methods ldmd,fdmd,mcc_leader,nbis,flare

# Fusion and search-level calibration: fit on training runs, report on
# held-out validation runs; the models go to data/calibration/
mix faulds.latent_fusion --train data/train --val-runs run-09,run-10

# How well each extractor finds the true minutiae
mix faulds.latent_minutiae data/train/run-01 --extractors nbis,fdd,leader
```

`mix help <task>` describes each task's options. Templates and score
matrices are cached under each dataset's `.cache/`.

LSA-R's loops run faster (about a fifth) when XLA keeps them on the
GPU as CUDA-graph nodes; XLA reads the flag when the process starts:

```
export XLA_FLAGS="--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
```
 No fitted fusion or
calibration models are distributed: fit them on your own training runs.

## Licence

Non-commercial use only: the [PolyForm Noncommercial License
1.0.0](LICENSE). Third-party code, models and their terms are listed in
[`NOTICE.md`](NOTICE.md); the Tsinghua FLARE/FDD and DMD/DMD++ models are
for academic, non-commercial use, and the ports here follow those terms.

Work using Faulds should cite the papers of the methods it uses
(`NOTICE.md`, "Citations").
