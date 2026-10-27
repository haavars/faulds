# Third-party notices

Faulds is non-commercial research, under the PolyForm Noncommercial
License 1.0.0 (`LICENSE`). This repository contains its own code and the
third-party code listed under "Included". Everything under "Fetched at
setup" is downloaded onto the user's machine by a setup task and is not
redistributed here; each stays under its own licence and terms.

No fingerprint data from NIST or any other dataset is in this repository,
and no trained weights from third parties. Datasets are obtained by each
user under their own terms (NIST SD302's, for example, forbid
redistribution).

## Included

| Component | Where | Licence |
|---|---|---|
| Ortex (elixir-nx/ortex), ported to `ort` 2.0.0-rc.15 | `vendor/ortex` (`VENDORED.md` lists the changes) | MIT, `vendor/ortex/LICENSE` |

## Ported or reimplemented from published code

These modules are this project's own Elixir code, written to reproduce
the cited work's results; they follow its source closely, so its notices
and terms apply.

| Module | From | Upstream licence and terms |
|---|---|---|
| `Faulds.Matchers.FlareNx`, `Faulds.Flare.Warp`, `Faulds.Minutiae.FddMap` | FLARE and FDD (Yu-Yy/FLARE, Tsinghua University) | LICENSE file Apache-2.0; the README states an "Academic Research License": academic and educational use only, commercial use strictly prohibited |
| `Faulds.Matchers.Dmd` (patches, LSA-R) | DMD / DMD++ (Yu-Yy/DMD, Tsinghua University) | Apache-2.0; the README: academic research only, not for commercial use |
| `Faulds.Minutiae.Leader` | LEADER in pyfing (raffaele-cappelli/pyfing) | MIT (code and weights) |
| `Faulds.Minutiae.FddMap`'s decoding | MSU-LatentAFIS (prip-lab/MSU-LatentAFIS) | MIT |
| `Faulds.Matchers.Mcc` | Cappelli, Ferrara, Maltoni, "Minutia Cylinder-Code" (IEEE TPAMI 2010), from the paper | — |
| `Faulds.SearchMetrics.tar_at_far/3`, `mate_ranks/3` (`:sort_order`) | DMD's `utils/get_eval_metric.py` | Apache-2.0 |

The Tsinghua models therefore stay academic and non-commercial.

## Fetched at setup, not redistributed

| Component | Licence and terms |
|---|---|
| FLARE, FLARE_ENH and DMD source and pretrained weights | As above: Apache-2.0 files, academic non-commercial use |
| pyfing (LEADER weights) | MIT |
| NBIS (`mindtct`, `bozorth3`) | NIST software, public domain in the US |
| Tsinghua N2N mate list; NIST SD302 errata | Apache-2.0 (DMD repository); NIST |

## Citations

Work using these components should cite their papers: FLARE (Pan et
al., IEEE TIFS 2026), FDD (Pan et al., arXiv 2311.18576), DMD and DMD++
(Pan et al., arXiv 2405.01199 and 2507.15297), LEADER (Cappelli and
Ferrara, arXiv 2602.15493), MCC (Cappelli et al., IEEE TPAMI 2010) and,
for results on SD302, the Nail to Nail Fingerprint Capture Challenge, as
SD302's terms require.
