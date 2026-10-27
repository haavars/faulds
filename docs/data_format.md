# Dataset format

The benchmark and training tools (`Faulds.LatentBenchmark`, `mix
faulds.latent_benchmark`, `mix faulds.latent_fusion`) read a dataset as
plain images and JSON files. Any dataset can be brought into this form by
a small converter; Faulds itself reads nothing else. Datasets live outside
the repository or under `data/` (gitignored), and are never committed.

## Layout

```
<root>/
  <run>/                       one folder per run (any name; optional level)
    <person>/                  one folder per person (any unique name)
      rolled/
        fgp01.png  fgp01.json  one image and one JSON per finger, 01–10
        ...
      latents/
        latent_1.png  latent_1.json
        latent_2.png  latent_2.json
        ...
```

- **The gallery** is every person's ten fingers (finger positions 1–10,
  ANSI/NIST numbering: 1 right thumb … 10 left little). A finger without
  an image is still a gallery entry, with an empty template.
- **The probes** are every `latents/latent_<n>.json` with its image.
  A probe's key is `<person>/L<n>`; its file path is matched by
  `--probes REGEX`, so the run folder can select runs.
- **Runs.** For training, each run folder is a search set of its own: its
  latents searched against its own people's rolled prints. Keep runs
  split by person, and never put a benchmark person in a training run.

## Images

PNG (or any format libvips reads), grayscale, 8 or 16 bits: dark ridges
on a light background. Images are used at their own resolution, given in
the JSON, and resampled to 500 ppi on reading (`Faulds.Resample`:
1000 → 500 ppi by NISTIR 7839's Gaussian and decimation, any other
resolution by Lanczos).

## JSON

Coordinates are in pixels of the image as stored, from the top left, `y`
down. Angles are in degrees, clockwise as seen (`y` down) from the `x`
axis, in the ANSI/INCITS 378 direction. On reading, coordinates are
scaled with the image to 500 ppi.

### Rolled print (`rolled/fgpNN.json`)

| Field | Type | |
|---|---|---|
| `ppi` | number | Resolution of the image |
| `finger` | integer | Finger position, 1–10 |
| `minutiae` | list, optional | The print's true minutiae: `id`, `x`, `y`, `angle` |

### Latent (`latents/latent_<n>.json`)

| Field | Type | |
|---|---|---|
| `ppi` | number | Resolution of the image |
| `finger` | integer | The finger the latent came from (its mate is `rolled/fgpNN` of the same person) |
| `minutiae` | list, optional | The latent's true minutiae: `id`, `x`, `y`, `angle`; optional `finger` (when the mark includes a neighbouring finger), `touch` (0 for the first touch), `visibility` (`visible`, `degraded`, `hidden`) |
| `mirrored` | boolean, optional | The image was mirrored back after capture, so its minutiae coordinates don't correspond to the rolled print's (no shared minutiae are counted) |
| `region` | string, optional | `centre`, `tip` or `side` |
| `rotation` | number, optional | The latent's rotation from upright, degrees |
| `analysis` | string, optional | `of value`, `limited value` or `no value` |
| `area_mm2` | number, optional | Area of the mark |
| `development` | string, optional | How the mark was developed |
| `minutiae_found` | integer, optional | Minutiae a reference extractor finds in it |
| `openlqm_vid` | number, optional | OpenLQM's value score |

Minutiae with the same `id` in a latent and in its rolled print are the
same physical minutia: they give the shared minutiae, the true pose and
the "findable" filter (`--min-shared N`). Visibility, where given, lets
extractors be scored against what the latent actually shows
(`mix faulds.latent_minutiae --visible`).

## Caches

Templates and score matrices are cached under each dataset root's
`.cache/` (gitignored), keyed by probe and gallery, so adding people or
methods computes only what's new. Delete it after changing an extractor.
