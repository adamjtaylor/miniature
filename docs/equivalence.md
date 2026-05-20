# Equivalence policy

Miniature is an original tool, not a rewrite of an upstream — so there is no
external "ground truth" to validate against. Instead, this document defines
what **internal equivalence** means: when we refactor, optimize, or port hot
paths, the output must not change beyond declared tolerances against a
**frozen reference set** of inputs and seeds.

This policy is what every perf PR validates against. It is the answer to "did
this speedup break anything?"

## Scope

In scope:

- All output PNG images (the primary product).
- The HDF5 sidecar written with `--save_data` — `mask`, `tissue_array`,
  `embedding`, and `colors/{colormap}`.
- All numerical metrics reported by `miniature-metrics`.

Out of scope:

- Per-iteration log output (timestamps, progress bars).
- Embedding plot PNGs written with `--plot_embedding` (visual sanity only).
- File modification times and HDF5 internal layout.

## Equivalence target

Stochastic methods (UMAP, t-SNE, optimization restarts) make byte-identical
output unattainable across machines or library versions. Equivalence is
therefore **metric-based** on a seeded run:

| Output | Metric | Threshold |
|--------|--------|-----------|
| Output PNG | SSIM vs. reference PNG | ≥ 0.99 |
| Output PNG | Mean ΔE2000 over tissue pixels | ≤ 1.0 |
| `embedding` (HDF5) | Trustworthiness (k=15) | ≥ baseline − 0.005 |
| `colors/*` (HDF5) | Perceptual trustworthiness (k=15) | ≥ baseline − 0.005 |
| Mask coverage | (`mask.sum()` − baseline) / baseline | within ±1% |

"Baseline" refers to the values recorded against the pinned dependency set
below, on the reference image set, with the seeds below.

A PR that fails any threshold must either:

1. Justify the change (and update the baseline numbers in this file in the
   same PR), or
2. Be rejected.

## Reference image set

These three files live in `data/` and ship with the repo:

| File | Size | Channels | Purpose |
|------|------|----------|---------|
| `WD-76845-003_ROI01.ome.tif` | 3.4 MB | 25 | Primary mIF reference (used in README examples) |
| `exemplar-001_small.tif` | 12 MB | multi | Medium-size benchmark target |
| `CMU-1-Small-Region.svs` | 1.9 MB | RGB | H&E / svs format coverage |

The primary reference for equivalence is `WD-76845-003_ROI01.ome.tif`. The
other two exist to catch format-handling regressions.

## Seeds and pinned versions

Stochastic methods must be seeded for equivalence runs. Current pins:

```
numpy=1.26.x
scipy=1.13.x
scikit-learn=1.5.x
scikit-image=0.24.x
umap-learn=0.5.x
colour-science=0.4.x
python=3.11
```

Seed: `42` everywhere a seed is accepted (`numpy.random.seed(42)`,
`UMAP(random_state=42)`, `TSNE(random_state=42)`, multistart Neldermead with
fixed seed in `ucie.py`).

When the pin set changes, the baseline must be re-captured in the same PR.

## Baseline numbers

> TODO: capture on first run after this policy lands. Update this table
> whenever pins change.

| Metric | Image | Baseline value | Captured on | Commit |
|--------|-------|----------------|-------------|--------|
| Trustworthiness (k=15) | `WD-76845-003_ROI01.ome.tif` | _pending_ | _pending_ | _pending_ |
| Perceptual trustworthiness LAB | `WD-76845-003_ROI01.ome.tif` | _pending_ | _pending_ | _pending_ |
| Perceptual trustworthiness OKLAB | `WD-76845-003_ROI01.ome.tif` | _pending_ | _pending_ | _pending_ |
| Wall time (load → LAB output) | `WD-76845-003_ROI01.ome.tif` | _pending_ | _pending_ | _pending_ |
| Peak RSS | `WD-76845-003_ROI01.ome.tif` | _pending_ | _pending_ | _pending_ |

## How to validate a PR

```bash
# 1. Generate reference outputs on main (once, store under /tmp/baseline/).
git checkout main
mkdir -p /tmp/baseline
miniature data/WD-76845-003_ROI01.ome.tif /tmp/baseline/out.png \
    --colormap ALL --save_data --plot_embedding

# 2. Generate candidate outputs on the perf branch.
git checkout <perf-branch>
mkdir -p /tmp/candidate
miniature data/WD-76845-003_ROI01.ome.tif /tmp/candidate/out.png \
    --colormap ALL --save_data --plot_embedding

# 3. Compare: SSIM, mean ΔE, metric deltas.
pytest tests/test_pipeline_benchmark.py -v -m benchmark
```

A small comparison helper script can be added later — for now, the
`miniature-metrics` numbers + visual diff of the PNGs is sufficient.

## When to revise this policy

- Adding a new colormap or output file → add it to the scope table.
- Pin update → re-capture baseline numbers and bump the pins block.
- New input modality (e.g. 3D / z-stack imaging) → add a reference file
  and define what equivalence means for it.
