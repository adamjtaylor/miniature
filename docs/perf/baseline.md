# Performance baseline

A captured snapshot of where time goes in the pipeline today. Use this as
the reference every Phase 1+ perf PR compares against. Refresh whenever the
dependency pin set in [`docs/equivalence.md`](../equivalence.md) changes.

## How to capture

```bash
# 1. Benchmarks (timing + RSS for the whole pipeline and the delta_E loop).
uv run pytest tests/test_pipeline_benchmark.py tests/test_delta_e_scaling.py \
    -v -m benchmark -s | tee docs/perf/_raw.txt

# 2. cProfile of a representative run for the function-level breakdown.
uv run python -m cProfile -o docs/perf/baseline.prof -m miniature.cli \
    data/WD-76845-003_ROI01.ome.tif /tmp/baseline.png \
    --colormap LAB --save_data

# 3. Inspect interactively (optional).
uv pip install snakeviz
uv run snakeviz docs/perf/baseline.prof
```

The raw `.prof` and `_raw.txt` files are gitignored — only the headline
numbers below live in version control.

## Hardware

| Component | Detail |
|-----------|--------|
| CPU | Apple M2 Max |
| RAM | 32 GB |
| OS | macOS 15.7.4 |
| Python | 3.11.13 (in `.venv`, project-local) |

## Pinned dependency versions

Captured from `uv pip list` on the run below.

```
numpy=1.26.4
scipy=1.17.1
scikit-learn=1.8.0
scikit-image=0.26.0
umap-learn=0.5.12
colour-science=0.4.6
tifffile=2026.3.3
h5py=3.16.0
matplotlib=3.10.9
pillow=12.2.0
```

## Whole-pipeline timing (LAB, 3D, `max_pixels=512*512`, seed=42)

From `tests/test_pipeline_benchmark.py`. All times in seconds; RSS in MB.

| Image | n_pixels | load | bg removal | UMAP | color | output | total | peak RSS |
|-------|----------|-----:|-----------:|-----:|------:|-------:|------:|---------:|
| `WD-76845-003_ROI01.ome.tif` | 27,073 | 0.016 | 0.019 | **10.704** | 0.005 | 0.001 | **10.746** | 772 |
| `exemplar-001_small.tif` | 654,687 | 0.021 | 0.096 | **859.968** | 0.100 | 0.025 | **860.210** | **5,580** |

Observations:

- **UMAP dominates** at 99.6% (small) / 99.97% (medium) of wall time.
  Everything else combined is rounding error. Algorithmic perf wins in
  pure-Python paths will be invisible against this; the big lever is the
  UMAP call itself — `n_jobs`, `low_memory`, or an alternative
  implementation.
- **Memory grows ~14×** between the two images (772 MB → 5.58 GB), and
  the medium image's `n_pixels` is only ~24× larger. The
  `np.array(zarray)` double-load in `core.py:132,140` is implicated;
  fixing it should ~halve the memory footprint of the load + background
  stages.
- The non-UMAP stages on the medium image take only **0.24s combined**,
  so Phase 1 wins outside UMAP register on memory, not on time.

## delta_E scaling

From `tests/test_delta_e_scaling.py`. Confirms O(N²) cost; per-call
Python + `colour.delta_E` dispatch dominates, not the math.

| N | `delta_e_distance_matrix` | `delta_e_pdist` |
|---|-------------------------:|----------------:|
| 200 | 1.156 s | 1.172 s |
| 1,000 | 28.738 s | 28.646 s |
| 3,000 | 256.429 s | 257.407 s |

Extrapolated to N=10k ≈ 47 min. Targeted by Phase 2.

## cProfile top time-consumers

> TODO: capture and paste once a fresh cProfile run is done. Expected
> top: `umap.umap_._optimize_layout_*`, NN descent, the spectral
> initialisation.

```
_pending_
```

## Notes

- The `exemplar-001_small.tif` file has no pyramid — `pull_pyramid`
  returns the full 1024×1024×12 plane, which is why `n_pixels` ends up
  at 654k after background removal (vs. the ~262k cap implied by
  `max_pixels=512*512`). The selector's fallback-to-finest behaviour
  may be worth a docstring note.
- `umap-learn` 0.5.x already defaults `low_memory=True` for large N, so
  there is no win to be had from setting it explicitly. Spot-checked on
  the medium image: 47–59s and ~3.4 GB regardless of explicit override,
  matching the auto behaviour.
- The `WD-76845-003_ROI01.ome.tif` pyramid selector lands on a small
  level (225×250 = 56k px, of which 27k are tissue) — the README example
  image. UMAP on 27k points is ~10s; on 655k points it's ~860s.
- Both images, single thread for UMAP. Investigating `n_jobs` is a
  cheap experiment in Phase 1.

## Phase tracking

When a Phase 1/2 PR lands, append a row for the same inputs on the new
code; the previous numbers stay in place as historical reference.

| Date | Commit | Image | Total | UMAP | Peak RSS | Notes |
|------|--------|-------|------:|-----:|---------:|-------|
| 2026-05-20 | (pre-Phase 1) | `WD-76845-003_ROI01.ome.tif` | 10.7 s | 10.7 s | 772 MB | baseline |
| 2026-05-20 | (pre-Phase 1) | `exemplar-001_small.tif` | 860.2 s | 860.0 s | 5,580 MB | baseline |
