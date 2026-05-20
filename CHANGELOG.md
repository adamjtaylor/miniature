# Changelog

All notable changes to Miniature are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `docs/equivalence.md` — internal equivalence policy for refactors and
  perf work (metric-based thresholds, frozen reference image set, seed
  policy, pinned dependency set).
- `docs/perf/baseline.md` — captured baseline numbers for the pipeline
  and the `delta_E` scaling on the pinned dependency set.
- `tests/test_pipeline_benchmark.py` — whole-pipeline wall-time + peak-RSS
  benchmark for the LAB / 3D path on the reference image set.
- `tests/test_delta_e_scaling.py` — N=200/1000/3000 sweep over the
  O(N²) perceptual-distance loop in `trustworthiness` and `metrics`.
- `CITATION.cff` for GitHub "Cite this repository" support.
- `CONTRIBUTING.md` documenting the equivalence policy and the
  "benchmark before merge" rule.
- AI assistance disclosure section in `README.md`.

### Changed

- `pyproject.toml` registers the `benchmark` pytest marker so the new
  benchmarks can be excluded from fast CI with `-m "not benchmark"`.

## [2.0.0] — pip-installable package, vectorised core

### Added

- Restructured as a `pip install miniature`-able package (`src/miniature/`).
- OKLab colour space support for 3D embeddings (better perceptual uniformity
  than CIELAB).
- UCIE-style rotation optimisation for LAB/RGB colour assignment.
- 2D bivariate colormaps: BREMM, CUBEDIAGONAL, SCHUMANN, STEIGER, TEULING2,
  ZIEGLER.
- Background removal via Otsu thresholding on log-summed channels.
- HDF5 sidecar output (`--save_data`) with mask, tissue array, embedding,
  and colormap arrays.
- Nextflow batch pipeline (`nextflow/main.nf`).
- Trustworthiness and perceptual-trustworthiness metrics via
  `miniature-metrics`.
- GitHub Actions test workflow and automated example-image regeneration.
- Half-space convex-hull check (replaces Delaunay) for ~5–20× speedup on
  UCIE optimisation.

### Changed

- Colour conversions vectorised via `colour-science` (removed per-pixel
  Python loops).
- Removed per-pixel multiprocessing overhead.
- Pyramid level selection respects `--max_pixels` budget.

## [1.0] — original implementation (2021)

- Initial implementation as an R function (`paint_miniature.R`).
- Parallel Python implementation added shortly after.
- Dockerised entry point.

[Unreleased]: https://github.com/adamjtaylor/miniature/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/adamjtaylor/miniature/releases/tag/v2.0.0
