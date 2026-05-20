"""End-to-end pipeline benchmark.

Measures wall-clock time for the full pipeline: load → background removal →
dimensionality reduction → color assignment → output. This is the baseline
that every Phase 1+ perf PR must compare against.

Marked as ``benchmark`` so it can be excluded from fast CI with
``pytest -m "not benchmark"``. Run explicitly with::

    pytest tests/test_pipeline_benchmark.py -v -m benchmark -s
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REFERENCE_IMAGES = {
    "small_mif": DATA_DIR / "WD-76845-003_ROI01.ome.tif",
    "medium_mif": DATA_DIR / "exemplar-001_small.tif",
}

SEED = 42


def _peak_rss_mb() -> float:
    """Peak resident set size in MB. Linux returns KB, macOS returns bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


@pytest.mark.benchmark
@pytest.mark.parametrize("image_key", list(REFERENCE_IMAGES.keys()))
def test_pipeline_lab_3d(image_key: str) -> None:
    """Whole pipeline timing: load → background → UMAP(3D) → LAB → PNG.

    Records wall time and peak RSS. Asserts an upper bound that should be
    comfortably met on current hardware; tighten as Phase 1 wins land.
    """
    path = REFERENCE_IMAGES[image_key]
    if not path.exists():
        pytest.skip(f"reference image missing: {path}")

    from miniature.core import (
        pull_pyramid,
        remove_background,
        run_umap,
        assign_colours_lab,
        make_rgb_image,
    )

    np.random.seed(SEED)

    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()

    zarray = pull_pyramid(str(path), max_pixels=512 * 512)
    t_load = time.perf_counter()

    tissue_array, mask = remove_background(zarray, pseudocount=1.0)
    t_bg = time.perf_counter()

    embedding = run_umap(tissue_array, n=3, metric="euclidean")
    t_umap = time.perf_counter()

    rgb = assign_colours_lab(embedding)
    t_color = time.perf_counter()

    _ = make_rgb_image(rgb, mask)
    t_total = time.perf_counter()

    rss_after = _peak_rss_mb()

    timings = {
        "load": t_load - t0,
        "background": t_bg - t_load,
        "umap": t_umap - t_bg,
        "color": t_color - t_umap,
        "output": t_total - t_color,
        "total": t_total - t0,
        "rss_peak_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "n_pixels": tissue_array.shape[0],
    }

    print(f"\n--- pipeline benchmark: {image_key} ---")
    for k, v in timings.items():
        if isinstance(v, float):
            print(f"  {k:<14} {v:>10.3f}")
        else:
            print(f"  {k:<14} {v:>10}")

    # Loose ceiling — fail only on catastrophic regression. Current baseline
    # is ~11s on small_mif and ~860s on medium_mif (UMAP-dominated). The
    # baseline numbers live in docs/perf/baseline.md.
    assert timings["total"] < 1800.0, (
        f"pipeline total time {timings['total']:.1f}s exceeds 30-minute ceiling"
    )


@pytest.mark.benchmark
def test_pipeline_lab_3d_smoke_correctness() -> None:
    """Cheap end-to-end correctness check used alongside the timer above.

    Confirms the pipeline produces non-empty tissue, an embedding of the
    expected shape, and an RGB image with values in [0, 1]. This is the
    sanity check that catches a "fast but wrong" regression before any
    equivalence comparison runs.
    """
    path = REFERENCE_IMAGES["small_mif"]
    if not path.exists():
        pytest.skip(f"reference image missing: {path}")

    from miniature.core import (
        pull_pyramid,
        remove_background,
        run_umap,
        assign_colours_lab,
        make_rgb_image,
    )

    np.random.seed(SEED)

    zarray = pull_pyramid(str(path), max_pixels=128 * 128)
    tissue_array, mask = remove_background(zarray, pseudocount=1.0)

    assert tissue_array.shape[0] > 0, "no tissue pixels selected"
    assert tissue_array.shape[1] == zarray.shape[0], "channel count mismatch"
    assert mask.sum() == tissue_array.shape[0], "mask/tissue count mismatch"

    embedding = run_umap(tissue_array, n=3, metric="euclidean")
    assert embedding.shape == (tissue_array.shape[0], 3)

    rgb = assign_colours_lab(embedding)
    assert rgb.shape == (tissue_array.shape[0], 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    image = make_rgb_image(rgb, mask)
    assert image is not None
