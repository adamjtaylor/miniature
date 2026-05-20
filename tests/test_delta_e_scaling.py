"""Scaling benchmark for the O(N²) delta_E loop.

This characterises the bottleneck that Phase 2 (Numba JIT / vectorised
colour-science) targets in ``trustworthiness.delta_e_distance_matrix`` and
``metrics.delta_e_pdist``. Captures wall time at N=200, 1000, 3000 so the
scaling shape (~quadratic) can be confirmed before any rewrite.

Marked as ``benchmark`` so it can be excluded from fast CI with
``pytest -m "not benchmark"``.

Larger N values (10k, 50k) are intentionally omitted from the default sweep
because the current pure-Python loop takes minutes at that size — run them
manually if you need a deeper curve.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

SEED = 42


def _random_rgb(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.random((n, 3))


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [200, 1000, 3000])
def test_delta_e_distance_matrix_scaling(n: int) -> None:
    """Time ``delta_e_distance_matrix`` at increasing N.

    Asserts a generous ceiling so this fails on catastrophic regression but
    is not flaky on slow CI runners. The interesting number is the printed
    timing, recorded in ``docs/perf/baseline.md``.
    """
    from miniature.trustworthiness import delta_e_distance_matrix

    rgb = _random_rgb(n)

    t0 = time.perf_counter()
    distances = delta_e_distance_matrix(rgb)
    elapsed = time.perf_counter() - t0

    assert distances.shape == (n, n)
    assert np.all(distances >= 0.0)
    assert np.allclose(distances, distances.T)

    print(f"\ndelta_e_distance_matrix  N={n:>5}  t={elapsed:>8.3f}s")

    # Loose ceiling: 5 minutes for N=3000. Tighten after Phase 2.
    assert elapsed < 300.0, (
        f"delta_e_distance_matrix(N={n}) took {elapsed:.1f}s — likely a regression"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [200, 1000, 3000])
def test_delta_e_pdist_scaling(n: int) -> None:
    """Time ``delta_e_pdist`` at increasing N. Same loop, condensed output."""
    from miniature.metrics import delta_e_pdist

    rgb = _random_rgb(n)

    t0 = time.perf_counter()
    distances = delta_e_pdist(rgb)
    elapsed = time.perf_counter() - t0

    expected_len = n * (n - 1) // 2
    assert distances.shape == (expected_len,)
    assert np.all(distances >= 0.0)

    print(f"\ndelta_e_pdist            N={n:>5}  t={elapsed:>8.3f}s")

    assert elapsed < 300.0, (
        f"delta_e_pdist(N={n}) took {elapsed:.1f}s — likely a regression"
    )


@pytest.mark.benchmark
def test_delta_e_consistency() -> None:
    """The two implementations should agree on identical inputs.

    Cheap correctness gate: any Phase 2 JIT replacement must preserve this
    relationship.
    """
    from miniature.metrics import delta_e_pdist
    from miniature.trustworthiness import delta_e_distance_matrix

    rgb = _random_rgb(50)
    matrix = delta_e_distance_matrix(rgb)
    condensed = delta_e_pdist(rgb)

    # Upper-triangular flattening of `matrix` should equal `condensed`.
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    np.testing.assert_allclose(upper, condensed, rtol=1e-10, atol=1e-10)
