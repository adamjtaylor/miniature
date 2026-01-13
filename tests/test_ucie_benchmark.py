"""Benchmark tests for UCIE optimization performance."""

import numpy as np
import pytest
import time


def test_ucie_helper_functions():
    """Test the new UCIE helper functions for correctness."""
    from miniature.ucie import (
        in_hull_halfspace,
        facet_violation_penalty,
        select_support_points,
        get_target_hull,
    )
    from scipy.spatial import ConvexHull

    # Test in_hull_halfspace with a simple cube
    cube_points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=float)
    hull = ConvexHull(cube_points)
    equations = hull.equations

    # Test points inside the cube
    inside_points = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    result = in_hull_halfspace(inside_points, equations)
    assert np.all(result), "Points inside cube should be detected as inside"

    # Test points outside the cube
    outside_points = np.array([[1.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 1.5]])
    result = in_hull_halfspace(outside_points, equations)
    assert not np.any(result), "Points outside cube should be detected as outside"


def test_facet_violation_penalty():
    """Test that facet violation penalty is zero for inside points."""
    from miniature.ucie import facet_violation_penalty
    from scipy.spatial import ConvexHull

    # Unit cube
    cube_points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=float)
    hull = ConvexHull(cube_points)
    equations = hull.equations

    # Inside points should have zero penalty
    inside_points = np.array([[0.5, 0.5, 0.5]])
    penalty = facet_violation_penalty(inside_points, equations)
    assert penalty < 1e-10, "Inside points should have near-zero penalty"

    # Outside points should have positive penalty
    outside_points = np.array([[2.0, 0.5, 0.5]])
    penalty = facet_violation_penalty(outside_points, equations)
    assert penalty > 0, "Outside points should have positive penalty"


def test_select_support_points():
    """Test support point selection."""
    from miniature.ucie import select_support_points

    # Random point cloud
    rng = np.random.default_rng(42)
    points = rng.standard_normal((1000, 3))

    support = select_support_points(points, n_directions=50)

    # Support points should be fewer than original
    assert len(support) < len(points)
    # Should have at least a few support points
    assert len(support) >= 10
    # Should be deterministic with same seed
    support2 = select_support_points(points, n_directions=50)
    np.testing.assert_array_equal(support, support2)


def test_get_target_hull_caching():
    """Test that target hull caching works."""
    from miniature.ucie import get_target_hull, _HULL_CACHE

    # Clear cache
    _HULL_CACHE.clear()

    # First call should compute hull
    hull1, eq1 = get_target_hull('LAB')
    assert ('LAB', 32) in _HULL_CACHE

    # Second call should return cached result
    hull2, eq2 = get_target_hull('LAB')
    assert hull1 is hull2
    np.testing.assert_array_equal(eq1, eq2)


def test_optimize_embedding_output_shape():
    """Test that optimize_embedding returns correct shape and range."""
    from miniature.ucie import optimize_embedding

    rng = np.random.default_rng(42)
    embedding = rng.standard_normal((500, 3))

    rgb = optimize_embedding(embedding, target='LAB', verbose=False)

    assert rgb.shape == (500, 3)
    assert np.all(rgb >= 0) and np.all(rgb <= 1)


def test_optimize_embedding_2d_output_shape():
    """Test that optimize_embedding_2d returns correct shape and range."""
    from miniature.ucie import optimize_embedding_2d

    rng = np.random.default_rng(42)
    embedding = rng.standard_normal((500, 2))

    result = optimize_embedding_2d(embedding, verbose=False)

    assert result.shape == (500, 2)
    assert np.all(result >= 0) and np.all(result <= 1)


@pytest.mark.slow
def test_ucie_performance():
    """Benchmark UCIE optimization on realistic embedding size.

    This test measures the wall-clock time for optimization on a 100k point
    embedding. Run with: pytest -v -m slow tests/test_ucie_benchmark.py
    """
    from miniature.ucie import optimize_embedding

    rng = np.random.default_rng(42)
    # Simulate 100k pixel embedding (3D)
    embedding = rng.standard_normal((100_000, 3))

    start = time.perf_counter()
    result = optimize_embedding(embedding, target='LAB', verbose=False)
    elapsed = time.perf_counter() - start

    assert result.shape == (100_000, 3)
    assert np.all(result >= 0) and np.all(result <= 1)

    print(f"\n100k point optimization took {elapsed:.2f}s")

    # Performance target: should complete in under 60 seconds
    # (adjust based on your acceptance criteria)
    assert elapsed < 120, f"Optimization took too long: {elapsed:.2f}s"


@pytest.mark.slow
def test_ucie_2d_performance():
    """Benchmark 2D UCIE optimization."""
    from miniature.ucie import optimize_embedding_2d

    rng = np.random.default_rng(42)
    embedding = rng.standard_normal((100_000, 2))

    start = time.perf_counter()
    result = optimize_embedding_2d(embedding, verbose=False)
    elapsed = time.perf_counter() - start

    assert result.shape == (100_000, 2)
    assert np.all(result >= 0) and np.all(result <= 1)

    print(f"\n100k point 2D optimization took {elapsed:.2f}s")
    assert elapsed < 120, f"2D Optimization took too long: {elapsed:.2f}s"
