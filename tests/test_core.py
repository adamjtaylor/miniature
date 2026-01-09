"""Basic tests for miniature package."""

import numpy as np
import pytest


def test_import():
    """Test that the package can be imported."""
    import miniature
    assert hasattr(miniature, "__version__")
    assert miniature.__version__ == "2.0.0"


def test_core_functions_exist():
    """Test that core functions are exported."""
    from miniature import (
        pull_pyramid,
        remove_background,
        run_pca,
        run_umap,
        assign_colours_lab,
        assign_colours_rgb,
        make_rgb_image,
    )


def test_assign_colours_lab():
    """Test LAB color assignment."""
    from miniature import assign_colours_lab

    # Create fake 3D embedding
    embedding = np.random.randn(100, 3)

    rgb = assign_colours_lab(embedding)

    assert rgb.shape == (100, 3)
    assert rgb.min() >= 0
    assert rgb.max() <= 1


def test_assign_colours_rgb():
    """Test RGB color assignment."""
    from miniature import assign_colours_rgb

    # Create fake 3D embedding
    embedding = np.random.randn(100, 3)

    rgb = assign_colours_rgb(embedding)

    assert rgb.shape == (100, 3)
    assert rgb.min() >= 0
    assert rgb.max() <= 1


def test_make_rgb_image():
    """Test RGB image creation."""
    from miniature import make_rgb_image

    # Create fake RGB values and mask
    rgb = np.random.rand(50, 3)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 2:7] = True  # 5x5 = 25 pixels, but we have 50 RGB values

    # Adjust mask to have exactly 50 True values
    mask = np.zeros((10, 10), dtype=bool)
    mask.flat[:50] = True

    image = make_rgb_image(rgb, mask)

    assert image.shape == (10, 10, 3)
    assert image.dtype == np.uint8


def test_run_pca():
    """Test PCA reduction."""
    from miniature import run_pca

    # Create fake pixel data
    data = np.random.randn(100, 10)

    embedding = run_pca(data, n=3)

    assert embedding.shape == (100, 3)


def test_run_umap_default():
    """Test UMAP reduction with default parameters."""
    from miniature import run_umap

    # Create fake pixel data
    data = np.random.randn(100, 10)

    embedding = run_umap(data, n=3, metric='euclidean')

    assert embedding.shape == (100, 3)


def test_run_umap_with_kwargs():
    """Test UMAP reduction with custom parameters."""
    from miniature import run_umap

    # Create fake pixel data
    data = np.random.randn(100, 10)

    # Test with custom parameters
    embedding = run_umap(
        data,
        n=2,
        metric='euclidean',
        n_neighbors=10,
        min_dist=0.0,
        random_state=42
    )

    assert embedding.shape == (100, 2)

    # Test reproducibility with random_state
    embedding2 = run_umap(
        data,
        n=2,
        metric='euclidean',
        n_neighbors=10,
        min_dist=0.0,
        random_state=42
    )

    np.testing.assert_array_almost_equal(embedding, embedding2, decimal=5)
