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
        assign_colours_oklab,
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


def test_assign_colours_oklab():
    """Test OKLab color assignment."""
    from miniature import assign_colours_oklab

    # Create fake 3D embedding
    embedding = np.random.randn(100, 3)

    rgb = assign_colours_oklab(embedding)

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


def test_resize_if_needed_no_limit():
    """Test that resize_if_needed returns array unchanged when max_pixels is None."""
    from miniature.core import resize_if_needed
    import zarr

    # Create fake multi-channel image (10 channels, 100x100)
    data = np.random.rand(10, 100, 100).astype(np.float32)
    zarray = zarr.array(data)

    result = resize_if_needed(zarray, max_pixels=None)

    assert result.shape == (10, 100, 100)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, data)


def test_resize_if_needed_within_limit():
    """Test that resize_if_needed returns array unchanged when within limit."""
    from miniature.core import resize_if_needed
    import zarr

    # Create fake multi-channel image (5 channels, 50x50 = 2500 pixels)
    data = np.random.rand(5, 50, 50).astype(np.float32)
    zarray = zarr.array(data)

    # Set limit higher than current size
    result = resize_if_needed(zarray, max_pixels=5000)

    assert result.shape == (5, 50, 50)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, data)


def test_resize_if_needed_exceeds_limit():
    """Test that resize_if_needed downsamples when exceeding max_pixels."""
    from miniature.core import resize_if_needed
    import zarr

    # Create fake multi-channel image (8 channels, 200x200 = 40000 pixels)
    data = np.random.rand(8, 200, 200).astype(np.float32)
    zarray = zarr.array(data)

    # Set limit to 10000 pixels (should scale by sqrt(10000/40000) = 0.5)
    result = resize_if_needed(zarray, max_pixels=10000)

    # Check that we have 8 channels
    assert result.shape[0] == 8
    # Check that total pixels is close to max_pixels (within 10% tolerance)
    new_pixels = result.shape[1] * result.shape[2]
    assert new_pixels <= 10000 * 1.1, f"Expected ~10000 pixels, got {new_pixels}"
    assert new_pixels >= 10000 * 0.9, f"Expected ~10000 pixels, got {new_pixels}"
    # Check data type preserved
    assert result.dtype == np.float32
    # Check aspect ratio preserved (200x200 is square, result should be approximately square)
    aspect_ratio = result.shape[1] / result.shape[2]
    assert 0.95 <= aspect_ratio <= 1.05, f"Aspect ratio not preserved: {aspect_ratio}"


def test_resize_if_needed_preserves_aspect_ratio():
    """Test that resize_if_needed preserves aspect ratio for non-square images."""
    from miniature.core import resize_if_needed
    import zarr

    # Create fake multi-channel image (4 channels, 100x200 = 20000 pixels, aspect 1:2)
    data = np.random.rand(4, 100, 200).astype(np.float32)
    zarray = zarr.array(data)

    # Set limit to 5000 pixels
    result = resize_if_needed(zarray, max_pixels=5000)

    # Check aspect ratio preserved (1:2)
    aspect_ratio = result.shape[2] / result.shape[1]  # width / height
    expected_aspect = 200 / 100  # = 2.0
    assert abs(aspect_ratio - expected_aspect) < 0.1, \
        f"Aspect ratio not preserved: got {aspect_ratio}, expected {expected_aspect}"
