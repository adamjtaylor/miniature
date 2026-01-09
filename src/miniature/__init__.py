"""
Miniature - Visual thumbnails from high-dimensional imaging.

Create visual representations of multiplexed tissue imaging data using
dimensionality reduction and perceptually meaningful color mappings.
"""

__version__ = "2.0.0"

from .core import (
    pull_pyramid,
    remove_background,
    keep_background,
    run_pca,
    run_umap,
    run_tsne,
    assign_colours_lab,
    assign_colours_rgb,
    assign_colours_2d,
    make_rgb_image,
)

__all__ = [
    "__version__",
    "pull_pyramid",
    "remove_background",
    "keep_background",
    "run_pca",
    "run_umap",
    "run_tsne",
    "assign_colours_lab",
    "assign_colours_rgb",
    "assign_colours_2d",
    "make_rgb_image",
]
