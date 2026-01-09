"""
Perceptual Trustworthiness Module

Measures how well a color mapping preserves local structure using
perceptual color difference (CIEDE2000).
"""

import numpy as np
import colour
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def delta_e_cie2000_pairwise(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """
    Calculate CIEDE2000 color difference between two RGB colors.

    Args:
        rgb1: First RGB color (3,)
        rgb2: Second RGB color (3,)

    Returns:
        CIEDE2000 delta E value
    """
    # Convert RGB to LAB
    lab1 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb1))
    lab2 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb2))

    # Calculate delta E 2000
    return colour.delta_E(lab1, lab2, method='CIE 2000')


def delta_e_distance_matrix(rgb_array: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise CIEDE2000 distances for an array of RGB colors.

    Args:
        rgb_array: (N, 3) array of RGB values

    Returns:
        (N, N) distance matrix
    """
    n = len(rgb_array)

    # Convert all RGB to LAB
    lab_array = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb_array))

    # Calculate pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = colour.delta_E(lab_array[i], lab_array[j], method='CIE 2000')
            distances[i, j] = d
            distances[j, i] = d

    return distances


def perceptual_trustworthiness(
        X: np.ndarray,
        X_embedded: np.ndarray,
        n_neighbors: int = 5,
        X_metric: str = "euclidean"
) -> float:
    """
    Calculate perceptual trustworthiness of a color embedding.

    Measures how well the local structure in the original high-dimensional
    space is preserved in the perceptual color space.

    Args:
        X: (n_samples, n_features) original data or (n_samples, n_samples)
           precomputed distance matrix
        X_embedded: (n_samples, 3) RGB values of the embedding
        n_neighbors: Number of neighbors to consider
        X_metric: Distance metric for original space

    Returns:
        Trustworthiness score in [0, 1]

    References:
        Jarkko Venna and Samuel Kaski. 2001. Neighborhood Preservation in
        Nonlinear Projection Methods: An Experimental Study. ICANN '01.
    """
    n_samples = X.shape[0]

    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2 "
            f"({n_samples / 2})"
        )

    # Distance matrix in original space
    dist_X = pairwise_distances(X, metric=X_metric)
    if X_metric == "precomputed":
        dist_X = dist_X.copy()

    # Exclude self from neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)

    # Distance matrix in perceptual color space
    dist_embedded = delta_e_distance_matrix(X_embedded)
    np.fill_diagonal(dist_embedded, np.inf)
    ind_embedded = np.argsort(dist_embedded, axis=1)[:, :n_neighbors]

    # Build inverted index
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]

    # Calculate ranks
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])

    # Normalize
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )

    return t
