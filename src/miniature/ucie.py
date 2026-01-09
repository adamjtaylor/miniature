"""
UCIE - Uniform Color in Embedding

This module optimizes the transformation of 3D embeddings to LAB color space
to maximize color space utilization while staying within valid sRGB gamut.
"""

import numpy as np
import colour
import estimagic as em
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist


def generate_lab_grid(
        l_lims=(0, 100),
        a_lims=(-128, 127),
        b_lims=(-128, 127),
        n_points=32
) -> np.ndarray:
    """
    Generate a grid of LAB colors within the sRGB gamut.

    Args:
        l_lims: Lightness range
        a_lims: a* range
        b_lims: b* range
        n_points: Number of points per dimension

    Returns:
        Array of valid LAB colors within sRGB gamut
    """
    rgb_values = np.linspace(0, 1, n_points)
    rgb_grid = np.array(np.meshgrid(rgb_values, rgb_values, rgb_values)).T.reshape(-1, 3)

    # Vectorized conversion to LAB
    xyz = colour.sRGB_to_XYZ(rgb_grid)
    lab_grid = colour.XYZ_to_Lab(xyz)

    # Filter by LAB limits
    mask = (
        (lab_grid[:, 0] >= l_lims[0]) & (lab_grid[:, 0] <= l_lims[1]) &
        (lab_grid[:, 1] >= a_lims[0]) & (lab_grid[:, 1] <= a_lims[1]) &
        (lab_grid[:, 2] >= b_lims[0]) & (lab_grid[:, 2] <= b_lims[1])
    )

    return lab_grid[mask]


def rotation_matrix(rot_x: float, rot_y: float, rot_z: float) -> np.ndarray:
    """Create a combined rotation matrix for rotations around x, y, z axes."""
    cx, sx = np.cos(rot_x), np.sin(rot_x)
    cy, sy = np.cos(rot_y), np.sin(rot_y)
    cz, sz = np.cos(rot_z), np.sin(rot_z)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return rz @ ry @ rx


def transform_points(
        points: np.ndarray,
        scale: float,
        rot_x: float, rot_y: float, rot_z: float,
        trans_x: float, trans_y: float, trans_z: float
) -> np.ndarray:
    """
    Apply scale, rotation, and translation to points.

    Args:
        points: (N, 3) array of points
        scale: Uniform scale factor
        rot_x, rot_y, rot_z: Rotation angles in radians
        trans_x, trans_y, trans_z: Translation offsets

    Returns:
        Transformed points
    """
    rot = rotation_matrix(rot_x, rot_y, rot_z)
    transformed = points @ rot.T * scale
    transformed += np.array([trans_x, trans_y, trans_z])
    return transformed


def in_hull(points: np.ndarray, hull: Delaunay) -> np.ndarray:
    """Test if points are inside a convex hull."""
    return hull.find_simplex(points) >= 0


def objective(params: list, source: np.ndarray, target_hull: Delaunay) -> float:
    """
    Objective function for optimization.

    Maximizes scale while penalizing points outside the target hull.
    """
    scale, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z = params

    transformed = transform_points(source, scale, rot_x, rot_y, rot_z,
                                   trans_x, trans_y, trans_z)

    is_inside = in_hull(transformed, target_hull)
    outside_points = transformed[~is_inside]

    if len(outside_points) == 0:
        penalty = 0
    else:
        # Distance to nearest hull point
        hull_points = target_hull.points[target_hull.convex_hull.flatten()]
        dist_mat = cdist(outside_points, hull_points)
        penalty = np.sum(np.min(dist_mat, axis=1) ** 2)

    # Maximize scale, minimize penalty
    loss = -scale + penalty
    print(f'Loss: {loss:.4f}', end='\r')
    return loss


def guess_initial(points: np.ndarray, polygon: np.ndarray) -> tuple:
    """Estimate initial transformation parameters."""
    centroid_polygon = np.mean(polygon, axis=0)
    centroid_points = np.mean(points, axis=0)

    range_points = np.ptp(points, axis=0)
    range_polygon = np.ptp(polygon, axis=0)

    initial_scale = np.min(range_polygon / (range_points + 1e-10)) * 0.8
    initial_rot = [0, 0, 0]

    # Apply initial rotation and scale to get translation
    rot = rotation_matrix(*initial_rot)
    scaled_points = points @ rot.T * initial_scale
    centroid_scaled = np.mean(scaled_points, axis=0)
    initial_translation = centroid_polygon - centroid_scaled

    return initial_translation, initial_scale, initial_rot


def lab_to_rgb_vectorized(lab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB using colour-science (vectorized)."""
    xyz = colour.Lab_to_XYZ(lab)
    rgb = colour.XYZ_to_sRGB(xyz)
    return np.clip(rgb, 0, 1)


def ucie(embedding: np.ndarray, save_plots: bool = False) -> np.ndarray:
    """
    Uniform Color in Embedding - map 3D embedding to LAB color space.

    This function finds an optimal rotation, scale, and translation to map
    a 3D embedding into the LAB color space while maximizing color space
    utilization and staying within the sRGB gamut.

    Args:
        embedding: (N, 3) array of embedding coordinates
        save_plots: Whether to save debug plots

    Returns:
        (N, 3) array of RGB values in [0, 1]
    """
    print('Generating target colorspace')
    lab_polygon = generate_lab_grid()

    # Remove outliers
    lims = np.quantile(embedding, [0.001, 0.999], axis=0)
    mask = np.all((embedding >= lims[0]) & (embedding <= lims[1]), axis=1)
    filtered = embedding[mask]

    if len(filtered) < 10:
        print("Warning: Too few points after filtering, using all points")
        filtered = embedding

    print('Finding convex hull of the embedding')
    try:
        hull = ConvexHull(filtered)
        embedding_polygon = filtered[hull.vertices]
    except Exception as e:
        print(f"Warning: Could not compute convex hull: {e}")
        # Use all points if hull fails
        embedding_polygon = filtered

    print('Estimating initial transformation')
    initial = guess_initial(embedding_polygon, lab_polygon)

    print('Calculating Delaunay triangulation of the target colorspace')
    lab_delaunay = Delaunay(lab_polygon)

    # Initial parameters: [scale, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    params = [
        initial[1],
        initial[2][0], initial[2][1], initial[2][2],
        initial[0][0], initial[0][1], initial[0][2]
    ]

    # Bounds
    lb = np.array(params) - 0.001
    ub = params.copy()
    ub[1] = ub[2] = ub[3] = 2 * np.pi  # Allow full rotation

    print('Running optimization')
    try:
        res = em.minimize(
            criterion=objective,
            params=params,
            criterion_kwargs={'source': embedding_polygon, 'target_hull': lab_delaunay},
            algorithm="scipy_neldermead",
            algo_options={'stopping.max_iterations': 500},
            soft_lower_bounds=lb,
            soft_upper_bounds=ub,
            multistart=True,
            multistart_options={"n_samples": 10, "n_cores": 4}
        )
        best = res.params
    except Exception as e:
        print(f"Warning: Optimization failed: {e}, using initial guess")
        best = params

    print('\nApplying the optimal transform')
    transformed = transform_points(
        embedding,
        best[0],
        best[1], best[2], best[3],
        best[4], best[5], best[6]
    )

    print('Converting to RGB (vectorized)')
    rgb = lab_to_rgb_vectorized(transformed)

    if save_plots:
        _save_debug_plots(embedding, transformed, rgb)

    print('UCIE complete')
    return rgb


def _save_debug_plots(embedding: np.ndarray, transformed: np.ndarray, rgb: np.ndarray):
    """Save debug visualization plots."""
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                  s=1, edgecolors='none', alpha=0.5)
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    ax1.set_zlabel('Dim 3')
    ax1.set_title('Original Embedding')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter3D(transformed[:, 0], transformed[:, 1], transformed[:, 2],
                  c=rgb, s=1, edgecolors='none')
    ax2.set_xlabel('L*')
    ax2.set_ylabel('a*')
    ax2.set_zlabel('b*')
    ax2.set_title('Transformed to LAB')

    plt.tight_layout()
    plt.savefig('ucie_debug.png', dpi=150)
    plt.close()
