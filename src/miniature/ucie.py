"""
UCIE - Uniform Color in Embedding

This module optimizes the transformation of embeddings to color spaces
to maximize color space utilization while staying within valid gamuts.

Supports:
- 3D embeddings: optimize into LAB or RGB colorspace
- 2D embeddings: optimize into unit square [0,1]² for colormap lookup
"""

import numpy as np
import colour
import estimagic as em
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# Module-level cache for LAB/RGB gamut hulls
_HULL_CACHE = {}


def in_hull_halfspace(points: np.ndarray, equations: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Test if points are inside convex hull using half-space inequalities.

    Uses ConvexHull.equations (facet normals + offsets) for fast vectorized test.
    A point is inside if Ax + b <= 0 for all facets.

    Args:
        points: (N, D) array of points to test
        equations: (M, D+1) array from ConvexHull.equations (A|b where Ax + b <= 0)
        tolerance: Small value for numerical stability

    Returns:
        Boolean array indicating which points are inside
    """
    # Compute Ax + b for all points and all facets
    # points @ A.T + b gives (N, M) matrix of constraint values
    return np.all(points @ equations[:, :-1].T + equations[:, -1] <= tolerance, axis=1)


def facet_violation_penalty(points: np.ndarray, equations: np.ndarray) -> float:
    """Compute squared hinge penalty for points violating hull constraints.

    For each point, compute max(Ax + b, 0)^2 summed over all violated facets.
    This is smooth, fast, and requires no distance matrix.

    Args:
        points: (N, D) array of points
        equations: (M, D+1) array from ConvexHull.equations

    Returns:
        Total squared violation penalty
    """
    # violations shape: (N_points, N_facets)
    violations = points @ equations[:, :-1].T + equations[:, -1]
    # Squared hinge: penalize positive values (outside hull)
    penalty = np.sum(np.maximum(violations, 0) ** 2)
    return penalty


def select_support_points(
    points: np.ndarray,
    n_directions: int = 100,
    seed: int = 42
) -> np.ndarray:
    """Select support points using directional extrema sampling.

    For each random unit direction, find the points with min/max projection.
    These points define the "support" of the point cloud's convex hull.

    Args:
        points: (N, D) array of points
        n_directions: Number of random directions to sample
        seed: Random seed for reproducibility

    Returns:
        (M, D) array of support points (M <= 2 * n_directions, typically ~200)
    """
    rng = np.random.default_rng(seed)
    D = points.shape[1]

    # Generate random unit directions
    directions = rng.standard_normal((n_directions, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Project points onto each direction
    projections = points @ directions.T  # (N, n_directions)

    # Find argmin and argmax for each direction
    min_indices = np.argmin(projections, axis=0)
    max_indices = np.argmax(projections, axis=0)

    # Unique support point indices
    support_indices = np.unique(np.concatenate([min_indices, max_indices]))

    return points[support_indices]


def get_target_hull(target: str, n_points: int = 32):
    """Get cached convex hull and equations for target colorspace.

    Args:
        target: 'LAB' or 'RGB'
        n_points: Grid resolution per dimension

    Returns:
        Tuple of (ConvexHull, equations array)
    """
    cache_key = (target, n_points)
    if cache_key not in _HULL_CACHE:
        if target == 'LAB':
            grid = generate_lab_grid(n_points=n_points)
        elif target == 'RGB':
            grid = generate_rgb_grid(n_points=n_points)
        else:
            raise ValueError(f"Unknown target: {target}")
        hull = ConvexHull(grid)
        _HULL_CACHE[cache_key] = (hull, hull.equations.copy())
    return _HULL_CACHE[cache_key]


def generate_rgb_grid(n_points: int = 32) -> np.ndarray:
    """
    Generate a grid of RGB colors (unit cube [0,1]³).

    Args:
        n_points: Number of points per dimension

    Returns:
        Array of RGB colors as vertices of the unit cube
    """
    rgb_values = np.linspace(0, 1, n_points)
    rgb_grid = np.array(np.meshgrid(rgb_values, rgb_values, rgb_values)).T.reshape(-1, 3)
    return rgb_grid


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


def objective(params: list, source: np.ndarray, hull_equations: np.ndarray) -> float:
    """
    Objective function for optimization using fast half-space penalty.

    Maximizes scale while penalizing points outside the target hull.
    Uses facet violation penalty instead of expensive cdist computation.

    Args:
        params: [scale, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        source: Support points from embedding hull
        hull_equations: ConvexHull.equations for target colorspace

    Returns:
        Loss value (negative scale + violation penalty)
    """
    scale, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z = params

    transformed = transform_points(source, scale, rot_x, rot_y, rot_z,
                                   trans_x, trans_y, trans_z)

    penalty = facet_violation_penalty(transformed, hull_equations)

    # Maximize scale, minimize penalty
    return -scale + penalty


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


def optimize_embedding(
        embedding: np.ndarray,
        target: str = 'LAB',
        save_plots: bool = False,
        verbose: bool = True,
        n_support_directions: int = 100
) -> np.ndarray:
    """
    Optimize embedding rotation to fit within a target colorspace.

    This function finds an optimal rotation, scale, and translation to map
    a 3D embedding into the target color space while maximizing color space
    utilization.

    Args:
        embedding: (N, 3) array of embedding coordinates
        target: Target colorspace, either 'LAB' or 'RGB'
        save_plots: Whether to save debug plots
        verbose: Whether to print progress messages
        n_support_directions: Number of directions for support point sampling

    Returns:
        (N, 3) array of RGB values in [0, 1]
    """
    if target not in ('LAB', 'RGB'):
        raise ValueError(f"Unknown target colorspace: {target}. Use 'LAB' or 'RGB'.")

    if verbose:
        print(f'Getting target colorspace hull ({target})')

    # Get cached hull and equations for target colorspace
    target_hull, hull_equations = get_target_hull(target)
    target_polygon = target_hull.points

    # Remove outliers
    lims = np.quantile(embedding, [0.001, 0.999], axis=0)
    mask = np.all((embedding >= lims[0]) & (embedding <= lims[1]), axis=1)
    filtered = embedding[mask]

    if len(filtered) < 10:
        if verbose:
            print("Warning: Too few points after filtering, using all points")
        filtered = embedding

    if verbose:
        print('Finding convex hull of the embedding')
    try:
        hull = ConvexHull(filtered)
        embedding_polygon = filtered[hull.vertices]
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute convex hull: {e}")
        # Use all points if hull fails
        embedding_polygon = filtered

    # Select support points for faster optimization
    support_points = select_support_points(embedding_polygon, n_directions=n_support_directions)
    if verbose:
        print(f'Using {len(support_points)} support points for optimization')

    if verbose:
        print('Estimating initial transformation')
    initial = guess_initial(support_points, target_polygon)

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

    if verbose:
        print('Running optimization')
    try:
        res = em.minimize(
            criterion=objective,
            params=params,
            criterion_kwargs={'source': support_points, 'hull_equations': hull_equations},
            algorithm="scipy_neldermead",
            algo_options={'stopping.max_iterations': 500},
            soft_lower_bounds=lb,
            soft_upper_bounds=ub,
            multistart=True,
            multistart_options={"n_samples": 10, "n_cores": 4}
        )
        best = res.params
    except Exception as e:
        if verbose:
            print(f"Warning: Optimization failed: {e}, using initial guess")
        best = params

    if verbose:
        print('Applying the optimal transform')
    transformed = transform_points(
        embedding,
        best[0],
        best[1], best[2], best[3],
        best[4], best[5], best[6]
    )

    # Convert to RGB based on target colorspace
    if target == 'LAB':
        if verbose:
            print('Converting LAB to RGB (vectorized)')
        rgb = lab_to_rgb_vectorized(transformed)
    else:  # RGB
        if verbose:
            print('Clamping RGB values')
        rgb = np.clip(transformed, 0, 1)

    if save_plots:
        _save_debug_plots(embedding, transformed, rgb)

    if verbose:
        print(f'Optimization complete (target: {target})')
    return rgb


def ucie(embedding: np.ndarray, save_plots: bool = False) -> np.ndarray:
    """
    Uniform Color in Embedding - map 3D embedding to LAB color space.

    This is a backwards-compatible alias for optimize_embedding(target='LAB').
    Consider using optimize_embedding() directly for more control.

    Args:
        embedding: (N, 3) array of embedding coordinates
        save_plots: Whether to save debug plots

    Returns:
        (N, 3) array of RGB values in [0, 1]
    """
    return optimize_embedding(embedding, target='LAB', save_plots=save_plots)


# 2D Optimization Functions
# -------------------------

def rotation_matrix_2d(angle: float) -> np.ndarray:
    """Create a 2D rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def transform_points_2d(
        points: np.ndarray,
        scale: float,
        angle: float,
        trans_x: float,
        trans_y: float
) -> np.ndarray:
    """
    Apply scale, rotation, and translation to 2D points.

    Args:
        points: (N, 2) array of points
        scale: Uniform scale factor
        angle: Rotation angle in radians
        trans_x, trans_y: Translation offsets

    Returns:
        Transformed points
    """
    rot = rotation_matrix_2d(angle)
    transformed = points @ rot.T * scale
    transformed += np.array([trans_x, trans_y])
    return transformed


def objective_2d(params: list, source: np.ndarray, hull_equations: np.ndarray) -> float:
    """
    Objective function for 2D optimization using fast half-space penalty.

    Maximizes scale while penalizing points outside the target hull.
    Uses facet violation penalty instead of expensive cdist computation.

    Args:
        params: [scale, angle, trans_x, trans_y]
        source: Support points from embedding hull
        hull_equations: ConvexHull.equations for target space

    Returns:
        Loss value (negative scale + violation penalty)
    """
    scale, angle, trans_x, trans_y = params

    transformed = transform_points_2d(source, scale, angle, trans_x, trans_y)

    penalty = facet_violation_penalty(transformed, hull_equations)

    # Maximize scale, minimize penalty
    return -scale + penalty


def guess_initial_2d(points: np.ndarray, polygon: np.ndarray) -> tuple:
    """Estimate initial 2D transformation parameters."""
    centroid_polygon = np.mean(polygon, axis=0)
    centroid_points = np.mean(points, axis=0)

    range_points = np.ptp(points, axis=0)
    range_polygon = np.ptp(polygon, axis=0)

    initial_scale = np.min(range_polygon / (range_points + 1e-10)) * 0.8
    initial_angle = 0.0

    # Apply initial rotation and scale to get translation
    rot = rotation_matrix_2d(initial_angle)
    scaled_points = points @ rot.T * initial_scale
    centroid_scaled = np.mean(scaled_points, axis=0)
    initial_translation = centroid_polygon - centroid_scaled

    return initial_translation, initial_scale, initial_angle


def optimize_embedding_2d(
        embedding: np.ndarray,
        save_plots: bool = False,
        verbose: bool = True,
        n_support_directions: int = 50
) -> np.ndarray:
    """
    Optimize 2D embedding rotation to fit within the unit square [0,1]².

    This function finds an optimal rotation, scale, and translation to map
    a 2D embedding into the unit square while maximizing space utilization.

    Args:
        embedding: (N, 2) array of embedding coordinates
        save_plots: Whether to save debug plots
        verbose: Whether to print progress messages
        n_support_directions: Number of directions for support point sampling

    Returns:
        (N, 2) array of coordinates in [0, 1]
    """
    if verbose:
        print('Generating target space (unit square)')

    # Target is the unit square [0,1]²
    target_polygon = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5], [0.5, 0.5]  # Add interior points
    ])

    # Compute ConvexHull for the unit square (simple, can be cached but fast enough)
    target_hull = ConvexHull(target_polygon)
    hull_equations = target_hull.equations

    # Remove outliers
    lims = np.quantile(embedding, [0.001, 0.999], axis=0)
    mask = np.all((embedding >= lims[0]) & (embedding <= lims[1]), axis=1)
    filtered = embedding[mask]

    if len(filtered) < 10:
        if verbose:
            print("Warning: Too few points after filtering, using all points")
        filtered = embedding

    if verbose:
        print('Finding convex hull of the 2D embedding')
    try:
        hull = ConvexHull(filtered)
        embedding_polygon = filtered[hull.vertices]
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute convex hull: {e}")
        embedding_polygon = filtered

    # Select support points for faster optimization
    support_points = select_support_points(embedding_polygon, n_directions=n_support_directions)
    if verbose:
        print(f'Using {len(support_points)} support points for 2D optimization')

    if verbose:
        print('Estimating initial 2D transformation')
    initial = guess_initial_2d(support_points, target_polygon)

    # Initial parameters: [scale, angle, trans_x, trans_y]
    params = [
        initial[1],
        initial[2],
        initial[0][0], initial[0][1]
    ]

    # Bounds
    lb = np.array(params) - 0.001
    ub = params.copy()
    ub[1] = 2 * np.pi  # Allow full rotation

    if verbose:
        print('Running 2D optimization')
    try:
        res = em.minimize(
            criterion=objective_2d,
            params=params,
            criterion_kwargs={'source': support_points, 'hull_equations': hull_equations},
            algorithm="scipy_neldermead",
            algo_options={'stopping.max_iterations': 500},
            soft_lower_bounds=lb,
            soft_upper_bounds=ub,
            multistart=True,
            multistart_options={"n_samples": 10, "n_cores": 4}
        )
        best = res.params
    except Exception as e:
        if verbose:
            print(f"Warning: 2D Optimization failed: {e}, using initial guess")
        best = params

    if verbose:
        print('Applying the optimal 2D transform')
    transformed = transform_points_2d(
        embedding,
        best[0],
        best[1],
        best[2], best[3]
    )

    # Clamp to [0, 1]
    result = np.clip(transformed, 0, 1)

    if verbose:
        print('2D Optimization complete')
    return result


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
