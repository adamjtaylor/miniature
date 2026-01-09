#!/usr/bin/env python
"""
Paint Miniature - Create thumbnails from high-dimensional imaging data.

This tool applies dimensionality reduction to multiplexed imaging data
and maps the low-dimensional embeddings to perceptually meaningful colors.
"""

import argparse
import sys
from pathlib import Path

import colour
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import umap
import zarr
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm

from .ucie import ucie

__version__ = "2.0.0"

# Colormap configuration
COLORMAP_DIR = Path(__file__).parent / "colormaps"

COLORMAPS_2D = {
    'BREMM': 'bremm.png',
    'CUBEDIAGONAL': 'cubediagonal.png',
    'SCHUMANN': 'schumann.png',
    'STEIGER': 'steiger.png',
    'TEULING2': 'teulingfig2.png',
    'ZIEGLER': 'ziegler.png'
}


def pull_pyramid(input_path: str, max_pixels: int = 512 * 512) -> zarr.Array:
    """
    Load an appropriate pyramid level from a TIFF file.

    Selects the smallest pyramid level that has at least max_pixels pixels.

    Args:
        input_path: Path to the input TIFF file
        max_pixels: Maximum number of pixels to allow

    Returns:
        Zarr array of the selected pyramid level
    """
    print(f"Opening image: {input_path}", file=sys.stderr)
    tiff = tifffile.TiffFile(input_path)
    ndim = tiff.series[0].ndim

    if ndim == 2:
        raise ValueError("Cannot handle 2-dimensional images (yet)")
    elif ndim != 3:
        raise ValueError(f"Cannot handle {ndim}-dimensional images")

    levels = tiff.series[0].levels
    selected_level_idx = 0

    for i, level in reversed(list(enumerate(levels))):
        print(f'{i}: {level.shape}')
        if i == 0:
            break
        next_level = levels[i - 1]
        n_pixels = np.prod(next_level.shape[1:])
        if n_pixels > max_pixels:
            selected_level_idx = i
            break

    selected_level = levels[selected_level_idx]
    selected_level_shape = selected_level.shape[1:]
    selected_level_npixels = np.prod(selected_level_shape)

    if selected_level_npixels > max_pixels:
        print(f'No level with less than {max_pixels} pixels found, using lowest level (-1)')

    print(f'Selected level {selected_level_idx} with {selected_level_npixels} pixels '
          f'{selected_level_shape} and {selected_level.shape[0]} channels')

    zarray = zarr.open(selected_level.aszarr())

    if not hasattr(zarray, 'shape'):
        print('Zarr array failed to load', file=sys.stderr)
        sys.exit(1)

    return zarray


def crop_center(input_path: str, size: int = 256) -> np.ndarray:
    """Extract a centered square region from the image."""
    store = tifffile.imread(input_path, aszarr=True)
    z = zarr.open(store, mode='r')

    center = np.array([z[0].shape[1] // 2, z[0].shape[2] // 2], dtype=int)

    xmin, xmax = center[0] - size, center[0] + size
    ymin, ymax = center[1] - size, center[1] + size

    return z[0][:, xmin:xmax, ymin:ymax]


def remove_background(zarray: zarr.Array, pseudocount: float) -> tuple:
    """
    Remove background pixels using Otsu thresholding.

    Args:
        zarray: Input image array (channels x height x width)
        pseudocount: Pseudocount for log transformation

    Returns:
        Tuple of (tissue_array, mask) where tissue_array is (n_pixels, n_channels)
    """
    print("Finding background")
    sum_image = np.array(zarray).sum(axis=0)
    print(f'Using pseudocount of {pseudocount}')
    log_image = np.log2(sum_image + pseudocount)
    thresh = threshold_otsu(log_image[log_image > np.log2(pseudocount)])
    binary = log_image > thresh
    cleaned = remove_small_objects(binary)
    print("Background removed")

    tissue_array = np.array(zarray)[:, cleaned].T
    print(f"Selected {tissue_array.shape[0]} of {zarray.shape[1] * zarray.shape[2]} pixels as tissue")
    print(f"Pixels x channels matrix: {tissue_array.shape}")

    return tissue_array, cleaned


def keep_background(zarray: zarr.Array) -> tuple:
    """Keep all pixels including background."""
    print("Preserving background")
    shape = zarray.shape[1:]
    everything = np.ones(shape, dtype=bool)
    tissue_array = np.array(zarray)[:, everything].T
    print(f"Pixels x channels matrix: {tissue_array.shape}")
    return tissue_array, everything


def run_pca(tissue_array: np.ndarray, n: int) -> np.ndarray:
    """Run PCA dimensionality reduction."""
    print("Running PCA")
    reducer = PCA(n_components=n)
    return reducer.fit_transform(tissue_array)


def run_umap(tissue_array: np.ndarray, n: int, metric: str) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    print("Running UMAP")
    reducer = umap.UMAP(n_components=n, metric=metric, verbose=True)
    return reducer.fit_transform(tissue_array)


def run_tsne(tissue_array: np.ndarray, n: int, metric: str) -> np.ndarray:
    """Run t-SNE dimensionality reduction."""
    print("Running t-SNE")
    reducer = TSNE(n_components=n, metric=metric, verbose=True)
    return reducer.fit_transform(tissue_array)


def assign_colours_lab(embedding: np.ndarray) -> np.ndarray:
    """
    Map 3D embedding to LAB color space (vectorized).

    Args:
        embedding: (n_pixels, 3) array of embedding coordinates

    Returns:
        (n_pixels, 3) array of RGB values in [0, 1]
    """
    print("Assigning LAB colours to embedding (vectorized)")

    # Scale dimensions to LAB ranges
    # a* and b* range: [-128, 127]
    # L* range: [0, 100]
    scaler_ab = MinMaxScaler(feature_range=(-128, 127))
    scaler_l = MinMaxScaler(feature_range=(0, 100))

    dim1 = scaler_ab.fit_transform(embedding[:, 0:1])
    dim2 = scaler_ab.fit_transform(embedding[:, 1:2])
    dim3 = scaler_l.fit_transform(embedding[:, 2:3])

    # LAB array: [L, a, b] - note: L is third dimension in embedding
    lab = np.hstack([dim3, dim1, dim2])

    # Vectorized LAB to RGB conversion using colour-science
    xyz = colour.Lab_to_XYZ(lab)
    rgb = colour.XYZ_to_sRGB(xyz)

    # Clamp to valid RGB range
    rgb = np.clip(rgb, 0, 1)

    print("LAB colors assigned")
    return rgb


def assign_colours_rgb(embedding: np.ndarray) -> np.ndarray:
    """
    Map 3D embedding directly to RGB (vectorized).

    Args:
        embedding: (n_pixels, 3) array of embedding coordinates

    Returns:
        (n_pixels, 3) array of RGB values in [0, 1]
    """
    print("Assigning RGB colours to embedding (vectorized)")

    scaler = MinMaxScaler(feature_range=(0, 1))

    dim1 = scaler.fit_transform(embedding[:, 0:1])
    dim2 = scaler.fit_transform(embedding[:, 1:2])
    dim3 = scaler.fit_transform(embedding[:, 2:3])

    rgb = np.hstack([dim3, dim1, dim2])

    print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print("RGB colors assigned")
    return rgb


def assign_colours_2d(embedding: np.ndarray, colormap_path: Path) -> np.ndarray:
    """
    Map 2D embedding to colors using a colormap image.

    Args:
        embedding: (n_pixels, 2) array of embedding coordinates
        colormap_path: Path to colormap PNG image

    Returns:
        (n_pixels, 3) array of RGB values in [0, 1]
    """
    print(f"Loading colormap: {colormap_path.name}")
    colormap_im = Image.open(colormap_path)
    width, height = colormap_im.size

    print("Assigning 2D colours to embedding")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = np.hstack([
        scaler.fit_transform(embedding[:, 0:1]),
        scaler.fit_transform(embedding[:, 1:2])
    ])

    # Map to colormap pixel coordinates
    x_coords = (scaled[:, 0] * (width - 1)).astype(int)
    y_coords = (scaled[:, 1] * (height - 1)).astype(int)

    # Extract colors from colormap
    colormap_array = np.array(colormap_im)
    rgb = colormap_array[y_coords, x_coords, :3] / 255.0

    return rgb


def make_rgb_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create an RGB image from color values and mask.

    Args:
        rgb: (n_pixels, 3) array of RGB values in [0, 1]
        mask: Boolean mask of tissue pixels

    Returns:
        (height, width, 3) uint8 RGB image
    """
    print("Painting miniature")
    rgb_shape = list(mask.shape) + [3]
    rgb_image = np.zeros(rgb_shape)
    rgb_image[mask] = rgb
    rgb_image = (rgb_image * 255).astype(np.uint8)

    print(f"Image shape: {rgb_image.shape}")
    print(f"RGB range: [{rgb_image.min()}, {rgb_image.max()}]")
    return rgb_image


def plot_embedding(embedding: np.ndarray, rgb: np.ndarray, output_path: Path):
    """Save a visualization of the embedding colored by assigned colors."""
    newp = output_path.parent / f"{output_path.stem}-embedding{output_path.suffix}"
    fig = plt.figure()

    if embedding.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=rgb, s=1, edgecolors='none'
        )
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
    else:
        ax = fig.add_subplot()
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=rgb, s=1, edgecolors='none'
        )
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    plt.savefig(newp)
    plt.close()


def save_data(h5_path: Path, mask: np.ndarray, tissue_array: np.ndarray = None,
              embedding: np.ndarray = None):
    """Save intermediate data to HDF5 file."""
    print(f"Saving data to {h5_path}")
    with h5py.File(h5_path, 'w') as h5file:
        h5file.create_dataset('mask', data=mask)
        if tissue_array is not None:
            h5file.create_dataset('tissue_array', data=tissue_array)
        if embedding is not None:
            h5file.create_dataset('embedding', data=embedding)


def main():
    parser = argparse.ArgumentParser(
        description='Create miniature thumbnails from high-dimensional OME-TIFF images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input', type=str,
                        help='Input OME-TIFF file')
    parser.add_argument('output', type=str, default='miniature.png',
                        help='Output PNG file')

    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}')

    parser.add_argument('-l', '--level', type=int, dest='level',
                        help='Image pyramid level to use (default: auto-select)')
    parser.add_argument('--max_pixels', type=int, dest='max_pixels',
                        default=512*512,
                        help='Maximum pixels when auto-selecting pyramid level')
    parser.add_argument('--crop', action='store_true',
                        help='Use 512x512 center crop instead of full image')
    parser.add_argument('--keep_bg', action='store_true',
                        help="Don't remove background")

    parser.add_argument('--dimred', type=str, dest='dimred', default='umap',
                        choices=['umap', 'tsne', 'pca'],
                        help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, dest='n_components', default=3,
                        help='Number of dimensions for embedding')
    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['braycurtis', 'canberra', 'chebyshev', 'correlation',
                                 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
                                 'mahalanobis', 'minkowski'],
                        help='Distance metric for UMAP/t-SNE')

    parser.add_argument('--log', action='store_true',
                        help='Log10 transform the data')
    parser.add_argument('--pseudocount', type=float, default=1.0,
                        help='Pseudocount for log transformation')
    parser.add_argument('--scaler', choices=['NoScaler', 'MinMaxScaler',
                                              'StandardScaler', 'RobustScaler'],
                        help='Scaling function to apply')

    parser.add_argument('--colormap', type=str, dest='colormap', default='ALL',
                        choices=['ALL', 'BREMM', 'SCHUMANN', 'STEIGER', 'TEULING2',
                                 'ZIEGLER', 'CUBEDIAGONAL', 'LAB', 'RGB', 'UCIE'],
                        help='Colormap for visualization')

    parser.add_argument('--save_data', action='store_true',
                        help='Save intermediate data to HDF5')
    parser.add_argument('--plot_embedding', action='store_true',
                        help='Save embedding visualization')

    args = parser.parse_args()

    output_path = Path(args.output)

    # Initialize HDF5 file if saving data
    h5file = None
    h5color = None
    if args.save_data:
        h5_path = output_path.parent / f"{output_path.stem}.h5"
        h5file = h5py.File(h5_path, 'w')
        h5color = h5file.create_group('colors')

    # Load image
    if args.crop:
        zarray = crop_center(args.input)
    elif args.level is not None:
        tiff = tifffile.TiffFile(args.input)
        zarray = zarr.open(tiff.series[0].levels[args.level].aszarr())
    else:
        zarray = pull_pyramid(args.input, max_pixels=args.max_pixels)

    # Handle 3-channel (RGB) images directly
    if zarray.shape[0] == 3:
        rgb_image = np.moveaxis(np.array(zarray), 0, -1)
        print(f"Saving RGB image as {args.output}")
        im = Image.fromarray(rgb_image.astype(np.uint8), 'RGB')
        im.save(args.output)
        if args.save_data:
            h5file.create_dataset('rgb_image', data=rgb_image)
            h5file.close()
        print("Complete!")
        return

    # Extract tissue pixels
    if args.keep_bg:
        tissue_array, mask = keep_background(zarray)
    else:
        print('Removing background')
        tissue_array, mask = remove_background(zarray, args.pseudocount)

    if args.save_data:
        h5file.create_dataset('mask', data=mask)

    # Apply log transform if requested
    if args.log:
        tissue_array = np.log10(tissue_array + args.pseudocount)

    # Apply scaling if requested
    if args.scaler and args.scaler != "NoScaler":
        scalers = {
            'MinMaxScaler': MinMaxScaler,
            'StandardScaler': StandardScaler,
            'RobustScaler': RobustScaler
        }
        print(f"Applying {args.scaler}")
        tissue_array = scalers[args.scaler]().fit_transform(tissue_array)

    if args.save_data:
        h5file.create_dataset('tissue_array', data=tissue_array)

    # Run dimensionality reduction
    if args.dimred == 'tsne':
        embedding = run_tsne(tissue_array, args.n_components, args.metric)
    elif args.dimred == 'umap':
        embedding = run_umap(tissue_array, args.n_components, args.metric)
    else:  # pca
        embedding = run_pca(tissue_array, args.n_components)

    if args.save_data:
        h5file.create_dataset('embedding', data=embedding)

    # Select colormaps based on dimensions
    if args.colormap == "ALL":
        if args.n_components == 3:
            selected_colormaps = ['LAB', 'RGB', 'UCIE']
        else:
            selected_colormaps = list(COLORMAPS_2D.keys())
    else:
        selected_colormaps = [args.colormap]

    # Generate images for each colormap
    for cmap in tqdm(selected_colormaps, desc="Generating colormaps"):
        print(f'\nUsing colormap: {cmap}')

        if args.n_components == 3:
            if cmap == 'LAB':
                rgb = assign_colours_lab(embedding)
            elif cmap == 'UCIE':
                rgb = ucie(embedding)
            elif cmap == 'RGB':
                rgb = assign_colours_rgb(embedding)
            else:
                print(f'Colormap {cmap} not valid for 3D embedding, skipping')
                continue
        else:
            if cmap not in COLORMAPS_2D:
                print(f'Colormap {cmap} not valid for 2D embedding, skipping')
                continue
            colormap_path = COLORMAP_DIR / COLORMAPS_2D[cmap]
            rgb = assign_colours_2d(embedding, colormap_path)

        rgb_image = make_rgb_image(rgb, mask)

        if args.save_data:
            h5color.create_dataset(cmap, data=rgb)

        # Determine output filename
        if args.colormap == "ALL":
            out_path = output_path.parent / f"{output_path.stem}_{cmap}{output_path.suffix}"
        else:
            out_path = output_path

        print(f"Saving image as {out_path}")
        im = Image.fromarray(rgb_image, 'RGB')
        im.save(out_path)

        if args.plot_embedding:
            plot_embedding(embedding, rgb, out_path)

    if args.save_data:
        h5file.close()

    print("\nComplete!")


if __name__ == "__main__":
    main()
