#!/usr/bin/env python
"""
Miniature Metrics - Calculate quality metrics for color embeddings.

Computes trustworthiness and perceptual trustworthiness metrics to evaluate
how well the dimensionality reduction and color mapping preserve structure.
"""

import argparse
from pathlib import Path

import colour
import h5py
import mantel
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness

from .trustworthiness import (
    perceptual_trustworthiness,
    delta_e_cie2000_pairwise
)

__version__ = "2.0.0"


def delta_e_pdist(rgb_array: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise CIEDE2000 distances using pdist convention.

    Args:
        rgb_array: (N, 3) array of RGB values

    Returns:
        Condensed distance matrix (as returned by scipy.spatial.distance.pdist)
    """
    n = len(rgb_array)

    # Convert RGB to LAB
    lab_array = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb_array))

    # Calculate condensed distance matrix
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = colour.delta_E(lab_array[i], lab_array[j], method='CIE 2000')
            distances.append(d)

    return np.array(distances)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate quality metrics for miniature color embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input', type=str,
                        help='HDF5 file from paint_miniature --save_data')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='Metric used for the embedding')
    parser.add_argument('--n', type=int, default=128,
                        help='Number of pixels to sample for metrics')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}")
    h5file = h5py.File(args.input, 'r')
    tissue_array = np.array(h5file['tissue_array'][:])
    embedding = np.array(h5file['embedding'][:])

    # Create output file
    output = h5py.File('metrics.h5', 'w')

    n = min(args.n, tissue_array.shape[0])
    sampled_rows = np.random.choice(tissue_array.shape[0], size=n, replace=False)

    # Embedding trustworthiness
    print(f'Calculating embedding trustworthiness from {n} pixels')
    trust = trustworthiness(
        tissue_array[sampled_rows, :],
        embedding[sampled_rows, :],
        metric=args.metric
    )
    print(f'Embedding trustworthiness = {trust:.4f}')
    output.create_dataset('embedding_trust', data=trust)

    # Distance matrices
    print('Calculating distance matrix in high-dimensional space')
    original_dist = pdist(tissue_array[sampled_rows, :], metric=args.metric)
    output.create_dataset(f'original_dist_{args.metric}', data=original_dist)

    print('Calculating distance matrix in low-dimensional space')
    embedding_dist = pdist(embedding[sampled_rows, :], metric='euclidean')
    output.create_dataset('embedding_dist', data=embedding_dist)

    # Process each colormap
    if 'colors' in h5file:
        for cmap_name in h5file['colors'].keys():
            rgb = np.array(h5file['colors'][cmap_name])
            rgb_sampled = rgb[sampled_rows, :]

            print(f'\n--- Colormap: {cmap_name} ---')

            # Perceptual trustworthiness
            print(f'Calculating perceptual trustworthiness from {n} pixels')
            ptrust = perceptual_trustworthiness(
                tissue_array[sampled_rows, :],
                rgb_sampled,
                X_metric=args.metric
            )
            print(f'Perceptual trustworthiness = {ptrust:.4f}')

            output_colors = output.create_group(cmap_name)
            output_colors.create_dataset('perceptual_trust', data=ptrust)

            # Embedding perceptual trustworthiness
            e_ptrust = perceptual_trustworthiness(
                embedding[sampled_rows, :],
                rgb_sampled
            )
            print(f'Embedding perceptual trustworthiness = {e_ptrust:.4f}')
            output_colors.create_dataset('perceptual_embedding_trust', data=e_ptrust)

            # Perceptual distance matrix
            print('Calculating distance matrix in perceptual space')
            perceptual_dist = delta_e_pdist(rgb_sampled)
            output_colors.create_dataset('perceptual_dist', data=perceptual_dist)

            # Correlation analyses
            sp_orig_perc = spearmanr(original_dist, perceptual_dist)
            sp_emb_perc = spearmanr(embedding_dist, perceptual_dist)
            sp_orig_emb = spearmanr(original_dist, embedding_dist)

            print(f'Spearman (original vs perceptual): r={sp_orig_perc.correlation:.4f}, p={sp_orig_perc.pvalue:.4e}')
            print(f'Spearman (embedding vs perceptual): r={sp_emb_perc.correlation:.4f}, p={sp_emb_perc.pvalue:.4e}')
            print(f'Spearman (original vs embedding): r={sp_orig_emb.correlation:.4f}, p={sp_orig_emb.pvalue:.4e}')

            # Mantel tests
            embedding_v_perceptual = mantel.test(embedding_dist, perceptual_dist)
            tissue_v_perceptual = mantel.test(original_dist, perceptual_dist)

            print(f'Mantel (embedding vs perceptual): r={embedding_v_perceptual.r:.4f}, p={embedding_v_perceptual.p:.4e}')
            print(f'Mantel (original vs perceptual): r={tissue_v_perceptual.r:.4f}, p={tissue_v_perceptual.p:.4e}')

            # Save plots
            input_path = Path(args.input)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(original_dist, perceptual_dist, marker='.', s=0.5, alpha=0.5)
            ax.set_xlabel(f'{args.metric} distance')
            ax.set_ylabel('Delta E 2000')
            ax.set_title(f'{cmap_name}: Original vs Perceptual')
            fig.savefig(input_path.parent / f"{input_path.stem}_{cmap_name}_orig_v_percept.png", dpi=150)
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(embedding_dist, perceptual_dist, marker='.', s=0.5, alpha=0.5)
            ax.set_xlabel('Euclidean distance (embedding)')
            ax.set_ylabel('Delta E 2000')
            ax.set_title(f'{cmap_name}: Embedding vs Perceptual')
            fig.savefig(input_path.parent / f"{input_path.stem}_{cmap_name}_emb_v_percept.png", dpi=150)
            plt.close()

    h5file.close()
    output.close()
    print('\nMetrics calculation complete!')


if __name__ == "__main__":
    main()
