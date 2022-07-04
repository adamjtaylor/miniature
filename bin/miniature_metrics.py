#!/usr/bin/env python
import mantel
import argparse
import h5py
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def delta_e_cie2000_metric(rgb1,rgb2):
        color1 = sRGBColor(rgb1[0], rgb1[1], rgb1[2])
        lab1 = convert_color(color1, LabColor)
        color2 = sRGBColor(rgb2[0], rgb2[1], rgb2[2])
        lab2 = convert_color(color2, LabColor)
        delta = delta_e_cie2000(lab1, lab2)
        return delta


def perceptual_trustworthiness(X, X_embedded, *, n_neighbors=5, X_metric="euclidean"):
    r"""Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as
    .. math::
        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.
    X_embedded : ndarray of shape (n_samples, n_components)
        Embedding of the training data in RGB space
    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.
    X_metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.
        .. versionadded:: 0.20
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.
    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.
    """
    n_samples = X.shape[0]
    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=X_metric)
    if X_metric == "precomputed":
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    # Here we use delta_e_cie2000
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors, metric = delta_e_cie2000_metric)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t

def main():
    parser = argparse.ArgumentParser(description = 'Paint a miniature from an OME-TIFF')
    
    parser.add_argument('input',
                        type=str,
                        help=' A h5 file containing output from paint_miniature with --save_data argument')
    parser.add_argument('--metric',
                        type=str,
                        help=' Metric used for the embedding')

    args = parser.parse_args()

    h5file = h5py.File(args.input, 'r')
    tissue_array  =np.array(h5file['tissue_array'][:])
    embedding = np.array(h5file['embedding'][:])

    output = h5py.File('metrics.h5','w')

            
    n = 128

    print(f'Calculating embedding trustworthiness from {n} pixels')
    sampled_rows = np.random.randint(tissue_array.shape[0], size = n)
    trust = trustworthiness(tissue_array[sampled_rows,:], embedding[sampled_rows,:], metric = args.metric)
    print(f'Trustworthiness = {trust}')

    output.create_dataset('embedding_trust', data = trust)

    sampled_rows = np.random.randint(tissue_array.shape[0], size = n)
    print('Calculating distance matrix in high dimensional space')
    original_dist = pdist(tissue_array[sampled_rows,:], metric = args.metric)

    output.create_dataset(f'original_dist_{args.metric}', data = original_dist)

    print('Calculating distance matrix in low dimensional space')
    embedding_dist = pdist(embedding[sampled_rows,:], metric = 'euclidean')
    output.create_dataset('embedding_dist', data = embedding_dist)

    for c in h5file['colors'].keys():
        rgb = np.array(h5file['colors'][c])
        print(f'Calculating perceptual trustworthiness of {c} from {n} pixels')
        ptrust = perceptual_trustworthiness(tissue_array[sampled_rows,:], rgb[sampled_rows,:], X_metric = args.metric)
        print(f'Perceptual trustworthiness of {c} = {ptrust}')
        output_colors = output.create_group(c)
        output_colors.create_dataset('perceptual_trust', data = ptrust)

        e_ptrust = perceptual_trustworthiness(embedding[sampled_rows,:], rgb[sampled_rows,:])
        print(f'Embedding perceptual trustworthiness of {c} = {e_ptrust}')
        output_colors.create_dataset('perceptual_embedding_trust', data = e_ptrust)

        print('Calculating distance matrix in perceptual space')
        perceptial_dist = pdist(rgb[sampled_rows,:], metric = delta_e_cie2000_metric)
        output_colors.create_dataset('perceptial_dist', data = perceptial_dist)

        sp_array_color = spearmanr(original_dist, perceptial_dist)
        print(sp_array_color)

        sp_array_color = spearmanr(embedding_dist, perceptial_dist)
        print(sp_array_color)

        sp_array_color = spearmanr(original_dist, embedding_dist)
        print(sp_array_color)
        #embedding_v_perceptial = mantel.test(low_d_dist,perceptial_dist)
        #tissue_v_perceptial = mantel.test(high_d_dist, perceptial_dist)
        #print('Mantel test of embedding vs deltE:')
        #print(embedding_v_perceptial)
        #output.writerow([args.input, f'{c}_lowd_mantel_p', embedding_v_perceptial.p])
        #print(f'p<0.05: {embedding_v_perceptial.p < 0.05}')
        #print('Mantel test of tissue array vs deltE:')
        #print(tissue_v_perceptial)
        #print(f'p<0.05: {tissue_v_perceptial.p < 0.05}')
        #newp = Path.joinpath(p.parent, p.stem+ c+ "_array_v_perceptial" +p.suffix)            
        #fig,ax = plt.subplots()
        #ax = plt.scatter(high_d_dist, perceptial_dist, marker = ".", s = 0.1)
        #plt.xlabel(f'{args.metric} distance')
        #plt.ylabel('deltaE 2000')
        #fig.savefig(newp)
        #newp = Path.joinpath(p.parent, p.stem+ c+ "_embedding_v_perceptial" +p.suffix) 
        #fig,ax = plt.subplots()
        #ax = plt.scatter(low_d_dist, perceptial_dist, marker = ".", s = 0.1)
        #plt.xlabel(f'euclidean distance')
        #plt.ylabel('deltaE 2000')
        #fig.savefig(newp)

    h5file.close()
    output.close()

if __name__ == "__main__":
    main()  