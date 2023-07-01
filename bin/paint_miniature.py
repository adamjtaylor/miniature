#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from cmath import isinf
import tifffile
import zarr
import sys
import umap
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from pathlib import Path
from PIL import Image
import mantel
import csv
from itertools import repeat
from ucie_module import ucie
        
def pull_pyramid(input, level):
    print("Loading image")
    tiff = tifffile.TiffFile(input, is_ome=False)
    tiff_levels = tiff.series[0].levels
    highest_level_tiff = tiff_levels[level]
    zarray = zarr.open(highest_level_tiff.aszarr())
    print("Opened image pyramid level:", level)
    print("Image dimensions:", zarray.shape)
    return(zarray)
    
def remove_background(zarray, pseudocount):
    print("Finding background")
    sum_image = np.array(zarray).sum(axis = 0)
    print(f'Using pseudocount of {pseudocount}')
    log_image = np.log2(sum_image + pseudocount)
    thresh = threshold_otsu(log_image[log_image > np.log2(pseudocount)])
    binary = log_image > thresh
    cleaned = remove_small_objects(binary)
    print("Background removed")
    def get_tissue(x):
        return x[cleaned]
    tissue_array = list(map(get_tissue, zarray))
    tissue_array = np.array(tissue_array).T
    print("Selected", tissue_array.shape[0], "of", zarray.shape[1]*zarray.shape[2], "pixels as tissue")
    print("Pixels x channels matrix prepared")
    print(tissue_array.shape)
    return tissue_array,cleaned
    
def keep_background(zarray):
    print("Preserving background")
    shape = zarray.shape[1:]
    everything = np.ones(shape, dtype=bool)
    def get_all(x):
        return x[everything]
    tissue_array = list(map(get_all, zarray))
    tissue_array = np.array(tissue_array).T
    print("Pixels x channels matrix prepared")
    print(tissue_array.shape)
    return tissue_array,everything
    
def run_pca(tissue_array, n):
    reducer = PCA(
        n_components = n
        )
    print("Running PCA")
    embedding = reducer.fit_transform(tissue_array)
    return(embedding)

def run_umap(tissue_array, n, metric):
    reducer = umap.UMAP(
        n_components = n,
        metric = metric,
        #min_dist = 0,
        verbose = True)
    print("Running UMAP")
    embedding = reducer.fit_transform(tissue_array)
    return(embedding)
    
def run_tsne(tissue_array, n, metric):
    reducer = TSNE(
        n_components = n,
        metric = metric,
        verbose = True)
    print("Running t-SNE")
    embedding = reducer.fit_transform(tissue_array)
    return(embedding)
    
def embedding_to_lab_to_rgb(x):
        #print("Converting embedding to LAB colour")
        lab = LabColor(x[2], x[0], x[1])
        #print("Converting LAB to RGB for display")
        rgb = convert_color(lab, sRGBColor)
        #print("Clamping RGB values")
        clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
        return clamped_rgb.get_value_tuple()

def embedding_to_rgb(x):
        #print("Converting embedding to RGB colour")
        rgb = sRGBColor(x[2], x[0], x[1])
        #print("Clamping RGB values")
        clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
        return clamped_rgb.get_value_tuple()
    
def assign_colours_lab(embedding):
    print("Assigning colours to pixels embedding in low dimensional space")
    print("Rescaling embedding")
    scaler = MinMaxScaler(feature_range = (-128,127))
    dim1 = scaler.fit_transform(embedding[:,0].reshape(-1,1))
    dim2 = scaler.fit_transform(embedding[:,1].reshape(-1,1))
    scaler = MinMaxScaler(feature_range = (0,100))
    dim3 = scaler.fit_transform(embedding[:,2].reshape(-1,1))
    
    rescaled_embedding = np.concatenate((dim1,dim2,dim3), axis = 1)
    rescaled_embedding_list = rescaled_embedding.tolist()
    
    rgb = list(map(embedding_to_lab_to_rgb, rescaled_embedding_list))
    rgb = np.array(rgb)
    print("LAB Color assigned")
    return(rgb)

def assign_colours_rgb(embedding):
    print("Assigning colours to pixels embedding in low dimensional space")
    print("Rescaling embedding")
    scaler = MinMaxScaler(feature_range = (0,1))
    dim1 = scaler.fit_transform(embedding[:,0].reshape(-1,1))
    dim2 = scaler.fit_transform(embedding[:,1].reshape(-1,1))
    dim3 = scaler.fit_transform(embedding[:,2].reshape(-1,1))
    
    rescaled_embedding = np.concatenate((dim1,dim2,dim3), axis = 1)
    rescaled_embedding_list = rescaled_embedding.tolist()
    
    # TODO use a different mapping function
    rgb = list(map(embedding_to_rgb, rescaled_embedding_list))
    rgb = np.array(rgb) 
    print(rgb.max())
    print("RGB assigned")
    return(rgb)

imgFolder = "./bin/colormaps/"

dimensions = {
    'width': 512,
    'height': 512
}

colormaps = {
    'BREMM':imgFolder + "bremm.png",
    'CUBEDIAGONAL':imgFolder + "cubediagonal.png",
    'SCHUMANN':imgFolder + "schumann.png",
    'STEIGER':imgFolder + "steiger.png",
    'TEULING2':imgFolder + "teulingfig2.png",
    'ZIEGLER':imgFolder + "ziegler.png"
}



ranges = {
    'x': [0, 1],
    'y': [0, 1]
}


def getScaledX(x):
    val = ((x+1) - (ranges['x'][0]+1)) / ((ranges['x'][1]+1) - (ranges['x'][0]+1))
    return round(val * (dimensions['width']-1));

def getScaledY(y):
    val = ((y+1) - (ranges['y'][0]+1)) / ((ranges['y'][1]+1) - (ranges['y'][0]+1))
    return round(val * (dimensions['width']-1))


def getColor(e,colormap_ar):
    scaled_x = getScaledX(e[0])
    scaled_y = getScaledY(e[1])
    rgb = colormap_ar.getpixel((scaled_x,scaled_y))
    return(rgb)


def assign_colours_2d(embedding,colormap_ar):
    print("Assigning colours to pixels embedding in low dimensional space")
    print("Rescaling embedding")
    scaler = MinMaxScaler(feature_range = (0,1))
    dim1 = scaler.fit_transform(embedding[:,0].reshape(-1,1))
    dim2 = scaler.fit_transform(embedding[:,1].reshape(-1,1))
    rescaled_embedding = np.concatenate((dim1,dim2), axis = 1)
    rescaled_embedding_list = rescaled_embedding.tolist()
    
    rgb = list(map(getColor, rescaled_embedding_list, repeat(colormap_ar)))
    rgb = np.array(rgb) / 255
    return(rgb)

def make_rgb_image(rgb, mask):
    print("Painting miniature")
    rgb_shape = list(mask.shape)
    rgb_shape.append(3)
    rgb_image = np.zeros(rgb_shape)
    rgb_image[mask] = rgb
    rgb_image = (rgb_image * 255).astype(np.uint8)
    print(rgb_image.shape)
    print(f'RBG max = {rgb_image.max()}')
    print(f'RBG min = {rgb_image.min()}')
    return(rgb_image)

def calc_metrics(high_d, low_d):
    K = 30

    numerical_trustworthiness = umap.validation.trustworthiness_vector(source = high_d, embedding = low_d, max_k = K)
    
def plot_embedding(embedding, rgb, output):
    p = Path(output)
    newp = Path.joinpath(p.parent, p.stem+'-embedding' +p.suffix)
    fig = plt.figure()
    if embedding.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(
            embedding[:,0], 
            embedding[:,1], 
            embedding[:,2], 
            c = rgb,
            s = 1,
            edgecolors = 'none'
            )
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
    if embedding.shape[1] == 2:
        ax = fig.add_subplot()
        ax.scatter(
            embedding[:,0], 
            embedding[:,1], 
            c = rgb,
            s = 1,
            edgecolors = 'none'
            )
    plt.savefig(newp)
    
def save_data(path, args, tissue_array, mask, embedding, rgb, rgb_image):
    print("Saving log file")
    h5file = h5py.File(path, 'w')
    #h5file.create_dataset('args', data = args)
    h5file.create_dataset('mask', data = mask)
    #h5file.create_dataset('tissue_array', data = tissue_array)
    #h5file.create_dataset('embedding', data = embedding)
    #h5file.create_dataset('rgb_array', data = rgb)
    #h5file.create_dataset('rgb_image', data = rgb_image)
    h5file.close()
    print(h5file)


def main():

    parser = argparse.ArgumentParser(description = 'Paint a miniature from an OME-TIFF')
    
    parser.add_argument('input',
                        type=str,
                        help=' a file name, seekable binary stream, or FileHandle for an OME-TIFF')
    
    parser.add_argument('output',
                        type=str,
                        default='data/miniature.png',
                        help='file name of output')
    
    parser.add_argument('-l', '--level',
                        type=int,
                        dest='level',
                        default=-1,
                        help='image pyramid level to use. defaults to -1 (highest)')
    
    parser.add_argument('--keep_bg',
                        default=False,
                        action='store_true',
                        help="Don't perform background removal")
                        
    parser.add_argument('--dimred',
                        type=str,
                        dest='dimred',
                        default='umap',
                        help='Dimensionality reduction method [umap, tsne, pca]') 

    parser.add_argument('--n_components',
                        type=int,
                        dest='n_components',
                        default=3,
                        help='Number of components for dim red') 
    parser.add_argument('--log',
                        default = False,
                        action= 'store_true',
                        help = 'Log10 transfrom the data with a pseudocount set with --pseudocount (default 1)')
    parser.add_argument('--scaler',
                        choices = ['NoScaler', 'MinMaxScaler', 'StandardScaler','RobustScaler'],
                        help = 'Scaling fucntion to use')
    parser.add_argument('--pseudocount',
                        type = float,
                        default = 1.,
                        help = 'Pseudocount for log transformation')
    parser.add_argument('--metric',
                        type = str,
                        default = "euclidean",
                        choices= ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule'],
                        help = 'Metric to  use for UMAP and tSNE')

                        
    parser.add_argument('--save_data',
                    dest='save_data',
                    default=False,
                    action = 'store_true',
                    help='Save a h5 file with intermediate data')
                    
    parser.add_argument('--plot_embedding',
                    default=False,
                    action='store_true',
                    help='Save a figure of the embedding')
    parser.add_argument('--existing',
                    type=str,
                    help='Path of a h5 file output by miniature to recolour an existing embedding')
    parser.add_argument('--colormap',
                    type=str,
                    dest='colormap',
                    default='ALL',
                    choices= ['ALL', 'BREMM', 'SCHUMANN', 'STEIGER', 'TEULING2', 'ZIEGLER','CUBEDIAGONAL','LAB','RGB','UCIE'],
                    help='Colormap to use. For 2D plots this can be BREMM, SCHUMANN, STEIGER, TEULING2, ZEIGLER. For 3D this can be RGB or LAB')
    
    args = parser.parse_args()

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    if args.existing:
        path = Path(args.existing)
        with h5py.File(path, 'r') as h:
            h5 = h5py.File(h, 'r')
            embedding = h5['embedding']
        print(embedding.shape())
        exit

    p = Path(args.output)
    logp = Path.joinpath(p.parent, p.stem+"-log.csv" )
    output = csv.writer(open(logp, 'w'))
    output.writerow(['input', 'key', 'value'])

    if args.save_data:
        h5_path = Path(args.output)
        h5_path = Path.joinpath(h5_path.parent, h5_path.stem+ ".h5")
        h5file = h5py.File(h5_path, 'w')
        h5color = h5file.create_group('colors')
    
    zarray = pull_pyramid(args.input, args.level)

    output.writerow([args.input, 'zarray_shape', zarray.shape])
    
    if zarray.shape[0] == 3:
        rgb_image = np.moveaxis(zarray, 0, -1)
        print("Saving image as " + args.output)
        output_path = args.output
        im = Image.fromarray(rgb_image, 'RGB')
        im.save(output_path)
        if args.save_data:
            h5file.create_dataset('rgb_image', data = rgb_image)
    else: 
        if args.keep_bg:
            tissue_array, mask = keep_background(zarray)
        elif args.keep_bg == False:
            print('Removing background')
            tissue_array, mask = remove_background(zarray, args.pseudocount)
        else:
            tissue_array, mask = keep_background(zarray)

        if args.save_data:
            h5file.create_dataset('mask', data = mask)
        
        if args.log:
            tissue_array = np.log10(tissue_array+args.pseudocount)

        def NoScaler (x):
            return x

        FUNCTION_MAP = {'MinMaxScaler' : MinMaxScaler,
                'StandardScaler' : StandardScaler,
                'RobustScaler': RobustScaler,
                'NoScaler': NoScaler
        }

        if args.scaler:
            if args.scaler == "NoScaler":
                pass
            else:
                scaling_func = FUNCTION_MAP[args.scaler]
                print(scaling_func)
                tissue_array = scaling_func().fit_transform(tissue_array)

        if args.save_data:
            h5file.create_dataset('tissue_array', data = tissue_array)
        
        output.writerow([args.input, 'tissue_array_shape', tissue_array.shape])

        if args.dimred == 'tsne':
            embedding = run_tsne(tissue_array, args.n_components, args.metric)
        if args.dimred == 'umap':
            embedding = run_umap(tissue_array, args.n_components, args.metric)
        if args.dimred == 'pca':
            embedding = run_pca(tissue_array, args.n_components)

        if args.save_data:
            h5file.create_dataset('embedding', data = embedding)

        if args.colormap == "ALL" and args.n_components == 3:
            selected_colormap = ['LAB','RGB','UCIE']
        elif args.colormap == "ALL" and args.n_components ==2 :
            selected_colormap = ['BREMM', 'SCHUMANN', 'STEIGER', 'TEULING2', 'ZIEGLER','CUBEDIAGONAL']
        else: 
            selected_colormap = [args.colormap]

        for c in selected_colormap:

            if args.n_components == 3:
                print('Colouring in 3D')
                print(f'Using colormap {c}')
                if c == 'LAB':
                    rgb = assign_colours_lab(embedding)
                    print('Colors assigned')
                elif c == 'UCIE':
                    rgb = ucie(embedding)
                elif c == 'RGB':
                    rgb = assign_colours_rgb(embedding)
                    print('Colors assigned')
                else:
                    print('If n_dimensions = 3, colormap must be LAB or RGB')
            if args.n_components == 2:
                print('Colouring in 2D')
                colormap = colormaps[c]
                colormap_im = Image.open(colormap)
                colormap_ar = colormap_im
                print(f'Loaded colormap {c}')
                rgb = assign_colours_2d(embedding,colormap_ar)
                print('Colors assigned')

            rgb_image = make_rgb_image(rgb, mask)

            if args.save_data:
                h5color.create_dataset(c, data = rgb)

            print(f"Saving image as {c}_{args.output}")
            p = Path(args.output)
            if args.colormap == "ALL":
                newp = Path.joinpath(p.parent, p.stem+"_" + c +p.suffix)
            else:
                newp = p
            im = Image.fromarray(rgb_image, 'RGB')
            im.save(newp)

            if args.plot_embedding:
                plot_embedding(embedding, rgb, newp)

    if args.save_data:
        h5file.close()

    
    print("Complete!")
    
if __name__ == "__main__":
    main()  
