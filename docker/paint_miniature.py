#!/usr/bin/env python
# coding: utf-8

import argparse
import tifffile
import zarr
import sys
import umap
import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from sklearn.manifold import TSNE
import h5py
from pathlib import Path



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def pull_pyramid(input, level):
    
    print("Loading image")
    
    tiff = tifffile.TiffFile(input, is_ome=False)
    tiff_levels = tiff.series[0].levels
    highest_level_tiff = tiff_levels[level]
    zarray = zarr.open(highest_level_tiff.aszarr())

    print("Opened image pyramid level:", level)
    print("Image dimensions:", zarray.shape)

    return(zarray)
    
def remove_background(zarray):
    print("Finding background")
    sum_image = np.array(zarray).sum(axis = 0)
    pseudocount = 1
    log_image = np.log2(sum_image + pseudocount)
    thresh = threshold_otsu(log_image)
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
    return(everything)
    tissue_array = list(map(get_all, zarray))
    tissue_array = np.array(tissue_array).T
    print("Pixels x channels matrix prepared")
    print(tissue_array.shape)
    return tissue_array,everything
    
def run_umap(tissue_array):
    reducer = umap.UMAP(
        n_components = 3,
        metric = "correlation",
        min_dist = 0,
        verbose = True)
    print("Running UMAP")
    embedding = reducer.fit_transform(tissue_array)
    return(embedding)
    
def run_tsne(tissue_array):
    reducer = TSNE(
        n_components = 3,
        metric = "correlation",
        square_distances = True,
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
    
def assign_colours(embedding):
    print("Assigning colours to pixels embedding in low dimensional space")
    print("Rescaling embedding")
    scaler = MinMaxScaler(feature_range = (-128,127))
    dim1 = scaler.fit_transform(embedding[:,0].reshape(-1,1))
    dim2 = scaler.fit_transform(embedding[:,1].reshape(-1,1))
    scaler = MinMaxScaler(feature_range = (10,80))
    dim3 = scaler.fit_transform(embedding[:,2].reshape(-1,1))
    
    rescaled_embedding = np.concatenate((dim1,dim2,dim3), axis = 1)
    rescaled_embedding_list = rescaled_embedding.tolist()
    
    rgb = list(map(embedding_to_lab_to_rgb, rescaled_embedding_list))
    rgb = np.array(rgb)
    print("Colours assigned")
    return(rgb)
    
def make_rgb_image(rgb, mask):
    print("Painting miniature")
    rgb_shape = list(mask.shape)
    rgb_shape.append(3)
    rgb_image = np.zeros(rgb_shape)
    rgb_image[mask] = rgb
    return(rgb_image)
    
def plot_embedding(embedding, rgb, output):
    p = Path(output)
    newp = Path.joinpath(p.parent, p.stem+'-embedding' +p.suffix)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(
        embedding[:,0], 
        embedding[:,1], 
        embedding[:,2], 
        c = rgb,
        edgecolors = 'none'
        )
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.savefig(newp)
    
def save_data(output, args, tissue_array, mask, embedding, rgb, rgb_image):
    print("Saving log file")
    p = Path(output)
    p.rename(p.with_suffix('.h5'))
    h5 = h5py.File(p, 'w')
    #h5.create_dataset('args', data = args)
    h5.create_dataset('mask', data = mask)
    h5.create_dataset('tissue_array', data = tissue_array)
    h5.create_dataset('embedding', data = embedding)
    h5.create_dataset('rgb_array', data = rgb)
    h5.create_dataset('rgb_image', data = rgb_image)

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
    
    parser.add_argument('-r', '--remove_bg',
                        type=str2bool,
                        dest='remove_bg',
                        default=True,
                        help='Attempt to remove background (defaults to True)')
                        
    parser.add_argument('--dimred',
                        type=str,
                        dest='dimred',
                        default='umap',
                        help='Dimensionality reduction method [umap, tsne]') 
                        
    parser.add_argument('--save_data',
                    type=str2bool,
                    dest='save_data',
                    default=False,
                    help='Save a h5 file with intermediate data')
                    
    parser.add_argument('--plot_embedding',
                    type=str2bool,
                    dest='plot_embedding',
                    default=False,
                    help='Save a figure of the embedding')
    
    args = parser.parse_args()
    
    zarray = pull_pyramid(args.input, args.level)
    
    if args.remove_bg == False:
        tissue_array, mask = keep_background(zarray)
    elif args.remove_bg == True:
        tissue_array, mask = remove_background(zarray)
    else:
        tissue_array, mask = keep_background(zarray)
    
    if args.dimred == 'tsne':
        embedding = run_tsne(tissue_array)
    if args.dimred == 'umap':
        embedding = run_umap(tissue_array)
        
    rgb = assign_colours(embedding)
    rgb_image = make_rgb_image(rgb, mask)
    
    print("Saving image as " + args.output)
    output_path = "data/" + args.output
    imsave(output_path, rgb_image)
    
    if args.save_data == True:
        save_data(output_path, args, tissue_array, mask, embedding, rgb, rgb_image)
    if args.plot_embedding == True:
        plot_embedding(embedding, rgb, output_path)

    print("Complete!")
    
if __name__ == "__main__":
    main()  
