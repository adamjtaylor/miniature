#!/usr/bin/env python
# coding: utf-8

# Import libraries

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

# ## Extract the highest level of the pyramid

# Select the last level of the image pyramid and load as a zarr array. Reshape into a pixels x channels array.

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description = 'Paint a miniature from a streaming s3 object')

parser.add_argument('-i', '--input',
                    type=str,
                    dest='path',
                    help='path to ome-tiff')

parser.add_argument('-o', '--output',
                    type=str,
                    dest='output',
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

args = parser.parse_args()


print("Loading image")
tiff = tifffile.TiffFile(args.path, is_ome=False)
tiff_levels = tiff.series[0].levels
highest_level_tiff = tiff_levels[args.level]

zarray = zarr.open(highest_level_tiff.aszarr())

print("Opened image pyramid level:", args.level)
print("Image dimensions:", zarray.shape)


# # Threshold the image
# Make a sum image. Log this with a pseudocount of the 1st percentile. Otsus threshold

print("Finding background")
sum_image = np.array(zarray).sum(axis = 0)
first_percentile = np.quantile(sum_image, 0.1)

if first_percentile == 0:
    pseudocount = 1
else: pseudocount = first_percentile

log_image = np.log2(sum_image + pseudocount)

thresh = threshold_otsu(log_image)

binary = log_image > thresh

cleaned = remove_small_objects(binary)

everything = np.ones_like(sum_image, dtype=bool)

def get_tissue(x):
    return x[cleaned]

def get_all(x):
    return x[everything]

if args.remove_bg == False:
    tissue_array = list(map(get_all, zarray))
    print("Preserving background")
    cleaned = everything
elif args.remove_bg == True:
    tissue_array = list(map(get_tissue, zarray))
    print("Removing background")
else:
    tissue_array = list(map(get_all, zarray))
    print("Preserving background")

tissue_array = np.array(tissue_array).T

print("Selected", tissue_array.shape[0], "of", zarray.shape[0]*zarray.shape[1], "pixels as tissue")


# ## Perform dimensionality reduction
# Reduce the data from channels x pixels to channels x 3 by UMAP with correlation distance

reducer = umap.UMAP(
    n_components = 3,
    metric = "correlation",
    min_dist = 0,
    verbose = True)

print("Running UMAP")

embedding = reducer.fit_transform(tissue_array)

# # Set the colours
print("Painting miniature")

scaler = MinMaxScaler(feature_range = (-128,127))
dim1 = scaler.fit_transform(embedding[:,0].reshape(-1,1))
dim2 = scaler.fit_transform(embedding[:,1].reshape(-1,1))
scaler = MinMaxScaler(feature_range = (10,80))
dim3 = scaler.fit_transform(embedding[:,2].reshape(-1,1))

rescaled_embedding = np.concatenate((dim1,dim2,dim3), axis = 1)
rescaled_embedding_list = rescaled_embedding.tolist()

def umap_to_lab_to_rgb(x):
    lab = LabColor(x[2], x[0], x[1])
    rgb = convert_color(lab, sRGBColor)
    clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
    return clamped_rgb.get_value_tuple()

rgb = list(map(umap_to_lab_to_rgb, rescaled_embedding_list))


rgb_shape = list(cleaned.shape)
rgb_shape.append(3)
rgb_image = np.zeros(rgb_shape)
rgb_image[cleaned] = np.array(rgb)

print("Saving image as " + args.output)
output_path = "data/" + args.output

imsave(output_path, rgb_image)

print("Complete!")
