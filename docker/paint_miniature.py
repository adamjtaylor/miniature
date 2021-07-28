#!/usr/bin/env python
# coding: utf-8

# Import libraries

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


path = sys.argv[1]
output = "data/" + sys.argv[2] if len(sys.argv) >=3  else "data/miniature.png"
level = int(sys.argv[3]) if len(sys.argv) >= 4 else -1
remove_bg = sys.argv[4] == True if len(sys.argv) >= 5 else True

print("Loading image")
tiff = tifffile.TiffFile(path, is_ome=False)
tiff_levels = tiff.series[0].levels
highest_level_tiff = tiff_levels[level]

zarray = zarr.open(highest_level_tiff.aszarr())

print("Opened image pyramid level:", level)
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

everything = np.ones_like(binary, dtype=bool)

def get_tissue(x):
    return x[cleaned]

def get_all(x):
    return x[everything]

if remove_bg == False:
    tissue_array = list(map(get_all, zarray))
    print("Preserving background")
    cleaned = everything
elif remove_bg == True:
    tissue_array = list(map(get_tissue, zarray))
    print("Removing background")
else:
    tissue_array = list(map(get_tissue, zarray))
    print("Removing background")

tissue_array = np.array(tissue_array).T


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

print("Saving image as " + output)

imsave(output, rgb_image)

print("Complete!")
