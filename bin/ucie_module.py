import numpy as np
import pandas as pd
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from scipy.spatial import ConvexHull, Delaunay, distance
from scipy.optimize import minimize
import estimagic as em
import matplotlib.pyplot as plt
import math
import h5py
from PIL import Image




n = 1
# Make RGB grid 

## Generate a one-dimensional array of coordinates ranging between 0 and 256
coords = np.arange(0, 257, 32)

## Use the meshgrid function to create all unique combinations of the coordinates
xx, yy, zz = np.meshgrid(coords, coords, coords)

## Create a boolean mask to select only the coordinates on the edges of the cube
edges = np.logical_or.reduce((xx == 0, xx == 256, yy == 0, yy == 256, zz == 0, zz == 256))

## Use the mask to select only the edge coordinates and reshape them into a three-column matrix
rgb_polygon = np.column_stack((xx[edges], yy[edges], zz[edges]))

# Convert RGB grid to CIELAB coordinates

def rgb_to_lab(rgb):
    rgb = rgb/256
    rgb = sRGBColor(rgb[0],rgb[1],rgb[2])
    lab = convert_color(rgb, LabColor)
    lab = lab.get_value_tuple()
    return lab


lab_polygon = np.array(list(map(rgb_to_lab, rgb_polygon)))


# Filter out dark colours
bright = lab_polygon[:,1] > 10
# Define transformation functions


def rotation(convex_cloud, rot_l, rot_a, rot_b):
    # Convert convex cloud to numpy array
    convex_cloud = np.array(convex_cloud)

    # Rotate around x-axis
    rot_matrix = np.array([[1, 0, 0],
                           [0, np.cos(rot_l), -np.sin(rot_l)],
                           [0, np.sin(rot_l), np.cos(rot_l)]])
    convex_cloud = np.dot(convex_cloud, rot_matrix)

    # Rotate around y-axis
    rot_matrix = np.array([[np.cos(rot_a), 0, np.sin(rot_a)],
                           [0, 1, 0],
                           [-np.sin(rot_a), 0, np.cos(rot_a)]])
    convex_cloud = np.dot(convex_cloud, rot_matrix)

    # Rotate around z-axis
    rot_matrix = np.array([[np.cos(rot_b), -np.sin(rot_b), 0],
                           [np.sin(rot_b), np.cos(rot_b), 0],
                           [0, 0, 1]])
    convex_cloud = np.dot(convex_cloud, rot_matrix)

    return convex_cloud.tolist()

def translation(convex_cloud, tr_l, tra, tr_b):
    # Convert convex cloud to numpy array
    convex_cloud = np.array(convex_cloud)

    # Translate the convex cloud
    translation_vector = np.array([tr_l, tra, tr_b])
    convex_cloud += translation_vector

    return convex_cloud.tolist()

def scaling(convex_cloud, s):
    # Convert convex cloud to numpy array
    convex_cloud = np.array(convex_cloud)

    # Scale the convex cloud
    scaling_vector = np.array([s, s, s])
    convex_cloud *= scaling_vector

    return convex_cloud.tolist()


# Function to decide if a point is within the hull

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

# Combine in a transformation function

def transform(params, source, target,target_hull):
    scale_factor = params[0]
    rot1 = params[1]
    rot2 = params[2]
    rot3 = params[3]
    t1 = params[4]
    t2 = params[5]
    t3 = params[6]
    
    r = np.array(rotation(source,rot1,rot2, rot3))
    s = np.array(scaling(r,scale_factor))
    t = np.array(translation(s,t1,t2,t3))

    is_in_hull = np.apply_along_axis(in_hull,1, t,hull = target_hull)
    outside = np.delete(t, np.where(is_in_hull)[0],0)
    inside = t[np.where(is_in_hull)[0]]
    #print(f'{len(outside)}/{len(is_in_hull)}')
    return(source, r,s,t, is_in_hull, outside, inside)

# Write an objective function for minimization

def objective(params, source, target, target_hull):
    p, r,s,t, is_in_hull, outside, inside = transform(params,source,target, target_hull)
    scale_factor = params[0]

    #if len(outside) < 0.2 * len(p):
    if len(outside) == 0:
        dist = 0
    else:
        dist_mat = distance.cdist(outside,target)
        dist = np.min(dist_mat, axis=1)
    
    a = 1
    f = (a*scale_factor) - np.sum(np.square(dist))
    print(f'Loss: {np.round(-f,4)}', end = '\r')
    return(-f)

def guess_initial(points,polygon):
    centroid_polygon = np.mean(polygon, axis = 0)
    centroid_points  = np.mean(points,  axis = 0)
    initial_translation = centroid_polygon- centroid_points

    min_points = np.min(points, axis = 0)
    max_points = np.max(points, axis = 0)
    range_points = max_points - min_points

    min_polygon = np.min(polygon, axis = 0)
    max_polygon = np.max(polygon, axis = 0)
    range_polygon = max_polygon - min_polygon

    initial_scale = np.min(range_polygon/range_points)
    #initial_scale = 0.5

    initial_rot = [0,0,0]
    r = np.array(rotation(points, initial_rot[0], initial_rot[1], initial_rot[2]))
    s = np.array(scaling(r,initial_scale))

    centroid_points  = np.mean(s,  axis = 0)
    initial_translation = centroid_polygon- centroid_points

    return(initial_translation,initial_scale,initial_rot)

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

def embedding_to_lab_to_rgb(x):
        #print("Converting embedding to LAB colour")
        lab = LabColor(x[0], x[1], x[2])
        #print("Converting LAB to RGB for display")
        rgb = convert_color(lab, sRGBColor)
        #print("Clamping RGB values")
        clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
        return clamped_rgb.get_value_tuple()


def ucie(embedding):
   
    lims = np.quantile(embedding,[0,1],axis=0)

    outliers = (embedding[:,0]> lims[0][0]) & (embedding[:,1]> lims[0][1]) & (embedding[:,2]> lims[0][2]) & (embedding[:,0] < lims[1][0]) & (embedding[:,1] < lims[1][1]) & (embedding[:,2] < lims[1][2])

    filtered = embedding[outliers]

    embedding_polygon = filtered[ConvexHull(filtered).vertices]

    initial = guess_initial(embedding_polygon,lab_polygon)

    lab_delaunay = Delaunay(lab_polygon)

    params = [initial[1], initial[2][0], initial[2][1], initial[2][2],initial[0][0], initial[0][1], initial[0][2]]

    lb = np.array(params) - 0.000001
    ub = params
    #ub[1] = ub[2] = ub[3] = np.round(360*(math.pi/180),5)
    ub[1] = ub[2] = ub[3] = 6.63225

    res = em.minimize(
        criterion=objective,
        params=params,
        criterion_kwargs={'source':embedding_polygon,'target':lab_polygon,'target_hull':lab_delaunay},
        algorithm="scipy_neldermead",
        #numdiff_options={"n_cores": 4},
        algo_options = {'stopping.max_iterations':1000},
        soft_lower_bounds=lb,
        soft_upper_bounds=ub,
        multistart = True,
        multistart_options={"n_samples":25, "n_cores": 4}
    )

    print(res)

    best = res.params

    embedding_polygon = embedding

    p, r,s,t, is_in_hull, outside, inside = transform(best, embedding_polygon, lab_polygon, lab_delaunay)
   

    rgb = list(map(embedding_to_lab_to_rgb, t))

    return(rgb)