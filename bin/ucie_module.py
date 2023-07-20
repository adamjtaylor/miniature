import numpy as np
import pandas as pd
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from scipy.spatial import ConvexHull, Delaunay, distance
from scipy.optimize import minimize
import estimagic as em
import matplotlib.pyplot as plt
import math
import h5py
from PIL import Image
import multiprocessing
import pyvista as pv


# Perpare the target colourspace (defauly is full sRGB)
def generate_lab_grid(
        l_lims = [0,100], 
        a_lims = [-128,127], 
        b_lims = [-128,127], 
        saturation_lims = [0,1],
        hue_lims = [0,365]
        ):
    rgb_values = np.linspace(0,256,32)
    rgb_grid =  np.array(np.meshgrid(rgb_values, rgb_values, rgb_values)).T.reshape(-1, 3)

    lab_grid = np.apply_along_axis(lambda x: convert_color(sRGBColor(x[0],x[1],x[2]),LabColor).get_value_tuple(), 1, rgb_grid/256)
    hsl_grid = np.apply_along_axis(lambda x: convert_color(sRGBColor(x[0],x[1],x[2]),HSLColor).get_value_tuple(), 1, rgb_grid/256)

    filtered_grid = lab_grid[ \
        (lab_grid[:,0] >= l_lims[0]) & \
        (lab_grid[:,0]  <= l_lims[1]) & \
        (lab_grid[:,1] >= a_lims[0]) & \
        (lab_grid[:,1] <= a_lims[1]) & \
        (lab_grid[:,2] >= b_lims[0]) & \
        (lab_grid[:,2] <= b_lims[1]) & \
        (hsl_grid[:,1] >= saturation_lims[0]) & \
        (hsl_grid[:,1] <= saturation_lims[1]) & \
        (hsl_grid[:,0] >= hue_lims[0]) & \
        (hsl_grid[:,0] <= hue_lims[1])   \
        ]
    return(filtered_grid)

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

def in_hull_pv(p, hull):
    """
    Test if points in `p` are within `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull should be a pyvista object where the delaunay_3d function has been applied.

    Returns an array of 1 (is in hull) or 0 (is outside hull)
    """

    p_space = pv.PolyData(p)
    is_inside = p_space.select_enclosed_points(hull,check_surface=False)['SelectedPoints']

    return(is_inside)


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
    #is_in_hull = np.array(in_hull_pv(t,target_hull), dtype=bool)

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


def ucie(embedding,):

    print('Generating target colorspace')
    lab_polygon = generate_lab_grid()
   
    lims = np.quantile(embedding,[0.1,0.9],axis=0)

    outliers = (embedding[:,0]> lims[0][0]) & (embedding[:,1]> lims[0][1]) & (embedding[:,2]> lims[0][2]) & (embedding[:,0] < lims[1][0]) & (embedding[:,1] < lims[1][1]) & (embedding[:,2] < lims[1][2])

    filtered = embedding[outliers]


    print('Finding convex hull of the embedding')
    embedding_polygon = filtered[ConvexHull(filtered).vertices]

    print('Estimating initial transformation')
    initial = guess_initial(embedding_polygon,lab_polygon)

    print('Calculating Delaunay triangulation of the target colorspace')
    lab_delaunay = Delaunay(lab_polygon)
    #lab_delaunay = pv.PolyData(lab_polygon).delaunay_3d(alpha = 1)
    #lab_surface = lab_delaunay.extract_surface()

    params = [initial[1], initial[2][0], initial[2][1], initial[2][2],initial[0][0], initial[0][1], initial[0][2]]

    lb = np.array(params) - 0.000001
    ub = params
    #ub[1] = ub[2] = ub[3] = np.round(360*(math.pi/180),5)
    ub[1] = ub[2] = ub[3] = 6.63225

    print('Running optimisation')
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

    print('Applying the optimal transform')
    p, r, s, t, is_in_hull, outside, inside = transform(best, embedding_polygon, lab_polygon, lab_delaunay)
   

    # Get the number of available CPU cores
    num_processes = multiprocessing.cpu_count() - 1
    print(f'Using {num_processes} cores ')

    # Create a multiprocessing Pool with the number of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Apply the function to the list of items using the Pool
    print(f'Assigning colours')
    rgb = list(pool.map(embedding_to_lab_to_rgb, t))

    # Close the Pool and wait for all the processes to complete
    pool.close()
    pool.join()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(
        embedding[:,0], 
        embedding[:,1], 
        embedding[:,2], 
        s = 1,
        edgecolors = 'none'
        )
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.savefig('initial_embedding.png')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(
        t[:,0], 
        t[:,1], 
        t[:,2], 
        c = rgb,
        s = 1,
        edgecolors = 'none'
        )
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.savefig('optimized_embedding.png')

    print('UCIE complete')

    return(rgb)