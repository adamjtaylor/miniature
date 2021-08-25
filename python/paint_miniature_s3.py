print("Setting up")

import argparse
import boto3
import io
import tifffile
from tifffile import TiffFile
import zarr
import numpy as np
import zarr
import sys
import umap
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from pathlib import Path

parser = argparse.ArgumentParser(description = 'Paint a miniature from a streaming s3 object')
parser.add_argument('-b', '--bucket',
                    dest='bucket',
                    type=str,
                    help='name of a s3 bucket that your profile has access to')
parser.add_argument('-k', '--key',
                    type=str,
                    dest='key',
                    help='key for an object in the s3 bucket defined by --bucket. Must be a .ome.tiff file')
parser.add_argument('-p', '--profile',
                    type=str,
                    dest='profile',
                    help='aws profile to use')
parser.add_argument('-l', '--level',
                    type=int,
                    dest='level',
                    default=-1,
                    help='image pyramid level to use. defaults to -1 (highest)')
parser.add_argument('--s3_bucket_type',
                    type=str,
                    dest='s3_bucket_type',
                    default="aws",
                    help='S3 bucket type, [aws, gcs]')

args = parser.parse_args()


# Define a streaming s3 object file class S3File(io.RawIOBase):
class S3File(io.RawIOBase):
    """
    https://alexwlchan.net/2019/02/working-with-large-s3-objects/
    """
    def __init__(self, s3_object):
        self.s3_object = s3_object
        self.position = 0

    def __repr__(self):
        return "<%s s3_object=%r>" % (type(self).__name__, self.s3_object)

    @property
    def size(self):
        return self.s3_object.content_length

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence (%r, should be %d, %d, %d)" % (
                whence, io.SEEK_SET, io.SEEK_CUR, io.SEEK_END
            ))

        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        if size == -1:
            # Read to the end of the file
            range_header = "bytes=%d-" % self.position
            self.seek(offset=0, whence=io.SEEK_END)
        else:
            new_position = self.position + size

            # If we're going to read beyond the end of the object, return
            # the entire object.
            if new_position >= self.size:
                return self.read()

            range_header = "bytes=%d-%d" % (self.position, new_position - 1)
            self.seek(offset=size, whence=io.SEEK_CUR)

        return self.s3_object.get(Range=range_header)["Body"].read()

    def readable(self):
        return True

# Stream the highest level image
print("Loading image")

if args.s3_bucket_type == "aws":
  session = boto3.session.Session(profile_name=args.profile)
  s3 = session.client('s3')
  s3_resource = session.resource('s3')

if args.s3_bucket_type == "gcs":
  print("Accessing GCS resource")
  s3_resource = boto3.resource("s3", region_name = "auto", endpoint_url = "https://storage.googleapis.com", 
    aws_access_key_id = os.environ.get('GC_KEY_ID')
    aws_secret_access_key = os.environ.get('GC_KEY_SECRET')
    )

print("Getting object")
s3_obj = s3_resource.Object(bucket_name=args.bucket, key=args.key)
print("Creating streaming s3 file")
s3_file = S3File(s3_obj)

print("Loading pyramid level", args.level)
with TiffFile(s3_file, is_ome=False) as tif:
  s = tif.series[0].levels[args.level]
  z = zarr.open(s.aszarr())

print("Image loaded with dimensions:", z.shape)
# Remove background

print("Removing background")

print("Bring array into memory")

a = np.array(z)

print("Getting sum image")

sum_image = a.sum(axis = 0)

pseudocount = 1
print("Log transform")

log_image = np.log2(sum_image + pseudocount)
print("Finding Otsu's threshold")

thresh = threshold_otsu(log_image)

binary = log_image > thresh
print("Removing small objects")

cleaned = remove_small_objects(binary)

def get_tissue(x):
    return x[cleaned]
    
print("Generating tissue array")

tissue_array = list(map(get_tissue, a))
tissue_array = np.array(tissue_array).T

print("Selected", tissue_array.shape[0], "of", a.shape[1]*a.shape[2], "pixels as tissue")
print("Pixels x channels matrix prepared")
print(tissue_array.shape)

# Dim reduction

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

print("Saving output")

output = Path(args.key)
output = output.with_suffix('.png')
output = "output/"+ output.name

print(output)

imsave(output, rgb_image)

print("Complete!")
