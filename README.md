# Miniature

> *Miniature (illuminated manuscript)*, a small illustration used to decorate an illuminated manuscript

An approach using dimensionality reduction to create thumnails for high-dimensional imaging.  
*Miniature* enables rapid visual assesment of molecular heterogneity within highly multiplexed images of complex tissues.

*Miniature* images embeds each pixel into low dimensional space by corelation distance, and colours them by conversion of their position in low-D space to LAB colour. Therefore areas of similar colour can be expected to have comperable marker expression (within a single image).

![image](https://user-images.githubusercontent.com/14945787/127400268-b6345cf4-a90c-4d77-9f83-6889de6763a5.png)

- Load highest (or specified) level of image pyramid
- Background removal by Otsu's threshold (optional)
- Remove small objects (Not currently implemented in R version)
- Reduce each pixel from n-D to 3-D by UMAP with correlation distance 
- Colour pixels by conversion of position in low-D space to LAB colour

## Docker

The docker-comppose.yml expects images to be avaliable in `../data` (a folder in the current `miniature` dir called `data`)
Images should be an multichannel `ome.tiff` containing a image pyramid. 
Output as a `.png`, `.jpeg` or `.tif`/`.tiff`

Clone the repository
```
git clone https://github.com/adamjtaylor/miniature
cd miniature
mkdir data
```
Run the docker container
```
cd docker
sudo docker-compose run --rm app
```

Or from the docker image
```
docker run -it --rm --platform linux/amd64 -v <local-path>:/data adamjtaylor/htan-artist
```

Once in the container run
```
python paint_miniature.py data/<input-file-name> <output-file-name>
```

For example

```
python paint_miniature.py 'data/HTA9_1_BA_L_ROI04.ome.tif' 'miniature.jpg'

```

Optional arguments allow for changing level used, preserving background, saving the 3D embedding plot, and saving the intermediate data (tissue mask, data matrix, embedding and colours as h5. Optionally t-SNE can be used but this is slower than UMAP

For example, to paint a miniature on the second higest level, preserving the background, using t-SNE and saving both the 3D embedding and intermediate data use

```
python paint_miniature.py 'data/HTA9_1_BA_L_ROI04.ome.tif' 'miniature.jpg' \
     --level -2 --remove_bg True, --dimred tsne --save_data True --plot_embedding True
````


## R
Follow the notebooks in the `notebooks` folder or use the `R/paint_miniature.R` script

## Examples

Image info | Background removed | Background retained |
---- | ---- | --- |
 40 channel CODEX | ![image](https://user-images.githubusercontent.com/14945787/127377527-1d93913e-5ca0-4115-9e78-4fc49fee0d93.png) | ![image](https://user-images.githubusercontent.com/14945787/127377555-f33270f7-bddc-4af1-8ff8-aac46c634a5b.png) |
 12 channel MxIF | ![image](https://user-images.githubusercontent.com/14945787/127377665-fe4a7dbd-2847-4a7c-9688-1928a65159a3.png) | ![image](https://user-images.githubusercontent.com/14945787/127377625-ae88c1da-c647-47f7-ab3d-8f627f2ebf2d.png) |
 28 channel IMC | ![image](https://user-images.githubusercontent.com/14945787/127378069-0f15d759-bb71-4a13-97c4-126efffa60af.png) | ![image](https://user-images.githubusercontent.com/14945787/127378051-634836b3-972f-4bae-bc4a-fdc05ded048b.png) |
 48 channel MIBI | ![image](https://user-images.githubusercontent.com/14945787/127505192-45cf2c1c-2596-4a80-975c-1a926cfdf357.png) | ![image](https://user-images.githubusercontent.com/14945787/127505226-d6d160a6-10c3-4cbf-8bea-82f870932a66.png) |
 12 channel MxIF | ![image](https://user-images.githubusercontent.com/14945787/127377800-c6351c50-f957-4154-8e47-fd83c1c6f202.png) | ![image](https://user-images.githubusercontent.com/14945787/127377769-013602da-17cd-4be2-bf32-c408b400abe0.png) |
 12 channel MxIF | ![image](https://user-images.githubusercontent.com/14945787/127377898-3639b4b1-54c1-4e0a-847a-f87f5fea7527.png) | ![image](https://user-images.githubusercontent.com/14945787/127377986-1bdfc4e1-5b9d-48a9-86ea-3c1e266f7d6e.png) |
 
