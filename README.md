# Miniature

> *Miniature (illuminated manuscript)*, a small illustration used to decorate an illuminated manuscript

An approach using dimensionality reduction to create thumnails for high-dimensional imaging

![image](https://user-images.githubusercontent.com/14945787/127029087-b0312bc3-299b-41ae-acf7-ffa226f81218.png)

- Load highest (or specified) level of image pyramid
- Background removal by Otsu's threshold (optional)
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

Once in the container run
```
python paint_miniature.py 'data/<input-file-name>' <output-file-name> <image-pyramid-level-index>
```

For example
```
python paint_miniature.py 'data/HTA9_1_BA_L_ROI04.ome.tif' 'miniature.jpg' -2
````

## R
Follow the notebooks in the `notebooks` folder or use the `R/paint_miniature.R` script

## Examples

![example image](https://github.com/adamjtaylor/miniature/blob/main/outputs/miniature.jpg?raw=true)
![example_image](https://github.com/adamjtaylor/miniature/blob/main/outputs/miniatur-_L2-crc20.png?raw=true)
![IMC image example with miniature](https://github.com/adamjtaylor/miniature/blob/main/outputs/HT060P1_REMAIN_ROI_04-miniature.png?s=100)
![example image](https://github.com/adamjtaylor/miniature/blob/main/outputs/HTA9_1_BA_M_ROI03-miniature.png?raw=true)
![example_image](https://github.com/adamjtaylor/miniature/blob/main/outputs/HTA9_1_POST_M_ROI02-miniature.png?raw=true)

