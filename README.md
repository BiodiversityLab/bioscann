# BIOSCANN
**BIOSCANN**: **BIO**diversity **S**egmentation and **C**lassification with **A**rtificial **N**eural **N**etworks

____
<div style="display: flex; justify-content: center; gap: 60px;">
    <img src="img/Uppsala_universitet_logo.jpg" alt="Image 2" width="50" />
    <img src="img/SciLifeLab_Logotype_Green_POS.png" alt="Image 2" width="200" />
    <img src="img/2560px-Skogsstyrelsen_logo.svg.png" alt="Image 1" width="200" />
</div>

_A collaboration between Uppsala University and The Swedish Forest Agency, funded by the Swedish Government and the SciLifeLab and Wallenberg Data-Driven Life Science (DDLS) program._

___

A BIOSCANN preprint is now available: https://doi.org/10.21203/rs.3.rs-4734879/v1

---

Check out the interactive data product for all of Sweden: https://gee-hcvf-andermann.projects.earthengine.app/view/hcvf-viewer

---

This tutorial shows the basic commands for running the `bioscann` pipeline to make predictions of conservation value across national scale (in this case applied to forests in Sweden). 
This includes the following main steps:
- installation of `bioscann` (installation tested on MacOS)
- preparation of training data
- model-training
- predictions of conservation value

## Download project from GitHub
Download the `bioscann` GitHub repo (download [here](https://github.com/BiodiversityLab/bioscann)) and navigate into the project folder using your command line. For the installation in the next step you need to have the installation manager `conda` installed (download [here](https://docs.anaconda.com/miniconda/)).

## Installation
You can install the software as a conda package by executing the command below in your command line. Before running the command, make sure you are located in the downloaded `bioscann` GitHub directory. For this to work the installation file `environment.yml` needs to be present in your directory.

**Install bioscann (tested on MacOS and Linux):**
```commandline
conda create -y -n bioscann
conda activate bioscann
conda config --add channels conda-forge
conda install -y python=3.8
conda install -y --file requirements_conda.txt
pip install -r requirements_pip.txt
```

From here on out, for any bioscann command you run, make sure you are connected to your bioscann conda environment, by running `conda activate bioscann` in your command line.


## Convert polygon data into individual instances
In this step we create the cropping windows for the individual instances that we use for model training and evaluation. For simplicity we show the `bioscann` wokflow for only one of the five separate bioregional subsets of the data, namely the southern boreal region.

```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/boreal_south \
    --output_path tutorial/processed_geodata/boreal_south/cropped_windows \
    --extent_size 1280 \
    --no_overlap
```


## Download environmental features for each instance
To download the remote-sensing derived environmental features for each instance (1.28 x 1.28 km cropping window), we need access to the data-server. For this it is required to create a user-account at https://www.skogsstyrelsen.se/sjalvservice/karttjanster/geodatatjanster/skaffa-anvandarkonto/ (application form in Swedish). For testing the data-download, it is possible to use a temporary access account with the following login-information:
- username: `skstemp_user`
- password: `S7Qawt3v`

For this script to run, it requires as input the folder with the cropped instances resulting from the previous command.
However, the upcoming data download step will take a long time to finish for all instances compiled in the previous step.
We therefore recommend to instead download the reduced selection of instances, provided in the `tutorial/precompiled` folder.
The input-data file paths provided in this tutorial will usually point to the files in this `tutorial/precompiled` folder, you will have to adjust them accordingly in case you want to use your own files.

**!Note!:** The set of environmental features in this tutorial differs slightly from the set used in the original implementation presented in the manuscript, due to updates on the data server. Instead of 11 channels in the original implementation, the data used in this tutorial consist of only 9 channels.

```commandline
python extract_geo_data.py \
    --output_path tutorial/processed_geodata/boreal_south/boreal_south_geodata  \
    --window_coordinates tutorial/precompiled/processed_geodata/boreal_south/cropped_windows \
    --configuration version_public_sat_2024 \
    --test_config version_2 \
    --testset_size 0.2 \
    --img-size 128 \
    --username skstemp_user \
    --password S7Qawt3v \
    --logging_off \
    --threads 10
```

To learn more about the api download of environmental features, check out the jupyter notebook in this repository [download_geodata_api.ipynb](https://github.com/BiodiversityLab/bioscann/blob/main/download_geodata_api.ipynb).

To explore the downloaded tiff files resulting from the command above, check out the notebook [display_downloaded_tiff.ipynb](https://github.com/BiodiversityLab/bioscann/blob/main/display_downloaded_tiff.ipynb).

## Train model
The next step is training the deep-learning model, which can be time-intensive depending on the size of the input data.
For the entire southern boreal dataset this will likely take more than 1 day, but it can be spead up considerably when running on a machine that can utilize GPU resources (requires manual installation and GPU-mounting of the pytorch machine learning library, not covered in this tutorial).
You can try out training your model using the reduced input data provided at `tutorial/precompiled/processed_geodata`.
However, since this model won't train very well on such little input data, we recommend to use one of our provided trained models (found in `tutorial/precompiled/best_model`) for the following steps.

**! Note !:** We are using the reduced precompiled data (`tutorial/precompiled/processed_geodata`) as input here to speed up model training. In case you want to use your own compiled large dataset, change the input paths for `--dataset`, `--validation`, and `--test_dataset` accordingly.

```commandline
python train_model.py \
    --batch_size 5 \
    --device cpu \
    --dataset tutorial/precompiled/processed_geodata/boreal_south/boreal_south_geodata \
    --validation tutorial/precompiled/processed_geodata/boreal_south/boreal_south_geodata/validation \
    --test_dataset tutorial/precompiled/processed_geodata/boreal_south/boreal_south_geodata/testset/ \
    --plot \
    --experiment_name trained_model_tutorial \
    --epochs 300 \
    --img_size 128 \
    --n_channels_per_layer 100,50,100 \
    --patience 20
```

The model will be stored in the `train/` directory in the main folder under the name provided as `--experiment_name`.

## Extract environmental features for target area for predictions
Now that we have a trained model, let us make predictions for an area of interest. The first step is to define the area and extract all needed environmental predictors for this area. You can define the area by providing the coordinates of the bottom left and the top right corner (using the [SWEREF 99](https://www.lantmateriet.se/en/geodata/gps-geodesi-och-swepos/reference-systems/three-dimensional-systems/SWEREF-99/) coordinate reference system). The script will draw a rectangle between the provided points, break up the area into 1.28 x 1.28 km tiles, and extract all environmental features for each tile.

The trained model can only be applied to sites in Sweden, as several of the environmental features are only available within the country boundaries. To make sure all your tiles are within Sweden you can provide a cropping polygon of Sweden that will be used to filter your tiles. For this, use the `--sweden_map` flag and point to the provided shape-file: `'data/sweden_polygon/Sweref_99_TM/shape/swemap_gpkg/swemap.gpkg'`. (Note: To make predictions for all of Sweden, you can add the `--auto_adjust_prediction_range` flag, which will identify all tiles within the bounds of the Sweden polygon, independent of the provided input coordinates).

In our implementation we download an additional buffer of 200 m around each tile to address the edge-effect (for explanation see manuscript). This is done by setting `--additional_offset -400`, adding a total of 400 m to the tile-size along the x and y-axis.

```commandline
python download_prediction_geodata.py \
  --coordinates 522375,6416051,532938,6425134 \
  --offset 1280 \
  --image_size 128 \
  --download_folder tutorial/prediction_geodata/download_folder \
  --configuration version_public_sat_2024 \
  --meters_per_pixel 10 \
  --image_scale 1 \
  --additional_offset -400 \
  --target_server https://geodata.skogsstyrelsen.se/arcgis/rest/ \
  --username skstemp_user \
  --password S7Qawt3v \
  --threads 10 \
  --sweden_map data/sweden_polygon/Sweref_99_TM/shape/swemap_gpkg/swemap.gpkg
```

## Make predictions with trained model
Now we use the trained model to make predictions for all tiles for which we downloaded the environmental features in the previous step. Since we applied a 200 m buffer to each tile, we need to remove this buffer from the predictions to convert each tile back to the target-tile-size. Therefore we apply the flag `--crop_corners 20`, which removes 20 pixels (at 10 m per pixel = 200 m) around each prediction image.

**! Note !:** If you trained your own model based on the environmental features you compiled during this tutorial, your model will be trained on 9 environmental data channels (see explanation above). If you instead use our pre-trained model, presented in the manuscript, you will be working with a model that is trained on 11 environmental data channels, instead. In the latter case, you won't be able to use our pre-trained model to make predictions for the data you downloaded in the previous step, because the number of channels does not match what the model has been trained on (this discrepancy is due to changes on the data-server, with certain features being discontinued). In the command below we use the precompiled prediction data and our pre-trained model. Both can be found in the `tutorial/precompiled` folder.

```commandline
python make_predictions.py \
--geodata_folder tutorial/precompiled/prediction_geodata/download_folder \
--output_path tutorial/model_predictions_tutorial \
--trained_model tutorial/precompiled/best_model/best_model.pth \
--crop_corners 20 \
--apply_mask \
--threads 10
```


## Merge predictions into one spatial raster
Now we have produced predictions for each individual tile with our trained model. The final step is to merge these predictions into one geo-referenced raster, using the command below.

```commandline
python merge_tiff_files.py \
    --input tutorial/model_predictions_tutorial/predictions \
    --outfile tutorial/model_predictions_tutorial/merged_predictions.tiff
```


## Plot the predictions
You can either load the resulting tiff file into QGIS for an interactive view, or plot it with python using the following code:
```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Path to the TIFF file
tiff_file = 'tutorial/model_predictions_tutorial/merged_predictions.tiff'
# Read the TIFF file using Pillow
with Image.open(tiff_file) as img:
    tiff_data = np.array(img)
# Plot the TIFF data using the Turbo colormap
plt.imshow(tiff_data, cmap='turbo')
plt.colorbar()
plt.title('Predictions of conservation value')
plt.show()
```
The values on the x-axis and y-axis are scaled the number of 10m pixels, i.e. 500 units along one of the axes corresponds to a distance of 5 km.

![Predictions of conservation value](img/conservation_value_predictions.png)
