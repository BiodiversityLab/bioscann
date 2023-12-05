# BIOSCANN
BIOSCANN: BIOdiversity Segmentation and Classification with Artificial Neural Networks

## Installation
```commandline
conda install -c conda-forge -y python=3.8
conda install -c conda-forge -y imagecodecs
conda install -c conda-forge -y fiona=1.9.3
conda install -c conda-forge -y pytorch
conda install -c conda-forge -y torchvision
conda install -c conda-forge -y mlflow
cd bin/AIRaster-dataprocessing
python -m pip install .
cd ../..
pip install -r requirements.txt
```

## Define cropping window around each polygon
#### Alpine
```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/alpine \
    --output_path data/processed_geodata/alpine/cropped_windows \
    --no_overlap
```

#### Boreal_east
```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/boreal_east \
    --output_path data/processed_geodata/boreal_east/cropped_windows \
    --no_overlap
```

#### Boreal_northwest
```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/boreal_northwest \
    --output_path data/processed_geodata/boreal_northwest/cropped_windows \
    --no_overlap
```

#### Boreal_south
```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/boreal_south \
    --output_path data/processed_geodata/boreal_south/cropped_windows \
    --no_overlap
```

#### Continental
```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/continental \
    --output_path data/processed_geodata/continental/cropped_windows \
    --no_overlap
```

## Extract spatial data for each cropping window
#### Alpine
```commandline
python extract_geo_data.py \
    --output_path datasets/testdata_2_geodata  \
    --window_coordinates data/processed_geodata/alpine/cropped_windows \
    --configuration version_public \
    --test_config version_1 \
    --testset_size 0.2 \
    --img-size 128 \
    --username uppun_user \
    --password 4sjHa2YQ \
    --logging_off
```
