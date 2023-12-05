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

## Prepare the polygon data for input
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