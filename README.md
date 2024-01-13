# BIOSCANN
BIOSCANN: BIOdiversity Segmentation and Classification with Artificial Neural Networks

## Installation
```commandline
conda install -c conda-forge -y python=3.8
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

#### Boreal_east
```commandline
python crop_windows_from_polygons.py \
    --input_path data/polygons/boreal_east \
    --output_path data/processed_geodata/boreal_east/cropped_windows \
    --no_overlap
```


## Extract spatial data for each cropping window

#### Boreal_east
```commandline
region="boreal_east"
python extract_geo_data.py \
    --output_path data/processed_geodata/${region}/${region}_geodata  \
    --window_coordinates data/processed_geodata/${region}/cropped_windows \
    --configuration version_public_sat \
    --test_config version_1 \
    --testset_size 0.2 \
    --img-size 128 \
    --username ***** \
    --password ***** \
    --logging_off \
    --threads 30
```

## Train model
#### Boreal_east
```commandline
region="boreal_east"
configuration="20,30,40,50,40,30,20"

python train_model.py \
    --batch 14 \
    --input 11 \
    --output 1 \
    --algorithm AttentionPixelClassifierFlex \
    --device gpu \
    --dataset data/processed_geodata/${region}/${region}_geodata \
    --validation data/processed_geodata/${region}/${region}_geodata/validation \
    --test_dataset data/processed_geodata/${region}/${region}_geodata/testset/ \
    --plot \
    --experiment_name ${region} \
    --epochs 300 \
    --img_size 128 \
    --mlflow \
    --n_channels_per_layer ${configuration} \
    --pfi \
    --patience 20
```

