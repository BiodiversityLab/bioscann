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