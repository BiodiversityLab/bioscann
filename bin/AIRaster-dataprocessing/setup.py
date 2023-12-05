#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='AIRasterDataprocessing',
      version='0.0.2',
      description='This package has shared components that can be used to manipulate rasters or vector data.',
      author='AI Raster teamet',
      author_email='user@email.com',
      packages=find_packages(),
      install_requires=[
        "geopandas>=0.9.0",
        "numpy>=1.21.6",
        "shapely==1.8.4",
        "scikit-learn>=1.1.2",
        "scipy>=1.5.2",
        "scikit-image>=0.19.3"
    ],
      license='LICENSE.txt',
    )