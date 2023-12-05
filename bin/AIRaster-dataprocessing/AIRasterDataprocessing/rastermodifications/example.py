import tifffile
import numpy as np
try:
    import gdal
except: 
    from osgeo import gdal

from AIRasterDataprocessing.rastermodifications.geometric_correction import geometric_correction
from AIRasterDataprocessing.rastermodifications.radiometric_correction import radiometric_correction

print("Testing geometric and radiometric correction".center(60,'-'))

im1_path='./data/id_0_1499299200000_0_0_ndvi.tiff'
im2_path='./data/id_0_1502928000000_0_1_ndvi.tiff'

im1=gdal.Open(im1_path).ReadAsArray()
im2=gdal.Open(im2_path).ReadAsArray()

im2_geom_corr = geometric_correction(im1, im2, upsample_factor=100)

im2_rad_corr = radiometric_correction(im1, im2)

# Note that those raster modifications are order dependent if chained
im2_geom_rad_corr = radiometric_correction(im1, im2_geom_corr)

im2_rad_geom_corr = geometric_correction(im1, im2_rad_corr, upsample_factor=100)

print(im2_geom_rad_corr==im2_rad_geom_corr)

# Test radiometric correction with a filter image.
# Load the filter image, should be one band only.
print("Testing radiometric correction with filter image".center(60,'-'))
path_to_filter_image = './data/499220_6390690_NMD_ogeneraliserad.tif'

ds = gdal.Open(path_to_filter_image, gdal.GA_ReadOnly)
rb = ds.GetRasterBand(1)
filter_image_data = rb.ReadAsArray()
filter_values = (101+np.arange(28)).tolist()
im2_rad_corr_filter = radiometric_correction(im1, im2, filter_image=filter_image_data, filter_val=filter_values)
print(im2_geom_rad_corr==im2_rad_corr_filter)