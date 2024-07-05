import os
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr, osr

def match_resolution_of_rasters(src_ds,ref_ds):
    # Get the GeoTransform and Projection from the reference raster
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    x_size = ref_ds.RasterXSize  # number of columns
    y_size = ref_ds.RasterYSize  # number of rows

    # Create a new raster to save the resampled output
    driver = gdal.GetDriverByName('GTiff')
    resampled_ds = driver.Create('resampled_prediction.tif', x_size, y_size, 1, gdal.GDT_Float32)
    resampled_ds.SetGeoTransform(geo_transform)
    resampled_ds.SetProjection(projection)

    # Perform the resampling - here we use bilinear interpolation; you can also use gdal.GRA_NearestNeighbour or other methods
    gdal.ReprojectImage(src_ds, resampled_ds, src_ds.GetProjection(), projection, gdal.GRA_NearestNeighbour)
    return resampled_ds

def create_filter_mask(filter_shapefile,target_raster):
    # Open the GeoPackage
    vector_ds = ogr.Open(filter_shapefile)
    if vector_ds is None:
        raise Exception("Could not open the GeoPackage.")
    # Assuming you know the layer name or just want to use the first layer
    layer = vector_ds.GetLayerByIndex(0)  # or vector_ds.GetLayer('layer_name')
    # Now proceed to open the prediction raster
    pred_ds = gdal.Open(target_raster)
    if pred_ds is None:
        raise Exception("Prediction raster could not be opened.")
    geo_transform = pred_ds.GetGeoTransform()
    projection = pred_ds.GetProjection()
    x_size = pred_ds.RasterXSize
    y_size = pred_ds.RasterYSize
    # Create a memory dataset for the mask
    mask_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(geo_transform)
    mask_ds.SetProjection(projection)
    band = mask_ds.GetRasterBand(1)
    band.Fill(0)  # Initialize raster with zeros
    band.SetNoDataValue(0)
    # Rasterize the GeoPackage layer
    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
    # Read the mask array
    mask_array = band.ReadAsArray()
    return mask_array


def fix_soil_moisture_tiff_file(input_tiff,output_tiff):
    # Open the input TIFF file
    dataset = gdal.Open(input_tiff, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    # Read the image data into a numpy array
    image = band.ReadAsArray()
    # Replace each pixel with the value 255 with the value 0
    image[image == 255] = 101
    # Get the georeference info
    transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    # Create the output TIFF file
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_tiff, band.XSize, band.YSize, 1, gdal.GDT_Byte)
    # Set the georeference info to the output file
    out_dataset.SetGeoTransform(transform)
    out_dataset.SetProjection(projection)
    # Write the modified image data to the output TIFF file
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(image)
    # Flush data to disk
    out_band.FlushCache()
    # Clean up
    del dataset
    del out_dataset
    print(f"Processed TIFF saved as {output_tiff}")


region = 'continental'
if region in ['continental','alpine']:
    predictions_file = 'predictions/batchsize_5_opposite_loss_all_sweden/sweden_all_%s_predictions_batchsize_5.tiff'%region
else:
    predictions_file = 'predictions/batchsize_5_opposite_loss_final/%s_batchsize_5.tiff'%region

output_file = 'predictions/cleaned_final/predictions_%s.tiff'%region
polygon_filter_file = 'data/custom_polygons/crop_polygon_soilmoisture.gpkg'
soil_moisture_raster_file = 'data/prediction_geodata/soil_moisture_high_res/soil_moisture_merged_high_res.tiff'
if not soil_moisture_raster_file.endswith('_fixed.tiff'):
    path_fixed_file = soil_moisture_raster_file.replace('.tiff','_fixed.tiff')
    if not os.path.exists(path_fixed_file):
        fix_soil_moisture_tiff_file(soil_moisture_raster_file,path_fixed_file)
    soil_moisture_raster_file = path_fixed_file

# load the predictions tiff file
pred_tiff_path = predictions_file
pred_ds = gdal.Open(pred_tiff_path)
pred_band = pred_ds.GetRasterBand(1)
pred_array = pred_band.ReadAsArray()


# load filter polygons shp file
filter_poly_path = polygon_filter_file
# Mask out values inside the polygons
mask_array = create_filter_mask(filter_poly_path,pred_tiff_path)
pred_array[mask_array == 1] = np.nan  # Replace with np.nan or appropriate nodata value


# remove all pixels where soil_moisture == 101 (water)
# Open the soil moisture raster (mask)
mask_ds = gdal.Open(soil_moisture_raster_file)
resampled_mask_ds = match_resolution_of_rasters(mask_ds,pred_ds)
mask_band = resampled_mask_ds.GetRasterBand(1)
mask_array = mask_band.ReadAsArray()

# Apply the mask: set prediction values to None where mask values are 101
pred_array[mask_array == 101] = np.nan  # Use np.nan or another appropriate nodata value

# Create a new file to save the filtered predictions
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(output_file, pred_ds.RasterXSize, pred_ds.RasterYSize, 1, pred_band.DataType)
out_ds.SetGeoTransform(pred_ds.GetGeoTransform())  # Same geotransform
out_ds.SetProjection(pred_ds.GetProjection())  # Same projection

# Write the modified array to the new raster
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(pred_array)
out_band.SetNoDataValue(np.nan)  # Set the nodata value
out_band.FlushCache()

# Close datasets
mask_ds = None
pred_ds = None
out_ds = None




