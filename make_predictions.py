import argparse
import utils.ImageRequest as ir
import utils.AttentionPixelClassifier as attentionPixelClassifier
import utils.mask_image as mi
import torch
from tifffile import imread
import urllib
import time

import geopandas as gpd
from shapely.geometry import MultiPolygon, Point

import cv2
import numpy as np
import pathlib
import platform
import os

from osgeo import gdal
import osgeo.osr as osr

import utils.RestAPIs as ra

import json
import errno
import shutil
import pdb
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

pltf = platform.system()
if pltf == 'Linux':
    print("on Linux system")
    pathlib.WindowsPath = pathlib.PosixPath
elif pltf == 'Windows':
    print("on Windows system")
    pathlib.PosixPath = pathlib.WindowsPath
else:
    print("on other than Linux or Windows system")
    pathlib.WindowsPath = pathlib.PosixPath

def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_model(opt,model_stored_as_dict=False):
    if opt.device != 'cpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if model_stored_as_dict:
        if str(opt.algorithm).lower() == 'attentionpixelclassifier':
            model = attentionPixelClassifier.AttentionPixelClassifier(
                input_numChannels=opt.input_channels[0],
                output_numChannels=opt.output_channels).to(device)
            image_size = opt.img_size[0]
        elif str(opt.algorithm).lower() == 'attentionpixelclassifierlite':
            model = attentionPixelClassifier.AttentionPixelClassifierLite(
                input_numChannels=opt.input_channels[0],
                output_numChannels=opt.output_channels).to(device)
            image_size = opt.img_size[0]
        elif str(opt.algorithm).lower() == 'attentionpixelclassifiermedium':
            model = attentionPixelClassifier.AttentionPixelClassifierMedium(
                input_numChannels=opt.input_channels[0],
                output_numChannels=opt.output_channels).to(device)
            image_size = opt.img_size[0]
        elif str(opt.algorithm).lower() == "attentionpixelclassifierlitedeep":
            model = attentionPixelClassifier.AttentionPixelClassifierLiteDeep(
                input_numChannels=opt.input_channels[0],
                output_numChannels=opt.output_channels).to(device)
            image_size = opt.img_size[0]
        elif str(opt.algorithm).lower() == "attentionpixelclassifierflex":
            n_channels_per_layer = opt.conv_layer_depth_info
            n_channels_per_layer = np.array(n_channels_per_layer.split(',')).astype(int)
            if opt.n_coefficients_per_upsampling_layer != None:
                n_coefficients_per_upsampling_layer = opt.n_coefficients_per_upsampling_layer
                n_coefficients_per_upsampling_layer = np.array(n_coefficients_per_upsampling_layer.split(',')).astype(
                    int)
            else:
                n_coefficients_per_upsampling_layer = opt.n_coefficients_per_upsampling_layer
            model = attentionPixelClassifier.AttentionPixelClassifierFlex(
                input_numChannels=opt.input_channels[0],
                output_numChannels=opt.output_channels,
                n_channels_per_layer=n_channels_per_layer,
                n_coefficients_per_upsampling_layer=n_coefficients_per_upsampling_layer
            ).to(device)
        model.load_state_dict(torch.load(opt.trained_model, map_location=device))
        model.eval()
    else:
        model = torch.load(opt.trained_model)
        model.eval()
    return model, device



def plot_pred(pred, image_name, in_ds, image_scale,crop_corners=0):
    if not isinstance(pred, np.ndarray) :
        pred = pred.cpu().detach().numpy()
        pred = pred.transpose((1, 2, 0))
    prediction = np.clip(pred, 0, 1)

    #cv2.imwrite(image_name.replace('.tiff','.png'),prediction*255)

    write_geotiff(image_name,prediction,in_ds, image_scale,crop_corners)


def read_geotiff(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def write_geotiff(filename, arr, in_ds, image_scale, crop_corners):
   # pdb.set_trace()
    #gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')
    arr_type = gdal.GDT_Float32
    
    if image_scale != 1:
        driver = gdal.GetDriverByName("MEM")
        out_ds = driver.Create('', arr.shape[1], arr.shape[0], 1, arr_type)
    else:
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type, options=['COMPRESS=LZW'])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3006)
    out_ds.SetProjection(srs.ExportToWkt())

    if crop_corners > 0:
        # Get original GeoTransform
        geo_transform = in_ds.GetGeoTransform()

        # Calculate new top-left X and Y coordinates
        new_top_left_x = geo_transform[0] + crop_corners * geo_transform[1]
        new_top_left_y = geo_transform[3] + crop_corners * geo_transform[5]

        # Create new GeoTransform
        new_geo_transform = (
            new_top_left_x,
            geo_transform[1],
            geo_transform[2],
            new_top_left_y,
            geo_transform[4],
            geo_transform[5]
        )
        out_ds.SetGeoTransform(new_geo_transform)
    else:
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    if len(arr.shape) > 2:
       arr = arr.squeeze(2)
    #pdb.set_trace()
    out_ds.GetRasterBand(1).WriteArray(arr)
    if image_scale != 1:
        gdal.Warp(filename,out_ds,options=gdal.WarpOptions(xRes=1/image_scale,yRes=1/image_scale))


def process_tile(args):
    index, coords, channels, opt = args
    land_cover_file = channels[-1]
    channels = channels[:-1]
    image_coordinates = list(np.array(coords.split('-')).astype(int))
    #print(channels)
    instance_tiff_folder = os.path.join(opt.output_path,'input')
    predictions_folder = os.path.join(opt.output_path,'predictions')
    ensure_dir(instance_tiff_folder)
    ensure_dir(predictions_folder)
    channel_names = [os.path.basename(i) for i in channels]
    image_name, number_of_channels = ir.compose_indata(instance_tiff_folder, channel_names, apis=ra.apis, image_name='input_%i.tiff'%index,target_dir=opt.geodata_folder)
    if number_of_channels == len(opt.selected_apis):
        if image_name != None:
            # img_path = channels[0][0]
            # print(img_path)
            if len(np.array(channels).shape) > 1:
                arr, ds = read_geotiff(channels[0][0])
            else:
                arr, ds = read_geotiff(channels[0])

            # image = imread(os.path.join(opt.input_path,image_name)).astype('uint8')
            image = imread(os.path.join(instance_tiff_folder, image_name))
            image = image / 255.0
            # print(image.shape)
            if len(image.shape) > 2:
                image = image.transpose(2, 0, 1)
            else:
                image = np.array([image])
            image = torch.from_numpy(image).to(opt.device)

            image = image[None, ...]
            # print(image.shape)

            pred = opt.model(image.float())
            pred = pred.cpu().detach().numpy()
            pred = pred[0,0,:,:]

            if opt.apply_mask:
                mask_image = np.ones((image.shape[-2], image.shape[-1]))
                land_cover_data_layer = imread(land_cover_file)
                mask_image = mi.create_filter_mask(land_cover_data_layer, mask_image, opt.json_data[opt.configuration]['prediction_masks'][0])
                pred = pred*mask_image

            pred = pred.round(decimals=8)
            if opt.crop_corners > 0:
                x=opt.crop_corners
                pred = pred[x:-x, x:-x]

            plot_pred(pred,
                      os.path.join(predictions_folder,'{}_{}.tiff'.format(opt.outfile_stem, index)),
                      ds,
                      float(1),
                      crop_corners=opt.crop_corners)
            #
            # plot_pred(np.array([mask_image]).transpose(1, 2, 0),
            #           os.path.join(predictions_folder,'{}_{}.tiff'.format('mask', index)),
            #           ds,
            #           float(1))
    print('Finished processing tile %i'%index)


def check_if_point_in_target_area(args):
    index, coords, channels, opt = args
    image_coordinates = list(np.array(coords.split('-')).astype(int))
    point1 = Point(image_coordinates[0], image_coordinates[1])
    point2 = Point(image_coordinates[2], image_coordinates[3])
    if opt.region_file:
        # Check if point is inside one of the polygons
        process_point = all([opt.geometry.contains(point) for point in [point1, point2]])
    else:
        process_point = True
    if not process_point:
        print('Skipping tile %i, extending from %i,%i to %i,%i as it is outside of the target area.' %(index, image_coordinates[1], image_coordinates[0], image_coordinates[3], image_coordinates[2]))
        return False
    else:
        return True

def organize_input_data_to_instances(folder_path,predefined_order):
    # Initialize an empty dictionary
    files_dict = {}
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            # Split the filename to separate the type and coordinates
            parts = filename.rsplit('-', 4)
            if len(parts) < 5:
                continue  # Skip filenames that do not have the expected format
            file_type = parts[0]
            coordinates = '-'.join(parts[1:5]).replace('.tiff','')

            # Check if the file type is in the predefined order list
            if file_type in predefined_order:
                # Ensure the coordinates key exists in the dictionary
                if coordinates not in files_dict:
                    files_dict[coordinates] = []

                # Append the full file path to the list for these coordinates
                files_dict[coordinates].append(os.path.join(folder_path, filename))

    # Remove keys where not all elements in predefined_order are present
    for coordinates in list(files_dict.keys()):
        file_types = [os.path.basename(fp).rsplit('-', 4)[0] for fp in files_dict[coordinates]]
        if not all(elem in file_types for elem in predefined_order):
            print('Not processing tile with coordinates %s because geodata could not be found for all target channels.'%coordinates)
            del files_dict[coordinates]
        else:
            # Sort the list of file paths according to predefined_order
            files_dict[coordinates].sort(key=lambda x: predefined_order.index(os.path.basename(x).rsplit('-', 4)[0]))
    return files_dict



def main(opt):

    start_time = time.time()
    print("main")
    ensure_dir(opt.output_path)
    # load configuration info
    conf_info_file = os.path.join(opt.geodata_folder, 'configuration_info.txt')
    with open(conf_info_file, 'r') as file:
        configuration = file.read()  # Read all the contents of the file into a variable
    print(configuration)
    opt.configuration = configuration

    # get list of required data channels
    selected_apis = []
    with open('utils/configurations.json',encoding='utf-8') as f:
        json_data = json.load(f)
    for api in json_data[opt.configuration]['apis']:
        print("api: {}".format(api))
        selected_apis.append(api)
    opt.json_data = json_data
    opt.selected_apis = selected_apis
    opt.input_channels = [len(selected_apis)]
    opt.output_channels = 1
    opt.algorithm = "AttentionPixelClassifierFlex"
    opt.n_coefficients_per_upsampling_layer = None
    instance_dict = organize_input_data_to_instances(opt.geodata_folder,selected_apis+['land_cover_types'])

    # load region file (polygon) in case it is provided by the user. Only points within this region will be used for prediction
    if opt.region_file:
        # Read the .gpkg file into a GeoDataFrame
        mp_path = opt.region_file
        gdf = gpd.read_file(mp_path)
        # Filter the GeoDataFrame based on a specific name
        target_name = opt.target_region
        selected_polygons = gdf[gdf[opt.region_id_field] == target_name]
        # To convert the filtered row into a Shapely geometry object
        geometry = selected_polygons.geometry.unary_union
    else:
        geometry = None
    opt.geometry = geometry

    # load the trained model
    model, device = load_model(opt,opt.model_stored_as_dict)
    opt.model = model
    opt.device = device

    args = [[i, coords, instance_dict[coords], opt] for i, coords in enumerate(instance_dict)]
    args = [arglist for arglist in args if check_if_point_in_target_area(arglist)]

    start_time_loop = time.time()
    print('Running parallel on %i threads'%opt.threads)
    # with ProcessPoolExecutor(max_workers=opt.threads) as executor:
    #     results = [executor.submit(process_tile, arglist) for arglist in args]
    results=[]
    with ThreadPoolExecutor(max_workers=opt.threads) as executor:
        results = list(executor.map(process_tile, args))
    # print(results)

    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_loop = end_time - start_time_loop
    print("Ran on %i threads" %opt.threads)
    print(f"Code executed in {elapsed_time} seconds")
    print(f"Loop executed in {elapsed_time_loop} seconds")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geodata_folder', type=str, default='data/prediction_geodata/download_folder')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--outfile_stem', type=str, default='prediction', help='Name_stem of output tiff-files containing predictions.')
    parser.add_argument('--trained_model', type=str, default='model.pth', help='model.pth path')
    parser.add_argument('--device', action='store', type=str, default='0')
    parser.add_argument('--threads', type=int, action='store', default=1)
    parser.add_argument('--region_file', action='store', default='')
    parser.add_argument('--region_id_field', action='store', default='EU_REGION')
    parser.add_argument('--target_region', action='store', default='')
    parser.add_argument('--conv_layer_depth_info', action='store', type=str, default='')
    parser.add_argument("--apply_mask", action="store_true", default=False)
    parser.add_argument("--model_stored_as_dict", action="store_true", default=False)
    parser.add_argument('--crop_corners', type=int, action='store', default=0,help='Specify number of pixels to be removed from the edge of the images (to remove edge-effect).')
    opt = parser.parse_args()
    main(opt)
#
# # below code is for trouble-shooting purposes only
# from types import SimpleNamespace
#
# opt = SimpleNamespace(
#     geodata_folder='data/prediction_geodata/test',
#     output_path='predictions/test_area_batchsize_5',
#     outfile_stem='prediction',
#     trained_model='train/modeltesting_models/100,50,100_5_weighted_loss.pth',
#     device='0',
#     threads=10,
#     region_file='',
#     region_id_field = 'EU_REGION',
#     target_region='',
#     model_stored_as_dict=False,
#     conv_layer_depth_info='',
#     apply_mask=True,
#     crop_corners=20
#     )