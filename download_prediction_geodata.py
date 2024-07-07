
import time
import json
import errno
import shutil
import os
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point
from osgeo import gdal
import osgeo.osr as osr
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import utils.ImageRequest as ir
import utils.mask_image as mi
import utils.RestAPIs as ra

def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def plot_pred(pred, image_name, in_ds, image_scale):
    if not isinstance(pred, np.ndarray) :
        pred = pred.cpu().detach().numpy()
        pred = pred.transpose((1, 2, 0))
    prediction = np.clip(pred, 0, 1)

    #cv2.imwrite(image_name.replace('.tiff','.png'),prediction*255)

    write_geotiff(image_name,prediction,in_ds, image_scale)


def read_geotiff(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds
    
def write_geotiff(filename, arr, in_ds, image_scale):
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
    #out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    arr = arr.squeeze(2)
    #pdb.set_trace()
    out_ds.GetRasterBand(1).WriteArray(arr)
    if image_scale != 1:
        gdal.Warp(filename,out_ds,options=gdal.WarpOptions(xRes=1/image_scale,yRes=1/image_scale))


def process_tile(args):
    y, x, index, opt = args

    print('Processing tile %i'%index)

    image_coordinates = [y, x, y + opt.offset, x + opt.offset]
    channels = ir.get_channels(image_coordinates, opt.download_folder, img_size=opt.image_size,
                               selected_apis=opt.selected_apis, apis=ra.apis, username=opt.username,
                               password=opt.password, meters_per_pixel=int(opt.meters_per_pixel),
                               get_image_coord=False)
    mi.download_landcover_channel(opt.json_data[opt.configuration]['prediction_masks'],
                                  image_coordinates,
                                  opt.download_folder,
                                  opt.image_size,
                                  opt.username,
                                  opt.password)
    # #print(channels)
    # if (len(channels) > 0):
    #     if channels[0] == None:
    #         print('No spatial data found for location', image_coordinates,'. Area either outside of range where spatial data is available or the remote server is not reachable.')
    #     else:
    #         image_name, number_of_channels = ir.compose_indata(opt.download_folder, channels, apis=ra.apis, image_name='input_%i.tiff'%index)
    #     if opt.save_input_img:
    #         ensure_dir(os.path.join(opt.output_path, 'indata'))
    #         shutil.copy(os.path.join(opt.download_folder, image_name),os.path.join(opt.output_path,'indata',image_name))
    #     # remove the input.tiff file after it's all done
    #     os.remove(os.path.join(opt.download_folder, image_name))
    #     # remove images in input dir
    #     for channel in channels:
    #         #print('Removing',channel)
    #         os.remove(channel)
    print('Finished processing tile %i'%index)


def check_if_point_in_target_area(args):
    y, x, index, opt = args
    image_coordinates = [y, x, y + opt.offset, x + opt.offset]
    point1 = Point(image_coordinates[0], image_coordinates[1])
    point2 = Point(image_coordinates[2], image_coordinates[3])
    if opt.sweden_map:
        process_point = all([opt.swe_poly_geometry.contains(point) for point in [point1, point2]])
    else:
        process_point = True
    if not process_point:
        print('Skipping tile %i, extending from %i,%i to %i,%i as it is outside of the target area.' %(index, image_coordinates[1], image_coordinates[0], image_coordinates[3], image_coordinates[2]))
        return False
    else:
        return True

def run_parallel_script(args):
    try_get_image_attempt = 0
    while True:
        try:
            process_tile(args)
            break
        except Exception as e:
            time.sleep(0.01)
            try_get_image_attempt += 1
            if try_get_image_attempt > 30:
                return
            else:
                print('Attempt %i: Retrying processing of tile %i.'%(try_get_image_attempt+1,args[2]))
                pass


def main(opt):

    start_time = time.time()
    print("main")
    ensure_dir(opt.download_folder)
    selected_apis = []

    # load sweden map if provided
    if opt.sweden_map:
        gdf = gpd.read_file(opt.sweden_map)
        target_polys = gdf[gdf[opt.swemap_id_field]==opt.swemap_target_id]
        swe_poly_geometry = target_polys.geometry.unary_union
    else:
        swe_poly_geometry = None

    print("using config {}".format(opt.configuration))
    output_file = os.path.join(opt.download_folder, 'configuration_info.txt')
    # Store configuration in info txt file with data
    with open(output_file, 'w') as file:
        file.write(str(opt.configuration))  # Convert the value to a string and write it to the file
    with open('utils/configurations.json',encoding='utf-8') as f:
        json_data = json.load(f)

    for api in json_data[opt.configuration]['apis']:
        print("api: {}".format(api))
        selected_apis.append(api)

    coordinates = opt.coordinates
    coordinates = coordinates.split(',')

    minY, minX, maxY, maxX = [int(coord) for coord in coordinates]
    if opt.sweden_map and opt.auto_adjust_prediction_range:
        minY, minX, maxY, maxX = np.array(swe_poly_geometry.bounds).astype(int)
        print('Adjusting spatial range of predictions to max range provided by imported map:',minY, minX, maxY, maxX)

    # index = 0
    # fail_index = 0
    # skip_index = 0

    opt.swe_poly_geometry = swe_poly_geometry
    opt.selected_apis = selected_apis
    opt.json_data = json_data

    args = [[y, x, i, opt] for i, (y, x) in enumerate(
        (y, x) for y in range(minY, maxY, opt.offset + int(opt.additional_offset)) for x in
        range(minX, maxX, opt.offset + int(opt.additional_offset)))]
    args = [arglist for arglist in args if check_if_point_in_target_area(arglist)]

    start_time_loop = time.time()
    print('Running parallel on %i threads'%opt.threads)
    # with ProcessPoolExecutor(max_workers=opt.threads) as executor:
    #     results = [executor.submit(process_tile, arglist) for arglist in args]
    results=[]
    with ThreadPoolExecutor(max_workers=opt.threads) as executor:
        results = list(executor.map(run_parallel_script, args))
    # print(results)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_loop = end_time - start_time_loop

    print("Ran on %i threads" %opt.threads)
    print(f"Code executed in {elapsed_time} seconds")
    print(f"Loop executed in {elapsed_time_loop} seconds")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', action='store', help='ex: 580500,6563500,588500,6565500') #Sweden:232771,6103725,932460,7680000
    parser.add_argument('--offset', type=int, action='store', default=1280, help='the offset for each tif image, ex 1000 (Meter)')
    parser.add_argument('--image_size', type=int, action='store', default=128)
    parser.add_argument('--download_folder', type=str, default='data/prediction_geodata/download_folder')
    parser.add_argument('--outfile', type=str, default='output', help='*.gpkg')
    parser.add_argument('--configuration', action='store', default='')
    parser.add_argument('--meters_per_pixel', action='store', default=10)
    parser.add_argument('--image_scale', action='store', default=1, help='set 0.5 to get half the image width and height')
    parser.add_argument('--additional_offset', type=int, action='store', default=0)
    parser.add_argument('--target_server', action='store', default='https://geodata.skogsstyrelsen.se/arcgis/rest/')
    parser.add_argument('--username', action='store', default='')
    parser.add_argument('--password', action='store', default='')
    parser.add_argument('--sweden_map', action='store', default='')
    parser.add_argument('--swemap_id_field', action='store', default='KKOD')
    parser.add_argument('--swemap_target_id', action='store', default='901')
    parser.add_argument('--auto_adjust_prediction_range', action='store_true', help='If this flag is activated, it will overwrite the spatial range defined with the --coordinates flag to the maximum range defined by the provided --sweden_map')
    parser.add_argument('--threads', type=int, action='store', default=1)
    opt = parser.parse_args()
    main(opt)

#
# # below code is for trouble-shooting purposes only
# from types import SimpleNamespace
#
# opt = SimpleNamespace(
#     coordinates='522626,6642549,528626,6648549',
#     offset=1280,
#     image_size=128,
#     download_folder='data/prediction_geodata/test',
#     configuration='version_public_sat',
#     meters_per_pixel=10,
#     image_scale=1,
#     additional_offset=-400,
#     target_server='https://geodata.skogsstyrelsen.se/arcgis/rest/',
#     username='xxxx',
#     password='xxxx',
#     sweden_map = '',
#     # sweden_map='data/sweden_polygon/Sweref_99_TM/shape/swemap_gpkg/swemap.gpkg',
#     swemap_id_field='KKOD',
#     swemap_target_id='901',
#     auto_adjust_prediction_range=False,
#     threads=10
#     )
