import sys
import os
import errno


import pandas as pd
import geopandas as geopd
import argparse
from PIL import Image
from tifffile import imsave, imread
import cv2

import utils.CreatePolygon as cp
import utils.PolygonRequests as PolygonRequests
import utils.ortophoto as op
import utils.RestAPIs as ra
import utils.save_gpkg as save_gpkg
import utils.ImageRequest as ir
import utils.helpers as hp
import utils.mask_image as mi
import utils.PolygonFilters as pf
import utils.PixelNormalization as PN
import utils.TestData as td
import json
import glob
import numpy as np
import pdb
import logging
import re
import concurrent.futures

logging.basicConfig(level='INFO')
log = logging.getLogger()

def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def append_lon_lat_arrays_to_channels(center_polygon,channels):
    # get corner points
    corner_points = np.array([i.exterior.coords.xy for i in center_polygon.geometry]).T
    corner_points = corner_points[:-1]
    # divide into n pixels np.linspace
    min_lon = corner_points.T[0][0].min()
    max_lon = corner_points.T[0][0].max()
    min_lat = corner_points.T[0][1].min()
    max_lat = corner_points.T[0][1].max()
    lons = np.linspace(min_lon, max_lon, int(opt.img_size))
    lats = np.linspace(min_lat, max_lat, int(opt.img_size))
    lon_array, lat_array = np.meshgrid(lons, lats[::-1])
    joined_lon_lat = np.array([lon_array,lat_array]).transpose(1,2,0)
    lonlat_normalized = PN.normalization_lon_lat(joined_lon_lat)
    # from matplotlib import pyplot as plt
    # plt.imshow(x, interpolation='nearest')
    # plt.show()
    final_lon_array = lonlat_normalized[:,:,0]
    final_lat_array = lonlat_normalized[:,:,1]
    path_base = '-'.join(channels[0].split('-')[:-1])
    filename_lon = path_base + '-lon.tiff'
    filename_lat = path_base + '-lat.tiff'
    ir.write_geotif_at_location(channels[0], filename_lon, [final_lon_array])
    ir.write_geotif_at_location(channels[0], filename_lat, [final_lat_array])
    channels = channels+[filename_lon,filename_lat]
    return channels


def create_annotation_and_mask_image(training_dataset_output_path, training_dataset_input_path ,polygons, center_polygon, json_data, image_size,meters_per_pixel, geo_image, mask_image, polygon_filtering_method):
    annotation_image = np.zeros((image_size,image_size))
    all_polygons_image = np.zeros((image_size,image_size))
    center_polygons_to_remove = []
    for i, poly in polygons.iterrows():
        p = op.get_image_polygon_training_data(poly, image_size=image_size,polygon_offset=center_polygon, meters_per_pixel=meters_per_pixel)

        exterior_coords = p.exterior.coords

        get_int_coords = lambda x: np.array(x).round().astype(np.int32)
        exterior_coords = [get_int_coords(exterior_coords)]
     
        
        #Only remove polygons thats completely inside image:
        whole_polygon_inside_image = True
        for coords in exterior_coords[0]:
            if coords[0] < 0 or coords[0] > image_size or coords[1] < 0 or coords[1] > image_size:
                whole_polygon_inside_image = False
        if whole_polygon_inside_image:
            center_polygons_to_remove.append(poly['TARGET_FID'])
            
        #polygons, probably high nature value:
        #for annotation_bypass_mask in annotation_bypass_masks:
        if pf.configurations[polygon_filtering_method]['annotation_filtering'](poly):
               cv2.fillPoly(annotation_image, exterior_coords, color=(255,255,255))
        cv2.fillPoly(all_polygons_image, exterior_coords, color=(255,255,255))

        interiors = p.interiors
        for interior in interiors:

            interior_coords = [get_int_coords(interior.coords)]
            cv2.fillPoly(all_polygons_image, interior_coords, color=(0,0,0)) 
            cv2.fillPoly(annotation_image, interior_coords, color=(0,0,0)) 


    cluster_id = center_polygon['TARGET_FID']

    annotation_image = annotation_image * mask_image
    #cv2.imwrite('{}/{}.png'.format(training_dataset_output_path,cluster_id),annotation_image)

    #pdb.set_trace()
    ir.write_geotif_at_location(geo_image,'{}/{}.tiff'.format(training_dataset_output_path,cluster_id), np.array([annotation_image]))

    mask_image = all_polygons_image * mask_image
    ir.write_geotif_at_location(geo_image,"{}/{}_mask.tiff".format(training_dataset_input_path,cluster_id), np.array([mask_image]))

    return center_polygons_to_remove

def do_work(extent_name,json_data,opt,train_annotation_path,testset_instances_dir,validationset_instances_dir,test_features_path,test_annotation_path,selected_test_apis,testdata_path,json_test_data,validation_features_path,validation_annotation_path,train_features_path,selected_apis):
    np.random.seed(1)
    log.info('Creating data from extent '+ extent_name)
    # this is the file containing the selected squares, stored as polygons
    extent_name = extent_name.replace('\\','/')
    original_polygons_name = os.path.split(extent_name)[0]+'/original_'+extent_name.split('/')[-1]
    polygon_ids = cp.read_gpkg(extent_name)
    original_polygons = cp.read_gpkg(original_polygons_name)
    original_polygons = pf.configurations[json_data[opt.configuration]['polygon_filtering_method']]['indata_filtering'](original_polygons,'all')
    # select random subset of polygons for test set
    n_instances_test = int(np.round(len(polygon_ids)*opt.testset_size))
    n_instances_validation = int(np.round(len(polygon_ids)*opt.validation_size))
    if opt.testset_size > 0:
        selected_test_ids = np.random.choice(polygon_ids.index.values,n_instances_test, replace=False)
    else:
        selected_test_ids = []
    if opt.validation_size > 0:
        left_over_polygons = polygon_ids.index.delete(selected_test_ids)
        selected_validation_ids = np.random.choice(left_over_polygons, n_instances_validation, replace=False)
    
    testset = polygon_ids.iloc[selected_test_ids]
    a=testset.index.values
    log.info(' '.join([extent_name, str(a)]))


    trainset = polygon_ids.drop(selected_test_ids, axis=0, inplace=False)
    validationset = polygon_ids.iloc[selected_validation_ids]
    # export test set dataframe to gpkg file
    testset_outfile = os.path.join(testset_instances_dir,'testset_'+os.path.basename(extent_name))
    if not testset.empty:
        save_gpkg.save(testset, testset_outfile)
    validationset_outfile = os.path.join(validationset_instances_dir, 'validationset_'+os.path.basename(extent_name))
    if not validationset.empty:
        save_gpkg.save(validationset, validationset_outfile)

    for index, cluster in enumerate(polygon_ids.iterrows()):
        # for each selected square
        center_polygon = cluster
        points_within_image_df = pd.DataFrame([{'geometry': center_polygon[1]['geometry'], 'TARGET_FID': extent_name.replace('\\','/').split('/')[-1].split('.')[0]+'_'+str(index)}])
        center_polygon = geopd.GeoDataFrame(points_within_image_df)
        if index in selected_test_ids:
            target_folder_features = test_features_path
            target_folder_annotations = test_annotation_path

        elif index in selected_validation_ids:
            target_folder_features = validation_features_path
            target_folder_annotations = validation_annotation_path
        else:
            target_folder_features = train_features_path
            target_folder_annotations = train_annotation_path

        mask_image = np.ones((int(opt.img_size), int(opt.img_size)))
        mask_image = mi.get_mask_image(json_data[opt.configuration]['prediction_masks'], mask_image, center_polygon, target_folder_features, username=opt.username, password=opt.password, img_size=int(opt.img_size))
        if mask_image.sum()==0:
            continue
        
        if index in selected_test_ids:
            #creating test data
            if opt.test_configuration:
                while True:
                    test_channels = ir.get_channels(center_polygon, "", img_size=int(opt.img_size), selected_apis=selected_test_apis, apis=ra.apis, username=opt.username, password=opt.password, meters_per_pixel=int(opt.meters_per_pixel))
                    success, _ = td.compose_testdata(testdata_path, test_channels, test_channels[0],img_size=int(opt.img_size), test_config=json_test_data[opt.test_configuration], center_polygon=center_polygon.iloc[0])
                    if success:
                        break

        while True:
            channels = ir.get_channels(center_polygon, "", img_size=int(opt.img_size), selected_apis=selected_apis, apis=ra.apis, username=opt.username, password=opt.password, meters_per_pixel=int(opt.meters_per_pixel))
            
            if opt.lonlat_features:
                channels = append_lon_lat_arrays_to_channels(center_polygon, channels)

            #log.info('Creating annotation and mask images')
            create_annotation_and_mask_image(target_folder_annotations,target_folder_features, original_polygons, center_polygon.iloc[0], json_data,image_size=opt.img_size,meters_per_pixel=int(opt.meters_per_pixel), geo_image=channels[0], mask_image=mask_image,polygon_filtering_method=json_data[opt.configuration]['polygon_filtering_method'])
            #log.info('Composing indata')
            success, _ = ir.compose_indata(target_folder_features, channels, apis=ra.apis)
            if success != False:
                break
def main(opt):
    if opt.logging_off:
        logging.disable(logging.CRITICAL)
    selected_apis = []
    selected_test_apis = []


    log.info('Using config' + opt.configuration)
    with open('utils/configurations.json',encoding='utf-8') as f:
        json_data = json.load(f)

    if json_data[opt.configuration]['apis'] != []: log.info('Using apis:')
    for api in json_data[opt.configuration]['apis']:
        #log.info(f'api: {api}')
        if opt.ai_image_server:
            selected_apis.append('ai_image_server_'+api)
        else:
            selected_apis.append(api)
    print(selected_apis)
    if opt.polygon_ids != []: log.info('Using polygons:')
    polygon_ids = []
    for c in opt.polygon_ids:
        log.info('Polygon '+ c)
        polygon_ids.append(int(c))

    train_annotation_path = opt.output_path+'/outdata'
    ensure_dir(train_annotation_path)
    train_features_path = opt.output_path+'/indata'
    ensure_dir(train_features_path)
    test_annotation_path = os.path.join(opt.output_path,'testset/outdata')
    ensure_dir(test_annotation_path)
    test_features_path = os.path.join(opt.output_path,'testset/indata')
    ensure_dir(test_features_path)
    testset_instances_dir = os.path.join(opt.output_path, 'testset/test_instances')
    ensure_dir(testset_instances_dir)
    ensure_dir('feature_layers')

    validationset_instances_dir = os.path.join(opt.output_path, 'validation/validation_instances')
    validation_annotation_path = opt.output_path+'/validation/outdata'
    validation_features_path = opt.output_path+'/validation/indata'
    ensure_dir(validationset_instances_dir)
    ensure_dir(validation_annotation_path)
    ensure_dir(validation_features_path)

    if opt.test_configuration:
        log.info('Using test config '+ opt.test_configuration)
        with open('utils/test_configurations.json', encoding='utf-8') as f:
            json_test_data = json.load(f)
        testdata_path = os.path.join(opt.output_path, 'testset', 'testdata')
        ensure_dir(testdata_path)
        for api in json_test_data[opt.test_configuration]['apis']:
            selected_test_apis.append(api)

        extents = glob.glob(os.path.join(opt.window_coordinates, "[!original_]*.gpkg"))
        # Filter out filenames ending with 'combined.gpkg' or 'combined_coarse.gpkg'
        extents = [f for f in extents if not f.endswith(('combined.gpkg', 'combined_coarse.gpkg'))]
        def sort_key(name):
            base = name.split('_')[-1]
            num = re.sub(r'\D', '', base)
            return int(num) if num.isdigit() else base
        extents.sort(key=sort_key)
        # Create info file describing the channels of the feature data
        save_apis_channel_data(extents[0], selected_apis, int(opt.meters_per_pixel), opt.output_path)

        # Download and compose data
        if opt.threads > 1:
            print('Running on ' + str(opt.threads) +' threads')
            with concurrent.futures.ThreadPoolExecutor(max_workers=opt.threads) as executor:
                results = [
                    executor.submit(do_work, extents_name, json_data, opt, train_annotation_path, testset_instances_dir,
                                    validationset_instances_dir, test_features_path, test_annotation_path,
                                    selected_test_apis, testdata_path, json_test_data, validation_features_path,
                                    validation_annotation_path, train_features_path, selected_apis) for
                    extents_index, extents_name in enumerate(extents)]
        else:
            print('Running on one core')
            for extents_index, extents_name in enumerate(extents):
                do_work(extents_name, json_data, opt, train_annotation_path, testset_instances_dir,
                        validationset_instances_dir, test_features_path, test_annotation_path, selected_test_apis,
                        testdata_path, json_test_data, validation_features_path, validation_annotation_path,
                        train_features_path,
                        selected_apis)  # for extents_index, extent_name in enumerate(extents) if extents_index>=0 ]

    # extents = glob.glob(os.path.join(opt.extents,"[!original_]*.gpkg"))
    # def sort_key(name):
    #     base = name.split('_')[-1]
    #     num = re.sub(r'\D', '', base)
    #     return int(num) if num.isdigit() else base
    #
    # extents.sort(key=sort_key)
    # print('sorting done')
    # print(extents)
    # # Create info file describing the channels of the feature data
    # save_apis_channel_data(extents[0], selected_apis, int(opt.meters_per_pixel), opt.output_path)
    # print('this')
    # # Download and compose data
    # if opt.threads>1:
    #     print(f'Startar med {opt.threads} threads')
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
    #         results = [
    #             executor.submit(do_work, extents_name,json_data,opt,train_annotation_path,testset_instances_dir,validationset_instances_dir,test_features_path,test_annotation_path,selected_test_apis,opt.extents,json_data,validation_features_path,validation_annotation_path,train_features_path, selected_apis) for extents_index, extents_name in enumerate(extents) if extents_index>0 ]
    # else:
    #     print('Startar med en core')
    #     for extents_index, extents_name in enumerate(extents):
    #         if extents_index >= 1:
    #             do_work( extents_name,json_data,opt,train_annotation_path,testset_instances_dir,validationset_instances_dir,test_features_path,test_annotation_path,selected_test_apis,opt.extents,json_data,validation_features_path,validation_annotation_path,train_features_path, selected_apis)# for extents_index, extent_name in enumerate(extents) if extents_index>=0 ]
    #



def save_apis_channel_data(extent_name, selected_apis, meters_per_pixel, output_path):
    '''Save the channel information for an extent'''
    output_file = os.path.join(output_path, 'channel_info.json')
    
    # Get channel info
    index = 0
    polygon_ids = cp.read_gpkg(extent_name)
    #print(polygon_ids)
    center_polygon = polygon_ids.iloc[index]
    #print(center_polygon)
    # Check if the polygon is a multipolygon or simple polygon
    polygon_geometry_type = center_polygon['geometry'].geom_type
    #print(polygon_geometry_type)
    if  polygon_geometry_type == 'MultiPolygon':
        # extract polygon out of multipolygon
        center_polygon = center_polygon[0]            

    # points_within_image_df = pd.DataFrame([{'geometry': center_polygon[1]['geometry'], 'TARGET_FID': extent_name.replace('\\','/').split('/')[-1].split('.')[0]+'_'+str(index)}])
    points_within_image_df = pd.DataFrame([{'geometry': center_polygon['geometry'], 'TARGET_FID': extent_name.replace('\\','/').split('/')[-1].split('.')[0]+'_'+str(index)}])
    #print(points_within_image_df)
    center_polygon = geopd.GeoDataFrame(points_within_image_df)
    #print(center_polygon)

    apis_channel_info = ir.get_channel_info(center_polygon, img_size=int(opt.img_size), selected_apis=selected_apis, apis=ra.apis, username=opt.username, password=opt.password, meters_per_pixel=meters_per_pixel)

    # Save information to json
    with open(output_file, "w") as f:
        json.dump(apis_channel_info, f, indent=4)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--apis', nargs='*', default=["ndvi"])
    parser.add_argument('--polygon_ids', nargs='+', default=[])
    parser.add_argument('--number_of_polygons',type=int, default=0)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--output_path', action='store', default='datasets')
    parser.add_argument('--start_index', action='store', type=int, default=5)
    parser.add_argument('--testset_size', action='store', type=float, default=0.1)
    parser.add_argument('--validation_size', action='store', type=float, default=0.2)
    parser.add_argument('--test_area', nargs='*',type=int, default=[389000,6752000,389000,6757000])
    parser.add_argument('--configuration', action='store', default='')
    parser.add_argument('--window_coordinates', action='store', default='')
    parser.add_argument('--polygons_path', action='store', default='geopackage')
    parser.add_argument('--lonlat_features',action='store_true')
    parser.add_argument('--meters_per_pixel', action='store', default=10)
    parser.add_argument('--test_configuration', action='store', default='')
    parser.add_argument('--threads', action='store', type=int, default=1)
    parser.add_argument('--ai_image_server', action='store_true')
    parser.add_argument('--logging_off', action='store_true')
    parser.add_argument('--target_server', action='store', default='https://geodata.skogsstyrelsen.se/arcgis/rest/')
    parser.add_argument('--username', action='store', default='')
    parser.add_argument('--password', action='store', default='')
    opt = parser.parse_args()

    main(opt)


# below code is for trouble-shooting purposes only:
# from types import SimpleNamespace
#
# opt = SimpleNamespace(
#     polygon_ids=[],
#     number_of_polygons=0,
#     img_size=128,
#     output_path="data/processed_geodata/alpine/alpine_geodata",
#     start_index=5,
#     testset_size=0.2,
#     validation_size=0.2,
#     test_area=[389000, 6752000, 389000, 6757000],
#     configuration="version_public_sat",
#     window_coordinates="data/processed_geodata/alpine/cropped_windows",
#     polygons_path='geopackage',
#     lonlat_features=False,
#     meters_per_pixel=10,
#     test_configuration="version_1",
#     threads=30,
#     ai_image_server=False,
#     logging_off=True,
#     target_server='https://geodata.skogsstyrelsen.se/arcgis/rest/',
#     username='uppun_user',
#     password='4sjHa2YQ'
# )
