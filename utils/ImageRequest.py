#import utils.ortophoto as op
from tifffile import imsave, imread
import numpy as np
import os, shutil
import requests
import logging
import json
import string

from osgeo import gdal
import osgeo.osr as osr

import time




def get_image_coordinates( geodf_separata_polygoner, offset=150):
    #cluster_ids = []
    coordinates = []
    for i, poly in enumerate(geodf_separata_polygoner.iterrows()):
        coord = poly[1]['geometry'].centroid.coords[0]
        #cluster_ids.append(poly[1]['TARGET_FID'])
        coordinates.append([coord[0]-offset, coord[1]-offset, coord[0]+offset, coord[1]+offset])     
    return coordinates
    
    
def get_channels(center_polygon,training_dataset_path,img_size, selected_apis, apis, username, password, meters_per_pixel=10, get_image_coord=True):
        #image_size = 512
    offset = img_size / 2
    #meters_per_pixel = 10
    offset = offset * meters_per_pixel
    channels = []
    if get_image_coord: 
        image_coordinates = get_image_coordinates(center_polygon,offset)
        for selected_api in selected_apis:
            channels.append(get_image(center_polygon, image_coordinates, 0, img_size, selected_api, apis, username, password))
    else:
        image_coordinates = center_polygon
        for selected_api in selected_apis:
            img_data = get_image_no_polygon(training_dataset_path, image_coordinates, 0, img_size, selected_api, apis, username, password)
            # if img_data == None:
            #     channels = [None]*len(selected_apis)
            #     return channels
            # else:
            channels.append(img_data)
    return channels


def get_channel_info(center_polygon, img_size, selected_apis, apis, username, password, meters_per_pixel:int) -> dict:
    '''Get the channel info for each selected api into a json'''
    apis_channel_info = {}

    offset = (img_size / 2) * meters_per_pixel
    image_coordinates = get_image_coordinates(center_polygon, offset)

    # Fetch information from each selected api
    for selected_api in selected_apis:
        image_path = get_image(center_polygon, image_coordinates, 0, img_size, selected_api, apis, username, password)
        image = imread(image_path)
        apis_channel_info[selected_api] = image.shape

    return apis_channel_info


# def download_tif_image(tiff_image_url, name='input.png'):
#     urllib.request.urlretrieve(tiff_image_url, name)

def get_image_url(url, params, username, password):
    # When making get image requests, use HTTP POST to not risk "URL too long" errors,
    # see https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/414
    post_headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, params=params, auth=(username, password), headers=post_headers)
    # Requests built in error check for status code
    response.raise_for_status()
    # Read the json data from response
    json_response = response.json()
    # Extra error check, ArcGis can return HTTP Status 200 but still fail, then we have to check the error object
    if "error" in json_response:
        error = json_response.get("error")
        raise Exception(f"Request failed with error: {error.get('message')}")
    if 'href' not in json_response:
        raise Exception("No image url returned")
    # Extract the image path on the server
    image_href = json_response['href']
    return image_href

# def download_tif_image(image_href, username, password, image_path='input.png'):
#     # Make request to get the image bytes
#     img_bytes = requests.get(image_href, auth=(username, password), stream=True)
#     # Save the image to disk
#     # Create image dir if not exists
#     os.makedirs(os.path.dirname(image_path), exist_ok=True)
#     # Read and write the image bytes to disk
#     with open(image_path, 'wb') as f:
#         img_bytes.raw.decode_content = True
#         shutil.copyfileobj(img_bytes.raw, f)


def download_tif_image(image_href, username, password, image_path='input.tif'):
    response = requests.get(image_href, auth=(username, password), stream=True)
    # Check if the response is OK
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    # Ensure the content type is TIFF
    if 'image/tiff' not in response.headers.get('Content-Type', ''):
        raise Exception("The downloaded file is not a TIFF image")
    # Ensure complete download
    content_length = response.headers.get('Content-Length')
    if content_length is not None:
        total_bytes = int(content_length)
        if total_bytes != len(response.content):
            raise Exception("Incomplete download of the TIFF image")
    # Create image dir if not exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    # Write the image bytes to disk
    with open(image_path, 'wb') as f:
        f.write(response.content)


def get_image(poly, image_coordinates, polygon_index, img_size,rest_api, apis, username, password):
    #image_name = "{}/id-{}-{}-{}.tiff".format(dataset_path,poly.iloc[0]['TARGET_FID'],polygon_index,rest_api)
    image_name = "id-{}-{}-{}-{}.tiff".format(poly.iloc[0]['TARGET_FID'],polygon_index,rest_api,image_coordinates[0])
    image_name = image_name.replace('\'','').replace('"','').replace('[','').replace(']','').replace(', ','-')
    image_name = os.path.join('feature_layers',image_name)
    if not os.path.isfile(image_name):
        params = apis[rest_api]['params'](image_coordinates[0], image_size=img_size)
        server_url = apis[rest_api]['url']()
        download_url = get_image_url(server_url, params, username, password)
        try_get_image_attempt = 0
        while True:
            try:
                print(download_url)
                print('Downloading image data...')
                download_tif_image(download_url, username, password, image_name)
                break
            except Exception as e:
                # print(rest_api)
                # print(params)
                # print(e)
                time.sleep(0.01)
                try_get_image_attempt += 1
                if try_get_image_attempt > 30:
                    return
                else:
                    pass
        print(try_get_image_attempt)
    return image_name


def get_image_no_polygon(dataset_path, image_coordinates, polygon_index, img_size,rest_api, apis, username, password):
    image_name = "{}-{}.tiff".format(rest_api,image_coordinates)
    image_name = image_name.replace('\'','').replace('"','').replace('[','').replace(']','').replace(', ','-')
    image_name = os.path.join(dataset_path,image_name)
    if not os.path.isfile(image_name):
        params = apis[rest_api]['params'](image_coordinates, image_size=img_size)
        server_url = apis[rest_api]['url']()
        download_url = get_image_url(server_url, params, username, password)
        try_get_image_attempt = 0
        while True:
            try:
                print(download_url)
                print('Downloading image data...')
                download_tif_image(download_url, username, password, image_name)
                break
            except Exception as e:
                # print(rest_api)
                # print(params)
                # print(e)
                time.sleep(0.01)
                try_get_image_attempt += 1
                if try_get_image_attempt > 30:
                    return
                else:
                    pass
        print(try_get_image_attempt)
    return image_name


def write_geotif_at_location(ref_image_filepath, out_image_filepath, list_of_numpy_arr):
    '''

    Writes a geotif at the same position as a reference image. 
    Each band in the geotif is added in the list as np.array 
    
    input:
        ref_image_filepath (string) - path to georeferences image
        out_image_filepath (string) - path to output image
        list_of_numpy_arr (list)  - list of 2d nparrys, shape should be of same size as shape of ref_image_filepath
    output:
        None
    '''
    logging.info(f'Writing geotif {out_image_filepath}')
    ds = gdal.Open(ref_image_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    
    driver = gdal.GetDriverByName('GTiff')
    srs = osr.SpatialReference()
    outdata = driver.Create(out_image_filepath, cols, rows, len(list_of_numpy_arr), gdal.GDT_Float32, options=['COMPRESS=LZW'])
    srs.ImportFromEPSG(3006)
    outdata.SetProjection(srs.ExportToWkt())
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    #print(ds.GetProjection())
    #outdata.SetProjection(ds.GetProjection())##sets same projection as input
    for i in range(len(list_of_numpy_arr)):
        outdata.GetRasterBand(i+1).WriteArray(list_of_numpy_arr[i])
        outdata.GetRasterBand(i+1).SetNoDataValue(10000)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band = None
    ds = None
    return None

def compose_indata(dataset_path, channels, apis, image_name='', mask_image=''):
    imgs = []
    for i, img in enumerate(channels):
        im = imread(img)
        logging.info(f'Reading file {img}')
        #normalize:
        feature_name = img.split('-')[-5].split('.')[0]
        if not feature_name in ['lon','lat']:
            im = apis[feature_name]['normalization'](im)
        if(len(im.shape) > 2):
            im = im.transpose(2,0,1)
            for chan in im:
                imgs.append(chan)
        else:
            imgs.append(im)
    imgs = np.array(imgs)

    if len(imgs) > 0:
        try:

            #for training
            if image_name == '':
                cluster_id = img.split('/')[-1].split('-')[1]
                #imsave(os.path.join(dataset_path,"{}.tiff".format(cluster_id)),imgs, compress='lzma')
                write_geotif_at_location(channels[0],os.path.join(dataset_path,"{}.tiff".format(cluster_id)), imgs)
                #if mask_image != '':
                #    write_geotif_at_location(channels[0],os.path.join(dataset_path,"{}_mask.tiff".format(cluster_id)), np.array([mask_image]))
                return image_name, len(imgs)
            else:
                #imsave(os.path.join(dataset_path,image_name),imgs, compress='lzma')
                write_geotif_at_location(channels[0],os.path.join(dataset_path,image_name), imgs)
                return image_name, len(imgs)

        except Exception as e:

            print(e)
            print(i) 
            return False, False