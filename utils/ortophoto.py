import urllib.request
import requests
from shapely.geometry import Polygon
import cv2
import numpy as np


from datetime import datetime

import os

import utils.PolygonRequests as pr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
from matplotlib import pyplot as plt
import errno
import json
import tqdm

def get_ortophoto_sentinel_2_3_0_request_params(coordinates_list, date,image_size=512, rendering_rule='ndvi'):
    renderingRule = ''
    if rendering_rule=='ndvi':
        renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(b3*0+b2-b1-500)/(b1+b2+500)",\
                "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
                "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [2, 3, 6]}}}}}}',

        #sentinel 2_3_0 SAVI , inte fått till riktigt
        # renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(1+0.25)*(b3*0+b2-b1)/(b1+b2+0.25)",\
        #         "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
        #         "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [2, 3, 6]}}}}}}',
    elif rendering_rule=='ndwi':
        #sentinel 2_3_0 NDWI
        renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(b3*0+b1-b2)/(b1+b2)",\
                 "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
                 "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [1, 3, 6]}}}}}}',

        #sentinel 2_2_0
        # renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(b3*0+b1-b2)/(b1+b2-2000)",\
        #               "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
        #               "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [3, 8, 12]}}}}}}',

    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compressionQuality='',
        bandIds='',
        sliceId='',
        renderingRule=renderingRule,
        mosaicRule='{"where":"ImageDate=date'+"'"+str(date)+"'"+'"}',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
 

    )
    return params
    

def get_tif_image_url_sentinel_2_3_0(params):
    #url = 'http://imgutv.svo.local/arcgis/rest/services/Sentinel2_2_0/ImageServer/exportImage'
    #url = 'http://imgutv.svo.local/arcgis/rest/services/Sentinel2_3_0/ImageServer/exportImage'
    url = 'http://193.183.28.189/arcgis/rest/services/Sentinel2_3_0/ImageServer/exportImage'
    resp = requests.post(url=url, params=params, verify=False)
    data = resp.json()
    #print(data)
    return data['href']

def get_ortophoto_request_params(coordinates_list, image_year, rendering_rule,image_size=512):
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='jpgpng',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        mosaicRule='{"where":"ImageYear='+str(image_year)+'"}',
        renderingRule='{"rasterfunction":"'+rendering_rule+'"}', #SKS_VisaRGB KraftigareFargmattnad SKS_VisaCIR
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params
    
    
def get_tif_image_url(params):
    #url = 'https://imgutv.svo.local/arcgis/rest/services/Ortofoto_2_0/ImageServer/exportImage'
    #url = 'http://imgutv.svo.local/arcgis/rest/services/Ortofoto_2_0/ImageServer/exportImage'
    url = 'http://193.183.28.189/arcgis/rest/services/Ortofoto_2_0/ImageServer/exportImage'
    resp = requests.get(url=url, params=params, verify=False)
    data = resp.json()
    return data['href']


def get_image_coordinates( geodf_separata_polygoner, offset=150):

    cluster_ids = []

    coordinates = []
    for i, poly in enumerate(geodf_separata_polygoner.iterrows()):
        coord = poly[1]['geometry'].centroid.coords[0]
        cluster_ids.append(poly[1]['TARGET_FID'])
        coordinates.append([coord[0]-offset, coord[1]-offset, coord[0]+offset, coord[1]+offset])     
    return coordinates


# def get_image_polygon_with_coords_offset(polygon, offset, image_size=512,polygon_offset=0):

#     center_offsets = [0,0]
#     if type(polygon_offset) != int:
#         center = polygon['geometry'].centroid.coords[0]
#         center_offset = polygon_offset['geometry'].centroid.coords[0]
#         center_offsets = [center[0]-center_offset[0], center[1]-center_offset[1]]

    
#     coords = polygon['geometry'].exterior.coords[:]
#     center = polygon['geometry'].centroid.coords[0]
#     new_coordinates = []
#     for c in coords:
#         co1 = 0
#         co2 = 0
#         if len(center_offsets) > 0:
#             co1 = center_offsets[0]
#             co2 = center_offsets[1]

#         new_coordinates.append(((c[0]-center[0]+co1)*image_size, image_size-(c[1]-center[1]+co2)*image_size))
#     return Polygon(new_coordinates)


def get_image_polygon_training_data(polygon, image_size=512,polygon_offset=0, meters_per_pixel=1):

    center_offsets = [0,0]
    if type(polygon_offset) != int:
        center = polygon['geometry'].centroid.coords[0]
        center_offset = polygon_offset['geometry'].centroid.coords[0]
        center_offsets = [center[0]-center_offset[0], center[1]-center_offset[1]]

    
    coords = polygon['geometry'].exterior.coords[:]
    interiors = polygon['geometry'].interiors
    center = polygon['geometry'].centroid.coords[0]
    new_coordinates = []
    for c in coords:
        co1 = 0
        co2 = 0
        if len(center_offsets) > 0:
            co1 = center_offsets[0]
            co2 = center_offsets[1]

        new_coordinates.append((((c[0]-center[0]+co1)/meters_per_pixel) + image_size/2, image_size-((c[1]-center[1]+co2)/meters_per_pixel)-image_size/2))

    new_interiors = []
    for interior in interiors:
        new_interiors.append([])
        inte = interior.coords[:]
        for c in inte:
            i1 = 0
            i2 = 0
            if len(center_offsets) > 0:
                i1 = center_offset[0]
                i2 = center_offset[1]
                new_interiors[-1].append((((c[0]-center[0]+co1)/meters_per_pixel) + image_size/2, image_size-((c[1]-center[1]+co2)/meters_per_pixel)-image_size/2))

    return Polygon(new_coordinates, new_interiors)



def get_image_polygon(polygon, offset=150, image_size=512,polygon_offset=0):

    center_offsets = [0,0]
    if type(polygon_offset) != int:
        center = polygon['geometry'].centroid.coords[0]
        center_offset = polygon_offset['geometry'].centroid.coords[0]
        center_offsets = [center[0]-center_offset[0], center[1]-center_offset[1]]

    
    coords = polygon['geometry'].exterior.coords[:]
    center = polygon['geometry'].centroid.coords[0]
    new_coordinates = []
    for c in coords:
        co1 = 0
        co2 = 0
        if len(center_offsets) > 0:
            co1 = center_offsets[0]
            co2 = center_offsets[1]
        new_coordinates.append(((c[0]-center[0]+offset+co1)/(offset*2)*image_size, image_size-(c[1]-center[1]+offset+co2)/(offset*2)*image_size))
        # if offset != 0:
        #     new_coordinates.append(((c[0]-center[0]+offset+co1)/(offset*2)*image_size, image_size-(c[1]-center[1]+offset+co2)/(offset*2)*image_size))
        # elif co1 != 0 and co2 != 0:
        #     new_coordinates.append(((c[0]-center[0]+co1)/(co1*2)*image_size, image_size-(c[1]-center[1]+co2)/(co2*2)*image_size))
        # else:
        #     print((c[0]-center[0]+co1))
        #     new_coordinates.append(((c[0]-center[0]+co1)*image_size, image_size-(c[1]-center[1]+co2)*image_size))
    return Polygon(new_coordinates)



def download_tif_image(url, params, username, password, image_path='input.png'):
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
    # Make request to get the image bytes
    img_bytes = requests.get(image_href, auth=(username, password), stream=True)
    # Save the image to disk
    try:
        # Create image dir if not exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Read and write the image bytes to disk
        with open(image_path, 'wb') as f:
            img_bytes.raw.decode_content = True
            shutil.copyfileobj(img_bytes.raw, f)
    except Exception as e:
        raise Exception(f"Error when saving image to file {image_path}. Shutil Error: {e}")



def save_polygon_to_image(df_separata_polygoner, source='input.png',destination='result.png'):


    alpha = 0.5 # that's your transparency factor
    path = source

    polys = get_image_polygon(df_separata_polygoner,polygon_offset=0)
    for i, poly in enumerate(polys):
        poly = poly[0]
        exterior = poly.exterior.coords

        image = cv2.imread(path)
        overlay = image.copy()
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exterior = [int_coords(exterior)]
        cv2.fillPoly(overlay, exterior, color=(50, 255, 200))
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.imwrite("images/{}_{}".format(df_separata_polygoner.loc[i]['cluster_id'],destination),overlay)
        #cv2.imshow("Polygon", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


def save_polygon_and_healthy_to_image(polygon, healthy_polygon,  source='input.png',destination='polygon.png', color=(50, 255, 200)):
    
    alpha = 0.5 # that's your transparency factor
    path = source

    poly = get_image_polygon(polygon,polygon_offset=0)

    healthy_poly = get_image_polygon(healthy_polygon,polygon_offset=polygon)


    exterior = poly.exterior.coords
    exterior_healthy = healthy_poly.exterior.coords
    image = cv2.imread(path)
    overlay = image.copy()
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(exterior)]
    exterior_healthy = [int_coords(exterior_healthy)]
    #exterior = np.concatenate(exterior, exterior_healthy)
    res = np.concatenate((exterior, exterior_healthy),axis=1)
    #cv2.fillPoly(overlay, res, color=(50, 255, 200))
    #cv2.polylines(overlay, res, isClosed=True,color=(50, 255, 200))
    cv2.polylines(overlay, res, isClosed=True,color=color)
    new_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.imwrite(destination,new_image)
    #cv2.imshow("Polygon", image)
    #cv2.waitKey(0)




def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def add_median_diagram(poly,current_date, source, destination):
    print("open: {}".format(source))
   
    
    fig, subplot = plt.subplots( 2,1, figsize=[10, 10],gridspec_kw={'height_ratios': [5, 1]})
    fig.tight_layout()

    timestamps = json.loads(poly['timestamps'])
    subplot[1].plot(timestamps,json.loads(poly['diff_smooth_median']), label='diff median')

    subplot[1].plot(timestamps,json.loads(poly['smooth_acorvi_median']), label='median')

    subplot[1].plot(timestamps,json.loads(poly['smooth_healthy_acorvi_median']), label='healthy median')
    if len(poly['change_dates']) > 0:
        change_date = json.loads(poly['change_dates'])[1]
        subplot[1].plot(change_date, [0.1], marker="o", markersize=4)
        subplot[1].axvline(change_date,color='red')
    subplot[1].plot(current_date,[0.1], marker='o',markersize=4)
    subplot[1].axvline(current_date,color='green')
        
    subplot[1].legend(loc='upper center')

    datum = json.loads(poly['datum'])
    split = (int)(len(datum)/5)
    subplot[1].plot(timestamps,json.loads(poly['smooth_healthy_acorvi_median']), label='healthy median')
    plt.sca(subplot[1])
    filtered_timestamps = timestamps[0::split]
    filtered_datum = datum[0::split]
    plt.xticks(filtered_timestamps,filtered_datum)

    im = plt.imread(source)
    subplot[0].imshow(im)
    subplot[0].axis('off')
    fig.savefig(source)
    plt.close(fig)
    print("save image: {}".format(destination))


def create_rgb_images(df_separata_polygoner,df_healthy_separata_polygoner,username, password):
    image_size = 512
    offset = 150
    image_coordinates = get_image_coordinates(df_separata_polygoner,offset)

    for i, poly in enumerate(df_separata_polygoner.iterrows()):

        ensure_dir("images/review/{}".format(poly[1]['cluster_id']))
        name_first = "images/review/{}/2019".format(poly[1]['cluster_id'])
        name_second = "images/review/{}/2021".format(poly[1]['cluster_id'])

        params = get_ortophoto_request_params(coordinates_list=image_coordinates[i], image_year='2019', rendering_rule='KraftigareFargmattnad',image_size=image_size)
        image_url = get_tif_image_url(params)
        download_tif_image(image_url, params, username, password, "{}_{}.png".format(name_first,i))
        save_polygon_and_healthy_to_image(poly[1],df_healthy_separata_polygoner.iloc[i], source="{}_{}.png".format(name_first,i),destination="{}_{}.png".format(name_first,i))

        params = get_ortophoto_request_params(coordinates_list=image_coordinates[i], image_year='2021', rendering_rule='KraftigareFargmattnad',image_size=image_size)
        image_url = get_tif_image_url(params)
        download_tif_image(image_url, params, username, password, "{}_{}.png".format(name_second,i))

        save_polygon_and_healthy_to_image(poly[1],df_healthy_separata_polygoner.iloc[i], source="{}_{}.png".format(name_second,i),destination="{}_{}.png".format(name_second,i))

def create_acorvi_timelaps(df_separata_polygoner,df_healthy_separata_polygoner, username, password,threshold=0.999):
    image_size = 512
    offset = 150
    image_coordinates = get_image_coordinates(df_separata_polygoner,offset)

    l = 0
    for i, poly in enumerate(df_separata_polygoner.iterrows()):

        bad_files = []
        good_dates = pr.get_cloudfree_dates(poly[1],threshold=threshold)
        for d in tqdm.tqdm(good_dates):
            l +=1
            date_time = datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d')
            #if date_time == '2021-03-30':
            # date_time = datetime.fromtimestamp(good_dates[-1]/1000).strftime('%Y-%m-%d')
                
            #print(date_time)
            name_first = "images/review/{}/{}/".format(poly[1]['cluster_id'],i)
            ensure_dir(name_first)

            params = get_ortophoto_sentinel_2_3_0_request_params(coordinates_list=image_coordinates[i], date=date_time,image_size=image_size, rendering_rule='ndvi')
            image_url = get_tif_image_url_sentinel_2_3_0(params)
            download_tif_image(image_url, params, username, password, "{}{}.png".format(name_first,d))


        
            try:
                k1 = Image.open("{}{}.png".format(name_first,d))
                k = k1.copy()
                im = Image.fromarray(np.array(k)*127+128)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                #im.save("{}_{}_{}_acorvi.png".format(name_first,date_time,i))
                im.save("{}{}.png".format(name_first,d))
                save_polygon_and_healthy_to_image(poly[1],df_healthy_separata_polygoner.iloc[i],"{}{}.png".format(name_first,d),"{}{}.png".format(name_first,d),color=(0,0,255))
                add_median_diagram(poly[1],d, "{}{}.png".format(name_first,d),"{}{}.png".format(name_first,d))
            except(Exception)as e:
                #print(e)
                #os.remove("{}_{}_{}_acorvi.png".format(name_first,date_time,i))
                #print("bad file")
                bad_files.append("{}{}.png".format(name_first,d))
                continue
                #break

            #if l > 20:
            #    break

        for bad_file in bad_files:
            os.remove(bad_file)
        video_path = "images/review/{}/video_review_{}.avi".format(poly[1]['cluster_id'],i)
        
        

        image_folder = 'images/review/{}/{}'.format(poly[1]['cluster_id'],i)
        sorted_files = []
        for root, dirs, filenames in os.walk(image_folder):
            files = [int(f.split('.')[0]) for f in filenames if f.split('.')[0].isnumeric()]
            files.sort()
            sorted_files = [os.path.join(root,str(f)+'.png') for f in files]

        if len(sorted_files) > 0:
            frame = cv2.imread(sorted_files[0])
            height, width, layers = frame.shape

            video = cv2.VideoWriter(video_path, 0, 15, (width,height))

            for image in sorted_files:
                video.write(cv2.imread(image))

            cv2.destroyAllWindows()
            video.release()

def create_review_data(df_separata_polygoner, df_healthy_separata_polygoner, threshold=0.999):
    create_rgb_images(df_separata_polygoner, df_healthy_separata_polygoner)
    create_acorvi_timelaps(df_separata_polygoner,df_healthy_separata_polygoner,threshold)
