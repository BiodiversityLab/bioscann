import utils.ortophoto as op
from tifffile import imsave, imread
import numpy as np
import os 

def get_channels(center_polygon,training_dataset_path,img_size, selected_apis, apis, get_image_coord=True):
        #image_size = 512
    offset = img_size / 2
    meters_per_pixel = 10
    offset = offset * meters_per_pixel
    channels = []
    if get_image_coord:
        image_coordinates = op.get_image_coordinates(center_polygon,offset)

        
        for selected_api in selected_apis:
            channels.append(get_image(training_dataset_path, center_polygon, image_coordinates, 0, img_size, selected_api, apis))
    else:
        image_coordinates = center_polygon
        for selected_api in selected_apis:
            channels.append(get_image_no_polygon(training_dataset_path, image_coordinates, 0, img_size, selected_api, apis))
    return channels


def get_image(dataset_path, poly, image_coordinates, polygon_index, img_size,rest_api, apis, username, password):

    print(poly)
    print(poly.iloc[0])
    print(poly.iloc[0]['TARGET_FID'])
    image_name = "{}/id_{}_{}_{}.tiff".format(dataset_path,poly.iloc[0]['TARGET_FID'],polygon_index,rest_api)
    params = apis[rest_api]['params'](image_coordinates[0], image_size=img_size)

    image_url = apis[rest_api]['url']()

    op.download_tif_image(image_url, params, username, password, image_name)

    return image_name


def get_image_no_polygon(dataset_path, image_coordinates, polygon_index, img_size,rest_api, apis, username, password):

    image_name = "{}/id_{}.tiff".format(dataset_path,rest_api)
    params = apis[rest_api]['params'](image_coordinates, image_size=img_size)

    image_url = apis[rest_api]['url']()

    op.download_tif_image(image_url, params, username, password, image_name)

    return image_name


def compose_indata(dataset_path, channels, image_name=''):
    imgs = []
    for i, img in enumerate(channels):
        im = imread(img)
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
                cluster_id = img.split('/')[-1].split('_')[1]
                imsave(os.path.join(dataset_path,"{}.tiff".format(cluster_id)),imgs, compress='lzma')
            else:
                imsave(os.path.join(dataset_path,image_name),imgs, compress='lzma')
                return image_name, len(imgs)

        except Exception as e:

            print(e)
            print(i) 