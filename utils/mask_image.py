import utils.ImageRequest as ir
from tifffile import imread
import utils.RestAPIs as ra

def create_filter_mask(feature_layer, filter_image, feature_layer_config):
    #print(feature_layer.shape)
    for index, channel in enumerate(feature_layer_config['channels']):
        if len(feature_layer.shape) > 2:
            for x in range(len(filter_image[0])):
                for y in range(len(filter_image[1])):
                    if len(feature_layer_config['mask_values']) != 0:
                        if(float)(feature_layer[x][y][channel]) not in feature_layer_config['mask_values'][index]:
                            filter_image[x][y] = 0
                    if len(feature_layer_config['mask_values_exclude']) != 0:
                        if(float)(feature_layer[x][y][channel]) in feature_layer_config['mask_values_exclude'][index]:
                            filter_image[x][y] = 0
    return filter_image


def mask_with_annotation(chan, mask_image):
    return chan * mask_image


def get_mask_image(indata_mask_configurations, mask_image, center_polygon, indata_path, img_size, username, password, get_image_coord=True):
    for feature_layer_config in indata_mask_configurations:
        feature_layer = ir.get_channels(center_polygon, indata_path, img_size, selected_apis=[feature_layer_config['api']], apis=ra.apis, username=username, password=password, get_image_coord=get_image_coord)
        feature_layer = imread(feature_layer[0])
        mask_image = create_filter_mask(feature_layer, mask_image, feature_layer_config)
    return mask_image

def download_landcover_channel(indata_mask_configurations, image_coordinates, indata_path, img_size, username, password):
    for feature_layer_config in indata_mask_configurations:
        feature_layer = ir.get_channels(image_coordinates, indata_path, img_size, selected_apis=[feature_layer_config['api']], apis=ra.apis, username=username, password=password, get_image_coord=False)
    return feature_layer