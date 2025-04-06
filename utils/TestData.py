import utils.TestDataCompose as tdc
import numpy as np
import utils.ImageRequest as ir
import os
import pdb


def compose_testdata(output_path, channels, ref_image, img_size, test_config, center_polygon):
    print("compose testdata")
    images = []
    image_paths = []
    try:
        for feature_composition in test_config['feature_preprocess_compositions']:
            #pdb.set_trace()
            mask_image = np.ones((img_size, img_size))
            mask_image = tdc.compose_method[feature_composition['compose']](channels, mask_image)
            # pdb.set_trace()
            #pdb.set_trace()
            image_name = os.path.join(output_path,'{}-{}.tiff'.format(center_polygon['TARGET_FID'],feature_composition['compose']))
            image = np.array([mask_image])
            images.append(image)
            ir.write_geotif_at_location(ref_image, image_name, image)
            image_paths.append(image_name)
    except AttributeError as e:
        print("AttributeError: 'NoneType' object has no attribute 'SetGeoTransform'")
        print("Exception message:", e)
        return False, False
    return images, image_paths