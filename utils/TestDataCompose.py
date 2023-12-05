from tifffile import imread
import pdb
import exifread


#import pdb
def marktacke_2_0_moist(channels, mask_image):
    print("moist")
    mask_values = [121,122,123,124]
  
    channel = [c for c in channels if 'land_cover_types' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    
    if len(img.shape) > 2:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y][0]) not in mask_values:
                    mask_image[x][y] = 0
    return mask_image


def marktacke_2_0_dry(channels, mask_image):
    print("dry")
    mask_values = [111,112,113,114]
  
    channel = [c for c in channels if 'land_cover_types' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    
    if len(img.shape) > 2:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y][0]) not in mask_values:
                    mask_image[x][y] = 0
    return mask_image
    

def marktacke_2_0_coniferous(channels, mask_image):
    mask_values = [111,112,113,121,122,123]
  
    channel = [c for c in channels if 'land_cover_types' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    
    if len(img.shape) > 2:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y][0]) not in mask_values:
                    mask_image[x][y] = 0
    return mask_image


def marktacke_2_0_deciduous(channels, mask_image):
    
    mask_values = [115,116,117,125,126,127]
    channel = [c for c in channels if 'land_cover_types' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    
    if len(img.shape) > 2:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y][0]) not in mask_values:
                    mask_image[x][y] = 0
    return mask_image

def marktacke_2_0_coniferous_deciduous_mix(channels, mask_image):
    
    mask_values = [114,124]
    channel = [c for c in channels if 'land_cover_types' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    
    if len(img.shape) > 2:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y][0]) not in mask_values:
                    mask_image[x][y] = 0
    return mask_image


def tradhojd_3_2_over_15_m(channels, mask_image):

    channel = [c for c in channels if 'treeheight' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    if len(img.shape) > 1:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y]) < 200:
                    mask_image[x][y] = 0
    return mask_image

def tradhojd_3_2_under_15_m(channels, mask_image):
    #pdb.set_trace()
    channel = [c for c in channels if 'treeheight' in c][0]
    img = imread(channel)
    #pdb.set_trace()
    if len(img.shape) > 1:
        for x in range(len(mask_image[0])):
            for y in range(len(mask_image[1])):
                if (img[x][y]) >= 200:
                    mask_image[x][y] = 0
    return mask_image

def latitude_north_of_7000000(channels, mask_image):
    tiff_file = open(channels[0], 'rb')
    coordinates = exifread.process_file(tiff_file)['Image Tag 0x8482']
    
    for x in range(len(mask_image[0])):
        for y in range(len(mask_image[1])):
            if coordinates.values[4][0] <= 7000000:
                mask_image[x][y] = 0
    return mask_image

def latitude_south_of_6590000(channels, mask_image):
    tiff_file = open(channels[0], 'rb')
    coordinates = exifread.process_file(tiff_file)['Image Tag 0x8482']
    
    for x in range(len(mask_image[0])):
        for y in range(len(mask_image[1])):
            if coordinates.values[4][0] >= 6590000:
                mask_image[x][y] = 0
    return mask_image


compose_method = {
    'marktacke_2_0_moist': marktacke_2_0_moist,
    'marktacke_2_0_dry': marktacke_2_0_dry,
    'marktacke_2_0_coniferous': marktacke_2_0_coniferous,
    'marktacke_2_0_deciduous': marktacke_2_0_deciduous,
    'marktacke_2_0_coniferous_deciduous_mix': marktacke_2_0_coniferous_deciduous_mix,
    'tradhojd_3_2_over_15_m': tradhojd_3_2_over_15_m,
    'tradhojd_3_2_under_15_m': tradhojd_3_2_under_15_m,
    'latitude_north_of_7000000': latitude_north_of_7000000,
    'latitude_south_of_6590000' : latitude_south_of_6590000
}