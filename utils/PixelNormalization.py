import numpy as np
import math
import datetime

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

vectorized_sigmoid = np.vectorize(sigmoid)

def min_max_scaler(image_array, min_value, max_value):
    image_array_scaled = (image_array - min_value) / (max_value - min_value) * 255
    return image_array_scaled

def rescale_land_cover_types(image_array):
    #print("marktackedata max:")
    #print(image_array.max())
    return image_array # don't actually rescale these values since we want to preserve the categorical values

def rescale_treeheight(image_array):
    min_value = 0
    max_value = 500
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_elevation(image_array):
    min_value = -322
    max_value = 2100
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_soil_moisture(image_array):
    min_value = 0
    max_value = 101
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_elevation_gradient(image_array):
    min_value = 0
    max_value = 89
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_max_temp(image_array):
    min_value = 0
    max_value = 40
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_sum_temp(image_array):
    min_value = 0
    max_value = 60
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_ditches(image_array):
    new_image_array = image_array
    return new_image_array

def rescale_soil_depth(image_array):
    min_value = 0
    max_value = 88
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_soil_type(image_array):
    min_value = 0
    max_value = 4
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_biomass(image_array):
    min_value = 0
    max_value = 1246
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

def rescale_leaves_present(image_array):
    new_image_array = image_array
    return new_image_array

def rescale_sattelite_img(image_array):
    min_value = 0
    max_value = 255
    new_image_array = min_max_scaler(image_array,min_value,max_value)
    return new_image_array

# def normalization_skogligagrunddata_2_0(image_array):
#     min_value = [0, 0, 0, 0, 0, 0, 0, 14393]
#     max_value = [3906, 501, 139, 115, 1246, 500, 100, 18318]
#     new_image_array = np.array([min_max_scaler(image_array[:,:,channel_id],min_value[channel_id],max_value[channel_id])
#                                 for channel_id
#                                 in np.arange(image_array.shape[2])]).transpose(1,2,0)
#     return new_image_array




def normalize_to_summer(day_in_year):
    days_to_summer = day_in_year-182
    normalized_day_in_year = abs(days_to_summer)/182
    return normalized_day_in_year

def days_in_year(days_since_june_1_1970):
    # Convert days since June 1, 1970 to a date object

    start_date = datetime.date(1970, 6, 1) + datetime.timedelta(days=days_since_june_1_1970)
    
    # Get the start of the current year
    current_year_start = datetime.date(start_date.year, 1, 1)
    # Calculate the number of days that have elapsed in the current year
    elapsed_days = (start_date - current_year_start).days
    
    # Return the result
    return elapsed_days


def normalization_lon_lat(image_array):
    print("normalize longitude and latitude")
    min_value = [260000, 6132000]
    max_value = [920000, 7690000]
    new_image_array = np.array([min_max_scaler(image_array[:,:,channel_id],min_value[channel_id],max_value[channel_id])
                                for channel_id
                                in np.arange(image_array.shape[2])]).transpose(1,2,0)
    return new_image_array


def none(image_array):
    return image_array