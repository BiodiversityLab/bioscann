import requests
import utils.PixelNormalization as PN
import ee
import geemap
import pandas as pd

def GEE_initialize(service_account,json_path):
    credentials = ee.ServiceAccountCredentials(service_account, json_path)
    ee.Initialize(credentials)

def standard_params(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2],
                                     coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson'
    )
    return params

def params_land_cover_types(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
#        renderingRule='{"rasterfunction":"SKS_NMD_ogeneraliserad"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_tree_height_alternative(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"SKS_Tradhojd_rod"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_tree_height(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2],coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='1',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_elevation_shading(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"Terrangskuggning"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_elevation_gradient(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"Lutning"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_soil_moisture(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='0',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson'
    )
    return params

def params_depth_to_water(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"SKS_DTWMarkfuktighet"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='0',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson'
    )

def params_peat_depth(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"SKS_Torv"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_soil_type(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"SKS_TorvKlassad"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_biomass(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2],coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='4',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_leaves_present(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"SKS_LovAvlov"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_leaves_present_2024(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2],coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='8',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_satellite_img(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"SvartVit"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='0',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson'
    )
    return params

#
# def params_bioclim(coordinates_list, image_year='', rendering_rule='', image_size=512):
#     coordinates_string = str()
#     params = dict(
#         bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
#         bboxSR='3006',
#         size='{},{}'.format(image_size,image_size),
#         imageSR='',
#         time='',
#         format='tiff',
#         pixelType='UNKNOWN','bio01',gee_output_file,meters_per_pixel,gee_geometry,crs
#         noData='',
#         noDataInterpretation='esriNoDataMatchAny',
#         interpolation='+RSP_BilinearInterpolation',
#         renderingRule='{"rasterfunction":"SvartVit"}',
#         compression='LZ77',
#         compressionQuality='',
#         bandIds='0',
#         sliceId='',
#         adjustAspectRatio='true',
#         validateExtent='false',
#         lercVersion='1',
#         compressionTolerance='',
#         f='pjson'
#     )
#     return params


def params_maxtemp(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2],coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='1713225600000',
        # time='1672527601000',
        # MinDate='2023/01/01 00:00:00 UTC',
        # MaxDate='2023/12/31 23:59:59 UTC',
        # ImageYear='2023',
        # NumDays='365',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        # renderingRule='None',
        renderingRule='{"rasterfunction":"Maxtemp"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='1',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_sumtemp(coordinates_list, image_year='', rendering_rule='', image_size=512):
    coordinates_string = str()
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size, image_size),
        imageSR='',
        time='1713225600000',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        renderingRule='{"rasterfunction":"TempSumma"}',
        compression='LZ77',
        compressionQuality='',
        bandIds='1',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params

def params_gee_ndvi(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['NDVI',coordinates_list,image_size,start_date,end_date]
    return params

def params_gee_temperature(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['bio01',coordinates_list,image_size,start_date,end_date]
    return params

def params_gee_temperature_seasonality(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['bio04',coordinates_list,image_size,start_date,end_date]
    return params

def params_gee_precipitation(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['bio12',coordinates_list,image_size,start_date,end_date]
    return params

def params_gee_precipitation_seasonality(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['bio15',coordinates_list,image_size,start_date,end_date]
    return params

def params_gee_hii(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['hii',coordinates_list,image_size,start_date,end_date]
    return params

def params_gee_elevation(coordinates_list, image_year='2024', rendering_rule='', image_size=512):
    start = '%s-01-01'%(image_year)
    end = '%s-12-31'%(image_year)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    params = ['ASTER/GDEM',coordinates_list,image_size,start_date,end_date]
    return params



def url_land_cover_types():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Marktackedata_2_0/ImageServer/exportImage'
    return url

# def url_treeheight():
#     url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Samverkan/Tradhojd/ImageServer/exportImage' #newer version
#     # url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Tradhojd/ImageServer/exportImage'
#     return url

def url_treeheight():
    url = "https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Tradhojd_3_1/ImageServer/exportImage"
    return url

def url_elevation():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Markhojdmodell_05m_1_0/ImageServer/exportImage'
    return url

def url_soil_moisture():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Markfuktighet_SLU_2_0/ImageServer/exportImage'
    return url

def url_depth_to_water():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Samverkan/Markfuktighet_DTW_1_1/ImageServer/exportImage'
    return url

def url_max_temp():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/CDS_Maxtemp_1_0/ImageServer/exportImage'
    return url

def url_max_temp_2024():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/GBB_SvarmTemp_1_0/ImageServer/exportImage'
    return url

def url_sum_temp():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/CDS_Tempsum_1_0/ImageServer/exportImage'
    return url

def url_sum_temp_2024():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/GBB_SvarmTemp_1_0/ImageServer/exportImage'
    return url

def url_ditches():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Diken_1_0/ImageServer/exportImage'
    return url

def url_soildata():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Skogsdatalabbet/Torvkarta_1_0/ImageServer/exportImage'
    return url

# def url_basic_forest_attributes():
#     url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/SkogligaGrunddata/ImageServer/exportImage'
#     return url

def url_basic_forest_attributes():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/SkogligaGrunddata_3_1/ImageServer/exportImage'
    return url

def url_sattelite_img():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Satellitdata_2_0/ImageServer/exportImage'
    return url

def get_satellite_data_gee(target_channel,gee_output_file,gee_geometry,meters_per_pixel,crs,start_date,end_date):
    def maskS2clouds(image):
        # This function is copied from GEE examples library and converted from JavaScript to Python
        # Function to mask clouds using the Sentinel-2 QA band.
        qa = image.select('QA60')
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        # Return the masked and scaled data, without the QA bands.
        return image.updateMask(mask).divide(10000).select("B.*").copyProperties(image, ["system:time_start"])
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) \
        .map(maskS2clouds)
    s2_collection = s2_collection.mean()
    if target_channel == 'NDVI':
        target_channel = s2_collection.normalizedDifference(['B8', 'B4']).rename("NDVI")
    else:
        target_channel = s2_collection.select(target_channel)
    geemap.ee_export_image(target_channel, gee_output_file, scale=meters_per_pixel, region=gee_geometry, crs=crs)

def get_climate_data_bioclim_gee(target_channel,gee_output_file,gee_geometry,meters_per_pixel,crs,start_date,end_date):
    bioclim_image = ee.Image('WORLDCLIM/V1/BIO')
    output_channel = bioclim_image.select(target_channel)
    geemap.ee_export_image(output_channel, gee_output_file, scale=meters_per_pixel, region=gee_geometry, crs=crs)

def get_hii_data_gee(target_channel,gee_output_file,gee_geometry,meters_per_pixel,crs,start_date,end_date):
    hii_coll = ee.ImageCollection("projects/HII/v1/" + target_channel).filter(ee.Filter.calendarRange(2020, 2020, 'year'))
    hii_image = hii_coll.limit(1, 'system:time_start', False).first()
    # img_collection = ee.ImageCollection.fromImages([hii_image])
    # stacked_image = img_collection.toBands()
    geemap.ee_export_image(hii_image, gee_output_file, scale=meters_per_pixel, region=gee_geometry, crs=crs)

def get_elevation_data_gee(target_channel,gee_output_file,gee_geometry,meters_per_pixel,crs,start_date,end_date):
    dem_image = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM')
    geemap.ee_export_image(dem_image, gee_output_file, scale=meters_per_pixel, region=gee_geometry, crs=crs)


#Storing all functions in an object
apis = {
    'land_cover_types': {'params': params_land_cover_types, 'url': url_land_cover_types,'normalization': PN.rescale_land_cover_types},
    'treeheight': {'params': params_tree_height, 'url': url_basic_forest_attributes,'normalization': PN.rescale_treeheight},
    "biomass": {'params': params_biomass, 'url': url_basic_forest_attributes, 'normalization': PN.rescale_biomass},
    "leaves_present": {'params': params_leaves_present, 'url': url_basic_forest_attributes, 'normalization': PN.rescale_leaves_present},
    'elevation_shading': {'params': params_elevation_shading, 'url': url_elevation, 'normalization': PN.rescale_elevation}, # this is only available for part of the country
    "elevation_gradient": {'params': params_elevation_gradient, 'url': url_elevation,'normalization': PN.rescale_elevation_gradient},
    "soil_moisture": {'params': params_soil_moisture, 'url': url_soil_moisture,'normalization': PN.rescale_soil_moisture},
    "depth_to_water":{'params': params_depth_to_water, 'url':url_depth_to_water, 'normalization': PN.rescale_depth_to_water},
    "max_temp": {'params': standard_params, 'url': url_max_temp,'normalization': PN.rescale_max_temp},
    "max_temp_2024": {'params': params_maxtemp, 'url': url_max_temp_2024, 'normalization': PN.rescale_max_temp},
    "sum_temp": {'params': standard_params, 'url': url_sum_temp,'normalization': PN.rescale_sum_temp},
    "sum_temp_2024": {'params': params_sumtemp, 'url': url_sum_temp_2024,'normalization': PN.rescale_sum_temp},
    "ditches": {'params': standard_params, 'url': url_ditches,'normalization': PN.rescale_ditches},
    "peat_depth": {'params': params_peat_depth, 'url': url_soildata,'normalization': PN.rescale_peat_depth},
    "soil_type": {'params': params_soil_type, 'url': url_soildata,'normalization': PN.rescale_soil_type},
    "satellite": {'params': params_satellite_img, 'url': url_sattelite_img,'normalization': PN.rescale_sattelite_img},
    "gee_ndvi": {'params': params_gee_ndvi, 'function': get_satellite_data_gee,'normalization': PN.rescale_gee_ndvi},
    "gee_temperature": {'params': params_gee_temperature, 'function': get_climate_data_bioclim_gee,'normalization': PN.rescale_gee_temperature},
    "gee_temperature_seasonality": {'params': params_gee_temperature_seasonality, 'function': get_climate_data_bioclim_gee,'normalization': PN.rescale_gee_temperature_seasonality},
    "gee_precipitation": {'params': params_gee_precipitation, 'function': get_climate_data_bioclim_gee,'normalization': PN.rescale_gee_precipitation},
    "gee_precipitation_seasonality": {'params': params_gee_precipitation_seasonality, 'function': get_climate_data_bioclim_gee,'normalization': PN.rescale_gee_precipitation_seasonality},
    "gee_hii": {'params': params_gee_hii, 'function': get_hii_data_gee, 'normalization': PN.rescale_gee_hii},
    "gee_elevation": {'params': params_gee_elevation, 'function': get_elevation_data_gee, 'normalization': PN.rescale_gee_elevation}
}
    # "era5_bioclim":{'params': params_bioclim, 'url': url_bioclim,'normalization': PN.rescale_bioclim},
    # "s2_monthly":{'params': params_s2, 'url': url_s2,'normalization': PN.rescale_s2},
    # "hii":{'params': params_hii, 'url': url_hii,'normalization': PN.rescale_hii},
    # "elevation_dem":{'params': params_elevation_dem, 'url': url_elevation_dem,'normalization': PN.rescale_elevation_dem}


