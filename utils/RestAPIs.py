import requests
import utils.PixelNormalization as PN



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

def params_tree_height(coordinates_list, image_year='', rendering_rule='', image_size=512):
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

def params_elevation(coordinates_list, image_year='', rendering_rule='', image_size=512):
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
        renderingRule='{"rasterfunction":"Terrangskuggning"}', #SKS_VisaRGB KraftigareFargmattnad SKS_VisaCIR
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
        bandIds='0',
        sliceId='',
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson'
    )
    return params

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
        renderingRule='{"rasterfunction":"Biomassa0"}',# renderingRule = '{"rasterfunction":"Biomassa"}'
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

def params_biomass_2024(coordinates_list, image_year='', rendering_rule='', image_size=512):
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


def params_bioclim(coordinates_list, image_year='', rendering_rule='', image_size=512):
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


def url_land_cover_types():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Marktackedata_2_0/ImageServer/exportImage'
    return url

def url_treeheight():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Samverkan/Tradhojd/ImageServer/exportImage' #newer version
    # url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Tradhojd/ImageServer/exportImage'
    return url

def url_treeheight_2024():
    url = "https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Tradhojd_3_1/ImageServer/exportImage"
    return url

def url_elevation():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/HojdmodellLaserdataSkog_1_0/ImageServer/exportImage'
    return url

def url_soil_moisture():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Markfuktighet_SLU_2_0/ImageServer/exportImage'
    return url

def url_elevation_gradient():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Lutning/ImageServer/exportImage'
    return url

def url_elevation_gradient_2024():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Lutning_1_0/ImageServer/exportImage'
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

def url_basic_forest_attributes():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/SkogligaGrunddata/ImageServer/exportImage'
    return url

def url_basic_forest_attributes_2024():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/SkogligaGrunddata_3_1/ImageServer/exportImage'
    return url

def url_sattelite_img():
    url = 'https://geodata.skogsstyrelsen.se/arcgis/rest/services/Publikt/Satellitdata_2_0/ImageServer/exportImage'
    return url


#Storing all functions in an object
apis = {
    'land_cover_types': {'params': params_land_cover_types, 'url': url_land_cover_types,'normalization': PN.rescale_land_cover_types},
    'treeheight': {'params': params_tree_height, 'url': url_treeheight, 'normalization': PN.rescale_treeheight},
    'treeheight_2024': {'params': params_tree_height, 'url': url_treeheight_2024,'normalization': PN.rescale_treeheight},
    'elevation': {'params': params_elevation, 'url': url_elevation, 'normalization': PN.rescale_elevation}, # this is only available for part of the country
    "soil_moisture": {'params': params_soil_moisture, 'url': url_soil_moisture,'normalization': PN.rescale_soil_moisture},
    "elevation_gradient": {'params': standard_params, 'url': url_elevation_gradient,'normalization': PN.rescale_elevation_gradient},
    "elevation_gradient_2024": {'params': standard_params, 'url': url_elevation_gradient_2024,'normalization': PN.rescale_elevation_gradient},
    "max_temp": {'params': standard_params, 'url': url_max_temp,'normalization': PN.rescale_max_temp},
    "max_temp_2024": {'params': params_maxtemp, 'url': url_max_temp_2024, 'normalization': PN.rescale_max_temp},
    "sum_temp": {'params': standard_params, 'url': url_sum_temp,'normalization': PN.rescale_sum_temp},
    "sum_temp_2024": {'params': params_sumtemp, 'url': url_sum_temp_2024,'normalization': PN.rescale_sum_temp},
    "ditches": {'params': standard_params, 'url': url_ditches,'normalization': PN.rescale_ditches},
    "peat_depth": {'params': params_peat_depth, 'url': url_soildata,'normalization': PN.rescale_peat_depth},
    "soil_type": {'params': params_soil_type, 'url': url_soildata,'normalization': PN.rescale_soil_type},
    "biomass": {'params': params_biomass, 'url': url_basic_forest_attributes,'normalization': PN.rescale_biomass},
    "biomass_2024": {'params': params_biomass_2024, 'url': url_basic_forest_attributes_2024, 'normalization': PN.rescale_biomass},
    "leaves_present": {'params': params_leaves_present, 'url': url_basic_forest_attributes,'normalization': PN.rescale_leaves_present},
    "leaves_present_2024": {'params': params_leaves_present_2024, 'url': url_basic_forest_attributes_2024,'normalization': PN.rescale_leaves_present},
    "satellite": {'params': params_satellite_img, 'url': url_sattelite_img,'normalization': PN.rescale_sattelite_img},
}
    # "era5_bioclim":{'params': params_bioclim, 'url': url_bioclim,'normalization': PN.rescale_bioclim},
    # "s2_monthly":{'params': params_s2, 'url': url_s2,'normalization': PN.rescale_s2},
    # "hii":{'params': params_hii, 'url': url_hii,'normalization': PN.rescale_hii},
    # "elevation_dem":{'params': params_elevation_dem, 'url': url_elevation_dem,'normalization': PN.rescale_elevation_dem}


