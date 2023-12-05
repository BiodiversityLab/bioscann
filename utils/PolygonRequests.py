import json
import requests
from datetime import datetime
import utils.save_gpkg as save_gpkg

def wkt_to_esri(wkt):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {'in_features': str([wkt]), "f": "json"}
    resp = requests.post(
        #'http://ingeagsinternutv/arcgis/rest/services/geoprocessingINGE/WKTToEsri/GPServer/Convert%20WKT%20to%20EsriJSON/execute',
        'http://10.16.240.101/arcgis/rest/services/geoprocessingINGE/WKTToEsri/GPServer/Convert%20WKT%20to%20EsriJSON/execute',
        headers=headers, data=data, verify=False)
    resp.raise_for_status()
    return resp.json()["results"][0]["value"][0]



def get_request_params(geom):
    return dict(
    geometryType='esriGeometryPolygon',
    geometry=geom,
    mosaicRule='',
    renderingRule='',
    pixelSize='',
    time='',
    processAsMultidimensional='false',
    f='pjson',
    )

def get_sclhistogram_between_dates_params(geom,early_date,late_date,threshold=0.1):
    return dict(
    where="BraData>={} AND Datum >='{}' AND Datum <= '{}'".format(threshold,early_date, late_date),
    timeRelation='esriTimeRelationOverlaps',
    spatialRel='esriSpatialRelIntersects',   
    units='esriSRUnit_Meter',
    geometryType='esriGeometryPolygon',
    outFields='Datum',
    returnGeometry='false',
    returnTrueCurves='false',
    returnIdsOnly='false',
    returnCountOnly='false',
    returnZ='false',
    returnM='false',
    returnDistinctValues='false',
    returnExtentOnly='false',
    sqlFormat='none',
    featureEncoding='esriDefault',
    geometry=geom,
    f='pjson',   
    )

def get_sclhistogram_between_dates(geom,early_date,late_date,threshold=0.1):
    early_date = str(early_date)
    early_date = early_date[0:4]+'-'+early_date[4:6]+'-'+early_date[6:8]
    late_date = str(late_date)
    late_date = late_date[0:4]+'-'+late_date[4:6]+'-'+late_date[6:8]
    params = get_sclhistogram_between_dates_params(geom, early_date,late_date,threshold=threshold)
    #resp = requests.post(url='http://ingeagsexterntest.svo.local/arcgis/rest/services/Skogsdatalabbet/SkogsdatalabbetVisaSclHistogram_3_0/MapServer/0/query',data=params, verify=False)
    resp = requests.post(url='http://10.16.240.116/arcgis/rest/services/Skogsdatalabbet/SkogsdatalabbetVisaSclHistogram_3_0/MapServer/0/query',data=params, verify=False)
    return resp.json()


def get_sclhistogram_params(geom,date,threshold=0.1):
    #print(date)
    return dict(
    where="BraData>={} AND Datum <='{}'".format(threshold,date),
    timeRelation='esriTimeRelationOverlaps',
    spatialRel='esriSpatialRelIntersects',   
    units='esriSRUnit_Meter',
    geometryType='esriGeometryPolygon',
    outFields='Datum',
    returnGeometry='false',
    returnTrueCurves='false',
    returnIdsOnly='false',
    returnCountOnly='false',
    returnZ='false',
    returnM='false',
    returnDistinctValues='false',
    returnExtentOnly='false',
    sqlFormat='none',
    featureEncoding='esriDefault',
    geometry=geom,
    f='pjson',   
    )


def get_sclhistogram(geometry, date,threshold=0.1):
    date = str(date)
    date = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
    params = get_sclhistogram_params(geometry, date,threshold=threshold)
    #resp = requests.post(url='http://ingeagsexterntest.svo.local/arcgis/rest/services/Skogsdatalabbet/SkogsdatalabbetVisaSclHistogram_3_0/MapServer/0/query',data=params, verify=False)
    resp = requests.post(url='http://10.16.240.116/arcgis/rest/services/Skogsdatalabbet/SkogsdatalabbetVisaSclHistogram_3_0/MapServer/0/query',data=params, verify=False)
    return resp.json()



def get_acorvi_params(geom, date, pixelSize):
    return dict(
        mosaicRule='',
        #Sentinel 2_3_0 Riktig acorvi (NDVI)
        renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(b3*0+b2-b1-500)/(b1+b2+500)",\
             "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
                "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [2, 3, 6]}}}}}}', #BandIDs: (2)b4=rött ljus, (3)b8=nir, 6=scl 

        #Sentinel 2_3_0 (NDMI)
        # renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(b3*0+b1-b2)/(b1+b2)",\
        #      "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
        #           "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [3, 4, 6]}}}}}}',
       
        #Sentinel 2_2_0 (NDMI)
        # renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(b3*0+b1-b2)/(b1+b2-2000)",\
        #      "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
        #           "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [3, 8, 12]}}}}}}',
       
        pixelSize='{},{}'.format(pixelSize,pixelSize),
        time=str(date),
        geometry=geom,
        geometryType='esriGeometryPolygon',
        f='pjson'
    )


def get_acorvi(geom, date, pixelSize):
     params = get_acorvi_params(geom,date,pixelSize)
    # resp = requests.post(url='http://imgutv.svo.local/arcgis/rest/services/Sentinel2_2_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
     #resp = requests.post(url='http://imgutv.svo.local/arcgis/rest/services/Sentinel2_3_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
     resp = requests.post(url='http://193.183.28.189/arcgis/rest/services/Sentinel2_3_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
     return resp.json()


def get_SAVI_params(geom, date, pixelSize):
    return dict(
        mosaicRule='',
        #Sentinel 2_3_0  (SAVI) troligtvis EJ korrekt
        renderingRule='{"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 0, "BandIndexes": "(1+0.5)*(b3*0+b2-b1-500)/(b1+b2+500+0.5)",\
             "Raster": {"rasterFunction": "Mask", "rasterFunctionArguments": {"NoDataValues": ["0", "0", "0 1 2 3 7 8 9 10 11"], "NoDataInterpretation": 0,\
                "Raster": {"rasterFunction": "ExtractBand", "rasterFunctionArguments": {"BandIDs": [2, 3, 6]}}}}}}', #BandIDs: (2)b4=rött ljus, (3)b8=nir, 6=scl 

        pixelSize='{},{}'.format(pixelSize,pixelSize),
        time=str(date),
        geometry=geom,
        geometryType='esriGeometryPolygon',
        f='pjson'
    )


def get_SAVI(geom, date, pixelSize):
     params = get_SAVI_params(geom,date,pixelSize)
    # resp = requests.post(url='http://imgutv.svo.local/arcgis/rest/services/Sentinel2_2_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
     #resp = requests.post(url='http://imgutv.svo.local/arcgis/rest/services/Sentinel2_3_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
     resp = requests.post(url='http://193.183.28.189/arcgis/rest/services/Sentinel2_3_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
     return resp.json()



def get_marktackedata_2_0_params(geom, date, pixelSize):
    return dict(
        mosaicRule='',
        renderingRule='{"name": "SKS_Granskog_generaliserad"}',
        time=str(date),
        geometry=geom,
        geometryType='esriGeometryPolygon',
        f='pjson'
    )
        #pixelSize='{},{}'.format(pixelSize,pixelSize),
#http://imgutv/svo.local/geo/GeoLeverans/Leverans_2021/NMD/NMD_2_0_testprodukter_20210614/33VVC_SodraSverige/Tillaggskikt/Tradslagsvisa_kontinuerliga_raster
#https://imgtest/arcgis/rest/services/Marktackedata_2_0/ImageServer/computeStatisticsHistograms
def get_marktackedata_2_0(geom, date, pixelSize):
    params = get_marktackedata_2_0_params(geom, date, pixelSize)
    #resp = requests.post(url='http://imgtest/arcgis/rest/services/Marktackedata_2_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
    resp = requests.post(url='http://10.16.240.115/arcgis/rest/services/Marktackedata_2_0/ImageServer/computeStatisticsHistograms', data=params, verify=False)
    return resp.json()

def get_ndvi_data(geodf_separata_polygoner):
    geodf_separata_polygoner['datum'] = None
    geodf_separata_polygoner['acorvi_median'] = None
    geodf_separata_polygoner['timestamps'] = None
    damaged_polygon_ndvis = []
    for i, damaged_polygon in enumerate(geodf_separata_polygoner.iterrows()):
        print("get ndvi")
        damaged_polygon_wkt = str(wkt_to_esri(damaged_polygon[1]['geometry'].wkt))
        response = get_sclhistogram(damaged_polygon_wkt, damaged_polygon[1]['date'])
        good_dates_of_5km = [features['attributes']['Datum'] for features in response['features']]

        damaged = {
            'acorvi_median': [],
            'statistics': [],
            'datum': [],
            'timestamps': [],
            'std': [],
            'cluster_id': []
        }
        acorvi_median = []
        dates = []
        timestamps = []        

        pixelSize = 5
        for date in reversed(good_dates_of_5km):
           # damaged_statistics = get_acorvi(damaged_polygon_wkt,date, pixelSize)
            #damaged_statistics = get_acorvi(damaged_polygon_wkt,date, pixelSize)
            damaged_statistics = get_SAVI(damaged_polygon_wkt,date, pixelSize)
            #damaged_statistics = get_NDWI(damaged_polygon_wkt,date, pixelSize)

            if 'statistics' in damaged_statistics.keys():
                if len(damaged_statistics['statistics']) != 0:
                    date_time = datetime.fromtimestamp(date/1000).strftime('%Y-%m-%d')
                    if int(damaged_statistics['statistics'][0]['count'])*pixelSize*pixelSize /damaged_polygon[1]['geometry'].area > 0.95:
                        damaged['acorvi_median'].append(damaged_statistics['statistics'][0]['median'])
                        damaged['std'].append(damaged_statistics['statistics'][0]['standardDeviation'])
                        damaged['datum'].append(date_time)
                        damaged['timestamps'].append(date)

                        dates.append(date_time)
                        timestamps.append(date)
                        acorvi_median.append(damaged_statistics['statistics'][0]['median'])

        #damaged['cluster_id'].append(damaged_polygon[1]['cluster_id'])
        damaged['cluster_id'] = damaged_polygon[1]['cluster_id']
        damaged_polygon_ndvis.append(damaged)
    geodf_separata_polygoner = save_gpkg.append_lists_to_dataframe(['acorvi_median', 'datum', 'timestamps'], damaged_polygon_ndvis, geodf_separata_polygoner)

    return damaged_polygon_ndvis, geodf_separata_polygoner#, geodf_separata_polygoner


def get_cloudfree_dates(damaged_polygon,threshold=0.1):
    damaged_polygon_wkt = str(wkt_to_esri(damaged_polygon['geometry'].wkt))
    response = get_sclhistogram(damaged_polygon_wkt, damaged_polygon['date'],threshold)
    good_dates_of_5km = [features['attributes']['Datum'] for features in response['features']]
    return good_dates_of_5km


def get_good_dates_of_coordinates_between_dates(damaged_polygon, early_date=20170530, late_date=20180530,threshold=0.1):
   # poly = damaged_polygon
   # esri_poly = str(wkt_to_esri(poly.wkt))
    esri_poly = str(wkt_to_esri(damaged_polygon['geometry'].wkt))
    #early_date_time=20170530
    #late_date_time=20180530

    response = get_sclhistogram_between_dates(esri_poly, early_date=early_date, late_date=late_date,threshold=threshold)

    good_dates = [features['attributes']['Datum'] for features in response['features']]
    return good_dates
