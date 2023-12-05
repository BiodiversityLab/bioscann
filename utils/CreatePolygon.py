import pandas as pd
import geopandas as geopd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint, Polygon

import utils.PolygonRequests


def read_gpkg(file_path, row=-1):
    points_df = geopd.read_file(file_path,rows=row)
    #points_df = geopd.read_file(file_path)
    points_df = points_df.set_crs(epsg=3006)
    return  points_df #geodf,


def make_geodf_polygon_from_extent(extent,crs='epsg:3006'):
    lat = [extent[0],extent[2],extent[2], extent[0],extent[0]]
    lon = [extent[3],extent[3],extent[1], extent[1],extent[3]]
    poly = Polygon(zip(lat,lon))
    geodf = geopd.GeoDataFrame(index=[0], crs=crs, geometry=[poly])    
    return geodf


def create_cluster_polygons_by_id_natura2000(all_polygons, ids, sparse = False):
    next = False
    lista = []
    #for cluster_id in evaluate_on:
    for cluster_id in ids:

        polygons = all_polygons.loc[all_polygons['TARGET_FID']==cluster_id]

        for polygon in polygons.iterrows():
            polygon1 = polygon[1]
            #if 'KARTERINGS' in polygon1:
             #   if '3 - Besökt i fält' == polygon1['KARTERINGS'] or '4 - Inventerad i fält' == polygon1['KARTERINGS']:

            if polygon1['geometry'].geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon
                
                for p1 in polygon1['geometry'].geoms:
                    lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": polygon1['NATURTYPSS'], "KARTERINGS":polygon1['KARTERINGS']})
                    if sparse:
                        next=True
                        break
            
        
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1['geometry'], 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": polygon1['NATURTYPSS'], "KARTERINGS":polygon1['KARTERINGS']})
                if sparse:
                    next=True
            if next:
                next = False
                break


    points_within_image_df = pd.DataFrame(lista)
    points_within_image_geodf = geopd.GeoDataFrame(points_within_image_df)
    return points_within_image_geodf


def create_cluster_polygons_by_id(points_df, ids, sparse = False):
    next = False
    lista = []
    #for cluster_id in evaluate_on:
    for cluster_id in ids:

        #print(cluster_id)
        #points = points_df.loc[points_df['cluster_id']==cluster_id]['geometry']
        points = points_df.loc[points_df['cluster_id']==cluster_id]['geometry']
        points_df_tmp = geopd.GeoDataFrame(geometry=points,crs='EPSG:3006')
        polygon1 = points.buffer(5).unary_union.simplify(1, preserve_topology=True)
        p = geopd.GeoSeries(polygon1)
       # p.plot()
        #plt.show()
       # points.plot()
       # plt.show()
        if polygon1.geom_type == 'MultiPolygon':
            # extract polygons out of multipolygon
            
            for p1 in polygon1.geoms:
                count = points_df_tmp.within(p1).sum()
                if count<=1:
                    continue
                else:
                    lista.append({'geometry':p1,'count':count, 'cluster_id':cluster_id})
                if sparse:
                    next=True
                    break
        if next:
            next = False
            break
        elif polygon1.geom_type == 'Polygon':
            #lista.append({'geometry':polygon1,'count':count})
            count = points_df_tmp.within(polygon1).sum()
            if count<=1:
                continue
            else:
                lista.append({'geometry':polygon1,'count':count, 'cluster_id':cluster_id})

 
    df_separata_polygoner = pd.DataFrame(lista)
    return geopd.GeoDataFrame(df_separata_polygoner)    


def create_clusters_on_all_polygon_list(points_df):

    cluster_ids = points_df['cluster_id'].unique()
    cluster_ids = [c for c in cluster_ids if not math.isnan(c)]
    lista = []
    #for cluster_id in evaluate_on:
    for cluster_id in cluster_ids:

        #print(cluster_id)
        #points = points_df.loc[points_df['cluster_id']==cluster_id]['geometry']
        points = points_df.loc[points_df['cluster_id']==cluster_id]['geometry']
        points_df_tmp = geopd.GeoDataFrame(geometry=points,crs='EPSG:3006')
        polygon1 = points.buffer(5).unary_union.simplify(1, preserve_topology=True)
        p = geopd.GeoSeries(polygon1)
        #p.plot()
        #plt.show()
        #points.plot()
        #plt.show()
        if polygon1.geom_type == 'MultiPolygon':
            # extract polygons out of multipolygon
            
            for p1 in polygon1.geoms:
                count = points_df_tmp.within(p1).sum()
                if count<=1:
                    continue
                else:
                    lista.append({'geometry':p1,'count':count, 'cluster_id':cluster_id})
        
        elif polygon1.geom_type == 'Polygon':
            #lista.append({'geometry':polygon1,'count':count})
            count = points_df_tmp.within(polygon1).sum()
            if count<=1:
                continue
            else:
                lista.append({'geometry':polygon1,'count':count, 'cluster_id':cluster_id})

 
    df_separata_polygoner = pd.DataFrame(lista)
    return geopd.GeoDataFrame(df_separata_polygoner)  

#def create_cluster_polygons(points_df, evaluate_on = [0]):
def create_cluster_polygons(points_df, evaluate_on_cluster_size = 5, number_of_clusters = 1, skip=[]):
   # np.unique(points_df.cluster_id.values)
    #points_df = points_df.dropna(axis=0, how='any', thresh=None, subset=['cluster_id'], inplace=False)
   # grps = points_df.groupby('cluster_id')
    #evaluate_on=[30, 32, 34]


    cluster = points_df.loc[points_df['cluster_count'] == evaluate_on_cluster_size]
    unique_ids = cluster['cluster_id'].unique()
    lista = []
    #for cluster_id in evaluate_on:
    for cluster_index in range(number_of_clusters):
        if cluster_index not in skip:
            if len(unique_ids) > cluster_index: 
                #print(cluster_id)
                #points = points_df.loc[points_df['cluster_id']==cluster_id]['geometry']
                points = points_df.loc[points_df['cluster_id']==unique_ids[cluster_index]]['geometry']

                points_df_tmp = geopd.GeoDataFrame(geometry=points,crs='EPSG:3006')
                polygon1 = points.buffer(5).unary_union.simplify(1, preserve_topology=True)
                p = geopd.GeoSeries(polygon1)
                p.plot()
                plt.show()
                points.plot()
                plt.show()
                if polygon1.geom_type == 'MultiPolygon':
                    # extract polygons out of multipolygon
                    
                    for p1 in polygon1.geoms:
                        count = points_df_tmp.within(p1).sum()
                        if count<=1:
                            continue
                        else:
                            lista.append({'geometry':p1,'count':count, 'cluster_id':unique_ids[cluster_index]})
                
                elif polygon1.geom_type == 'Polygon':
                    #lista.append({'geometry':polygon1,'count':count})
                    count = points_df_tmp.within(polygon1).sum()
                    if count<=1:
                        continue
                    else:
                        lista.append({'geometry':polygon1,'count':count, 'cluster_id':unique_ids[cluster_index]})

    
    df_separata_polygoner = pd.DataFrame(lista)
    return geopd.GeoDataFrame(df_separata_polygoner)


def get_all_dates_from_polygons(utfall, geodf_separata_polygoner):
    
    #utfall_flygfoto15_2021_pt = './data/utfall_flygfoto15_2021_pt.shp'
    utfall_flygfoto15_2021_pt = geopd.read_file(utfall)

    geopoints = geopd.points_from_xy(x = utfall_flygfoto15_2021_pt['E'], y = utfall_flygfoto15_2021_pt['N'])
    gpd2_pts = MultiPoint(geopoints)
    geodf_separata_polygoner['date'] = 0
    for i, polygon in enumerate(geodf_separata_polygoner.iterrows()):

        center_point = polygon[1]['geometry'].centroid.coords[0]
        center_point = Point(center_point)
        nearest = nearest_points(center_point, gpd2_pts)
        nearest_point_row = utfall_flygfoto15_2021_pt.loc[utfall_flygfoto15_2021_pt['N']==nearest[1].y].loc[utfall_flygfoto15_2021_pt['E']==nearest[1].x]
        geodf_separata_polygoner.loc[i,['date']]=nearest_point_row.iloc[0]['DATUM']
        
    return geodf_separata_polygoner

