import glob
import os
import geopandas as geopd
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
import errno
import time
from shapely.geometry import MultiPoint, Polygon, Point


def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def take_along_edge(numElems, polygon):
    points = []
    # minx, miny, maxx, maxy = polygon.bounds
    x, y = polygon.exterior.coords.xy
    idx = idx = np.round(np.linspace(0, len(x) - 1, numElems)).astype(int)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    _points = []
    for _x, _y in zip(x, y):
        _points.append(Point(_x, _y))
    return _points


def create_polygon_from_extent(bbox):
    bb_minx, bb_maxx, bb_miny, bb_maxy = bbox
    coords = ((bb_minx, bb_miny), (bb_minx, bb_maxy), (bb_maxx, bb_maxy), (bb_maxx, bb_miny), (bb_minx, bb_miny))
    return Polygon(coords)


def center_bounds_in_extent(bounds, extent_size=1280):
    bound_minx, bound_miny, bound_maxx, bound_maxy = bounds
    if bound_maxx - bound_minx > extent_size:
        print('Error x size')
    if bound_maxy - bound_miny > extent_size:
        print('Error y size')
        print(bound_maxy - bound_miny)
    half_xsize = np.floor((bound_maxx - bound_minx) / 2)
    half_ysize = np.floor((bound_maxy - bound_miny) / 2)
    bbox_minx = bound_minx - (np.floor(extent_size / 2) - half_xsize)
    bbox_miny = bound_miny - (np.floor(extent_size / 2) - half_ysize)
    bbox_maxx = bbox_minx + extent_size
    bbox_maxy = bbox_miny + extent_size
    return np.array([bbox_minx, bbox_maxx, bbox_miny, bbox_maxy])


def round_w_base(x, base=5, round_type=np.round):
    return base * round_type(x / base)


def point_clustering_to_boxes(searchstring, output_path, im_size=1280, no_overlap=False, write_eval=False):
    if not no_overlap:
        bboxes = []
        print(len(glob.glob(searchstring)))
        for file in glob.glob(searchstring):
            print('Prcessing data from file:', file)
            geo_df = geopd.GeoDataFrame.from_file(file, geometry='geometry', crs="EPSG:3006")
            if 'cluster' in geo_df.columns:
                pass
            else:
                geo_df['cluster'] = 1
            grouped = geo_df.groupby('cluster')
            for name, group_df in grouped:
                points = []
                print('Försökte skapa extents för cluster:', name)
                print(group_df.shape)
                for i in range(group_df.shape[0]):
                    # TODO: Kanske kan det i vissa fall vara bra att sprida punkter över ett polygon istället för att ta längs kanten.
                    # I så fall lägg till följande så att man kan välja i anrop
                    # Within polygon
                    # random_points = generate_random_points_within_polygon(10, group_df.iloc[i]['geometry'] )
                    # points.extend(random_points)
                    # Along each polygon edge
                    # pdb.set_trace()
                    if group_df.iloc[i]['geometry'] != None:
                        if type(group_df.iloc[i]['geometry']) == Point:
                            # group_df.iloc[i]['geometry'].coords.xy
                            points.append(group_df.iloc[i]['geometry'])
                        elif type(group_df.iloc[i]['geometry']) == Polygon:
                            points.extend(take_along_edge(10, group_df.iloc[i]['geometry']))
                        else:
                            for geom in list(group_df.iloc[i]['geometry']):
                                points.extend(take_along_edge(10, geom))
                pnt_df = geopd.GeoDataFrame(points, crs="EPSG:3006", columns=['geometry'])
                pnt_df['x'] = pnt_df.geometry.x
                pnt_df['y'] = pnt_df.geometry.y
                X = pnt_df[['x', 'y']].values
                X = X.astype(np.int32)
                dx = np.abs(X[:, 0][..., np.newaxis] - X[:, 0][np.newaxis, ...])
                dy = np.abs(X[:, 1][..., np.newaxis] - X[:, 1][np.newaxis, ...])
                distance_matrix = np.max([dx, dy], axis=0)
                X = None
                dx = None
                dy = None
                aff = 'precomputed'
                clustering = AgglomerativeClustering(n_clusters=None, affinity=aff, memory=None, connectivity=None,
                                                     compute_full_tree='auto', linkage='complete',
                                                     distance_threshold=im_size,
                                                     compute_distances=False).fit(distance_matrix)
                bboxes = []
                limited_bboxes = []
                for label in np.unique(clustering.labels_):
                    idx = np.argwhere(clustering.labels_ == label)
                    temp_df = pnt_df.iloc[idx.ravel(), [pnt_df.columns.get_loc("geometry")]]
                    multipoint = MultiPoint(temp_df['geometry'].values)
                    bounds = multipoint.bounds
                    bbox = center_bounds_in_extent(bounds, extent_size=im_size)
                    bbox = round_w_base(bbox, base=10, round_type=np.round)
                    bboxes.append(bbox)
                    limited_bboxes.append(bounds)
                _bboxes = []
                for i in range(len(bboxes)):
                    _bboxes.append({'geometry': create_polygon_from_extent(bboxes[i]),
                                    'geometry_inner': create_polygon_from_extent(limited_bboxes[i])})
                gdf_bboxes = geopd.GeoDataFrame(_bboxes, geometry='geometry', crs="EPSG:3006")
                gdf_bboxes = gdf_bboxes.drop(columns=['geometry_inner'])
                gdf_bboxes.to_file(f"{output_path}/bboxes_{name}.gpkg", driver="GPKG")
                group_df.to_file(f"{output_path}/original_bboxes_{name}.gpkg", driver="GPKG")
                if write_eval:
                    gdf_inner_bboxes = geopd.GeoDataFrame(_bboxes, geometry='geometry_inner', crs="EPSG:3006")
                    gdf_inner_bboxes = gdf_inner_bboxes.drop(columns=['geometry'])
                    gdf_inner_bboxes.to_file(f"bboxes_inner_{aff}.gpkg", driver="GPKG")
                    geo_df.to_file(f"{output_path}/original.gpkg", driver="GPKG")
                    pnt_df.to_file(f"{output_path}/points.gpkg", driver="GPKG")
    else:
        for file in glob.glob(searchstring):
            print('Prcessing data from file:', file)
            geo_df = geopd.GeoDataFrame.from_file(file, geometry='geometry', crs="EPSG:3006")
            if 'cluster' in geo_df.columns:
                pass
            else:
                geo_df['cluster'] = 1
            grouped = geo_df.groupby('cluster')
            for name, group_df in grouped:
                all_polygons = []
                all_polygons.extend(list(group_df['geometry'].values))
                all_bounds = geopd.GeoSeries(all_polygons).bounds
                x_min = all_bounds['minx'].min()
                x_max = all_bounds['maxx'].max()
                y_min = all_bounds['miny'].min()
                y_max = all_bounds['maxy'].max()
                x_min -= x_min % im_size
                x_max += x_max % im_size
                y_min -= y_min % im_size
                y_max += y_max % im_size
                grid = []
                for x in range(int(x_min), int(x_max), im_size):
                    for y in range(int(y_min), int(y_max), im_size):
                        lat = [x, x + im_size, x + im_size, x, x]
                        lon = [y, y, y + im_size, y + im_size, y]
                        poly = Polygon(zip(lat, lon))
                        geodf = geopd.GeoDataFrame(index=[0], crs='EPSG:3006', geometry=[poly])
                        grid.append(geodf)
                intersecting_cells = []
                start_time = time.time()
                indexer = group_df.sindex
                overlaps = []
                for cell in grid:
                    overlaps = geopd.sjoin(cell, group_df, op='intersects')
                    if not overlaps.empty:
                        intersecting_cells.append(cell)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Tid som tagits: {elapsed_time:.6f} sekunder")
                gdf_bboxes = geopd.GeoDataFrame(pd.concat(intersecting_cells), crs='EPSG:3006')
                gdf_bboxes.to_file(f"{output_path}/bboxes_{name}.gpkg", driver="GPKG")
                group_df.to_file(f"{output_path}/original_bboxes_{name}.gpkg", driver="GPKG")


def coarse_clustering_of_points(searchstring,out_folder,max_n=1500):
    ensure_dir(out_folder)
    # Generate sample data
    rows=1500
    for file in glob.glob(searchstring):
        geo_df = geopd.GeoDataFrame.from_file(file, geometry='geometry',crs="EPSG:3006")
        pnt_df = geopd.GeoDataFrame(geo_df.geometry.centroid,columns=['geometry'])
       # pdb.set_trace()
        pnt_df['x'] = pnt_df.geometry.x
        pnt_df['y'] = pnt_df.geometry.y
        X = pnt_df[['x','y']].values
        X = X.astype(np.uint32)
        np.random.seed(0)
        max_count=max_n+1
        n_cluster=int(np.ceil(pnt_df.shape[0]/max_n))
        while max_count>max_n:
          #  pdb.set_trace()
            print(n_cluster)
            kmeans = MiniBatchKMeans(n_clusters=n_cluster,random_state=0,batch_size=1024,max_iter=10).fit(X)
            cluster_ids = kmeans.predict(X)
           # pdb.set_trace()
            geo_df['cluster'] = cluster_ids
            max_count=geo_df['cluster'].value_counts().max()
            n_cluster+=int(np.ceil(n_cluster*0.2))
        geo_df.to_file(f"{out_folder}/{os.path.basename(file).split('.')[0]}_coarse.gpkg",  driver="GPKG")