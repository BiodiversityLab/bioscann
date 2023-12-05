import glob
import geopandas as geopd
import numpy as np
from shapely.geometry import MultiPoint,Polygon,Point

from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':
    from utils.utils import take_along_edge, crete_polygon_from_extent,center_bounds_in_extent,round_w_base
else:
    from .utils.utils import take_along_edge, crete_polygon_from_extent,center_bounds_in_extent,round_w_base

import glob
import geopandas as geopd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint, Polygon, Point

from sklearn.cluster import AgglomerativeClustering

from .utils.utils import take_along_edge, crete_polygon_from_extent, center_bounds_in_extent, round_w_base
import time


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
                    _bboxes.append({'geometry': crete_polygon_from_extent(bboxes[i]),
                                    'geometry_inner': crete_polygon_from_extent(limited_bboxes[i])})
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