import os
import numpy as np
import errno
from shapely.geometry import MultiPoint,Polygon,Point
import random
import geopandas as geopd
def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def center_bounds_in_extent(bounds, extent_size=1280):
    bound_minx, bound_miny, bound_maxx, bound_maxy = bounds
    if bound_maxx-bound_minx> extent_size:
        print('Error x size')
        
    if bound_maxy-bound_miny> extent_size:
        print('Error y size')
        print(bound_maxy-bound_miny)
        
    half_xsize = np.floor((bound_maxx-bound_minx)/2)
    half_ysize = np.floor((bound_maxy-bound_miny)/2)

    bbox_minx = bound_minx - (np.floor(extent_size/2) - half_xsize)
    bbox_miny = bound_miny - (np.floor(extent_size/2) - half_ysize)
    
    bbox_maxx = bbox_minx + extent_size
    bbox_maxy = bbox_miny + extent_size
    
    return np.array([bbox_minx, bbox_maxx, bbox_miny, bbox_maxy])
    

def round_w_base(x, base=5,round_type=np.round):
        return base * round_type(x/base)



def create_bounding_boxes(bounds,imlength=1280):
    _bboxes = []
    # round the bounds out from the real bound
    ll_c = round_w_base(bounds[:2], base=imlength,round_type=np.floor)
    ur_c = round_w_base(bounds[-2:], base=imlength,round_type=np.ceil)

    nx=int((ur_c[0]-ll_c[0])/imlength)
    ny=int((ur_c[1]-ll_c[1])/imlength)
    print(f'    Need to look through {nx*ny} tiles')
    for i in range(0,nx+1):
        for j in range(0,ny+1):
            
            bb_minx = ll_c[0]+i*imlength
            bb_miny = ll_c[1]+j*imlength
            bb_maxx = ll_c[0]+(i+1)*imlength
            bb_maxy = ll_c[1]+(j+1)*imlength

            coords = ((bb_minx, bb_miny), (bb_minx, bb_maxy), (bb_maxx, bb_maxy), (bb_maxx, bb_miny), (bb_minx, bb_miny))
            polygon = Polygon(coords)
            _bboxes.append({'geometry':polygon})
    
    gdf_bboxes = geopd.GeoDataFrame(_bboxes, geometry='geometry',crs="EPSG:3006")
    gdf_bboxes['intersects'] = 0
    return gdf_bboxes


def generate_random_points_within_polygon(number, polygon):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

def take_along_edge(numElems, polygon):
    points = []
    #minx, miny, maxx, maxy = polygon.bounds
    x,y = polygon.exterior.coords.xy
    idx = idx = np.round(np.linspace(0, len(x) - 1, numElems)).astype(int)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    _points = []
    for _x,_y in zip(x,y):
        _points.append(Point(_x,_y))
        
    return _points

def max_of_x_and_y_distance(p1,p2):
    return np.max([np.abs(p1[0]-p2[0]),np.abs(p1[1]-p2[1])])

def crete_polygon_from_extent(bbox):
    bb_minx, bb_maxx, bb_miny, bb_maxy = bbox
    
    coords = ((bb_minx, bb_miny), (bb_minx, bb_maxy), (bb_maxx, bb_maxy), (bb_maxx, bb_miny), (bb_minx, bb_miny))
    return Polygon(coords)