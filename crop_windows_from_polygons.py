import os
import argparse
from AIRasterDataprocessing.createcontainingextents.coarse_clustring import coarse_clustering_of_points
from AIRasterDataprocessing.createcontainingextents.point_clustring import point_clustering_to_boxes
import glob
import geopandas as geopd
import pandas as pd
import utils.save_gpkg as save_gpkg
import utils.CreatePolygon as cp


def convert_polygon(geodf_polygoner):
    lista = []
    
    for index, polygon in enumerate(geodf_polygoner.iterrows()):
        
        polygon1 = polygon[1]

       # print(polygon1)
        if polygon1['geometry'].geom_type == 'MultiPolygon':
            # extract polygons out of multipolygon
            for p1 in polygon1['geometry'].geoms:
                p = {'geometry':p1}
                lista.append(p)
        elif polygon1['geometry'].geom_type == 'Polygon':
            p = {'geometry':polygon1['geometry']}
            lista.append(p)
    df_separata_polygoner = pd.DataFrame(lista)
    all_polygons = geopd.GeoDataFrame(df_separata_polygoner) 
    return all_polygons

def main(opt):
    print("main")
    no_overlap = opt.no_overlap
    files = []
    for root, dirs, filenames in os.walk(opt.input_path):
        for filename in filenames:
            if filename.endswith('.gpkg'):
                print(filename)
                files.append(os.path.join(root, filename))
    combined = []
    for f in files:
        gpkg = cp.read_gpkg(f)
        print(len(gpkg))
        gpkg['filename'] = os.path.basename(f)
        polygons = gpkg
        #polygons = gpkg
        polygons = polygons.rename(columns={'ID': 'ID2'})
        polygons = polygons.rename(columns={'Id': 'Id3'})
        combined.append(polygons)

    combined = pd.concat(combined)
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    combined_filename = os.path.join(opt.output_path, 'combined.gpkg')
    save_gpkg.save(combined, combined_filename)

    max_n_in_cluster = 1000 
    #Gor forst en grov clustringsalgorithm pa alla polygoner
    coarse_clustering_of_points(combined_filename, opt.output_path, max_n_in_cluster)

    #Gor en finare uppdelning utifran den grova indelningen
    print(os.path.join(opt.output_path,'*coarse.gpkg'))
    point_clustering_to_boxes(os.path.join(opt.output_path,'*_coarse.gpkg'), opt.output_path, im_size=int(opt.extent_size), no_overlap=no_overlap, write_eval=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--gpkgs', nargs='*', default=['*.gpkg'])
    parser.add_argument('--input_path', action='store', default='data/polygons/alpin')
    parser.add_argument('--output_path', action='store', default='geopackage')
    parser.add_argument('--extent_size', action='store', default='1280')
    parser.add_argument('--no_overlap', action='store_true', help='set no overlap if no extent is allowed to overlap')
    opt = parser.parse_args()

    main(opt)

# below code is for trouble-shooting purposes only:
# from types import SimpleNamespace
# # Create an opt object with the desired attributes
# opt = SimpleNamespace(input_path='data/polygons/alpin', output_path='geopackage', extent_size='1280', no_overlap=True)
# # Use this opt object for testing
# main(opt)
