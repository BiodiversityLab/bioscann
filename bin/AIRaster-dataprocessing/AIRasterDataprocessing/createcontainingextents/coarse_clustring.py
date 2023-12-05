import glob
import os

import geopandas as geopd
import numpy as np
from sklearn.cluster import MiniBatchKMeans


from .utils.utils import ensure_dir
#import pdb

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

        #TODO: Gör kontroll om något av klustrena innehåler med än max_n punkter
        #if pnt_df.shape[0]>max_n:
        #    pdb.set_trace()
       # geo_df.to_file(f"./out/{os.path.basename(file).split('.')[0]}_coarse.gpkg",  driver="GPKG")
        geo_df.to_file(f"{out_folder}/{os.path.basename(file).split('.')[0]}_coarse.gpkg",  driver="GPKG")