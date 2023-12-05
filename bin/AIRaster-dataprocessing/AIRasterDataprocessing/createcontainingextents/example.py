import argparse

from AIRasterDataprocessing.createcontainingextents.coarse_clustring import coarse_clustering_of_points
from AIRasterDataprocessing.createcontainingextents.point_clustring import point_clustering_to_boxes


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_ss', default='./data/*.gpkg')
    parser.add_argument('--imsize', type=int, default=5120)
    parser.add_argument('--output_path', action='store', default='./out')
    parser.add_argument('--max_n_in_cluster', type=int, default=1500)
    parser.add_argument('--write_eval', type=bool, default=False)
    opt = parser.parse_args()
    
    searchstring = './data/*.gpkg'
    
    #Gör först en grov clustringsalgorithm på alla polygoner
    coarse_clustering_of_points(opt.input_file_ss, opt.output_path,opt.max_n_in_cluster)

    #Gör en finare uppdelning utifrån den grova indelningen
    point_clustering_to_boxes(f'{opt.output_path}/*coarse.gpkg', opt.output_path, im_size=opt.imsize, write_eval=opt.write_eval)