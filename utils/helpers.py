def create_synthetic_user_input():
    class opt:
        pass
    opt = opt()
#    opt.apis = ['skogligagrunddata_dev_3_0']
    opt.polygon_ids = []
    opt.number_of_polygons = 0
    opt.img_size = 256
    opt.output_path = 'datasets_small_set'
    opt.start_index = 5
    opt.meters_per_pixel = 10
    opt.filter_on_marktackedata_2_0 = []
    opt.configuration = 'version_1_all_forest'
    opt.extents = 'extents'
    opt.testset_size = 0.1
#    opt.test_area = [389000,6752000,389000,6757000]#[628110,7416525,660013,7436367]
    opt.lonlat_features = True
    opt.polygons_path = 'geopackage'
    return(opt)
