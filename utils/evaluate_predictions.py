import numpy as np
from PIL import Image
import glob, os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import tifffile

def read_tiff_to_array(tif_path):
    #im = Image.open(tif_path)
    #imarray = np.array(im)
    imarray = tifffile.imread(tif_path)
    if len(imarray.shape)>2:
        n_channels = imarray.shape[-1]
    else:
        n_channels = 1
        imarray = imarray.reshape(list(imarray.shape)+[1])
    if 'markfuktighet' in tif_path:
        n_channels = 1
        if tif_path.endswith('_KON.tiff'):
            imarray = imarray[:,:,0]
        else:
            imarray = imarray[:, :, 1]
        imarray = imarray.reshape(list(imarray.shape)+[1])
    return imarray, n_channels

def get_tiff_files_from_dir(dir):
    return glob.glob(os.path.join(dir, '*.tiff'))

# read prediction tiff file
pred_dir = 'pred_test'
prediction = os.path.join(pred_dir,'output/output_0.tiff')
prediction_array, __ = read_tiff_to_array(prediction)
# read predictor tiff file
feature_dir = os.path.join(pred_dir,'input')
feature_tiffs = get_tiff_files_from_dir(feature_dir)
#target_feature = 'tradhojd'
#target_file = [feature for feature in feature_tiffs if target_feature in feature][0]
layer_names = []
layer_arrays = []
for i,target_file in enumerate(feature_tiffs):
    if 'marktacke' in target_file:
        pass
    else:
        feature_array, dimensions = read_tiff_to_array(target_file)
        name = os.path.basename(target_file).replace('.tiff','')
        for i in np.arange(dimensions):
            layer_name = name+'_%i'%i
            layer_array = feature_array[:,:,i]
            layer_names.append(layer_name)
            layer_arrays.append(layer_array)

fig = plt.figure(figsize = (20,20))
n_rows = 6
for i,feature_array in enumerate(layer_arrays):
    # extract pixel values as flat array
    x = feature_array.flatten()
    y = prediction_array.flatten()
    name = layer_names[i]
    # # plot scatterplot
    # corr, _ = pearsonr(x, y)
    # print('Pearsons correlation: %.3f' % corr)
    # corr, _ = spearmanr(x, y)
    # print('Spearmans correlation: %.3f' % corr)
    model = LinearRegression().fit(x.reshape([len(x),1]), y)
    # r_sq = model.score(x.reshape([len(x),1]), y)
    # print(f"coefficient of determination: {r_sq}")
    reg_line = model.intercept_ + (x * model.coef_)
    subplot = fig.add_subplot(n_rows,int(np.ceil(len(layer_arrays)/n_rows)),i+1)
    subplot.scatter(x,y,s=2)
    subplot.plot(x,reg_line,'red')
    subplot.set_ylim(-0.1,1.1)
    #subplot.text(0.5, 0.1, 'TEST',ha='center', va='center')
    plt.title(name)
fig.savefig(os.path.join(pred_dir,'linreg.png'),dpi=300)#bbox_inches='tight', dpi = 300)

