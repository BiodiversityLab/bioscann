import geopandas as geopd
import numpy as np
import os
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



file_path = 'geopackage/Stat_Svea_Barr.gpkg'
sveaskog_stats_df = geopd.read_file(file_path)
sveaskog_stats_df.columns
age_array = sveaskog_stats_df.ALDER.values
nv_mean_array = sveaskog_stats_df.MAX.values
nv_std_array = sveaskog_stats_df.STD.values

fig=plt.figure(figsize = (8,8))
plt.hist(age_array,np.arange(0,max(age_array))[::10])
fig.savefig(os.path.join('plots','sveaskog_polygon_ages_hist.png'),dpi=300)#bbox_inches='tight', dpi = 300)

# linear regression
x = age_array
y = nv_mean_array

# with sklearn
# model = LinearRegression().fit(x.reshape([len(x), 1]), y)
# # r_sq = model.score(x.reshape([len(x),1]), y)
# # print(f"coefficient of determination: {r_sq}")
# reg_line = model.intercept_ + (x * model.coef_)

# with scipy.stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
reg_line = intercept + slope*x

fig=plt.figure(figsize = (8,8))
plt.scatter(age_array,nv_mean_array,s=1)
plt.plot(x, reg_line, 'red')
fig.savefig(os.path.join('plots','linreg_age_naturvarden.png'),dpi=300)#bbox_inches='tight', dpi = 300)


# test for significant correlation
pearsonr(x, y)

# Here’s how to interpret the output:
# If the correlation coefficient is close to 1, this tells us that there is a
# strong positive association between the two variables.
# And if the corresponding p-value is less than .05, we conclude that there is a
# statistically significant association between the two variables.
