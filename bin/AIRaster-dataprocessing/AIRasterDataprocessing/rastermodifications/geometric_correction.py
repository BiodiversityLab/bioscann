
from skimage.registration import phase_cross_correlation
from scipy.interpolate import RegularGridInterpolator
import numpy as np
def geometric_correction(im1, im2, upsample_factor=100):
    shifts, error, _ = phase_cross_correlation(im1, im2,upsample_factor=100)
    
    x = np.arange(0, np.shape(im2)[0], 1)
    y = np.arange(0, np.shape(im2)[1], 1)
    
    interp = RegularGridInterpolator((x, y), im2,bounds_error=False, fill_value=None)

    xg, yg = np.meshgrid(x-shifts[0], y-shifts[1], indexing='ij')
    points = np.column_stack((xg.ravel(),yg.ravel()))
    znew = interp(points)
    znew = znew.reshape(im2.shape)
    return znew