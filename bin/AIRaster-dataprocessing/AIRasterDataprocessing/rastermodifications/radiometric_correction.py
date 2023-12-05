import numpy as np
from sklearn.linear_model import LinearRegression

def radiometric_correction(im1, im2, cut_value=0.7, itt=5, filter_image=None, filter_val=None):
    """Calculate and match the radiometric properties of two images.
    
    Arguments
    ----------
    im1 -- the firt image for the correction process as a numpy array
    im2 -- the second image for the correction process as a numpy array
    cut_value -- the cut value for deciding on how many values to interpolate (default 0.7)
    itt -- how many iterations to run the correction for before the model can be used for prediction (default 5)
    filter_image -- a 1-channel image as an numpy array used to filter out which parts of the images that should be used for correction (default None)
    filter_val -- a list of values corresponding to pixel values in the filter_image that should be used for the correction (default None)
    """    
   # Check if im1 and im2 are the same. In that case we don't need to do any matching.
    if np.sum(np.abs(im1-im2))==0:
        return im1

    im1copy = im1.copy()
    im2copy = im2.copy()

    # If we have a filter image we must applay the filter before we do the radiometric_match
    if filter_image is not None:
 
        # Create a filter mask from the filter_image, mask is invertet because we want to keep the data in the filter_val
        mask = np.isin(filter_image, filter_val)
 
        im1_masked = im1[mask] # Extract all values where the mask have a corresponding True value, the result will be flatten.
        im2_masked = im2[mask] # Extract all values where the mask have a corresponding True value, the result will be flatten.
        
        im1copy = im1_masked
        im2copy = im2_masked

    # Flatten the image into a 1D array. If using filter image this is already done.
    im1_f = im1copy.flatten() 
    im2_f = im2copy.flatten() 
    model = LinearRegression()

    no_of_values_to_interpolate = int(len(im1_f)*(cut_value))
    for i in range(itt):
        idx_to_reg_match = np.argsort(np.abs(im1_f-im2_f))[:no_of_values_to_interpolate]
        im1_f_rm = im1_f[idx_to_reg_match]
        im2_f_rm = im2_f[idx_to_reg_match]
        im1_f_rm = im1_f_rm[..., np.newaxis]

        model.fit(im1_f_rm,im2_f_rm)

        im1_f = model.predict(im1_f[..., np.newaxis])
        
    return model.predict(im1.flatten()[..., np.newaxis]).reshape(im1.shape)
