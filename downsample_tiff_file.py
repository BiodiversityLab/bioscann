import argparse
from osgeo import gdal

def main(opt):
    # Retrieve the options
    input_file = opt.input_file
    output_file = opt.output_file
    downsample_factor = opt.downsample_factor

    # Open the original TIFF file
    original_tiff = gdal.Open(input_file, gdal.GA_ReadOnly)

    # Get original dimensions
    original_width = original_tiff.RasterXSize
    original_height = original_tiff.RasterYSize

    # Calculate new dimensions
    new_width = original_width // downsample_factor
    new_height = original_height // downsample_factor

    # Perform the downsampling
    gdal.Warp(output_file, original_tiff, width=new_width, height=new_height, resampleAlg=gdal.GRA_Bilinear)

    # Close the dataset
    original_tiff = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downsample a TIFF file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input TIFF file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output TIFF file')
    parser.add_argument('--downsample_factor', type=int, required=True, help='Downsampling factor')

    opt = parser.parse_args()
    main(opt)
