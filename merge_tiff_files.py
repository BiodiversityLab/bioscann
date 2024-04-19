import argparse
from osgeo import gdal
import os


def main(opt):
    # Directory containing TIFF files
    tiff_dir = opt.input_folder

    # Pattern to match in the filenames
    pattern = opt.filename_pattern

    # List TIFF files that contain the pattern
    tiff_files = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tiff') and pattern in f]

    # Check if there are any files to process
    if not tiff_files:
        raise ValueError("No TIFF files found with the specified pattern.")

    # Output filename
    output_filename = opt.outfile

    # Append .tiff if not present
    if not output_filename.lower().endswith('.tiff'):
        output_filename += '.tiff'

    # Create a GDAL VRT file that references the filtered TIFF files
    vrt_filename = os.path.join(os.path.dirname(output_filename),"merged.vrt")
    gdal.BuildVRT(vrt_filename, tiff_files)

    # Convert the VRT to a single TIFF file
    gdal.Translate(output_filename, vrt_filename, format="GTiff",
                   outputType=gdal.GDT_Float32)

    # Delete the temporary VRT file
    os.remove(vrt_filename)
    print("Merging complete.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='predictions/predictions')
    parser.add_argument('--outfile', type=str, default='merged_tiffs.tiff')
    parser.add_argument('--filename_pattern', action='store', default='')
    opt = parser.parse_args()

    main(opt)


# # below code is for trouble-shooting purposes only:
# from types import SimpleNamespace
#
# opt = SimpleNamespace(
#     input_folder='predictions/alpine/predictions',
#     outfile='predictions/alpine/merged_predictions_alpine.tiff',
#     filename_pattern=''
# )