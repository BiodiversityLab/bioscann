import os
import re
import argparse
import shutil


def overlaps(window, tile):
    return not (tile[2] < window[0] or tile[0] > window[2] or
                tile[3] < window[1] or tile[1] > window[3])


def find_overlapping_files(folder_path, cropping_window):
    overlapping_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            coords = list(map(int, re.findall(r'\d+', filename)[-4:]))  # x_min, y_min, x_max, y_max
            if overlaps(cropping_window, coords):
                overlapping_files.append(filename)
    return overlapping_files


def copy_files(files, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for file in files:
        shutil.copy(os.path.join(src_folder, file), dst_folder)


def main(opt):
    overlapping_files = find_overlapping_files(opt.folder_path, (opt.x_min, opt.y_min, opt.x_max, opt.y_max))
    print("Overlapping files:")
    for file in overlapping_files:
        print(file)
    # Copy the overlapping files to the output folder
    if overlapping_files:
        copy_files(overlapping_files, opt.folder_path, opt.output_folder)
        print(f"Copied {len(overlapping_files)} files to {opt.output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and copy .tiff files overlapping with a specified window.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing .tiff files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for copying files')
    parser.add_argument('--x_min', type=int, required=True, help='Minimum X coordinate of the cropping window')
    parser.add_argument('--y_min', type=int, required=True, help='Minimum Y coordinate of the cropping window')
    parser.add_argument('--x_max', type=int, required=True, help='Maximum X coordinate of the cropping window')
    parser.add_argument('--y_max', type=int, required=True, help='Maximum Y coordinate of the cropping window')
    opt = parser.parse_args()
    main(opt)