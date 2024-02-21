import os
import re
import argparse
import geopandas as gpd
from shapely.geometry import box
import concurrent.futures
import threading
import shutil


def overlaps(tile_bbox, polygon):
    return tile_bbox.intersects(polygon)


def check_file(filename, folder_path, geom):
    if filename.endswith('.tiff'):
        try:
            coords = list(map(int, re.findall(r'\d+', filename)[-4:]))  # x_min, y_min, x_max, y_max
            tile_bbox = box(coords[0], coords[1], coords[2], coords[3])
            if overlaps(tile_bbox, geom):
                return os.path.join(folder_path, filename)
        except ValueError:
            pass
    return None


def find_overlapping_files(folder_path, geom):
    overlapping_files = []
    for filename in os.listdir(folder_path):
        result = check_file(filename, folder_path, geom)
        if result:
            overlapping_files.append(result)
    return overlapping_files


def copy_file(source, destination, lock, counter, update_interval=10000):
    shutil.copy(source, destination)
    with lock:
        counter['count'] += 1
        if counter['count'] % update_interval == 0:
            print(f"Copied {counter['count']} files...")


def copy_files_parallel(files, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    lock = threading.Lock()
    counter = {'count': 0}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            future = executor.submit(copy_file, file, dst_folder, lock, counter)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            pass  # Handle exceptions or additional logging here
    print(f"Total {counter['count']} files copied.")


def main(opt):
    # Load polygon geometry if a polygon file is provided
    if opt.polygon_file:
        gdf = gpd.read_file(opt.polygon_file)
        polygon_geom = gdf.unary_union
    else:
        polygon_geom = box(opt.x_min, opt.y_min, opt.x_max, opt.y_max)
    overlapping_files = find_overlapping_files(opt.folder_path, polygon_geom)
    # print("Overlapping files:")
    # for file in overlapping_files:
    #     print(file)
    # Copy the overlapping files to the output folder
    if overlapping_files:
        copy_files(overlapping_files, opt.output_folder)
        print(f"Copied {len(overlapping_files)} files to {opt.output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find and copy .tiff files overlapping with a specified window or polygon.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing .tiff files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for copying files')
    parser.add_argument('--polygon_file', type=str, help='Path to the polygon file (.gpkg format)')
    parser.add_argument('--x_min', type=int, help='Minimum X coordinate of the cropping window')
    parser.add_argument('--y_min', type=int, help='Minimum Y coordinate of the cropping window')
    parser.add_argument('--x_max', type=int, help='Maximum X coordinate of the cropping window')
    parser.add_argument('--y_max', type=int, help='Maximum Y coordinate of the cropping window')
    opt = parser.parse_args()
    # Check if either polygon file or coordinates are provided
    if not opt.polygon_file and (opt.x_min is None or opt.y_min is None or opt.x_max is None or opt.y_max is None):
        parser.error("Either --polygon_file or coordinates (--x_min, --y_min, --x_max, --y_max) must be provided.")
    main(opt)
