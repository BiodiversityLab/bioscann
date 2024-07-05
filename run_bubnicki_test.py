import os
import pandas as pd
from run_testset_prediction import get_file_dict,check_complete_instances,load_instance_tifffiles,compile_labels_and_predictions_per_pixel,produce_df_from_test_stats
from osgeo import gdal
import matplotlib.pyplot as plt
from itertools import islice
import concurrent.futures
import numpy as np

def load_geotiff(filepath):
    # Open the georeferenced TIFF file
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    if not dataset:
        print("Failed to open file %s"%filepath)
    else:
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        # Get geotransform and projection
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        # print("Data loaded successfully.")
    return(dataset,data,geotransform,projection)


# def load_predictions_for_target_area(predictions_file,dataset,projection):
#     # Open the second georeferenced TIFF file
#     second_dataset = gdal.Open(predictions_file, gdal.GA_ReadOnly)
#     if not second_dataset:
#         print("Failed to open predictions file")
#     else:
#         # Ensure the second dataset has the same projection
#         if second_dataset.GetProjection() != projection:
#             print("Projections do not match. Consider reprojecting one of the files.")
#         else:
#             # Create an in-memory raster to which we will warp the second dataset
#             mem_driver = gdal.GetDriverByName('MEM')
#             resampled_dataset = mem_driver.Create('', dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
#             resampled_dataset.SetGeoTransform(dataset.GetGeoTransform())
#             resampled_dataset.SetProjection(dataset.GetProjection())
#
#             # Perform the nearest neighbor resampling
#             gdal.ReprojectImage(second_dataset, resampled_dataset, second_dataset.GetProjection(),
#                                 dataset.GetProjection(), gdal.GRA_NearestNeighbour)
#             # Read the resampled data
#             band = resampled_dataset.GetRasterBand(1)
#             predictions_data = band.ReadAsArray()
#             # scale to range between 0 and 1
#             predictions_data = predictions_data/100
#     return predictions_data


def load_predictions_for_target_area(predictions_file,dataset,projection):
    # Open the second georeferenced TIFF file
    second_dataset = gdal.Open(predictions_file, gdal.GA_ReadOnly)
    if not second_dataset:
        print("Failed to open predictions file")
    else:
        # Ensure the second dataset has the same projection
        if second_dataset.GetProjection() != projection:
            print("Projections do not match. Consider reprojecting one of the files.")
        else:
            # Get the no-data value from the original dataset's band
            original_band = second_dataset.GetRasterBand(1)
            no_data_value = original_band.GetNoDataValue()
            # original_data = original_band.ReadAsArray()
            # Create an in-memory raster to which we will warp the second dataset
            mem_driver = gdal.GetDriverByName('MEM')
            resampled_dataset = mem_driver.Create('', dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
            resampled_dataset.SetGeoTransform(dataset.GetGeoTransform())
            resampled_dataset.SetProjection(dataset.GetProjection())
            # If no-data value is available, set it for the resampled dataset
            if no_data_value is not None:
                resampled_dataset.GetRasterBand(1).SetNoDataValue(no_data_value)
            # Perform the nearest neighbor resampling
            gdal.ReprojectImage(second_dataset, resampled_dataset, second_dataset.GetProjection(),dataset.GetProjection(), gdal.GRA_NearestNeighbour, options=['INIT_DEST=NO_DATA'])
            # Read the resampled data
            band = resampled_dataset.GetRasterBand(1)
            predictions_data = band.ReadAsArray()
            # scale to range between 0 and 1
            predictions_data = predictions_data/100
    return predictions_data


def get_test_stats_for_tifffile(args_list):
    i, instance, filepaths, predictions_tiff, total_instances = args_list
    features_file, mask_file, label_file = filepaths
    dataset, data, geotransform, projection = load_geotiff(features_file)
    prediction = load_predictions_for_target_area(predictions_tiff,dataset,projection)
    # # Plotting the data
    # plt.figure(figsize=(10, 6))
    # plt.imshow(label_image, cmap='viridis')  # 'viridis' is just one of many color maps
    # plt.colorbar(label='Pixel values')  # Add a color bar to show the data scale
    # plt.title('Resampled Raster Data')
    # plt.xlabel('Pixel X Coordinate')
    # plt.ylabel('Pixel Y Coordinate')
    # plt.show()
    feature_image,mask_image,label_image = load_instance_tifffiles(filepaths)
    pixel_labels, pixel_preds = compile_labels_and_predictions_per_pixel(prediction, mask_image, label_image, no_data_value = 2.55, remove_no_data_pixels = False)
    #true_positives, true_negatives, false_positives, false_negatives, total_pixels, target_thresholds = calculate_stats(prediction, mask_image, label_image, target_class_threshold)
    print(f'\rFinished instance %i/%i' %(i,total_instances), end='', flush=True)
    # return (true_positives, true_negatives, false_positives, false_negatives, total_pixels, target_thresholds)
    return(pixel_labels, pixel_preds)


def main():
    rootdir = 'data/processed_geodata'
    run_pred = True
    region_list = ['alpine', 'boreal_east', 'boreal_northwest', 'boreal_south', 'continental']
    for region in region_list:
        npy_outfile_labels = 'results/modeltesting/bubnicki_labels_%s.npy' % region
        npy_outfile_preds = 'results/modeltesting/bubnicki_preds_%s.npy' % region
        if run_pred:
            input_folder = os.path.join(rootdir,region,region+'_geodata','testset')
            bubnicki_predictions = 'predictions/bubnicki_HCVFSw.tiff'
            threads = 1

            # load the test instances
            file_dict = get_file_dict(input_folder)  # order of files for each key is: feature, mask, label
            file_dict = check_complete_instances(file_dict)
            sorted_keys = sorted(file_dict.keys())

            # #line below is for trouble-shooting, should be commented out when running the script properly
            # file_dict = dict(islice(file_dict.items(), 100))

            total_instances = len(file_dict.keys())
            # for i, instance in enumerate(file_dict.keys()):
            #     print(i)
            #     get_test_stats_for_tifffile(instance, file_dict[instance],bubnicki_predictions)

            if threads > 1:
                print("Running in parallel on %i cores." %threads)
                # Function to be executed in the pool
                # Using ProcessPoolExecutor to parallelize the loop
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    # Submit all tasks to the executor
                    futures = [executor.submit(get_test_stats_for_tifffile,[i, instance, file_dict[instance], bubnicki_predictions, total_instances]) for i, instance in enumerate(sorted_keys)]
                    # Collect results as they complete
                    stats_list = [future.result() for future in concurrent.futures.as_completed(futures)]
            else:
                stats_list = [get_test_stats_for_tifffile([i, instance, file_dict[instance], bubnicki_predictions, total_instances]) for i, instance in enumerate(sorted_keys)]

            print()  # Move to the next line after the loop completes

            label_list = []
            pred_list = []
            for i in stats_list:
                label_list.append(i[0])
                pred_list.append(i[1])
            all_labels = np.concatenate(label_list)
            all_preds = np.concatenate(pred_list)
            np.save(npy_outfile_labels, all_labels)
            np.save(npy_outfile_preds,all_preds)
            # np.unique(all_labels, return_counts=True)
        else:
            all_labels = np.load(npy_outfile_labels)
            all_preds = np.load(npy_outfile_preds)

        plt.hist(all_preds[all_preds>0],99)
        df_stats = produce_df_from_test_stats(all_labels,all_preds)
        # Save the DataFrame to CSV
        df_stats = df_stats.round(4)
        df_stats.to_csv('results/modeltesting/bubnicki_%s.csv'%region, index=False)



    # combine regional models stats for overall sweden stats_______________________________________
    # Load all CSV files
    alpine = pd.read_csv('results/modeltesting/bubnicki_alpine.csv', skiprows=[1])
    boreal_east = pd.read_csv('results/modeltesting/bubnicki_boreal_east.csv', skiprows=[1])
    boreal_northwest = pd.read_csv('results/modeltesting/bubnicki_boreal_northwest.csv', skiprows=[1])
    boreal_south = pd.read_csv('results/modeltesting/bubnicki_boreal_south.csv', skiprows=[1])
    continental = pd.read_csv('results/modeltesting/bubnicki_continental.csv', skiprows=[1])
    outfile = 'results/modeltesting/bubnicki_combined_sweden.csv'

    # Combine the dataframes by summing the relevant columns
    dfs = [alpine, boreal_east, boreal_northwest, boreal_south, continental]

    combined_df = pd.concat(dfs).groupby('Threshold').sum().reset_index()

    # Calculate metrics
    combined_df['Accuracy'] = combined_df['Correct Pixels'] / combined_df['Total Pixels']
    combined_df['Precision'] = combined_df['True Positives'] / (combined_df['True Positives'] + combined_df['False Positives'])
    combined_df['Recall'] = combined_df['True Positives'] / (combined_df['True Positives'] + combined_df['False Negatives'])
    combined_df['F1 Score'] = 2 * (combined_df['Precision'] * combined_df['Recall']) / (combined_df['Precision'] + combined_df['Recall'])
    combined_df.fillna(0.000000, inplace=True)

    # Find the row with the highest F1 score
    max_f1_row = combined_df.loc[combined_df['F1 Score'].idxmax()]

    # Create a new DataFrame with the max F1 score row as the first row
    max_f1_df = pd.DataFrame([max_f1_row])

    # Append the rest of the combined_df to the max_f1_df
    new_combined_df = pd.concat([max_f1_df, combined_df], ignore_index=True)

    # write to file
    new_combined_df.to_csv(outfile, index=False)


main()


    #
    #
    #
    # # random code bits below
    # predictions_file = 'predictions/bubnicki_HCVFSw.tiff'
    # dataset = gdal.Open(predictions_file, gdal.GA_ReadOnly)
    #
    # # Check if the dataset was successfully opened
    # if dataset is None:
    #     raise ValueError("Could not open the TIFF file.")
    #
    # # Assuming you're interested in the first band
    # band = dataset.GetRasterBand(1)
    #
    # # Read data into an array
    # array = band.ReadAsArray()
    #
    # # Retrieve the no-data value
    # no_data_value = band.GetNoDataValue()
    #
    # # Count occurrences of the no-data value
    # if no_data_value is not None:
    #     no_data_count = np.count_nonzero(array == no_data_value)
    # else:
    #     raise ValueError("No no-data value defined for this raster band.")
    #
    # print(f'Number of no-data pixels: {no_data_count}')
