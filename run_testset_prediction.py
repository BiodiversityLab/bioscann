import os
import numpy as np
import torch
from tifffile import imread
import matplotlib.pyplot as plt
import csv
import argparse
import concurrent.futures
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


# Function to extract the sample ID from the filename
def extract_sample_id(filename):
    return filename.split('_mask')[0].split('.tiff')[0]

def plot_image(input_array):
    plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed
    plt.imshow(input_array, cmap='gray')  # Use cmap='gray' for grayscale images, remove for colored images
    plt.axis('off')  # Hide the axes
    plt.show()

def get_file_dict(base_path):
    geodata_folder = os.path.join(base_path, 'indata')
    label_folder = os.path.join(base_path, 'outdata')
    # Initialize a dictionary to store sample_id as key and list of corresponding files as values
    sample_files = {}
    # Read the feature and mask files
    for filename in os.listdir(geodata_folder):
        if filename.endswith('.tiff'):
            full_path = os.path.join(geodata_folder, filename)
            sample_id = extract_sample_id(filename)
            if sample_id not in sample_files:
                sample_files[sample_id] = [None, None, None]  # Feature, Mask, Label
            if '_mask' in filename:
                sample_files[sample_id][1] = full_path  # Mask file
            else:
                sample_files[sample_id][0] = full_path  # Feature file
    # Read the label files
    for filename in os.listdir(label_folder):
        if filename.endswith('.tiff'):
            full_path = os.path.join(label_folder, filename)
            sample_id = extract_sample_id(filename)
            if sample_id in sample_files:
                sample_files[sample_id][2] = full_path  # Label file
            else:
                # If a label has no corresponding feature or mask, initialize with label only
                sample_files[sample_id] = [None, None, full_path]
    return sample_files

def check_complete_instances(file_dict):
    # Check for missing files in each dictionary entry
    missing_files_info = []
    incomplete_ids = []  # List to hold IDs of incomplete instances
    for sample_id, files in file_dict.items():
        # Check if any of the three files (Feature, Mask, Label) is None
        missing_files = [ftype for ftype, fpath in zip(['Feature', 'Mask', 'Label'], files) if fpath is None]
        if missing_files:
            missing_files_info.append((sample_id, missing_files))
            incomplete_ids.append(sample_id)  # Add to list if missing any files
    # Remove incomplete instances from the dictionary
    for sample_id in incomplete_ids:
        del file_dict[sample_id]
    # Print information about entries with missing files
    if missing_files_info:
        print("Entries with missing files:")
        for sample_id, missing in missing_files_info:
            missing_str = ', '.join(missing)
            print(f"{sample_id}: Missing {missing_str}")
        print(f"Removed {len(incomplete_ids)} incomplete instances.")
    else:
        print("All entries have complete files.")
    return file_dict  # Optionally return the modified dictionary

def compile_labels_and_predictions_per_pixel(prediction, mask_image, label_image, no_data_value = None, remove_no_data_pixels = True):
    valid_preds = prediction[mask_image > 0]
    valid_labels = label_image[mask_image > 0]
    if no_data_value:
        if remove_no_data_pixels:
            valid_labels = valid_labels[valid_preds != no_data_value]
            valid_preds = valid_preds[valid_preds != no_data_value]
        else:
            # code the pixels where we have the no-data value in the prediction raster to nan
            valid_preds[valid_preds == no_data_value] = np.nan
    return(valid_labels, valid_preds)

def calc_auc_threshold(labels, preds):
    precision, recall, thresholds = precision_recall_curve(labels,preds)
    # Calculate AUC for Precision-Recall Curve
    pr_auc = auc(recall, precision)
    # # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Finding the best threshold
    best_threshold_index = np.argmax(precision * recall)  # Maximize geometric mean of precision and recall
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold}")

def calculate_stats(prediction, mask_image, label_image, target_class_threshold):
    if target_class_threshold == 0:
        target_thresholds = np.arange(0.1,1.,0.1)
    else:
        target_thresholds = [target_class_threshold]
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    total_pixels = []
    for th in target_thresholds:
        # round predictions based on threshold
        pred_binary = (prediction >= th).astype(int)
        # Filter arrays based on mask
        valid_preds = pred_binary[mask_image > 0]
        valid_labels = label_image[mask_image > 0]
        # Calculate metrics
        true_positives.append(np.sum((valid_preds == 1) & (valid_labels == 1)))  # where we predict 1 and it is 1
        true_negatives.append(np.sum((valid_preds == 0) & (valid_labels == 0)))  # where we predict 0 and it is 0
        false_positives.append(np.sum((valid_preds == 1) & (valid_labels == 0)))  # where we predict 1 but it is 0
        false_negatives.append(np.sum((valid_preds == 0) & (valid_labels == 1)))  # where we predict 0 but it is 1
        # calculate accuracy, F1, precision, recall
        # correct_pixels = true_positives + true_negatives
        total_pixels.append(len(valid_labels))
    return (true_positives, true_negatives, false_positives, false_negatives, total_pixels, target_thresholds)

def load_instance_tifffiles(files):
    features_file, mask_file, label_file = files
    # load feature data
    feature_image = imread(features_file)
    feature_image = feature_image / 255.0
    # load mask
    mask_image = imread(mask_file)
    mask_image = mask_image / 255.0
    # plot_image(mask_image)
    # load label
    label_image = imread(label_file)
    label_image = label_image / 255.0
    # plot_image(label_image)
    return (feature_image,mask_image,label_image)


def load_instance_and_calculate_stats(args_list):
    instance, files, model, target_class_threshold, i, total_instances = args_list
    feature_image, mask_image, label_image = load_instance_tifffiles(files)
    # prepare feature image for input in model
    feature_image = feature_image.transpose(2, 0, 1)
    feature_image = torch.from_numpy(feature_image).to(torch.device("cpu"))
    feature_image = feature_image[None, ...]
    #plot_image(feature_image[0,2,:,:])
    # make predictions
    prediction = model(feature_image.float())
    prediction = prediction.cpu().detach().numpy()
    prediction = prediction[0, 0, :, :]
    # plot_image(prediction)

    pixel_labels, pixel_preds = compile_labels_and_predictions_per_pixel(prediction, mask_image, label_image)
    #true_positives, true_negatives, false_positives, false_negatives, total_pixels, target_thresholds = calculate_stats(prediction, mask_image, label_image, target_class_threshold)
    print(f'\rFinished instance %i/%i' %(i,total_instances), end='', flush=True)
    # return (true_positives, true_negatives, false_positives, false_negatives, total_pixels, target_thresholds)
    return(pixel_labels, pixel_preds)

def calculate_metrics_for_threshold(all_labels, all_preds, threshold):
    pred_binary = (all_preds >= threshold).astype(int)
    true_positives = np.sum((pred_binary == 1) & (all_labels == 1))
    true_negatives = np.sum((pred_binary == 0) & (all_labels == 0))
    false_positives = np.sum((pred_binary == 1) & (all_labels == 0))
    false_negatives = np.sum((pred_binary == 0) & (all_labels == 1))
    total_pixels = len(pred_binary)
    correct_pixels = true_positives + true_negatives
    accuracy = correct_pixels / total_pixels
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return [threshold, true_positives, true_negatives, false_positives, false_negatives, total_pixels,
            correct_pixels,
            accuracy, precision, recall, f1_score]

def produce_df_from_test_stats(all_labels_in,all_preds_in):
    all_labels = all_labels_in.copy()
    all_preds = all_preds_in.copy()
    # sum(np.isnan(all_preds)) / len(all_preds)
    # Check where predictions are NaN
    nan_indices = np.isnan(all_preds)
    # # For NaN predictions, set to 0 where label is 1, and 1 where label is 0, to treat them as false predictions
    # all_preds[nan_indices] = 1 - all_labels[nan_indices]
    # replace the nan pixels with random values, as we have no information for them
    np.random.seed(1234)
    all_preds[nan_indices] = np.random.random(sum(nan_indices))
    # plt.hist(all_preds, 99)

    # Assuming precision_list, recall_list, thresholds_list are defined as in your snippet
    precision_list, recall_list, thresholds_list = precision_recall_curve(all_labels, all_preds)
    f1_list = 2 * (precision_list * recall_list) / (precision_list + recall_list)
    # Find the maximum value ignoring NaNs
    max_f1 = np.nanmax(f1_list)
    # Find the index of the maximum value
    best_threshold_index = np.where(f1_list == max_f1)[0][0]  # np.where returns a tuple of arrays, one for each dimension
    # best_threshold_index = np.argmax(f1_list)
    best_threshold = thresholds_list[best_threshold_index]

    # Calculate AUC for Precision-Recall Curve
    pr_auc = auc(recall_list, precision_list)

    # Calculate stats for the best threshold
    best_stats = calculate_metrics_for_threshold(all_labels, all_preds, best_threshold)

    # Create a DataFrame to hold all stats
    columns = ['Threshold', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Total Pixels',
               'Correct Pixels', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    df_stats = pd.DataFrame([best_stats], columns=columns)

    # Generate stats for thresholds ranging from 0 to 1 at 0.01 increments
    for threshold in np.arange(0, 1.01, 0.01):
        stats = calculate_metrics_for_threshold(all_labels, all_preds, threshold)
        df_stats = pd.concat([df_stats, pd.DataFrame([stats], columns=columns)])
    return df_stats


def main(opt):

    if not os.path.exists(os.path.dirname(opt.output_file)):
        os.makedirs(os.path.dirname(opt.output_file))

    # load the model
    model = torch.load(opt.trained_model_path)
    # model.eval()

    # load the test instances
    file_dict = get_file_dict(opt.input_folder) # order of files for each key is: feature, mask, label
    file_dict = check_complete_instances(file_dict)
    sorted_keys = sorted(file_dict.keys())

    # #two lines below are for testing
    # from itertools import islice
    # file_dict = dict(islice(file_dict.items(), 100))

    total_instances = len(file_dict.keys())

    if opt.cores > 1:
        print("Running in parallel on %i cores."%opt.cores)
        # Function to be executed in the pool
        # Using ProcessPoolExecutor to parallelize the loop
        with concurrent.futures.ProcessPoolExecutor(max_workers=opt.cores) as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(load_instance_and_calculate_stats, [instance, file_dict[instance], model, opt.target_class_threshold, i, total_instances]) for i, instance in enumerate(sorted_keys)]

            # Collect results as they complete
            stats_list = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        stats_list = [load_instance_and_calculate_stats([instance, file_dict[instance], model, opt.target_class_threshold, i, total_instances]) for i, instance in enumerate(sorted_keys)]

    print()  # Move to the next line after the loop completes

    label_list = []
    pred_list = []
    for i in stats_list:
        label_list.append(i[0])
        pred_list.append(i[1])
    all_labels = np.concatenate(label_list)
    all_preds = np.concatenate(pred_list)

    df_stats = produce_df_from_test_stats(all_labels,all_preds)
    # Save the DataFrame to CSV
    df_stats = df_stats.round(4)
    df_stats.to_csv(opt.output_file, index=False)
    print("CSV file has been saved with model performance metrics.")
    if opt.save_predictions_to_file:
        np.save(opt.output_file.split('.')[0]+'_preds.npy',all_preds)
        np.save(opt.output_file.split('.')[0] + '_labels.npy', all_labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model testing configuration")
    # Adding arguments
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing data')
    parser.add_argument('--output_file', type=str, help='Path to the output file where the results will be saved')
    parser.add_argument('--trained_model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--target_class_threshold', type=float, default=0.5, help='Threshold value to classify target classes. When set to 0, the program will run multiple threshold (for evaluating optimal threshold)')
    parser.add_argument('--cores', type=int, default=1, help='Number of computing cores you want to use for parallelizing the test set predictions.')
    parser.add_argument('--save_predictions_to_file', action="store_true", default=False)
    opt = parser.parse_args()
    main(opt)


#
# from types import SimpleNamespace
#
# # Hardcoded options for troubleshooting purposes
# opt = SimpleNamespace(
#     input_folder='data/processed_geodata/boreal_south/boreal_south_geodata/testset',
#     output_file='results/test_stats/boreal_south_testset.csv',
#     trained_model_path='train/first_round/boreal_south/100,50,100/best_model.pth',
#     target_class_threshold=0,
#     cores = 1,
#     save_predictions_to_file=True
# )

