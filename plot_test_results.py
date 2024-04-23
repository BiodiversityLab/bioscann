import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import os
import matplotlib
# Set the global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'


def read_test_stats_from_csv_files(directory,threshold,extract_best_threshold):
    # List to store results
    results = []
    # Loop through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Check if the file ends with '_testset.csv'
            filepath = os.path.join(directory, filename)
            data = pd.read_csv(filepath)
            if extract_best_threshold:
                threshold = data.values[0,0]
            # Filter the row where Threshold is 0.5
            filtered_data = data[data['Threshold'] == threshold]
            if not filtered_data.empty:
                # Extract needed metrics
                accuracy = filtered_data['Accuracy'].values[0]
                precision = filtered_data['Precision'].values[0]
                recall = filtered_data['Recall'].values[0]
                f1_score = filtered_data['F1 Score'].values[0]
                pr_auc = auc(data['Recall'].values[1:], data['Precision'].values[1:])
                # Extract part of the filename before '_testset.csv'
                base_name = filename.replace('_testset.csv', '')
                # Append the results
                results.append({
                    'experiment_name': base_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1_score,
                    'AUC': pr_auc
                })
    # Convert results to a DataFrame for better visualization
    output_df = pd.DataFrame(results)
    return output_df


test_stat_file = 'results/test_stats/boreal_south_batchsize_5_testset.csv'
df = pd.read_csv(test_stat_file)

# Plotting
plt.figure(figsize=(10, 6))
# Plot Accuracy
plt.plot(df['Threshold'].values, df['Accuracy'].values, marker='o', label='Accuracy', linestyle='-', color='blue')
# Plot Precision
plt.plot(df['Threshold'].values, df['Precision'].values, marker='s', label='Precision', linestyle='--', color='green')
# Plot Recall
plt.plot(df['Threshold'].values, df['Recall'].values, marker='^', label='Recall', linestyle='-.', color='red')
# Adding title, labels, grid, and legend
plt.title('Model Performance Metrics vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metrics')
plt.grid(True)
plt.legend()  # Adds a legend to identify the lines
plt.show()
#
#
# # Calculating Precision-Recall values
# precision, recall, thresholds = precision_recall_curve(df['True Positives'].values + df['False Negatives'].values, df['True Positives'].values / (df['True Positives'].values + df['False Negatives'].values))
#
# # Calculate AUC for Precision-Recall Curve
# pr_auc = auc(recall, precision)
#
# # Plot Precision-Recall Curve
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Finding the best threshold
# best_threshold_index = np.argmax(precision * recall)  # Maximize geometric mean of precision and recall
# best_threshold = thresholds[best_threshold_index]
# print(f"Best Threshold: {best_threshold}")



# Set the directory containing the CSV files
directory = 'results/modeltesting'
metric = 'Accuracy'

# Initialize a figure for plotting
plt.figure(figsize=(10, 8))

# Variables to track the highest precision and corresponding file at threshold 0.6
max_precision = 0
max_precision_file = ""

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        if not 'reguXXXXlar_loss' in filename:
            # Construct the full file path
            filepath = os.path.join(directory, filename)
            # Read the CSV file, skip the first row (index row 0)
            data = pd.read_csv(filepath, skiprows=[1])
            if metric == 'Precision':
                # Replace 0.0 precision with 1.0
                data['%s'%metric] = data['%s'%metric].replace(0.0, 1.0)
            # Extract the 'Threshold' and 'Precision' columns
            thresholds = data['Threshold']
            precision = data['%s'%metric]
            # Plot the data
            plt.plot(thresholds.values, precision.values, label=f'{filename[:-4]}')  # remove '.csv' from filename for the label

            # Check precision at threshold 0.6
            precision_at_threshold = data[data['Threshold'] == 0.05]['%s'%metric].iloc[0]
            if precision_at_threshold > max_precision:
                max_precision = precision_at_threshold
                max_precision_file = filename

# Configure plot settings
plt.title('%s vs. Threshold'%metric)
plt.xlabel('Threshold')
plt.ylabel('%s'%metric)
plt.legend()
plt.grid(True)
# Save the plot to a PDF file
plt.savefig('results/%s_vs_threshold.pdf'%metric, bbox_inches='tight')

# Show the plot
plt.show()

# Print the file with the highest precision at threshold 0.6
print(f"The highest %s at threshold 0.05 is {max_precision}, found in file {max_precision_file}."%metric)



# AUC
directory = 'results/modeltesting'

auc_values = []
model_names = []
# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        filepath = os.path.join(directory, filename)
        # Read the CSV file, skip the first row (index row 0)
        data = pd.read_csv(filepath, skiprows=[1])
        pr_auc = auc(data['Recall'].values, data['Precision'].values)
        auc_values.append(pr_auc)
        modelname = f'{filename[:-4]}'.replace('_testset','')
        model_names.append(modelname)

# Initialize a figure for plotting
plt.figure(figsize=(10, 8))
# Create a barplot
plt.bar(model_names,auc_values)

# Set the tick labels on the x-axis to vertical
plt.xticks(rotation='vertical')
# Set y-axis ticks every 0.1
plt.yticks(np.arange(0, 1.1, 0.1))
# Add gridlines
plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5)

# Title and labels (optional)
plt.title('Barplot with Vertical Tick Labels')
plt.xlabel('Groups')
plt.ylabel('Values')

# Save the plot to a PDF file
plt.savefig('results/AUC_scores.pdf', bbox_inches='tight')

# Show the plot
plt.show()




# load the csv file data in the target_dir_________________________________
# Directory containing the CSV files
directory = 'results/modeltesting'
threshold = 0.5
extract_best_threshold = False

output_df = read_test_stats_from_csv_files(directory,threshold,extract_best_threshold)

if 'test_stats' in directory:
    order_of_keys = ['boreal_south', 'boreal_east', 'boreal_northwest', 'alpine', 'continental']
    # Create a categorical type based on the defined order of keys
    output_df['experiment_name'] = pd.Categorical(output_df['experiment_name'], categories=order_of_keys, ordered=True)
    # Sort the DataFrame by 'Filename Base'
    output_df = output_df.sort_values('experiment_name')
    # Format 'experiment_name' for display
    formatted_experiment_names = [name.replace('_', ' ').capitalize() for name in output_df['experiment_name']]
else:
    formatted_experiment_names = [name for name in output_df['experiment_name']]

# Set the position of the bars
barWidth = 0.2
r1 = np.arange(len(output_df['experiment_name']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Create the bar plot
plt.figure(figsize=(10, 5))
plt.bar(r1, output_df['Accuracy'], color='b', width=barWidth, edgecolor='grey', label='Accuracy', zorder=3)
plt.bar(r2, output_df['F1 Score'], color='g', width=barWidth, edgecolor='grey', label='F1', zorder=3)
plt.bar(r3, output_df['Precision'], color='r', width=barWidth, edgecolor='grey', label='Precision', zorder=3)
plt.bar(r4, output_df['Recall'], color='c', width=barWidth, edgecolor='grey', label='Recall', zorder=3)

# Add xticks on the middle of the group bars
plt.xlabel('Experiment Name', fontweight='bold')
plt.ylabel('Metrics Value', fontweight='bold')
plt.xticks([r + barWidth + barWidth / 2 for r in range(len(r1))], formatted_experiment_names)

# Create legend & Show graphic
# plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0, 1)  # Scale y-axis from 0 to 1

plt.xticks(rotation='vertical')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)

# Save the plot to a PDF file
if extract_best_threshold:
    plt.savefig(os.path.join(directory, 'grouped_metrics_plot_testset_threshold_optimal.pdf'),bbox_inches='tight')
else:
    plt.savefig(os.path.join(directory,'grouped_metrics_plot_testset_threshold_%.2f.pdf'%threshold), bbox_inches='tight')

# plt.show()

# Identify float columns in the DataFrame
float_cols = output_df.select_dtypes(include=['float64']).columns
# Round only the float columns to 3 decimal places
output_df[float_cols] = output_df[float_cols].round(3)
output_df.values
# write to file
if extract_best_threshold:
    output_df.to_csv(os.path.join(directory, 'regional_model_scores_testset_threshold_optimal.txt'), sep='\t', index=False)
else:
    output_df.to_csv(os.path.join(directory,'regional_model_scores_testset_threshold_%.2f.txt'%threshold), sep='\t', index=False)






# Directory containing the CSV files
directory = 'results/single_bar_plots'
threshold = 0.5
extract_best_threshold = False
metric = 'Accuracy'

output_df = read_test_stats_from_csv_files(directory,threshold,extract_best_threshold)

# Initialize a figure for plotting
plt.figure(figsize=(10, 8))
# Create a barplot
plt.bar(output_df.experiment_name.values,output_df[metric].values)

# Set the tick labels on the x-axis to vertical
plt.xticks(rotation='vertical')
# Set y-axis ticks every 0.1
plt.yticks(np.arange(0, 1.1, 0.1))
# Add gridlines
plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5)

# Title and labels (optional)
plt.title('Barplot with Vertical Tick Labels')
plt.xlabel('Groups')
plt.ylabel('Values')

# Save the plot to a PDF file
plt.savefig('results/single_bar_plots/%s_scores.pdf'%metric, bbox_inches='tight')

# Show the plot
plt.show()
