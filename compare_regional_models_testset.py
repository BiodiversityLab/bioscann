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


test_stat_file = 'results/modeltesting/bubnicki_combined_sweden.csv'
# test_stat_file = 'results/final_test_stats_regional_models/combined_sweden.csv'
# test_stat_file = 'results/modeltesting/testset_performance/100,50,100_5_regular_loss_testset.csv'
# test_stat_file = 'results/modeltesting_corrected_loss/boreal_south_batchsize_5_corrected_loss_100,50,100_testset.csv'
df = pd.read_csv(test_stat_file)

thresholds = df['Threshold'].values[1:]
precision = df['Precision'].replace(0.0, 1.0).values[1:]
recall = df['Recall'].values[1:]
f1_score = df['F1 Score'].values[1:]
accuracy = df['Accuracy'].values[1:]

selected_thresholds = [0.1,0.5,0.9]


plt.figure(figsize=(5, 3))
# Plot Precision, replacing 0.0 with 1.0
plt.plot(thresholds, precision, label='Precision', linestyle='-', color='#5E1675'.lower())
# Plot Recall
plt.plot(thresholds, recall, label='Recall', linestyle='-', color='#EE4266'.lower())
# Plot F1
plt.plot(thresholds, f1_score, label='F1', linestyle='-', color='#FFD23F'.lower())
# Adding title, labels, and legend
#plt.title('Model Performance Metrics vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metrics Value')
plt.legend(loc='lower center')  # Adds a legend to identify the lines
# Setting grid
plt.xticks(ticks=np.arange(0, 1.1, 0.1))  # Set x-ticks at 0.1 intervals
plt.yticks(ticks=np.arange(0, 1.1, 0.1))  # Set y-ticks at 0.1 intervals
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

plt.axvline(x=selected_thresholds[0], color='grey', linestyle='--', linewidth=1)  # Vertical line at x=0.1
plt.axvline(x=selected_thresholds[1], color='grey', linestyle='--', linewidth=1)  # Vertical line at x=0.5
plt.axvline(x=selected_thresholds[2], color='grey', linestyle='--', linewidth=1)  # Vertical line at x=0.9

plt.ylim(0.,1.05)
plt.show()
plt.savefig('results/test_stats/%s_pr_curve.png'%os.path.basename(test_stat_file).replace('_testset.csv',''), bbox_inches='tight', dpi=900)


selected_acc = np.array([accuracy[thresholds==selected_thresholds[0]],accuracy[thresholds==selected_thresholds[1]],accuracy[thresholds==selected_thresholds[2]]]).flatten()
selected_pre = np.array([precision[thresholds==selected_thresholds[0]],precision[thresholds==selected_thresholds[1]],precision[thresholds==selected_thresholds[2]]]).flatten()
selected_rec = np.array([recall[thresholds==selected_thresholds[0]],recall[thresholds==selected_thresholds[1]],recall[thresholds==selected_thresholds[2]]]).flatten()
selected_f1s = np.array([f1_score[thresholds==selected_thresholds[0]],f1_score[thresholds==selected_thresholds[1]],f1_score[thresholds==selected_thresholds[2]]]).flatten()

# make the multibarplot plot
# Set the position of the bars
barWidth = 0.2
r1 = np.arange(len(selected_thresholds))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Create the bar plot
plt.figure(figsize=(5, 3))
plt.bar(r1, selected_acc, color='#337357'.lower(), width=barWidth, edgecolor='grey', label='Accuracy', zorder=3)
plt.bar(r2, selected_pre, color='#5E1675'.lower(), width=barWidth, edgecolor='grey', label='Precision', zorder=3)
plt.bar(r3, selected_rec, color='#EE4266'.lower(), width=barWidth, edgecolor='grey', label='Recall', zorder=3)
plt.bar(r4, selected_f1s, color='#FFD23F'.lower(), width=barWidth, edgecolor='grey', label='F1 score', zorder=3)

# Add xticks on the middle of the group bars
plt.xlabel('Threshold')
plt.ylabel('Metrics Value')
plt.xticks([r + barWidth + barWidth / 2 for r in range(len(r1))], np.array(selected_thresholds).astype(str))
#plt.xticks(rotation='vertical')

# Create legend & Show graphic
# plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0, 1)  # Scale y-axis from 0 to 1

plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
plt.ylim(0.,1.05)
plt.savefig( 'results/test_stats/%s_barplot_metrics_thresholds.png'%os.path.basename(test_stat_file).replace('_testset.csv',''),bbox_inches='tight', dpi=900)





#

# Calculating Precision-Recall values
# precision, recall, thresholds = precision_recall_curve(df['True Positives'].values[1:] + df['False Negatives'].values[1:], df['True Positives'].values[1:] / (df['True Positives'].values[1:] + df['False Negatives'].values[1:]))

# Calculate AUC for Precision-Recall Curve
pr_auc = auc(df['Recall'].values[1:], df['Precision'].replace(0.0, 1.0).values[1:])

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(df['Recall'].values[1:], df['Precision'].replace(0.0, 1.0).values[1:], label=f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# # Finding the best threshold
# best_threshold_index = np.argmax(precision * recall)  # Maximize geometric mean of precision and recall
# best_threshold = thresholds[best_threshold_index]
# print(f"Best Threshold: {best_threshold}")



# Set the directory containing the CSV files
directory = 'results/test_stats'
metric = 'F1 Score'

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
plt.savefig('results/test_stats/%s_vs_threshold.pdf'%metric, bbox_inches='tight')

# Show the plot
plt.show()

# Print the file with the highest precision at threshold 0.6
print(f"The highest %s at threshold 0.05 is {max_precision}, found in file {max_precision_file}."%metric)



# AUC
directory = 'results/test_stats'

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
plt.savefig('results/test_stats/AUC_scores.pdf', bbox_inches='tight')

# Show the plot
plt.show()



# combine regional models stats for overall sweden stats_______________________________________
regional_models = False
if regional_models:
    # Load all CSV files
    alpine = pd.read_csv('results/final_test_stats_regional_models/alpine.csv', skiprows=[1])
    boreal_east = pd.read_csv('results/final_test_stats_regional_models/boreal_east.csv', skiprows=[1])
    boreal_northwest = pd.read_csv('results/final_test_stats_regional_models/boreal_northwest.csv', skiprows=[1])
    boreal_south = pd.read_csv('results/final_test_stats_regional_models/boreal_south.csv', skiprows=[1])
    continental = pd.read_csv('results/final_test_stats_regional_models/continental.csv', skiprows=[1])
    outfile = 'results/final_test_stats_regional_models/combined_regional.csv'
else:
    # Load all CSV files
    alpine = pd.read_csv('results/final_test_stats_regional_models/sweden_alpine.csv', skiprows=[1])
    boreal_east = pd.read_csv('results/final_test_stats_regional_models/sweden_boreal_east.csv', skiprows=[1])
    boreal_northwest = pd.read_csv('results/final_test_stats_regional_models/sweden_boreal_northwest.csv', skiprows=[1])
    boreal_south = pd.read_csv('results/final_test_stats_regional_models/sweden_boreal_south.csv', skiprows=[1])
    continental = pd.read_csv('results/final_test_stats_regional_models/sweden_continental.csv', skiprows=[1])
    outfile = 'results/final_test_stats_regional_models/combined_sweden.csv'

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








# load the csv file data in the target_dir_________________________________
# Directory containing the CSV files
directory = 'results/final_test_stats_regional_models'
threshold = 0.5
extract_best_threshold = True

output_df = read_test_stats_from_csv_files(directory,threshold,extract_best_threshold)

# Keys to group by
order_of_keys = ['boreal_south', 'boreal_east', 'boreal_northwest', 'alpine', 'continental', 'bubnicki']

# Create a column to indicate group based on the order of keys
def get_group(experiment_name):
    for key in order_of_keys:
        if key in experiment_name:
            return key
    return 'other'

output_df['group'] = output_df['experiment_name'].apply(get_group)

# Sort the dataframe by group
output_df = output_df.sort_values(['group','experiment_name']).reset_index(drop=True)
formatted_experiment_names = [name for name in output_df['experiment_name']]
metrics = ['Accuracy','F1 Score','Precision','Recall','AUC']

# make the multibarplot plot
# Set the position of the bars
barWidth = 0.15
r1 = np.arange(len(output_df['experiment_name']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Create the bar plot
plt.figure(figsize=(10, 5))
plt.bar(r1, output_df[metrics[0]], color='b', width=barWidth, edgecolor='grey', label=metrics[0], zorder=3)
plt.bar(r2, output_df[metrics[1]], color='g', width=barWidth, edgecolor='grey', label=metrics[1], zorder=3)
plt.bar(r3, output_df[metrics[2]], color='r', width=barWidth, edgecolor='grey', label=metrics[2], zorder=3)
plt.bar(r4, output_df[metrics[3]], color='c', width=barWidth, edgecolor='grey', label=metrics[3], zorder=3)
plt.bar(r5, output_df[metrics[4]], color='y', width=barWidth, edgecolor='grey', label=metrics[4], zorder=3)

# Add xticks on the middle of the group bars
plt.xlabel('Model', fontweight='bold')
plt.ylabel('Metrics Value', fontweight='bold')
plt.xticks([r + barWidth + barWidth / 2 for r in range(len(r1))], formatted_experiment_names)
plt.xticks(rotation='vertical')

# Create legend & Show graphic
# plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0, 1)  # Scale y-axis from 0 to 1

plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)

# Save the plot to a PDF file
if extract_best_threshold:
    plt.savefig(os.path.join(directory, 'grouped_metrics_plot_testset_threshold_optimal.pdf'),bbox_inches='tight')
else:
    plt.savefig(os.path.join(directory,'grouped_metrics_plot_testset_threshold_%.2f.pdf'%threshold), bbox_inches='tight')

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





# plot subplots by metric________________________________________________________________
directory = 'results/final_test_stats_regional_models'
file_path = os.path.join(directory,'regional_model_scores_testset_threshold_0.50.txt')
# file_path = 'results/final_test_stats_regional_models/regional_model_scores_testset_threshold_optimal.txt'
threshold = file_path.split('_')[-1].split('.txt')[0]
data = pd.read_csv(file_path, sep='\t')

# Define the metrics and colors
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
colors = ['b', 'c', 'r', 'g', 'y']
alpha_value = 0.6

# Define the unique groups and their positions
unique_groups = data['group'].unique()
group_positions = {group: [] for group in unique_groups}

# Assign positions for each group
position = 0
for group in unique_groups:
    indices = data[data['group'] == group].index.tolist()
    group_positions[group] = [position, position + 1]
    position += 2.5  # Move to the next group with more space in between

# Create subplots with grouped bars
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    for group, positions in group_positions.items():
        group_data = data[data['group'] == group][metric]
        axes[idx].bar(positions[0], group_data.iloc[0], width=1, color=colors[idx], edgecolor='black')
        axes[idx].bar(positions[1], group_data.iloc[1], width=1, color=colors[idx], alpha=alpha_value, edgecolor='black')
        # axes[idx].bar(positions, data[data['group'] == group][metric], width=1, color=colors[idx],edgecolor='black')
    axes[idx].set_title(metric)
    axes[idx].tick_params(axis='x', rotation=90)
    axes[idx].set_xticks([np.mean(positions) for positions in group_positions.values()])
    axes[idx].set_xticklabels(unique_groups)
    axes[idx].set_ylim(0, 1)
    axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axes[idx].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(directory,'metrics_subplots_threshold_%s.pdf'%threshold), bbox_inches='tight')




# plot subplots by region________________________________________________________________
# Load the data
directory = 'results/final_test_stats_regional_models'
# file_path = os.path.join(directory,'regional_model_scores_testset_threshold_0.50.txt')
file_path = os.path.join(directory,'regional_model_scores_testset_threshold_optimal.txt')
threshold = file_path.split('_')[-1].split('.txt')[0]
data = pd.read_csv(file_path, sep='\t')

# Define the metrics and colors
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
colors = ['b', 'c', 'r', 'g', 'y']
alpha_value = 0.6  # Opacity for second bar of each pair

# Define the unique groups
unique_groups = data['group'].unique()

# Create a figure and axes for the subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = axes.flatten()

# Bar width
bar_width = 1

for i, group in enumerate(unique_groups):
    group_data = data[data['group'] == group]
    experiments = group_data['experiment_name'].unique()
    positions = np.arange(len(metrics)) * (len(experiments) + 0.5)  # Initial positions for the first experiment

    for idx, metric in enumerate(metrics):
        for j, experiment in enumerate(experiments):
            experiment_data = group_data[group_data['experiment_name'] == experiment]
            # Adjust bar positions and plot
            pos = positions[idx] + j * (bar_width)  # Shift position for each experiment
            axes[i].bar(pos, experiment_data[metric].values[0], width=bar_width, color=colors[idx], alpha=alpha_value if j % 2 == 1 else 1, edgecolor='black')

    # Set x-ticks to be in the middle of each metric group
    axes[i].set_xticks(positions + (len(experiments) - 1) * bar_width / 2)
    axes[i].set_xticklabels(metrics)
    axes[i].set_title(f"Group: {group}")
    axes[i].set_ylim(0, 1)
    axes[i].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axes[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(directory,'region_subplots_threshold_%s.pdf'%threshold), bbox_inches='tight')





# plot subplots by region as individual files________________________________________________________________
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = 'Arial'
# Load the data
directory = 'results/final_test_stats_regional_models'
# file_path = os.path.join(directory,'regional_model_scores_testset_threshold_0.50.txt')
file_path = os.path.join(directory,'regional_model_scores_testset_threshold_optimal.txt')
threshold = file_path.split('_')[-1].split('.txt')[0]
data = pd.read_csv(file_path, sep='\t')

# Define the metrics and colors
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
# colors = ['b', 'c', 'r', 'g', 'y']
# colors = ['lightgrey','lightgrey','lightgrey','lightgrey','lightgrey']
colors = ['#DBE2B5','#DEB5DB','#C0F4C3','#94D1E2','#9888EA','lightgrey']
alpha_value = 1.0  # Opacity for second bar of each pair

# Define the unique groups
unique_groups = data['group'].unique()

# Bar width
bar_width = 1

if not os.path.exists(directory):
    os.makedirs(directory)

for i, group in enumerate(unique_groups):
    fig, ax = plt.subplots(figsize=(10, 5))
    group_data = data[data['group'] == group]
    experiments = group_data['experiment_name'].unique()
    positions = np.arange(len(metrics)) * (len(experiments) + 0.5)  # Initial positions for the first experiment

    for idx, metric in enumerate(metrics):
        for j, experiment in enumerate(experiments):
            experiment_data = group_data[group_data['experiment_name'] == experiment]
            if group =='other':
                pos = positions[idx] + j * (bar_width)  # Shift position for each experiment
                if j % 2 == 0:  # Apply multi-color effect to every first bar of the pair
                    # Number of color segments
                    num_segments = len(colors[:-1])
                    segment_height = experiment_data[metric].values[0] / num_segments
                    # Plot each segment
                    for i, color in enumerate(colors[:-1]):
                        # Each segment starts at the top of the previous segment
                        bottom = i * segment_height
                        ax.bar(pos, segment_height, width=bar_width, bottom=bottom, color=color,
                               edgecolor=color)
                    ax.bar(pos, experiment_data[metric].values[0], width=bar_width, color='none', edgecolor='black')

                else:
                    ax.bar(pos, experiment_data[metric].values[0], width=bar_width, color='lightgrey',
                           alpha=alpha_value, edgecolor='black')

            else:
                # Adjust bar positions and plot
                pos = positions[idx] + j * (bar_width)  # Shift position for each experiment
                ax.bar(pos, experiment_data[metric].values[0], width=bar_width, color='lightgrey' if j % 2 == 1 else colors[i],
                       alpha=alpha_value if j % 2 == 1 else 1, edgecolor='black')

    # Set x-ticks to be in the middle of each metric group
    ax.set_xticks(positions + (len(experiments) - 1) * bar_width / 2)
    ax.set_xticklabels(metrics)
    # ax.set_title(f"Group: {group}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    # Save each plot as a PNG file
    file_name = os.path.join(directory,'individual_plots', f'{group}_metrics_grouped_paired_threshold_{threshold}.png')
    plt.savefig(file_name, bbox_inches='tight', dpi=900)
    plt.close()  # Close the figure to free up memory





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



# plot the modle compariosn between our model and bubnicki state-of-the-art
data = {
    "Models": ['State of the art model', 'Our model - "High precision"', 'Our model - "Generalist"'],
    "Accuracy": [0.704, 0.881, 0.911],
    "Precision": [0.443, 0.941, 0.809],
    "Recall": [0.752, 0.557, 0.841],
    "F1 Score": [0.558, 0.699, 0.824],
    "AUC Score": [0.482, 0.860, 0.860]
}


# Creating DataFrame
df = pd.DataFrame(data)
# Reorder the rows
df = df.reindex([1, 2, 0]).reset_index(drop=True)

rows_to_plot = [1,2]
overlay_idx=0


fig, ax = plt.subplots(figsize=(5, 7))

# Colors for each model
#colors = ['#80cdc1', '#018571', '#a6611a']
colors = ['#FECC02','#006AA7','grey']
alpha_values = [0.7, 1.0, 1.0]  # Transparency levels

# Positions for the bars
positions = np.arange(len(df.columns) - 1)
width = 0.3
num_models = len(rows_to_plot)

# Create bars for each metric
for i, idx in enumerate(rows_to_plot):
    model = df.loc[idx, 'Models']
    metrics = df.loc[idx, df.columns != 'Models']
    ax.bar(positions + i * width, metrics, width, label=model, color=colors[idx], edgecolor='black',
           alpha=alpha_values[idx])

# Plot the overlay index (e.g., index 0) on top with full opacity
if overlay_idx is not None:
    model = df.loc[overlay_idx, 'Models']
    metrics = df.loc[overlay_idx, df.columns != 'Models']
    # Overlaying on the position of row index 1
    overlay_position = rows_to_plot.index(rows_to_plot[0])
    ax.bar(positions + overlay_position * width, metrics, width, label=model, color=colors[overlay_idx],
           edgecolor='black', alpha=alpha_values[overlay_idx])

# Adjust x-ticks to be in the center of the groups of bars
ax.set_xticks(positions + width * (num_models - 1) / 2)
ax.set_xticklabels(df.columns[1:])  # Metric names
# ax.legend()

plt.xticks(rotation=0)  # Keep metric names horizontal for readability
plt.yticks(ticks=np.arange(0, 1.1, 0.1))  # Set y-ticks at 0.1 intervals
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
plt.show()
plt.savefig('results/modeltesting/plots/comparison_bubnicki_all_sweden.png', bbox_inches='tight', dpi=900)


#________________________________make plot with only one of our models compared to reference model______________
fig, ax = plt.subplots(figsize=(5, 7))

new_df = df.copy()
new_df = new_df.loc[1:,]
# Colors for each model
#colors = ['#80cdc1', '#018571', '#a6611a']
colors = ['#006AA7','#FECC02','grey']
alpha_values = [1.0, 1.0, 1.0]  # Transparency levels
rows_to_plot = [1,2]

# Positions for the bars
positions = np.arange(len(new_df.columns) - 1)
width = 0.3

# Create bars for each metric
for i, idx in enumerate(rows_to_plot):
    model = new_df.loc[idx, 'Models']
    metrics = new_df.loc[idx, new_df.columns != 'Models']
    ax.bar(positions + i * width, metrics, width, label=model, color=colors[idx], edgecolor='black',
           alpha=alpha_values[idx])

# Adjust x-ticks to be in the center of the groups of bars
ax.set_xticks(positions + width * (num_models - 1) / 2)
ax.set_xticklabels(new_df.columns[1:])  # Metric names
# ax.legend()

plt.xticks(rotation=0)  # Keep metric names horizontal for readability
plt.yticks(ticks=np.arange(0, 1.1, 0.1))  # Set y-ticks at 0.1 intervals
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
plt.show()
plt.savefig('results/modeltesting/plots/comparison_bubnicki_all_sweden_2_model.png', bbox_inches='tight', dpi=900)



