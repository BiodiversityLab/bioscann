import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve
from run_testset_prediction import produce_df_from_test_stats
import numpy as np
import matplotlib
# Set the global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'

def read_test_stats_from_csv_files(directory,threshold,extract_best_threshold, plot_p_r_curve=False):
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
                    'AUC': pr_auc,
                    'Best threshold': data.values[0,0]
                })
            if plot_p_r_curve:
                plt.figure(figsize=(8, 5))
                plt.plot(data['Recall'].values[1:], data['Precision'].replace(0.0, 1.0).values[1:], marker='o')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.grid(True)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.savefig('results/modeltesting/plots/precision_recall_curve_%s.pdf' % base_name, bbox_inches='tight')
    # Convert results to a DataFrame for better visualization
    output_df = pd.DataFrame(results)
    return output_df





# Directory containing the CSV files
directory = 'results/modeltesting/testset_performance' # csv files produced with run_testset_prediction.py
directory='/Users/toban562/Desktop/tmp'
threshold = 0.5
extract_best_threshold = False
output_df = read_test_stats_from_csv_files(directory,threshold,extract_best_threshold,plot_p_r_curve=False)

# Identify float columns in the DataFrame
float_cols = output_df.select_dtypes(include=['float64']).columns
# Round only the float columns to 3 decimal places
output_df[float_cols] = output_df[float_cols].round(4)
if extract_best_threshold:
    txtfile_name = 'results/modeltesting/testset_performance/modeltestscores_test_set_threshold_best.txt'
else:
    txtfile_name = 'results/modeltesting/testset_performance/modeltestscores_test_set_threshold_%.4f.txt'%threshold
output_df.to_csv(txtfile_name, sep='\t', index=False)


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
plt.xticks([r + barWidth + barWidth / 2 for r in range(len(r1))], output_df.experiment_name.values)
plt.xticks(rotation='vertical')

# Create legend & Show graphic
# plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0, 1)  # Scale y-axis from 0 to 1

plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)

# Save the plot to a PDF file
if extract_best_threshold:
    grouped_plot_file = 'results/modeltesting/plots/grouped_metrics_threshold_best.pdf'
else:
    grouped_plot_file = 'results/modeltesting/plots/grouped_metrics_threshold_%.4f.pdf'%threshold

plt.savefig(grouped_plot_file, bbox_inches='tight')

plt.show()


for metric in metrics:
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
    if extract_best_threshold:
        barplot_file = 'results/modeltesting/plots/%s_threshold_best.pdf'%metric
    else:
        barplot_file = 'results/modeltesting/plots/%s_threshold_%.4f.pdf'%(metric,threshold)

    plt.savefig(barplot_file, bbox_inches='tight')

    # Show the plot
    plt.show()





bins = 100
threshold = 0.9

npy_file = 'results/test_stats/boreal_south_batchsize_5_testset.npy'
preds = np.load(npy_file)
#plt.hist(preds,100,log=True)
plt.figure(figsize=(5, 3))
counts, bin_edges, patches = plt.hist(preds, bins=bins, log=True, color='#C0C0C0')  # Default grey color
# Color the bars based on the threshold
for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge >= threshold:
        patch.set_facecolor('#659C5D')  # Green color
plt.ylim(bottom=10000, top=10000000)
plt.title('%.4f'%(len(preds[preds>threshold])/len(preds)))
plt.savefig( 'results/test_stats/%s_pred_hist_threshold_%.1f.png'%(os.path.basename(npy_file).replace('_testset.npy',''),threshold),bbox_inches='tight', dpi=900)

npy_file = 'results/test_stats/boreal_south_regular_loss_batchsize_5_testset.npy'
preds = np.load(npy_file)
# plt.hist(preds,100,log=True)
plt.figure(figsize=(5, 3))
counts, bin_edges, patches = plt.hist(preds, bins=bins, log=True, color='#C0C0C0')  # Default grey color
# Color the bars based on the threshold
for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge >= threshold:
        patch.set_facecolor('#659C5D')  # Green color
plt.ylim(bottom=10000, top=10000000)
plt.title('%.4f'%(len(preds[preds>threshold])/len(preds)))
plt.savefig( 'results/test_stats/%s_pred_hist_threshold_%.1f.png'%(os.path.basename(npy_file).replace('_testset.npy',''),threshold),bbox_inches='tight', dpi=900)


threshold = 0.9
base_dir = 'results/modeltesting/'
region_list = ['alpine','boreal_east','boreal_northwest','boreal_south','continental']
preds = []
for region in region_list:
    npy_file = os.path.join(base_dir,'bubnicki_preds_%s.npy'%region)
    tmp_preds = np.load(npy_file)
    preds.append(tmp_preds)
preds = np.concatenate(preds)
valid_preds = preds[~np.isnan(preds)]
# plt.hist(preds,100,log=True)
plt.figure(figsize=(5, 3))
counts, bin_edges, patches = plt.hist(valid_preds, bins=bins, log=True, color='#C0C0C0')  # Default grey color
# Color the bars based on the threshold
for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge >= threshold:
        patch.set_facecolor('#659C5D')  # Green color
plt.ylim(bottom=10000, top=40000000)
plt.title('%.4f'%(len(valid_preds[valid_preds>threshold])/len(valid_preds)))
plt.savefig( 'results/test_stats/bubnicki_pred_hist_threshold_%.1f.png'%threshold,bbox_inches='tight', dpi=900)


threshold = 0.9
base_dir = 'results/testset_predictions_final_model_with_saved_arrays'
region_list = ['alpine','boreal_east','boreal_northwest','boreal_south','continental']
preds = []
labels = []
for region in region_list:
    preds_npy_file = os.path.join(base_dir,'sweden_all_batchsize_5_100,50,100_%s_testset_preds.npy'%region)
    tmp_preds = np.load(preds_npy_file)
    preds.append(tmp_preds)
    labels_npy_file = os.path.join(base_dir,'sweden_all_batchsize_5_100,50,100_%s_testset_labels.npy'%region)
    tmp_labels = np.load(labels_npy_file)
    labels.append(tmp_labels)
preds = np.concatenate(preds)
labels = np.concatenate(labels)
valid_preds = preds[~np.isnan(preds)]
# plt.hist(preds,100,log=True)
plt.figure(figsize=(5, 3))
counts, bin_edges, patches = plt.hist(valid_preds, bins=bins, log=True, color='#C0C0C0')  # Default grey color
# Color the bars based on the threshold
for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge >= threshold:
        patch.set_facecolor('#659C5D')  # Green color
plt.ylim(bottom=10000, top=40000000)
plt.title('%.4f'%(len(valid_preds[valid_preds>threshold])/len(valid_preds)))
plt.savefig( 'results/test_stats/sweden_all_pred_hist_threshold_%.1f.png'%threshold,bbox_inches='tight', dpi=900)




