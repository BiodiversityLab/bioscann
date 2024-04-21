import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import numpy as np
# Set the global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'

model_testing = True
training_history = False

# Initialize the MLflow client
client = MlflowClient()
# mlflow.set_tracking_uri('your_tracking_uri')

if not training_history:
    # #Specify the experiment ID or name
    # experiment_id = '10'#'450601850519144894' #'10'
    # #Query all runs from the experiment
    # runs = mlflow.search_runs([experiment_id])
    # filtered_runs = runs
    if model_testing:
        target_runs = [
            'f37b7d88f3734138a29db8d27a26c3b1',
            '2bb766687b0f416aac3d0bc27c6f103c',
            'cd58b9807dc44d339ab573f8b938e781',
            '9301c49d85a341e3a6e5f8af52f49bf9',
            'c08ad75e6ddd43e39a855194881df7e5',
            '1874024118f14411aad13ed9b5b120a3',
            'adde051ef5674fe9b73601b47eac4df1',
            '147a52315d544dfe80b798fae06eff64',
            'b913a3e16178489f98e6568d71222037',
            'eefb351a68414670beec213f9cab9487',
            'c9bf11ee679e4fb3948a266ee1fd031a',
            '3b0e76bbe9004e2f9c5682215feccb86',
            '4fdd6eda52a1489faf88eb413a59ee48'
            ]
    else:
        #main regional models
        target_runs = ['1810ee63cf3544f7beffb7e465f15536',
                       '1874024118f14411aad13ed9b5b120a3',
                       '4e3f21d8bcbe4c5b8f11111630b91695',
                       '7779d3583c8a418f93e798e058c3993f',
                       '94b233e5b1d541cbba7a980f74ad71de'
                       ]

    # filtered_runs = pd.concat([runs[runs['run_id'] == run_id] for run_id in target_runs if run_id in runs['run_id'].values], ignore_index=True)
    # filtered_runs = runs[runs['run_id'].isin(target_runs)]
    # filtered_runs.columns
    # output_df = filtered_runs[['params.experiment_name',
    #                            'params.n_channels_per_layer',
    #                            'params.batch_size',
    #                            'metrics.val_binary_accuracy',
    #                            'metrics.val_binary_F1',
    #                            'metrics.val_binary_precision',
    #                            'metrics.val_binary_recall',
    #                            'metrics.val_loss',
    #                            'metrics.train_loss']]

    # Fetch run data for each run ID
    # Prepare a list to hold all run data
    all_run_data = []

    # Fetch each run's data
    for run_id in target_runs:
        run = client.get_run(run_id)  # Retrieve the run from MLflow
        # Extract parameters and metrics
        run_data = {
            'experiment_name': run.data.params.get('experiment_name', 'N/A'),
            'n_channels_per_layer': run.data.params.get('n_channels_per_layer', 'N/A'),
            'batch_size': run.data.params.get('batch_size', 'N/A'),
            'val_binary_accuracy': run.data.metrics.get('val_binary_accuracy', None),
            'val_binary_F1': run.data.metrics.get('val_binary_F1', None),
            'val_binary_precision': run.data.metrics.get('val_binary_precision', None),
            'val_binary_recall': run.data.metrics.get('val_binary_recall', None),
            'val_loss': run.data.metrics.get('val_loss', None),
            'train_loss': run.data.metrics.get('train_loss', None)
        }
        all_run_data.append(run_data)  # Add to list

    # Create a DataFrame from the list of run data
    output_df = pd.DataFrame(all_run_data)

    if model_testing:
        # Identify float columns in the DataFrame
        float_cols = output_df.select_dtypes(include=['float64']).columns
        # Round only the float columns to 3 decimal places
        output_df[float_cols] = output_df[float_cols].round(3)
        output_df.to_csv('results/modeltesting/modeltestscores.txt', sep='\t', index=False)
    else:
        # Sort DataFrame to put 'boreal_south' first and maintain the order for the rest
        # Assuming 'boreal_south' is the desired first experiment, otherwise adjust as needed
        output_df['sort_key'] = output_df['experiment_name'].apply(lambda x: 0 if x == 'boreal_south' else 1)
        output_df.sort_values(by='sort_key', inplace=True)
        output_df.drop(columns='sort_key', inplace=True)



        # Format 'experiment_name' for display
        formatted_experiment_names = [name.replace('_', ' ').capitalize() for name in output_df['experiment_name']]


        # Set the position of the bars
        barWidth = 0.2
        r1 = np.arange(len(output_df['experiment_name']))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]

        # Create the bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(r1, output_df['val_binary_accuracy'], color='b', width=barWidth, edgecolor='grey', label='Accuracy', zorder=3)
        plt.bar(r2, output_df['val_binary_F1'], color='g', width=barWidth, edgecolor='grey', label='F1', zorder=3)
        plt.bar(r3, output_df['val_binary_precision'], color='r', width=barWidth, edgecolor='grey', label='Precision', zorder=3)
        plt.bar(r4, output_df['val_binary_recall'], color='c', width=barWidth, edgecolor='grey', label='Recall', zorder=3)

        # Add xticks on the middle of the group bars
        plt.xlabel('Experiment Name', fontweight='bold')
        plt.ylabel('Metrics Value', fontweight='bold')
        plt.xticks([r + barWidth + barWidth/2 for r in range(len(r1))], formatted_experiment_names)

        # Create legend & Show graphic
        #plt.legend()
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.ylim(0, 1)  # Scale y-axis from 0 to 1

        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)

        # Save the plot to a PDF file
        plt.savefig('results/grouped_metrics_plot.pdf', bbox_inches='tight')

        plt.show()

        # Identify float columns in the DataFrame
        float_cols = output_df.select_dtypes(include=['float64']).columns
        # Round only the float columns to 3 decimal places
        output_df[float_cols] = output_df[float_cols].round(3)
        output_df.values
        # write to file
        output_df.to_csv('results/regional_model_scores.txt',sep='\t',index=False)

    #
    # # Also select the training history of selected runs
    # selected_metric = 'train_loss'
    #
    # for run_id in filtered_runs.run_id:
    #     metric_history = client.get_metric_history(run_id, selected_metric)
    #     # Extracting epoch numbers and corresponding values
    #     epochs = [metric.step for metric in metric_history]
    #     values = [metric.value for metric in metric_history]
    #     filtered_epochs, filtered_values = zip(*[(epoch, value) for epoch, value in zip(epochs, values) if not math.isnan(value)])
    #     plt.plot(filtered_epochs,filtered_values)
    #
    #
    #
    # data_to_export = filtered_runs[['run_id',selected_metric]]
    #
    # # Filter and reformat the data as needed
    # # For example, you might want to include specific metrics or parameters
    # data_to_export = filtered_runs[['run_id', 'metrics.your_metric_name', 'params.your_param_name']]
    #
    # # Export to CSV
    # data_to_export.to_csv('exported_mlflow_data.csv', index=False)

else:


    # boreal_east
    run_id = '1810ee63cf3544f7beffb7e465f15536'
    # boreal_south
    run_id = '1874024118f14411aad13ed9b5b120a3'
    # boreal_northwest
    run_id = '4e3f21d8bcbe4c5b8f11111630b91695'
    # alpine
    run_id = '7779d3583c8a418f93e798e058c3993f'
    # continental
    run_id = '94b233e5b1d541cbba7a980f74ad71de'

    # Retrieve metric history for both 'train_loss' and 'val_loss'
    def get_metric_history(run_id, metric_name):
        return client.get_metric_history(run_id, metric_name)

    train_loss_history = get_metric_history(run_id, 'train_loss')
    val_loss_history = get_metric_history(run_id, 'val_loss')

    # Extract values and epochs (assuming timestamps represent epochs for simplicity)
    train_epochs = [metric.step for metric in train_loss_history]
    train_values = [metric.value for metric in train_loss_history]
    val_epochs = [metric.step for metric in val_loss_history]
    val_values = [metric.value for metric in val_loss_history]

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_values, label='Train Loss', marker='o', linestyle='-')
    plt.plot(val_epochs, val_values, label='Validation Loss', marker='x', linestyle='--')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    # regular loss function
    run_id = '4fdd6eda52a1489faf88eb413a59ee48'

