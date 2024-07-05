import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import numpy as np
# Set the global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'

# Initialize the MLflow client
client = MlflowClient()
# mlflow.set_tracking_uri('your_tracking_uri')


model_testing = False
training_history = True

if training_history: # pick a run_id
    # # boreal_east
    # run_id = '1810ee63cf3544f7beffb7e465f15536'
    # # boreal_south
    # run_id = '1874024118f14411aad13ed9b5b120a3'
    # # boreal_northwest
    # run_id = '4e3f21d8bcbe4c5b8f11111630b91695'
    # # alpine
    # run_id = '7779d3583c8a418f93e798e058c3993f'
    # # continental
    # run_id = '94b233e5b1d541cbba7a980f74ad71de'
    # # boreal south regular loss batchsize 50
    # run_id = '91655c64c5b54a66b1944a8e541d59a3'
    # # boreal south regular loss batchsize 5
    # run_id = 'a3277f8551b34ba79d54d38e7e1e37af'
    # # regular loss function
    # run_id = '4fdd6eda52a1489faf88eb413a59ee48'

    # final regional models, opposite weighted loss
    # alpine batchsize 5
    run_id = '30a2fb4fe17d4473afa37d924f71ad71'
    # continental batchsize 5
    run_id = '94b233e5b1d541cbba7a980f74ad71de'
    # boreal northwest batchsize 5
    run_id = '4b9420f09bd44d7bb86bda21ae25e3da'
    # boreal east batchsize 5
    run_id = '8976be82eec04296b88340372b5b1070'
    # boreal south batchsize 5
    run_id = 'd2ea3db2c2f04c09b99718baf55d1326'
    # # boreal all batchsize 5
    # run_id = '77f0751be2694bcabb7e0ee74cfbaf06'
    # sweden all batchsize 5
    run_id = 'ab55f164d4d74f7ba00c66e28c9fc094'


    # final regional models, regular loss
    # alpine, batchsize 5, regular loss
    run_id = '7f7346499de74de59a3a4e74bfc6b364'
    # continental, batchsize 5, regular loss
    run_id = 'f074c4a9a6df444d81faf27adae75024'
    # boreal_northwest, batchsize 5, regular loss
    run_id = 'ec16f265612c44a5acb3dacab1bb2164'
    # boreal_east, batchsize 5, regular loss
    run_id = 'c130b16f0a41416194480c707f0688e1'
    # boreal south regular loss batchsize 5
    run_id = 'a3277f8551b34ba79d54d38e7e1e37af'


    # modeltesting
    # 100,50,100, batchsize 5
    run_id = 'eb09ad9072804a1cbf62758945099865'
    # 100,50,100, batchsize 10
    run_id = '9535fc16ce244719a553498b92cfe6f3'
    # 100,50,100, batchsize 25
    run_id = '83c527ca51224eaca98513b48005b3b6'
    # 100,50,100, batchsize 50
    run_id = '106112d8ea2c4a1aabc7b2f382bf6c44'

    # 75,75,75, batchsize 15
    run_id = 'e16d78e286d744e19312b372cb89e8ae'
    # 50,100,50, batchsize 15
    run_id = '0d6c5c103fcc4ad287e85d2d5e02323a'
    # 100,50,100, batchsize 15
    run_id = 'abe2df7cef8f4a7ab6cc09f5c7472148'

    # 15,15,15, batchsize 15
    run_id = '8a5108b90d0348f8ba8b9702ec3f6f28'
    # 10,20,10, batchsize 15
    run_id = '968b0c7f7e8a47eea66f9a4b24d74a48'
    # 20,10,20, batchsize 15
    run_id = '9eaba72661824ab78327d175876f5e16'

    # 25,25,25,25,25, batchsize 15
    run_id = '234b482cdc184568938073b466fd159a'
    # 10,20,40,20,10, batchsize 15
    run_id = '625b8246bb0e4d19add1c5d8c54d6729'
    # 40,20,10,20,40, batchsize 15
    run_id = 'b8272fdf53834b8da70ad625c52344f6'

    run = client.get_run(run_id)  # Retrieve the run from MLflow
    print(run.info.status)

if training_history:
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

else:
    # #Specify the experiment ID or name
    # experiment_id = '10'#'450601850519144894' #'10'
    # #Query all runs from the experiment
    # runs = mlflow.search_runs([experiment_id])
    # filtered_runs = runs
    if model_testing:
        modeltesting_weighted_loss_new = [
            'eb09ad9072804a1cbf62758945099865',
            '9535fc16ce244719a553498b92cfe6f3',
            '83c527ca51224eaca98513b48005b3b6',
            '106112d8ea2c4a1aabc7b2f382bf6c44',
            'e16d78e286d744e19312b372cb89e8ae',
            '0d6c5c103fcc4ad287e85d2d5e02323a',
            'abe2df7cef8f4a7ab6cc09f5c7472148',
            '8a5108b90d0348f8ba8b9702ec3f6f28',
            '968b0c7f7e8a47eea66f9a4b24d74a48',
            '9eaba72661824ab78327d175876f5e16',
            '234b482cdc184568938073b466fd159a',
            '625b8246bb0e4d19add1c5d8c54d6729',
            'b8272fdf53834b8da70ad625c52344f6'
            ]
        modeltesting_regular_loss =[
            'a3277f8551b34ba79d54d38e7e1e37af',
            '4fdd6eda52a1489faf88eb413a59ee48',
            '91655c64c5b54a66b1944a8e541d59a3'
        ]
        modeltesting_weighted_loss_old = [
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
            '3b0e76bbe9004e2f9c5682215feccb86'
            ]
        target_runs = modeltesting_weighted_loss_new + modeltesting_regular_loss + modeltesting_weighted_loss_old
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
        output_df.to_csv('results/modeltesting/modeltestscores_validation_set.txt', sep='\t', index=False)
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
        plt.xticks(rotation='vertical')
        # Create legend & Show graphic
        #plt.legend()
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.ylim(0, 1)  # Scale y-axis from 0 to 1

        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)

        # Save the plot to a PDF file
        plt.savefig('results/modeltesting/grouped_metrics_plot_validation_set.pdf', bbox_inches='tight')

        plt.show()

        # Identify float columns in the DataFrame
        float_cols = output_df.select_dtypes(include=['float64']).columns
        # Round only the float columns to 3 decimal places
        output_df[float_cols] = output_df[float_cols].round(3)
        output_df.values
        # write to file
        output_df.to_csv('results/regional_model_scores_validation_set.txt',sep='\t',index=False)

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




