import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import pandas as pd
import math

client = MlflowClient()
# Set MLflow Tracking URI if necessary
# mlflow.set_tracking_uri('your_tracking_uri')

# Specify the experiment ID or name
experiment_id = '10'#'450601850519144894' #'10'

# Query all runs from the experiment
runs = mlflow.search_runs([experiment_id])
#filtered_runs = runs

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
    '3b0e76bbe9004e2f9c5682215feccb86'
    ]
filtered_runs = pd.concat([runs[runs['run_id'] == run_id] for run_id in target_runs if run_id in runs['run_id'].values], ignore_index=True)
#filtered_runs = runs[runs['run_id'].isin(target_runs)]
filtered_runs.columns

output_df = filtered_runs[['params.n_channels_per_layer',
                           'params.batch_size',
                           'metrics.val_binary_accuracy',
                           'metrics.val_binary_F1',
                           'metrics.val_binary_precision',
                           'metrics.val_binary_recall',
                           'metrics.val_loss',
                           'metrics.train_loss']]
# Identify float columns in the DataFrame
float_cols = output_df.select_dtypes(include=['float64']).columns
# Round only the float columns to 3 decimal places
output_df[float_cols] = output_df[float_cols].round(3)
output_df.values
# write to file
output_df.to_csv('results/modeltesting/modeltestscores.txt',sep='\t',index=False)


# Also select the training history of selected runs
selected_metric = 'train_loss'

for run_id in filtered_runs.run_id:
    metric_history = client.get_metric_history(run_id, selected_metric)
    # Extracting epoch numbers and corresponding values
    epochs = [metric.step for metric in metric_history]
    values = [metric.value for metric in metric_history]
    filtered_epochs, filtered_values = zip(*[(epoch, value) for epoch, value in zip(epochs, values) if not math.isnan(value)])
    plt.plot(filtered_epochs,filtered_values)



data_to_export = filtered_runs[['run_id',selected_metric]]

# Filter and reformat the data as needed
# For example, you might want to include specific metrics or parameters
data_to_export = filtered_runs[['run_id', 'metrics.your_metric_name', 'params.your_param_name']]

# Export to CSV
data_to_export.to_csv('exported_mlflow_data.csv', index=False)
