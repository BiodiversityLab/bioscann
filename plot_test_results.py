import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc, precision_recall_curve


test_stat_file = 'results/test_stats/boreal_south_testset.csv'
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


# Calculating Precision-Recall values
precision, recall, thresholds = precision_recall_curve(df['True Positives'].values + df['False Negatives'].values, df['True Positives'].values / (df['True Positives'].values + df['False Negatives'].values))

# Calculate AUC for Precision-Recall Curve
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
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

