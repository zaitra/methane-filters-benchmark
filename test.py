import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
def plot_curve(ax, x, y, thresholds, x_label, y_label, title, metric_label, breakpoints):
    """Generic function to plot AUPRC or ROC AUC with threshold annotations."""
    
    ax.plot(x, y, marker='o', linestyle='-', label=metric_label)

    # Find closest indices for annotation
    indices = [min(np.argmin(np.abs(x - b)), len(thresholds) - 1) for b in breakpoints]

    # Annotate selected thresholds
    for i in indices:
        ax.text(x[i], y[i], f'{thresholds[i]:.4f}', fontsize=8, ha='right')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid()



# Define paths
note = "BY_COLUMNS_STARCOP-MAG1C"
output_data_path = os.path.join("data", note)
output_results_path = os.path.join("outputs", note)
os.makedirs(output_results_path, exist_ok=True)
filters = ["mf", "ace", "cem", "mag1c"]  # Filter outputs to process
label_name = "label.npy"
auprc_scores = {}
roc_scores = {}

# Get all tile directories
tile_dirs = sorted([d for d in os.listdir(output_data_path) if os.path.isdir(os.path.join(output_data_path, d))])

# Process each filter separately to avoid high memory usage
for f in filters:
    print(f"Processing filter: {f}")
    
    # Initialize lists for the current filter
    ground_truths = []
    filter_results = []

    for tile in tile_dirs:
        tile_folder = os.path.join(output_data_path, tile)

        # Load ground truth label
        label_path = os.path.join(tile_folder, label_name)
        if not os.path.exists(label_path):
            print(f"Skipping {tile}, label file missing.")
            continue
        
        label = np.load(label_path).flatten()
        ground_truths.append(label)

        # Load the filter output
        filter_path = os.path.join(tile_folder, f"{f}.npy")
        if os.path.exists(filter_path):
            filter_results.append(np.load(filter_path).flatten())
        else:
            print(f"Skipping {tile}, {f} filter file missing.")
            continue

    if not filter_results:
        print(f"No valid data for {f}, skipping AUPRC calculation.")
        continue

    # Convert lists to numpy arrays
    ground_truths = np.concatenate(ground_truths)  # Full ground truth mask
    filter_results = np.concatenate(filter_results)  # Full detections

    # Compute Precision-Recall curve and AUPRC
    precision, recall, thresholds_auprc = precision_recall_curve(ground_truths, filter_results, drop_intermediate=True)
    auprc = auc(recall, precision)
    auprc_scores[f] = auprc
    print(f"AUPRC for {f}: {auprc:.4f}")

    fpr, tpr, thresholds_roc = roc_curve(ground_truths, filter_results, drop_intermediate=True)
    roc_auc = auc(fpr, tpr)
    roc_scores[f] = roc_auc
    print(f"ROC AUC for {f}: {roc_auc:.4f}")
    
    # Free memory
    del ground_truths, filter_results
        # Define breakpoints for annotation (evenly spaced recall/FPR values)
    breakpoints = np.linspace(0, 1, 11)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Plot AUPRC
    plot_curve(
        axes[0], recall, precision, thresholds_auprc,
        "Recall", "Precision", f"Precision-Recall Curve for {f}",
        f'{f} (AUPRC = {auprc:.4f})', breakpoints
    )

    # Plot ROC AUC
    plot_curve(
        axes[1], fpr, tpr, thresholds_roc,
        "False Positive Rate", "True Positive Rate", f"ROC Curve for {f}",
        f'{f} (ROC AUC = {roc_auc:.4f})', breakpoints
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_results_path,f"{f}.png"))
    

# Convert AUPRC and ROC-AUC scores to DataFrames
auprc_df = pd.DataFrame.from_dict(auprc_scores, orient="index", columns=["AUPRC"])
roc_df = pd.DataFrame.from_dict(roc_scores, orient="index", columns=["ROC-AUC"])

# Merge both DataFrames on index (filter name)
combined_df = pd.concat([auprc_df, roc_df], axis=1)

# Save the combined DataFrame to CSV
combined_df.to_csv(os.path.join(output_results_path, f"scores.csv"))