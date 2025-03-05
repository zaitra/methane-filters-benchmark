import os
import numpy as np
import tifffile as tiff
import pandas as pd
from sklearn.metrics import average_precision_score

# Define paths
output_data_path = "data"
filters = ["mf", "ace", "cem", "mag1c"]  # Filter outputs to process
label_name = "label.npy"
auprc_scores = {}

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

    # Compute AUPRC for the current filter
    auprc_scores[f] = average_precision_score(ground_truths, filter_results)
    print(f"AUPRC for {f}: {auprc_scores[f]:.4f}")

    # Free memory
    del ground_truths, filter_results

# Save AUPRC scores to CSV
auprc_df = pd.DataFrame.from_dict(auprc_scores, orient="index", columns=["AUPRC"])
auprc_df.to_csv(os.path.join(output_data_path, "auprc_scores.csv"))

print("Processing complete. AUPRC scores saved.")