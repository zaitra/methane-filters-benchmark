import os, glob

import numpy
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import pylab as plt

from baseline import Mag1cBaseline

# Download the dataset from https://zenodo.org/records/7863343 or https://huggingface.co/datasets/previtus/STARCOP_allbands_Eval
dataset_root = "/home/jherec/methane-filters-benchmark/data/WHOLE_IMAGE_STARCOP-MAG1C"
csv_file = "/home/jherec/starcop_big/STARCOP_allbands/test.csv"

### CHANGE THIS: ###############################################################################################
product = "mag1c.npy" # Which tif file is loaded

# Normalisation parameters
normalizer_params = {'offset': 0, 'factor': 1, 'clip': (0, 1)}
# This uses the following formula:
#   clamp( (input_product - offset) / factor, clip_min, clip_max)
#   ( so for this mag1c settings it divides the values by 1750 and then clips between <0,2>

threshold = 300 # Which threshold is applied to it ace 0.035
baseline_model = Mag1cBaseline(mag1c_threshold = threshold, normalizer_params = normalizer_params)
################################################################################################################

show = False
sort_by_plume_size = False
# for debug might be useful:
# show = True
# sort_by_plume_size = True

df = pd.read_csv(os.path.join(dataset_root, csv_file))
# optionally sort so that we get large plumes first (for visualisation)
if sort_by_plume_size:
    df = df.sort_values(["has_plume", "qplume"], ascending=False)


# Constants
EASY_HARD_THRESHOLD = 1000  # This is used to differentiate between weak / stong events
# When I doublechecked the code, what we do is not splitting by the qplume value
# ... instead we consider a plume "weak" (/"strong") if the binary mask we made has less (/more) than 1000 pixels
NUMBER_OF_PIXELS_PRED = 10  # ~ value we used in Starcop

predictions_weak = []
labels_weak = []
predictions_strong = []
labels_strong = []
predictions_noplume = []
labels_noplume = []

predictions_classification = []
labels_classification = []

# i = 0
invalid = 0
for idx, item in df.iterrows():
    print("processing", item["id"])

    # Load products:
    mf_path = os.path.join(dataset_root, item["id"], product)
    y_path = os.path.join(dataset_root, item["id"], "label.npy")
    mask_path = os.path.join(dataset_root, item["id"], "mf.npy") # used to get the validity mask
    try:
        mf_data = np.load(mf_path)
        mf_data = np.expand_dims(mf_data, axis=0)
    except FileNotFoundError:
        invalid +=1
        continue
    y_data = np.load(y_path)
    y_data = np.expand_dims(y_data, axis=0)
    mask_data = np.load(mask_path)
    # Identify columns where all values are zero
    col_mask = (mask_data == 0).all(axis=0)  # True if entire column is zero

    # Create the mask: 1 if any value is nonzero, else 0
    mask = np.where(col_mask, 0, 1)

    # Expand mask to match original shape
    mask_data = np.tile(mask, (mask_data.shape[0], 1))
    mask_data = np.expand_dims(mask_data, axis=0)
    """with rio.open(mf_path) as src:
        mf_data = src.read()
    with rio.open(y_path) as src:
        y_data = src.read()
    with rio.open(mask_path) as src:
        mask_data = src.read()
        mask_data = np.where(mask_data == 0, 0, 1)"""

    # Determine easy / hard split
    label_pixels_plume = np.sum(y_data)
    tile_has_plume = label_pixels_plume > 0
    difficulty = "easy" if label_pixels_plume > EASY_HARD_THRESHOLD else "hard"

    event_type = "noplume"
    if tile_has_plume and difficulty == "easy":
        event_type = "strong"
    if tile_has_plume and difficulty == "hard":
        event_type = "weak"

    # Use the pytorch lightning module
    batch = {}
    batch["input"] = torch.tensor(mf_data).unsqueeze(0)
    batch["output"] = torch.tensor(y_data).unsqueeze(0)

    batch = baseline_model.batch_with_preds(batch)

    if show:
        path = os.path.join("/home/jherec/methane-filters-benchmark/outputs/trash", item["id"])
        plt.imshow(batch["prediction"][0][0]) # < the normalised product
        plt.title("Product")
        plt.savefig(path + "product.png")
        plt.imshow(batch["pred_binary"][0][0]) # < the prediction after thr and morpho
        plt.title("Product after thr and morpho.")
        plt.savefig(path + "product_after.png")
        plt.imshow(batch["output"][0][0]) # < label
        plt.title("Label")
        plt.savefig(path + "label.png")
        plt.imshow(mask_data[0])  # < mask
        plt.title("Valid mask")
        plt.savefig(path + "mask.png")

    gt = y_data[0]
    pred = numpy.asarray(batch["pred_binary"][0][0])

    # Semantic segmentation result
    # Masking
    mask = mask_data[0].flatten()
    gt = gt.flatten()
    pred = pred.flatten()

    gt_masked = []
    pred_masked = []
    for px_idx, px_mask in enumerate(mask):
        # not efficient, I know
        if px_mask == 1: # if valid pixel
            gt_masked.append(gt[px_idx])
            pred_masked.append(pred[px_idx])

    if event_type == "noplume":
        labels_noplume += gt_masked
        predictions_noplume += pred_masked
    elif event_type == "weak":
        labels_weak += gt_masked
        predictions_weak += pred_masked
    elif event_type == "strong":
        labels_strong += gt_masked
        predictions_strong += pred_masked

    # Tile classification result
    # if we predict more than 10 pixels, then we assign the tile the predition that there is a plume
    tile_pred_has_plume = np.sum(pred) > NUMBER_OF_PIXELS_PRED
    predictions_classification.append(int(tile_pred_has_plume))
    labels_classification.append(int(tile_has_plume))

predictions = np.asarray(predictions_strong + predictions_weak + predictions_noplume)
labels = np.asarray(labels_strong + labels_weak + labels_noplume)
predictions_strong = np.asarray(predictions_strong)
labels_strong = np.asarray(labels_strong)
predictions_weak = np.asarray(predictions_weak)
labels_weak = np.asarray(labels_weak)

predictions_classification = np.asarray(predictions_classification)
labels_classification = np.asarray(labels_classification)

# Scores:
from sklearn.metrics import confusion_matrix

def round_to(n, digits=3):
    if np.isnan(n): return n
    m = pow(10,digits)
    return str(int(100 * n * m) / m)

def metric_prec_recall_f1(ground_truths, P_thresholded):
    cm = confusion_matrix(ground_truths.flatten(), P_thresholded.flatten())

    tn, fp, fn, tp = cm.ravel()
    tp = max(tp, 0.0000000000001)
    fn = max(fn, 0.0000000000001)
    fp = max(fp, 0.0000000000001)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*(recall * precision) / (recall + precision)

    iou = tp / (tp + fp + fn)
    print("Recall", round_to(recall)+", Precision", round_to(precision)+", F1", round_to(f1))
    print("IoU", round_to(iou))

    return recall, precision, f1

def tile_FPR(ground_truths, P_thresholded):
    """ FP / (FP + TN)"""
    cm = confusion_matrix(ground_truths.flatten(), P_thresholded.flatten())
    tn, fp, fn, tp = cm.ravel()
    fpr_for_tiles = fp / (fp + tn)
    print("FPR (tile)", round_to(fpr_for_tiles))
    return fpr_for_tiles
print(product)
print(threshold)
print("All:")
metric_prec_recall_f1(labels.flatten(), predictions.flatten())
tile_FPR(labels_classification.flatten(), predictions_classification.flatten())

print("Strong:")
metric_prec_recall_f1(labels_strong.flatten(), predictions_strong.flatten())
print("Weak:")
metric_prec_recall_f1(labels_weak.flatten(), predictions_weak.flatten())

print("Invalid:")
print(invalid)

"""
I am getting:

All:
Recall 58.422, Precision 30.574, F1 40.141
IoU 25.11
FPR (tile) 75.428

Strong:
Recall 59.844, Precision 77.41, F1 67.503
IoU 50.947

Weak:
Recall 52.76, Precision 32.149, F1 39.953
IoU 24.963

Which matches what I am reporting in my thesis:
                 AUPRC ↑ F1 ↑  (strong) (weak)  Prec. ↑ Rec. ↑ IoU ↑  FPR ↓
MF thr. baseline N/A     40.14   67.50   39.95  30.57   58.42  25.11  75.43

"""
"""
CEM - 0.007
All:
Recall 26.674, Precision 29.682, F1 28.098
IoU 16.345
FPR (tile) 88.571
Strong:
Recall 23.019, Precision 74.996, F1 35.226
IoU 21.378
Weak:
Recall 35.41, Precision 31.156, F1 33.147
IoU 19.866
Invalid:
5
CEM - 0.005
All:
Recall 38.734, Precision 15.668, F1 22.311
IoU 12.556
FPR (tile) 98.285
Strong:
Recall 34.231, Precision 54.729, F1 42.118
IoU 26.677
Weak:
Recall 49.495, Precision 15.684, F1 23.82
IoU 13.52
Invalid:
5
CEM - 0.004
All:
Recall 46.65, Precision 9.666, F1 16.014
IoU 8.703
FPR (tile) 99.428
Strong:
Recall 41.525, Precision 41.006, F1 41.264
IoU 25.995
Weak:
Recall 58.898, Precision 9.353, F1 16.143
IoU 8.78
Invalid:
5
Mag1c whole image - 500
All:
Recall 45.595, Precision 31.623, F1 37.345
IoU 22.96
FPR (tile) 95.428
Strong:
Recall 45.568, Precision 77.94, F1 57.512
IoU 40.362
Weak:
Recall 45.659, Precision 28.443, F1 35.051
IoU 21.25
Invalid:
5
mag1c whole image- 300
All:
Recall 54.519, Precision 19.993, F1 29.257
IoU 17.135
FPR (tile) 98.857
Strong:
Recall 53.134, Precision 67.052, F1 59.287
IoU 42.134
Weak:
Recall 57.829, Precision 18.04, F1 27.501
IoU 15.943
Invalid:
5
ace.npy
0.05
All:
Recall 23.657, Precision 29.03, F1 26.069
IoU 14.988
FPR (tile) 97.714
Strong:
Recall 19.935, Precision 82.243, F1 32.092
IoU 19.113
Weak:
Recall 32.549, Precision 31.799, F1 32.17
IoU 19.168
Invalid:
5
ace.npy
0.04
All:
Recall 28.147, Precision 19.487, F1 23.029
IoU 13.013
FPR (tile) 99.428
Strong:
Recall 23.917, Precision 70.739, F1 35.748
IoU 21.764
Weak:
Recall 38.254, Precision 21.164, F1 27.251
IoU 15.775
Invalid:
5
ace.npy
0.03
All:
Recall 34.171, Precision 10.526, F1 16.095
IoU 8.751
FPR (tile) 99.428
Strong:
Recall 29.463, Precision 52.48, F1 37.739
IoU 23.258
Weak:
Recall 45.42, Precision 11.261, F1 18.048
IoU 9.919
Invalid:
5
mf.npy
0.007
All:
Recall 26.427, Precision 29.367, F1 27.82
IoU 16.157
FPR (tile) 90.285
Strong:
Recall 22.719, Precision 75.171, F1 34.893
IoU 21.133
Weak:
Recall 35.289, Precision 31.104, F1 33.065
IoU 19.807
Invalid:
5
mf.npy
0.005
All:
Recall 38.504, Precision 15.725, F1 22.33
IoU 12.568
FPR (tile) 97.142
Strong:
Recall 33.891, Precision 54.525, F1 41.801
IoU 26.423
Weak:
Recall 49.527, Precision 15.827, F1 23.989
IoU 13.629
Invalid:
5
mf.npy
0.004
All:
Recall 46.541, Precision 9.787, F1 16.174
IoU 8.798
FPR (tile) 99.428
Strong:
Recall 41.363, Precision 41.209, F1 41.286
IoU 26.012
Weak:
Recall 58.917, Precision 9.506, F1 16.371
IoU 8.915
Invalid:
5
"""