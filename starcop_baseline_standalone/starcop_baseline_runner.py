import os, glob

import numpy
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import pylab as plt
from tqdm import tqdm
import io
import sys

from baseline import Mag1cBaseline

# Download the dataset from https://zenodo.org/records/7863343 or https://huggingface.co/datasets/previtus/STARCOP_allbands_Eval
#dataset_root = "/home/jherec/methane-filters-benchmark/data/WHOLE_IMAGE_STARCOP-MAG1C_SPED_UP_1573-2481_PRECISION-64"
csv_file = "/home/jherec/starcop_big/STARCOP_allbands/train.csv"

### CHANGE THIS: ###############################################################################################
#product = "cem.tif" # Which tif file is loaded

# Normalisation parameters
#normalizer_params = {'offset': 0, 'factor': 1750, 'clip': (0, 2)}
normalizer_params = {'offset': 0, 'factor': 1, 'clip': (0, 1)}
# This uses the following formula:
#   clamp( (input_product - offset) / factor, clip_min, clip_max)
#   ( so for this mag1c settings it divides the values by 1750 and then clips between <0,2>

#threshold = 0.0038 # Which threshold is applied to it 0.0035 - CEM (probably MF also)
#baseline_model = Mag1cBaseline(mag1c_threshold = threshold, normalizer_params = normalizer_params)

################################################################################################################
def main(dataset_root, product_threshold):
    product, threshold = product_threshold
    if "mag1c" in product:
        normalizer_params = {'offset': 0, 'factor': 1750, 'clip': (0, 2)}
    else:
        normalizer_params = {'offset': 0, 'factor': 1, 'clip': (0, 1)}
    baseline_model = Mag1cBaseline(mag1c_threshold = threshold, normalizer_params = normalizer_params)
    dataset_v = dataset_root.split("/")[-1]
    print(threshold, product, dataset_v)
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
    for idx, item in tqdm(df.iterrows(), total=len(df)):
        #print("processing", item["id"])

        # Load products:
        mf_path = os.path.join(dataset_root, item["id"], product)
        y_path = os.path.join("/home/jherec/starcop_big/STARCOP_allbands", item["id"], "labelbinary.tif")
        mask_path = os.path.join(dataset_root, item["id"], "valid_mask.tif") # used to get the validity mask
        with rio.open(mf_path) as src:
            mf_data = src.read()
        with rio.open(y_path) as src:
            y_data = src.read()
        with rio.open(mask_path) as src:
            mask_data = src.read()
            mask_data = np.where(mask_data == 0, 0, 1)

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
    dataset_v_splitted = dataset_v.split("_")

    whole_image, starcop_mag1c, sped_up, bit_depth_precision, wv_range, channel_n, _, _ = dataset_v_splitted
    whole_image = True if "WHOLE" in whole_image.upper() else False
    starcop_mag1c = True if "STARCOP" in starcop_mag1c.upper() else False
    sped_up = True if "SPED" in sped_up else False
    bit_depth_precision = f"float{bit_depth_precision.replace("PRECISION-","")}"
    channel_n = int(channel_n.replace("CHANNEL-N-", ""))
    metrics_dict = {
        "METHOD": product.replace(".tif", "").upper(),
        "THRESHOLD": threshold,
        "WHOLE_IMAGE": whole_image,
        "STARCOP_MAG1C": starcop_mag1c,
        "SPED_UP": sped_up,
        "PRECISION": bit_depth_precision,
        "WAVELENGTH_RANGE":wv_range,
        "CHANNEL_N": channel_n
        }

    def round_to(n, digits=3):
        if np.isnan(n): return n
        m = pow(10,digits)
        return str(int(100 * n * m) / m)

    def metric_prec_recall_f1(ground_truths, P_thresholded):
        cm = confusion_matrix(ground_truths.flatten(), P_thresholded.flatten())

        tn, fp, fn, tp = cm.ravel()
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1 = 2*(recall * precision) / (recall + precision)

        iou = tp / (tp + fp + fn)
        print("Recall", round_to(recall)+", Precision", round_to(precision)+", F1", round_to(f1))
        print("IoU", round_to(iou))

        return tn, fp, fn, tp, recall, precision, f1, iou

    def tile_FPR(ground_truths, P_thresholded):
        """ FP / (FP + TN)"""
        cm = confusion_matrix(ground_truths.flatten(), P_thresholded.flatten())
        tn, fp, fn, tp = cm.ravel()
        fpr_for_tiles = fp / (fp + tn)
        print("FPR (tile)", round_to(fpr_for_tiles))
        return tn, fp, fn, tp, fpr_for_tiles

    def add_metrics_to_metric_dict(suffix, metrics, metrics_dict):
        if len(metrics) > 5:
            names = ["TN", "FP", "FN", "TP", "Recall", "Precision", "F1-score", "Iou"]
            names = [f"{n}_{suffix}_seg" for n in names]
        else:
            names = ["TN", "FP", "FN", "TP", "FPR"]
            names = [f"{n}_{suffix}_clas" for n in names]
        names_metrics = dict(zip(names,metrics))
        metrics_dict = metrics_dict | names_metrics
        return metrics_dict



    print("All:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels.flatten(), predictions.flatten())
    metrics_dict = add_metrics_to_metric_dict("all_", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    tn, fp, fn, tp, fpr_for_tiles = tile_FPR(labels_classification.flatten(), predictions_classification.flatten())
    metrics_dict = add_metrics_to_metric_dict("all_tile_", [tn, fp, fn, tp, fpr_for_tiles], metrics_dict)
    print("Strong:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels_strong.flatten(), predictions_strong.flatten())
    metrics_dict = add_metrics_to_metric_dict("strong_", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    print("Weak:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels_weak.flatten(), predictions_weak.flatten())
    metrics_dict = add_metrics_to_metric_dict("weak_", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    return metrics_dict

if __name__ == "__main__":
    DEBUG = True
    all_metrics = []
    if DEBUG:
        dataset_roots = ["/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_STARCOP-MAG1C_SPED-UP_PRECISION-64_2122-2488_CHANNEL-N-72_EXPORT_VERSION"]
        products_threshold = [("cem.tif", 0.004)]
    else:
        dataset_roots = [
            "/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_STARCOP-MAG1C_SPED-UP_PRECISION-64_1573-2481_CHANNEL-N-122",
            "/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_STARCOP-MAG1C_SPED-UP_PRECISION-64_2122-2488_CHANNEL-N-72",
            ]
        products_threshold = []
        for x in ["cem.tif", "mf.tif"]:
            for i in [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006]:
                products_threshold.append((x,i))
        for x in ["ace.tif"]:
            for i in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
                products_threshold.append((x,i))
        for x in ["mag1c.tif"]:
            for i in [500]:
                products_threshold.append((x,i))
    
    
    for x in dataset_roots:
        for i in products_threshold:
            all_metrics.append(main(x,i))

    # Convert list of dicts to DataFrame and save as CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv("metrics.csv", index=False)

    print("CSV file saved successfully!")

        
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