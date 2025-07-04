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
MODEL = False
# Download the dataset from https://zenodo.org/records/7863343 or https://huggingface.co/datasets/previtus/STARCOP_allbands_Eval
#dataset_root = "/home/jherec/methane-filters-benchmark/data/WHOLE_IMAGE_STARCOP-MAG1C_SPED_UP_1573-2481_PRECISION-64"
csv_file = "/mnt/nfs/starcop_big/STARCOP_allbands/test.csv"
output_csv = "final_data_matrics.csv"
if not os.path.exists(output_csv):
    # Create an empty DataFrame
    empty_df = pd.DataFrame(columns=["col1", "col2", "col3"])
    # Save it to CSV
    empty_df.to_csv(output_csv, index=False)

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
    show = False
    sort_by_plume_size = False
    # for debug might be useful:
    # show = True
    # sort_by_plume_size = True


    df = pd.read_csv(csv_file)
    # optionally sort so that we get large plumes first (for visualisation)
    if sort_by_plume_size:
        df = df.sort_values(["has_plume", "qplume"], ascending=False)

    check_df = pd.read_csv(output_csv)
    dataset_v_splitted = dataset_v.split("_")
    whole_image, mag1c, sped_up, bit_depth_precision, wv_range, select_strategy, channel_n,= dataset_v_splitted
    whole_image = True if "WHOLE" in whole_image.upper() else False
    sampled_percentage = 1 if "SAMPLED" not in mag1c.upper() else mag1c.split("-")[-1]
    mag1c = "STARCOP" if "STARCOP" in mag1c.upper() else "GENERATED"
    sped_up = True if "SPED" in sped_up else False
    bit_depth_precision = f"float{bit_depth_precision.replace("PRECISION-","")}"
    channel_n = int(channel_n.replace("CHANNEL-N-", ""))
    metrics_dict = {
        "METHOD": product.replace(".tif", "").upper(),
        "THRESHOLD": threshold,
        "WHOLE_IMAGE": whole_image,
        "MAG1C": mag1c if "mag1c" in product.lower() else "-",
        "SAMPLED": sampled_percentage if "mag1c" in product.lower() else "-",
        "SPED_UP": sped_up,
        "PRECISION": bit_depth_precision,
        "WAVELENGTH_RANGE":wv_range,
        "CHANNEL_N": channel_n,
        "SELECT_STRATEGY": select_strategy
        }
    
    # Define the columns to check (same as keys in metrics_dict)
    columns_to_check = list(metrics_dict.keys())

    # Find matching row(s)
    if not check_df.empty:
        matching_rows = check_df[(check_df[columns_to_check] == pd.Series(metrics_dict)).all(axis=1)]

        # If a match is found, extract the first row as a dictionary
        if not matching_rows.empty:
            existing_entry = matching_rows.iloc[0].to_dict()  # Convert first matching row to dictionary
            print("Matching entry found:", existing_entry)
            return existing_entry

    # Constants
    EASY_HARD_THRESHOLD = 1000  # This is used to differentiate between weak / stong events
    # When I doublechecked the code, what we do is not splitting by the qplume value
    # ... instead we consider a plume "weak" (/"strong") if the binary mask we made has less (/more) than 1000 pixels
    NUMBER_OF_PIXELS_PRED = 10  # ~ value we used in Starcop
    predictions_weak_original = []
    predictions_weak = []
    labels_weak = []
    predictions_strong_original = []
    predictions_strong = []
    labels_strong = []
    predictions_noplume_original = []
    predictions_noplume = []
    labels_noplume = []

    predictions_classification = []
    labels_classification = []

    # i = 0
    for idx, item in tqdm(df.iterrows(), total=len(df)):
        #print("processing", item["id"])

        # Load products:
        mf_path = os.path.join(dataset_root, item["id"], product)
        y_path = os.path.join("/mnt/nfs/starcop_big/STARCOP_allbands", item["id"], "labelbinary.tif")
        mask_path = os.path.join("/mnt/nfs/starcop_big/STARCOP_allbands", item["id"], "TOA_AVIRIS_460nm.tif") # used to get the validity mask
        with rio.open(mf_path) as src:
            mf_data = src.read()
            if MODEL:
                mf_data = np.clip(mf_data.astype(np.float32)/3500, 0, 2)
        with rio.open(y_path) as src:
            y_data = src.read()
        with rio.open(mask_path) as src:
            mask_data = src.read()
            mask_data = np.where(mask_data == 0, 0, 1)
        if MODEL:
            with rio.open(os.path.join("/mnt/nfs/starcop_big/STARCOP_allbands", item["id"], "TOA_AVIRIS_460nm.tif")) as src:
                b = src.read()
                b = np.clip(b.astype(np.float32)/60, 0, 2)
            with rio.open(os.path.join("/mnt/nfs/starcop_big/STARCOP_allbands", item["id"], "TOA_AVIRIS_550nm.tif")) as src:
                g = src.read()
                g = np.clip(g.astype(np.float32)/60, 0, 2)
            with rio.open(os.path.join("/mnt/nfs/starcop_big/STARCOP_allbands", item["id"], "TOA_AVIRIS_640nm.tif")) as src:
                r = src.read()
                r = np.clip(r.astype(np.float32)/60, 0, 2)
        
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
        batch["output"] = torch.tensor(y_data).unsqueeze(0)
        if MODEL:
            batch["input"] = torch.tensor(np.concatenate([mf_data,r,g,b], axis=0)).unsqueeze(0)
            batch = baseline_model.batch_with_preds_model(batch)
        else:
            batch["input"] = torch.tensor(mf_data).unsqueeze(0)
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
        original_data = mf_data.flatten()

        gt_masked = []
        pred_masked = []
        original_masked = []
        for px_idx, px_mask in enumerate(mask):
            # not efficient, I know
            if px_mask == 1: # if valid pixel
                gt_masked.append(gt[px_idx])
                pred_masked.append(pred[px_idx])
                original_masked.append(original_data[px_idx])

        if event_type == "noplume":
            labels_noplume += gt_masked
            predictions_noplume += pred_masked
            predictions_noplume_original += original_masked
        elif event_type == "weak":
            labels_weak += gt_masked
            predictions_weak += pred_masked
            predictions_weak_original += original_masked
        elif event_type == "strong":
            labels_strong += gt_masked
            predictions_strong += pred_masked
            predictions_strong_original += original_masked

        # Tile classification result
        # if we predict more than 10 pixels, then we assign the tile the predition that there is a plume
        tile_pred_has_plume = np.sum(pred) > NUMBER_OF_PIXELS_PRED
        predictions_classification.append(int(tile_pred_has_plume))
        labels_classification.append(int(tile_has_plume))

    predictions = np.asarray(predictions_strong + predictions_weak + predictions_noplume)
    predictions_original = np.asarray(predictions_strong_original + predictions_weak_original + predictions_noplume_original)
    labels = np.asarray(labels_strong + labels_weak + labels_noplume)

    predictions_strong = np.asarray(predictions_strong)
    predictions_strong_original = np.asarray(predictions_strong_original)
    labels_strong = np.asarray(labels_strong)

    predictions_weak = np.asarray(predictions_weak)
    predictions_weak_original = np.asarray(predictions_weak_original)
    labels_weak = np.asarray(labels_weak)

    predictions_classification = np.asarray(predictions_classification)
    labels_classification = np.asarray(labels_classification)

    # Scores:
    from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
    suffixes = ["all", "strong", "weak"]
    for idx,labels_predictions in enumerate(zip([labels, labels_strong, labels_weak],[predictions_original, predictions_strong_original, predictions_weak_original])):
        precision, recall, thresholds_auprc = precision_recall_curve(labels_predictions[0], labels_predictions[1], drop_intermediate=True)
        auprc = auc(recall, precision)
        metrics_dict[f"AUPRC_{suffixes[idx]}"] = auprc
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
        # Find the index of the highest F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_auprc[best_idx] if best_idx < len(thresholds_auprc) else 1.0  # Handle edge case

        # Store the best threshold and its F1 score as a tuple
        metrics_dict[f"BEST_THRESHOLD_{suffixes[idx]}"] = best_threshold
        metrics_dict[f"BEST_F1-SCORE_{suffixes[idx]}"] = f1_scores[best_idx]
        print(f"AUPRC_{suffixes[idx]}: {auprc:.4f}")
        print(f"BEST_THRESHOLD_F1_{suffixes[idx]}: {best_threshold:.4f}, {f1_scores[best_idx]:.4f}")
    
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
    metrics_dict = add_metrics_to_metric_dict("all", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    tn, fp, fn, tp, fpr_for_tiles = tile_FPR(labels_classification.flatten(), predictions_classification.flatten())
    metrics_dict = add_metrics_to_metric_dict("all_tile", [tn, fp, fn, tp, fpr_for_tiles], metrics_dict)
    print("Strong:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels_strong.flatten(), predictions_strong.flatten())
    metrics_dict = add_metrics_to_metric_dict("strong", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    print("Weak:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels_weak.flatten(), predictions_weak.flatten())
    metrics_dict = add_metrics_to_metric_dict("weak", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    return metrics_dict

if __name__ == "__main__":
    DEBUG = True
    all_metrics = []
    if DEBUG:
        dataset_roots = ["/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_2122-2488_SELECT-ALL_CHANNEL-N-72"]
        products_threshold = [("mag1c_tile_sampled-0.01.tif", 500)]
    else:
        dataset_roots = [
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-125',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_2122-2488_SELECT-ALL_CHANNEL-N-72',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-110',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-EVENLY-SPACED_CHANNEL-N-10',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-EVENLY-SPACED_CHANNEL-N-25',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-100',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.05_SPED-UP_PRECISION-64_2122-2488_SELECT-ALL_CHANNEL-N-72',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-TRANSMITTANCE_CHANNEL-N-35',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-90',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-72',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-50',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-TRANSMITTANCE_CHANNEL-N-72',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-TRANSMITTANCE_CHANNEL-N-50',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-TRANSMITTANCE_CHANNEL-N-10',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-35',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-10',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-TRANSMITTANCE_CHANNEL-N-25',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-HIGHEST-VARIANCE_CHANNEL-N-25',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-EVENLY-SPACED_CHANNEL-N-50',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.1_SPED-UP_PRECISION-64_2122-2488_SELECT-ALL_CHANNEL-N-72',
            '/home/jherec/methane-filters-benchmark/data/WHOLE-IMAGE_TILE-AND-SAMPLED-MAG1C-0.01_SPED-UP_PRECISION-64_200-2600_SELECT-EVENLY-SPACED_CHANNEL-N-35']
        products_threshold = []
        for x in ["cem.tif", "mf.tif"]:
            for i in [0.004]:#[0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006]:
                products_threshold.append((x,i))
        for x in ["ace.tif"]:
            for i in [0.03]:#[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
                products_threshold.append((x,i))
        for x in ["mag1c_tile.tif", "mag1c_tile_sampled.tif"]:
            for i in [300]:
                products_threshold.append((x,i))
    
    
    for x in tqdm(dataset_roots, total=len(dataset_roots), desc="Dataset version:"):
        print(x.split("/")[-1])
        for i in tqdm(products_threshold, total=len(products_threshold),desc="Product:"):
            if "sampled" in i[0]:
                i = (f"mag1c_tile_sampled-{x.split("/")[-1].split("_")[1].split("-")[-1]}.tif",i[1])
            print(i)
            all_metrics.append(main(x,i))
            #save after each row
            df = pd.DataFrame(all_metrics)
            df.to_csv(output_csv, index=False)

            print("CSV file saved successfully!")