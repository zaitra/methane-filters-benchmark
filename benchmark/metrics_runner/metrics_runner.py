import os
import argparse

import numpy
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import pylab as plt
from tqdm import tqdm
import yaml

from baseline import Mag1cBaseline


def open_file(path):
    if ".npy" in path:
        file = np.load(path)
    else:
        with rio.open(path) as src:
            file = src.read()
    return file

def main(dataset_root, product_threshold, config):
    product, threshold = product_threshold
    if "mag1c" in product:
        normalizer_params = {'offset': 0, 'factor': 1750, 'clip': (0, 2)}
    else:
        normalizer_params = {'offset': 0, 'factor': 1, 'clip': (0, 1)}
    baseline_model = Mag1cBaseline(mag1c_threshold = threshold, normalizer_params = normalizer_params)
    dataset_v = dataset_root.split("/")[-1]
    show = False
    sort_by_plume_size = False

    df = pd.read_csv(config["csv_file"])
    if sort_by_plume_size:
        df = df.sort_values(["has_plume", "qplume"], ascending=False)

    check_df = pd.read_csv(config["output_csv"])
    metrics_dict = {
        "METHOD": product.replace(".tif", "").replace(".npy","").upper() + dataset_v,
        "THRESHOLD": threshold,}
    if config["STARCOP"]:
        dataset_v_splitted = dataset_v.split("_")
        whole_image, mag1c, sped_up, bit_depth_precision, wv_range, select_strategy, channel_n = dataset_v_splitted
        whole_image = True if "WHOLE" in whole_image.upper() else False
        sampled_percentage = 1 if "SAMPLED" not in mag1c.upper() else mag1c.split("-")[-1]
        mag1c = "STARCOP" if "STARCOP" in mag1c.upper() else "GENERATED"
        sped_up = True if "SPED" in sped_up else False
        bit_depth_precision = f"float{bit_depth_precision.replace("PRECISION-","")}"
        channel_n = int(channel_n.replace("CHANNEL-N-", ""))
        metrics_dict = metrics_dict | {
            "WHOLE_IMAGE": whole_image,
            "MAG1C": mag1c if "mag1c" in product.lower() else "-",
            "SAMPLED": sampled_percentage if "mag1c" in product.lower() else "-",
            "SPED_UP": sped_up,
            "PRECISION": bit_depth_precision,
            "WAVELENGTH_RANGE":wv_range,
            "CHANNEL_N": channel_n,
            "SELECT_STRATEGY": select_strategy
            }

        columns_to_check = list(metrics_dict.keys())

        if not check_df.empty:
            matching_rows = check_df[(check_df[columns_to_check] == pd.Series(metrics_dict)).all(axis=1)]

            if not matching_rows.empty:
                existing_entry = matching_rows.iloc[0].to_dict()
                print("Matching entry found:", existing_entry)
                return existing_entry

    # Constants
    EASY_HARD_THRESHOLD = 1000
    NUMBER_OF_PIXELS_PRED = config["NUMBER_OF_PIXELS_PRED"]
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

    indices_no_plume = []
    indices_strong = []
    indices_weak = []

    for idx, item in tqdm(df.iterrows(), total=len(df)):
        # Load products:
        mf_path = os.path.join(dataset_root, item["id"], product)
        y_path = os.path.join(config["DATASET_PATH"], item["id"], config["LABEL_NAME"])
        mask_path = os.path.join(config["DATASET_PATH"], item["id"], config["MASK_NAME"])

        mf_data = open_file(mf_path)
        y_data = open_file(y_path)
        mask_data = open_file(mask_path)
        mask_data = np.where(mask_data <= 0, 0, 1)
        if config["PRELOADED_MODEL_INFERENCE"] and config["LOGIT_TO_PROBABILITIES"]:
            mf_data = 1 / (1 + np.exp(-mf_data))
        if config["expand_dim_of_mf"]:
            mf_data = np.expand_dims(mf_data, axis=0)
        if config["expand_dim_of_y_data"]:
            y_data = np.expand_dims(y_data, axis=0)

        # Determine easy / hard split
        label_pixels_plume = np.sum(y_data)
        tile_has_plume = label_pixels_plume > 0
        if config["STARCOP"]:
            difficulty = "easy" if label_pixels_plume > EASY_HARD_THRESHOLD else "hard"
        else:
            difficulty = item["difficulty"]

        event_type = "noplume"
        if tile_has_plume and difficulty == "easy":
            event_type = "strong"
        if tile_has_plume and difficulty == "hard":
            event_type = "weak"
        # Use the pytorch lightning module
        batch = {}
        batch["output"] = torch.tensor(y_data).unsqueeze(0)
        batch["input"] = torch.tensor(mf_data).unsqueeze(0)
        if config["USE_MORPHOLOGICAL_BASELINE"]:
            batch = baseline_model.batch_with_preds(batch)
        else:
            batch["pred_binary"] = batch["input"] > threshold

        if show:
            path = os.path.join("/home/jherec/methane-filters-benchmark/outputs/trash", item["id"])
            plt.imshow(batch["prediction"][0][0])
            plt.title("Product")
            plt.savefig(path + "product.png")
            plt.imshow(batch["pred_binary"][0][0])
            plt.title("Product after thr and morpho.")
            plt.savefig(path + "product_after.png")
            plt.imshow(batch["output"][0][0])
            plt.title("Label")
            plt.savefig(path + "label.png")
            plt.imshow(mask_data[0])
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
            if px_mask == 1:
                gt_masked.append(gt[px_idx])
                pred_masked.append(pred[px_idx])
                original_masked.append(original_data[px_idx])

        if event_type == "noplume":
            labels_noplume += gt_masked
            predictions_noplume += pred_masked
            predictions_noplume_original += original_masked
            indices_no_plume.append(idx)
        elif event_type == "weak":
            labels_weak += gt_masked
            predictions_weak += pred_masked
            predictions_weak_original += original_masked
            indices_weak.append(idx)
        elif event_type == "strong":
            labels_strong += gt_masked
            predictions_strong += pred_masked
            predictions_strong_original += original_masked
            indices_strong.append(idx)

        # Tile classification result
        pred_pixels_count = np.sum(pred_masked)
        tile_pred_has_plume = pred_pixels_count > NUMBER_OF_PIXELS_PRED
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
    print(labels.shape, labels_strong.shape, labels_weak.shape,predictions_original.shape, predictions_strong_original.shape, predictions_weak_original.shape)
    # Scores:
    from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
    suffixes = ["all", "strong", "weak"]
    for idx,labels_predictions in enumerate(zip([labels, labels_strong, labels_weak],[predictions_original, predictions_strong_original, predictions_weak_original])):
        precision, recall, thresholds_auprc = precision_recall_curve(labels_predictions[0], labels_predictions[1], drop_intermediate=True)
        auprc = auc(recall, precision)
        metrics_dict[f"AUPRC_{suffixes[idx]}"] = auprc
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_auprc[best_idx] if best_idx < len(thresholds_auprc) else 1.0

        metrics_dict[f"BEST_THRESHOLD_{suffixes[idx]}"] = best_threshold
        metrics_dict[f"BEST_F1-SCORE_{suffixes[idx]}"] = f1_scores[best_idx]
        print(f"AUPRC_{suffixes[idx]}: {auprc:.4f}")
        print(f"BEST_THRESHOLD_F1_{suffixes[idx]}: {best_threshold:.4f}, {f1_scores[best_idx]:.4f}")

    def round_to(n, digits=3):
        if np.isnan(n): return "NaN"
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
        print(tn, fp, fn, tp )
        fpr_for_tiles = fp / (fp + tn)
        print("FPR (tile)", round_to(fpr_for_tiles))
        recall = tp/(tp+fn) if tp+fn > 0 else 0
        precision = tp/(tp+fp) if tp+fp > 0 else 0
        f1 = 2*(recall * precision) / (recall + precision)

        iou = tp / (tp + fp + fn)
        print("Recall (tile)", round_to(recall)+", Precision (tile)", round_to(precision)+", F1 (tile)", round_to(f1))
        print("IoU (tile)", round_to(iou))
        return tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles

    def add_metrics_to_metric_dict(suffix, metrics, metrics_dict):
        if len(metrics) == 8:
            names = ["TN", "FP", "FN", "TP", "Recall", "Precision", "F1-score", "Iou"]
            names = [f"{n}_{suffix}_seg" for n in names]
        else:
            names = ["TN", "FP", "FN", "TP", "Recall", "Precision", "F1-score", "Iou", "FPR"]
            names = [f"{n}_{suffix}_clas" for n in names]
        names_metrics = dict(zip(names,metrics))
        metrics_dict = metrics_dict | names_metrics
        return metrics_dict

    print("All:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels.flatten(), predictions.flatten())
    metrics_dict = add_metrics_to_metric_dict("all", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles = tile_FPR(labels_classification, predictions_classification)
    metrics_dict = add_metrics_to_metric_dict("all_class", [tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles], metrics_dict)
    print("Strong:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels_strong.flatten(), predictions_strong.flatten())
    metrics_dict = add_metrics_to_metric_dict("strong", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    combined_indices = indices_no_plume+indices_strong
    tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles = tile_FPR(labels_classification[combined_indices], predictions_classification[combined_indices])
    metrics_dict = add_metrics_to_metric_dict("strong_class", [tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles], metrics_dict)
    print("Weak:")
    tn, fp, fn, tp, recall, precision, f1, iou = metric_prec_recall_f1(labels_weak.flatten(), predictions_weak.flatten())
    metrics_dict = add_metrics_to_metric_dict("weak", [tn, fp, fn, tp, recall, precision, f1, iou], metrics_dict)
    combined_indices = indices_no_plume+indices_weak
    tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles = tile_FPR(labels_classification[combined_indices], predictions_classification[combined_indices])
    metrics_dict = add_metrics_to_metric_dict("weak_class", [tn, fp, fn, tp, recall, precision, f1, iou, fpr_for_tiles], metrics_dict)
    return metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metrics evaluation.")
    parser.add_argument("--config", type=str, default="cfg/starcop.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_csv = config["output_csv"]
    if not os.path.exists(output_csv):
        empty_df = pd.DataFrame(columns=["col1", "col2", "col3"])
        empty_df.to_csv(output_csv, index=False)

    dataset_roots = config["dataset_roots"]
    products_threshold = [tuple(pt) for pt in config["products_threshold"]]

    all_metrics = []
    for x in tqdm(dataset_roots, total=len(dataset_roots), desc="Dataset version:"):
        print(x.split("/")[-1])
        for i in tqdm(products_threshold, total=len(products_threshold), desc="Product:"):
            if "sampled" in i[0]:
                i = (f"mag1c_tile_sampled-{x.split('/')[-1].split('_')[1].split('-')[-1]}.tif", i[1])
            print(i)
            all_metrics.append(main(x, i, config))
            df = pd.DataFrame(all_metrics)
            df.to_csv(output_csv, index=False)
            print("CSV file saved successfully!")
