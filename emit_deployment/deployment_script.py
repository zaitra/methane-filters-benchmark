"""
Example command with all options:
python3 deployment_script.py \
    --data-dir /tmp/emmc/jonas/TGRS/data_selected_50_variance \
    --spectrum-path /tmp/emmc/jonas/TGRS/data_selected_50_variance/emit_50_selected_methane_spectrum.npy \
    --output-dir /tmp/emmc/jonas/TGRS/data_selected_50_variance \
    --tile-for-inference \
    --tile-for-mag1c-sas \
    --save-mag1c-sas-output \
    --save-valid-mask \
    --mark-timestamps
"""
import os
import time
import csv
import argparse
from pathlib import Path
import numpy as np
import rasterio
from onboard_methane_detection.mag1c_sas_base import compute_base_mag1c_SAS
from onboard_methane_detection.processing.utils import (
    compute_valid_mask, 
    preprocess_image, 
    compute_sampling_indices, 
    postprocess_result as postprocess_result_sas,
    tile_image,
    stitch_tiles,
)
from onboard_methane_detection.inference.utils import initialize_model, normalize_image, run_sole_inference, postprocess_result, pad_to_32, reverse_32_padding
SAMPLE_RATIO = 0.01

def load_scene_bands(scene_dir):
    """Load all exported TIFF bands from a scene directory into a (C, H, W) array."""
    # Find all tif files representing bands
    tif_files = list(Path(scene_dir).glob("*nm.tif"))
    if not tif_files:
        return None, None, None
        
    # Sort files by wavelength integer to keep the spectrum channel order intact
    tif_files.sort(key=lambda x: int(x.stem.replace("nm", "")))
    
    bands = []
    meta = None
    for i, tf in enumerate(tif_files):
        with rasterio.open(tf) as src:
            bands.append(src.read(1))
            if i == 0:
                meta = src.meta.copy()
                
    image = np.stack(bands, axis=0)
    
    rgb_bands = []
    for color in ["red", "green", "blue"]:
        with rasterio.open(Path(scene_dir) / f"{color}.tif") as src:
            rgb_bands.append(src.read(1))
    rgb = np.stack(rgb_bands, axis=0)
            
    return np.clip(image, 0, None), np.clip(rgb, 0, None), meta

def run_deployment(data_dir, spectrum_path, output_dir, tile_for_inference, tile_for_mag1c_sas, save_mag1c_sas_output, save_valid_mask, mark_timestamps):
    timing_data = [['id', 'action_name', 'marker', 'timestamp']]
    current_id = "Global"

    if mark_timestamps: timing_data.append([current_id, 'prep_init', 'START', time.time()])
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing model...")
    
    if tile_for_inference:
        model, input_name = initialize_model(dynamic_output_size=False)
    else:
        model, input_name = initialize_model(dynamic_output_size=True)

    print(f"Loading methane spectrum from {spectrum_path}")
    methane_spectrum = np.load(spectrum_path)
    
    
    # Iterate through each exported scene 
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if mark_timestamps: timing_data.append([current_id, 'prep_init', 'END', time.time()])
    
    for scene_dir in scene_dirs:
        current_id = scene_dir.name
        #if current_id not in ["EMIT_L1B_RAD_001_20231025T061531_2329804_012"]:
        #    continue
        if mark_timestamps: timing_data.append([current_id, 'load_scene', 'START', time.time()])
        image, rgb, meta = load_scene_bands(scene_dir)
        
        if image is None:
            continue
        if mark_timestamps: timing_data.append([current_id, 'load_scene', 'END', time.time()])
            
        print(f"\nProcessing {scene_dir.name}")
        if not tile_for_mag1c_sas:
            if mark_timestamps: timing_data.append([current_id, 'mag1c_prep', 'START', time.time()])
            c, h, w = image.shape
            
            valid_mask_flat = compute_valid_mask(image)
            if np.any(valid_mask_flat):
                image_batched = preprocess_image(image, valid_mask_flat)
                
                indices = compute_sampling_indices(image_batched.shape[1], SAMPLE_RATIO)
                if mark_timestamps: timing_data.append([current_id, 'mag1c_prep', 'END', time.time()])
                
                if mark_timestamps: timing_data.append([current_id, 'mag1c_compute', 'START', time.time()])
                raw_result = compute_base_mag1c_SAS(image_batched, methane_spectrum, indices)
                if mark_timestamps: timing_data.append([current_id, 'mag1c_compute', 'END', time.time()])
                
                if mark_timestamps: timing_data.append([current_id, 'mag1c_post', 'START', time.time()])
                result_sas = postprocess_result_sas(raw_result, valid_mask_flat, (1, h, w))
                if mark_timestamps: timing_data.append([current_id, 'mag1c_post', 'END', time.time()])
            else:
                result_sas = np.zeros((1, h, w), dtype=np.float32)
                if mark_timestamps: timing_data.append([current_id, 'mag1c_prep', 'END', time.time()])
                
            # Concatenate RGB and result, normalize, and clip
            if mark_timestamps: timing_data.append([current_id, 'normalize', 'START', time.time()])
            combined = np.concatenate([rgb, result_sas], axis=0, dtype=np.float32)
            normalized = normalize_image(combined, sensor_or_factors="emit")
            if mark_timestamps: timing_data.append([current_id, 'normalize', 'END', time.time()])
        else:
            if mark_timestamps: timing_data.append([current_id, 'tile_image', 'START', time.time()])
            tiles, tiling_info = tile_image(image)
            tiles_rgb, tiling_info_rgb = tile_image(rgb)
            if mark_timestamps: timing_data.append([current_id, 'tile_image', 'END', time.time()])
            if mark_timestamps: timing_data.append([current_id, 'mag1c_prep', 'START', time.time()])
            c, h, w = tiles[0].shape
            
            valid_masks_flat = [compute_valid_mask(tile) for tile in tiles]
            
            images_batched = [preprocess_image(tile, valid_mask_flat) if np.any(valid_mask_flat) else None for tile, valid_mask_flat in zip(tiles, valid_masks_flat)]
            
            indices = [compute_sampling_indices(image_batched.shape[1], SAMPLE_RATIO) if image_batched is not None else None for image_batched in images_batched]
            if mark_timestamps: timing_data.append([current_id, 'mag1c_prep', 'END', time.time()])
            
            if mark_timestamps: timing_data.append([current_id, 'mag1c_compute', 'START', time.time()])
            raw_results = [compute_base_mag1c_SAS(image_batched, methane_spectrum, indices) if indices is not None else None for image_batched, indices in zip(images_batched, indices)]
            if mark_timestamps: timing_data.append([current_id, 'mag1c_compute', 'END', time.time()])
            
            if mark_timestamps: timing_data.append([current_id, 'mag1c_post', 'START', time.time()])
            results_sas = [postprocess_result_sas(raw_result, valid_mask_flat, (1, h, w)) if raw_result is not None else np.zeros((1, h, w), dtype=np.float32) for raw_result, valid_mask_flat in zip(raw_results, valid_masks_flat)]
            if mark_timestamps: timing_data.append([current_id, 'mag1c_post', 'END', time.time()])
                
            # Concatenate RGB and result, normalize, and clip
            if mark_timestamps: timing_data.append([current_id, 'normalize', 'START', time.time()])
            combined = [np.concatenate([rgb, result_sas], axis=0, dtype=np.float32) for rgb, result_sas in zip(tiles_rgb, results_sas)]
            tiles = [normalize_image(combined, sensor_or_factors="emit") for combined in combined]
            if mark_timestamps: timing_data.append([current_id, 'normalize', 'END', time.time()])
        if tile_for_inference:
            if not tile_for_mag1c_sas:
                if mark_timestamps: timing_data.append([current_id, 'tile_image', 'START', time.time()])
                tiles, tiling_info = tile_image(normalized)
                if mark_timestamps: timing_data.append([current_id, 'tile_image', 'END', time.time()])
            if mark_timestamps: timing_data.append([current_id, 'inference_tiles', 'START', time.time()])
            raw_tiles_results = [run_sole_inference(model, input_name, tile) for tile in tiles]
            if mark_timestamps: timing_data.append([current_id, 'inference_tiles', 'END', time.time()])
            if mark_timestamps: timing_data.append([current_id, 'post_tiles', 'START', time.time()])
            processed_tiles = [postprocess_result(raw_tile_result, logits_to_probs=True) for raw_tile_result in raw_tiles_results]
            if mark_timestamps: timing_data.append([current_id, 'post_tiles', 'END', time.time()])
                
            if mark_timestamps: timing_data.append([current_id, 'stitch_tiles', 'START', time.time()])
            tiling_info['original_shape'] = (1, image.shape[1], image.shape[2])
            final_output = stitch_tiles(processed_tiles, tiling_info)
            if mark_timestamps: timing_data.append([current_id, 'stitch_tiles', 'END', time.time()])
        else:
            if mark_timestamps: timing_data.append([current_id, 'pad_image_32', 'START', time.time()])
            padded_clipped, pad_info = pad_to_32(normalized)
            if mark_timestamps: timing_data.append([current_id, 'pad_image_32', 'END', time.time()])
            if mark_timestamps: timing_data.append([current_id, 'inference_image', 'START', time.time()])
            raw_result = run_sole_inference(model, input_name, padded_clipped)
            if mark_timestamps: timing_data.append([current_id, 'inference_image', 'END', time.time()])
            if mark_timestamps: timing_data.append([current_id, 'post_image', 'START', time.time()])
            padded_final_output = postprocess_result(raw_result, logits_to_probs=True)
            final_output = reverse_32_padding(padded_final_output, pad_info)
            if mark_timestamps: timing_data.append([current_id, 'post_image', 'END', time.time()])


        # Save output maintaining the spatial metadata
        if mark_timestamps: timing_data.append([current_id, 'create_out_dir', 'START', time.time()])
        out_scene_dir = output_dir / scene_dir.name
        out_scene_dir.mkdir(parents=True, exist_ok=True)
        if mark_timestamps: timing_data.append([current_id, 'create_out_dir', 'END', time.time()])

        if save_mag1c_sas_output:
            if mark_timestamps: timing_data.append([current_id, 'save_mag1c', 'START', time.time()])
            if tile_for_mag1c_sas:
                result_sas= stitch_tiles(results_sas, tiling_info)
            suffix = "_tiling" if (tile_for_mag1c_sas) else ""
            sas_out_path = out_scene_dir / f"mag1c_sas{suffix}.tif"
            sas_meta = meta.copy()
            sas_meta.update({
                "count": 1,
                "dtype": "float32"
            })
            with rasterio.open(sas_out_path, "w", **sas_meta) as dst:
                dst.write(result_sas.astype(np.float32))
            if mark_timestamps: timing_data.append([current_id, 'save_mag1c', 'END', time.time()])
        
        # Save output maintaining the spatial metadata
        if mark_timestamps: timing_data.append([current_id, 'save_inference', 'START', time.time()])
        if tile_for_inference and tile_for_mag1c_sas:
            suffix = "_tiling"
        elif tile_for_inference and not tile_for_mag1c_sas:
            suffix = "_tiling_inference_only"
        else:
            suffix = ""
        out_path = out_scene_dir / f"inference{suffix}.tif"
        
        meta.update({
            "count": final_output.shape[0],
            "dtype": "float32"
        })
        
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(final_output.astype(np.float32))
        if mark_timestamps: timing_data.append([current_id, 'save_inference', 'END', time.time()])

        if save_valid_mask:
            if mark_timestamps: timing_data.append([current_id, 'save_valid_mask', 'START', time.time()])
            mask_out_path = out_scene_dir / "valid_mask.tif"
            mask_meta = meta.copy()
            mask_meta.update({
                "count": 1,
                "dtype": "uint8"
            })
            suffix = "_tiling" if (tile_for_mag1c_sas) else ""
            mask_out_path = out_scene_dir / f"valid_mask{suffix}.tif"
            if tile_for_mag1c_sas:
                valid_mask_2d = stitch_tiles([valid_mask_flat.reshape((1,h,w)) for valid_mask_flat in valid_masks_flat], tiling_info)
            else:
                valid_mask_2d = valid_mask_flat.reshape((h, w))
                valid_mask_2d = np.expand_dims(valid_mask_2d, axis=0)
            with rasterio.open(mask_out_path, "w", **mask_meta) as dst:
                dst.write(valid_mask_2d.astype(np.uint8))
            if mark_timestamps: timing_data.append([current_id, 'save_valid_mask', 'END', time.time()])

    if mark_timestamps:
        if tile_for_inference and tile_for_mag1c_sas:
            csv_path = "./timestamps_tiling.csv"
        else:
            csv_path = "./timestamps.csv"
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(timing_data)
        print(f"Timing data saved to {csv_path}")
        # Compute duration statistics per action from the saved CSV (pair START/END)
        try:
            events = []
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 4:
                        continue
                    try:
                        ts = float(row[3])
                    except Exception:
                        continue
                    events.append((row[0], row[1], row[2], ts))

            # Pair START/END for each (id, action) in chronological order
            pending = {}  # key -> list of start times
            durations = {}  # action -> list of durations
            for ev in events:
                _id, action, marker, ts = ev
                key = (_id, action)
                if str(marker).upper() == 'START':
                    pending.setdefault(key, []).append(ts)
                elif str(marker).upper() == 'END':
                    if pending.get(key):
                        start_ts = pending[key].pop()
                        dur = ts - start_ts
                        durations.setdefault(action, []).append(dur)

            if durations:
                print("\nTiming summary (seconds) per action:")
                for action in sorted(durations.keys()):
                    vals = np.array(durations[action], dtype=np.float64)
                    cnt = vals.size
                    vmin = float(np.min(vals)) if cnt else 0.0
                    vmax = float(np.max(vals)) if cnt else 0.0
                    vmean = float(np.mean(vals)) if cnt else 0.0
                    vmed = float(np.median(vals)) if cnt else 0.0
                    print(f"- {action}: count={cnt} min={vmin:.2f} max={vmax:.2f} mean={vmean:.2f} median={vmed:.2f}")
            else:
                print("No paired START/END timing events found in timestamps.csv")
        except Exception as e:
            print(f"Failed to compute timing summary: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run methane detection deployment.")
    parser.add_argument("--data-dir", type=str, default="/home/jherec/emit-dataset/data_selected_50_variance", help="Input data directory")
    parser.add_argument("--spectrum-path", type=str, default="/home/jherec/emit-dataset/data_selected_50_variance/emit_50_selected_methane_spectrum.npy", help="Methane spectrum path")
    parser.add_argument("--output-dir", type=str, default="/home/jherec/emit-dataset/data_selected_50_variance", help="Output directory")
    parser.add_argument("--tile-for-inference", action="store_true", help="Enable tiling for inference")
    parser.add_argument("--tile-for-mag1c-sas", action="store_true", help="Enable tiling for mag1c SAS")
    parser.add_argument("--save-mag1c-sas-output", action="store_true", help="Save intermediate mag1c SAS outputs")
    parser.add_argument("--save-valid-mask", action="store_true", help="Save the valid mask output")
    parser.add_argument("--mark-timestamps", action="store_true", help="Enable timing measurements")
    args = parser.parse_args()

    run_deployment(
        data_dir=args.data_dir,
        spectrum_path=args.spectrum_path,
        output_dir=args.output_dir,
        tile_for_inference=args.tile_for_inference,
        tile_for_mag1c_sas=args.tile_for_mag1c_sas,
        save_mag1c_sas_output=args.save_mag1c_sas_output,
        save_valid_mask=args.save_valid_mask,
        mark_timestamps=args.mark_timestamps,
    )