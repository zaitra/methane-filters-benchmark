"""
Example call with all options:
python3 prepare_emit_data.py \
    --csv-path /home/jherec/emit-dataset/dataset_info.csv \
    --data-dir /home/jherec/emit-dataset/data/data \
    --output-dir /home/jherec/emit-dataset/data_selected_50_highest_transmittance \
    --num-bands 50 \
    --strategy highest-transmittance \
    --dry-run
"""
import csv
from pathlib import Path
import numpy as np
import netCDF4
from tqdm import tqdm
import tifffile
import rasterio
from pyproj import Transformer
from onboard_methane_detection import generate_methane_spectrum, select_the_bands_by_transmittance
from georeader.readers import emit
import shutil

# For memory usage monitoring
import psutil
import os
import gc

BGR_WV = [460, 550, 640]

def find_nc(sample_dir):
    out = list(sample_dir.glob("*.nc")) + list(sample_dir.glob("**/*.nc"))
    return out[0] if out else None

def check_wavelengths_fwhm(csv_path, data_dir):
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    
    first_wavelengths = None
    first_fwhm = None
    
    all_same = True
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    for row in tqdm(rows, desc="Checking NC files"):
        sid = row['id']
        sample_dir = data_dir / sid
        nc_p = find_nc(sample_dir)
        
        if nc_p:
            ds = netCDF4.Dataset(nc_p)
            wavelengths = ds.groups["sensor_band_parameters"].variables["wavelengths"][:]
            fwhm = ds.groups["sensor_band_parameters"].variables["fwhm"][:]
            ds.close()
            
            if first_wavelengths is None:
                first_wavelengths = wavelengths
                first_fwhm = fwhm
            else:
                if not np.allclose(first_wavelengths, wavelengths, equal_nan=True):
                    print(f"Difference in wavelengths found in {sid}")
                    all_same = False
                if not np.allclose(first_fwhm, fwhm, equal_nan=True):
                    print(f"Difference in FWHM found in {sid}")
                    all_same = False
                    
    if all_same:
        print("All NC files have the same wavelengths and fwhm.")
    else:
        print("Differences were found.")
        
    return first_wavelengths, first_fwhm

def create_selected_bands_dataset(csv_path, data_dir, output_dir, num_bands=50, strategy='highest-transmittance', dry_run=False):
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Checking wavelengths and FWHM...")
    wavelengths, fwhm = check_wavelengths_fwhm(csv_path, data_dir)
    
    # Convert MaskedArrays to standard numpy arrays to drop the mask wrapper entirely
    wavelengths = np.array(wavelengths)
    fwhm = np.array(fwhm)
    
    print("Generating CH4 spectrum...")
    ch4_spectrum = generate_methane_spectrum(wavelengths, fwhm)
    np.save(output_dir / "emit_methane_spectrum.npy", ch4_spectrum)
    
    print(f"Selecting {num_bands} bands using {strategy}...")
    selected_w, selected_t = select_the_bands_by_transmittance(wavelengths, ch4_spectrum, num_bands, strategy)
    
    # Sort selected bands by wavelength
    sort_idx = np.argsort(selected_w)
    selected_w = selected_w[sort_idx]
    selected_t = selected_t[sort_idx]
    
    selected_indices = []
    for w in selected_w:
        idx = np.where(wavelengths == w)[0][0]
        selected_indices.append(idx)
    selected_indices = np.array(selected_indices)
        
    rgb_colors = ["blue", "green", "red"]
    rgb_indices = []
    actual_rgb_wavelengths = []
    for target_w in BGR_WV:
        idx = np.argmin(np.abs(wavelengths - target_w))
        rgb_indices.append(int(idx))
        actual_rgb_wavelengths.append(float(wavelengths[idx]))
    
    np.save(output_dir / "emit_50_selected_methane_spectrum.npy", np.array(selected_t))
    
    # Save a JSON file with selected indices for reference
    import json
    with open(output_dir / "selected_bands.json", "w") as f:
        json.dump({
            "indices": selected_indices.tolist(),
            "wavelengths": selected_w.tolist(),
            "methane_transmittance": selected_t.tolist(),
            "rgb_wavelengths": actual_rgb_wavelengths,
            "rgb_indices": rgb_indices
        }, f, indent=4)
    
    total_size_bytes = 0
    max_scene_size_bytes = 0
    biggest_scene_name = ""
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    for row in tqdm(rows, desc="Creating new dataset"):
        sid = row['id']
        #if sid != "EMIT_L1B_RAD_001_20231025T061531_2329804_012":
        #    continue
        sample_dir = data_dir / sid
        nc_p = find_nc(sample_dir)
        label_p = sample_dir / "label.npy"
        
        if not nc_p:
            continue
            
        if not dry_run:
            out_sample_dir = output_dir / sid
            if out_sample_dir.exists():
                shutil.rmtree(out_sample_dir)
            out_sample_dir.mkdir(parents=True)
            
        # Load directly using Georeader
        rst = emit.EMITImage(str(nc_p))
        
        # Compute proper affine transform for the raw swath data from lat/lon grids
        ds_nc = netCDF4.Dataset(nc_p)
        lon = ds_nc.groups["location"].variables["lon"][:]
        lat = ds_nc.groups["location"].variables["lat"][:]
        ds_nc.close()
        
        h, w = lon.shape
        y, x = np.mgrid[:h, :w]

        A = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])

        ax, bx, cx = np.linalg.lstsq(A, lon.ravel(), rcond=None)[0]
        ay, by, cy = np.linalg.lstsq(A, lat.ravel(), rcond=None)[0]

        transform = rasterio.Affine(ax, bx, cx,
                        ay, by, cy)
        lon_pred = ax * x + bx * y + cx
        lat_pred = ay * x + by * y + cy

        err = np.sqrt((lon_pred - lon)**2 + (lat_pred - lat)**2)
        print("mean error (deg):", err.mean())
        print("max error (deg):", err.max())
        
        spatial_ref = "EPSG:4326"
        data = rst.load_raw() # loads as (bands, rows, cols)
        selected_data = data[selected_indices, :, :]
        
        if total_size_bytes == 0:
            print(f"\nBand data dtype: {selected_data.dtype}")
        
        scene_size_bytes = selected_data.nbytes
        total_size_bytes += scene_size_bytes
        if scene_size_bytes > max_scene_size_bytes:
            max_scene_size_bytes = scene_size_bytes
            biggest_scene_name = sid
            
        if dry_run:
            continue
        
        # Save selected bands with georeferencing
        for i, w in enumerate(selected_w):
            w_int = int(w)
            band_data = selected_data[i, :, :]
            
            kwargs = {
                'driver': 'GTiff',
                'height': band_data.shape[0],
                'width': band_data.shape[1],
                'count': 1,
                'dtype': band_data.dtype,
                'transform': transform,
                'crs': spatial_ref
            }
                
            with rasterio.open(out_sample_dir / f"{w_int}nm.tif", 'w', **kwargs) as dst:
                dst.write(band_data, 1)

        # Save RGB bands as well
        for color, idx in zip(rgb_colors, rgb_indices):
            band_data = data[idx, :, :]
            
            kwargs = {
                'driver': 'GTiff',
                'height': band_data.shape[0],
                'width': band_data.shape[1],
                'count': 1,
                'dtype': band_data.dtype,
                'transform': transform,
                'crs': spatial_ref
            }
                
            with rasterio.open(out_sample_dir / f"{color}.tif", 'w', **kwargs) as dst:
                dst.write(band_data, 1)

        # Copy label
        if label_p.exists():
            shutil.copy(label_p, out_sample_dir / "label.npy")

        gc.collect()
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY] RSS: {mem_mb:.2f} MB after processing {sid}")
            
    print(f"\n--- Output Size Estimation ---")
    print(f"Maximum single scene size: {max_scene_size_bytes / (1024**2):.2f} MB (Scene: {biggest_scene_name})")
    print(f"Total dataset size: {total_size_bytes / (1024**3):.2f} GB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare EMIT dataset with selected bands.")
    parser.add_argument("--csv-path", type=str, default="/home/jherec/emit-dataset/dataset_info.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--data-dir", type=str, default="/home/jherec/emit-dataset/data/data", help="Path to the raw EMIT data directory.")
    parser.add_argument("--output-dir", type=str, default="/home/jherec/emit-dataset/data_selected_50_highest_transmittance", help="Path to the output directory.")
    parser.add_argument("--num-bands", type=int, default=50, help="Number of bands to select (default: 50).")
    parser.add_argument("--strategy", type=str, default="highest-transmittance", choices=["highest-transmittance", "highest-variance", "evenly-spaced"], help="Band selection strategy (default: highest-transmittance).")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Only estimate output size without writing files.")
    args = parser.parse_args()

    create_selected_bands_dataset(
        args.csv_path,
        args.data_dir,
        args.output_dir,
        num_bands=args.num_bands,
        strategy=args.strategy,
        dry_run=args.dry_run,
    )