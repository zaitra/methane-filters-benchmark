import spectral as spy
import numpy as np
import time
import argparse
from pysptools.detection.detect import ACE as ACE_original, CEM as CEM_original, MatchedFilter as MatchedFilterOriginal
from sped_up_filters import ACE_optimized, CEM_optimized, MatchedFilterOptimized, mag1c_SAS, mag1c_tile
import os
import spectral.io.envi as envi
import subprocess
import sys
import csv
from statistics import mean, median
from time import sleep
import importlib.util
_mag1c_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mag1c_fork", "mag1c", "mag1c.py")
_mag1c_spec = importlib.util.spec_from_file_location("mag1c_mod", _mag1c_path)
_mag1c_mod = importlib.util.module_from_spec(_mag1c_spec)
_mag1c_spec.loader.exec_module(_mag1c_mod)
get_censor_mask = _mag1c_mod.get_censor_mask

NODATA = -9999

def measure_morphological_baseline(tensor, threshold, repetitions=100):
    import torch
    from kornia.morphology import dilation as kornia_dilation
    from kornia.morphology import erosion as kornia_erosion

    kernel_torch = torch.nn.Parameter(torch.from_numpy(np.array([[0, 1, 0],
                                                                  [1, 1, 1],
                                                                  [0, 1, 0]])).float(), requires_grad=False)

    def binary_opening(x):
        eroded = torch.clamp(kornia_erosion(x.float(), kernel_torch), 0, 1) > 0
        return torch.clamp(kornia_dilation(eroded.float(), kernel_torch), 0, 1) > 0

    def apply_threshold(pred):
        # https://programtalk.com/python-more-examples/kornia.morphology.dilation.bool/
        return binary_opening(pred > threshold).long()

    print("Computing Morphological Baseline...")
    times = []

    for _ in range(repetitions):
        start_time = time.time()
        result = apply_threshold(tensor)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    stats = {
        "min": min(times),
        "max": max(times),
        "mean": mean(times),
        "median": median(times)
    }

    print(f"Morphological Baseline Computation Done! Min: {stats['min']:.4f}s, Max: {stats['max']:.4f}s, Mean: {stats['mean']:.4f}s, Median: {stats['median']:.4f}s")
    return stats


def load_hyperspectral_image(hdr_path):
    """Load hyperspectral image using Spectral Python (SPy)."""
    print(hdr_path)
    # Load the ENVI image
    img = spy.open_image(hdr_path).load()

    # Extract metadata
    metadata = img.metadata

    # Get wavelengths (convert from string list to float)
    wavelengths = metadata.get("wavelength", None)
    if wavelengths:
        wavelengths = [float(w) for w in wavelengths]

    # Get FWHM (convert from string list to float)
    fwhm = metadata.get("fwhm", None)
    if fwhm:
        fwhm = [float(f) for f in fwhm]

    return img, wavelengths, fwhm

# Modify measure_process to repeat measurements and collect statistics
def measure_process(name, function, hyperspectral_img, methane_spectrum, SAS=False, repetitions=100, use_addition=False):
    print(f"Computing {name}...")
    times = []

    for _ in range(repetitions):
        if not SAS:
            if ("ace" in name.lower() or "matchedfilter" in name.lower()) and not "original" in name.lower():
                start_time = time.time()
                result = function(hyperspectral_img, methane_spectrum, addition=use_addition)
                end_time = time.time()
            else:
                start_time = time.time()
                result = function(hyperspectral_img, methane_spectrum)
                end_time = time.time()
        else:
            pixel_N = hyperspectral_img.shape[1]
            sample_size = int(0.01 * pixel_N)
            step_size = pixel_N // sample_size
            indices = np.arange(0, pixel_N, step_size)[:sample_size]

            start_time = time.time()
            result = function(hyperspectral_img, methane_spectrum, indices)
            end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    stats = {
        "min": min(times),
        "max": max(times),
        "mean": mean(times),
        "median": median(times)
    }

    print(f"{name} Computation Done! Min: {stats['min']:.4f}s, Max: {stats['max']:.4f}s, Mean: {stats['mean']:.4f}s, Median: {stats['median']:.4f}s")
    return result, stats

# Modify measure_mag1c to repeat measurements and collect statistics
def measure_mag1c(name, args, repetitions=100):
    print(f"Computing {name}...")
    times = []

    for _ in range(repetitions):
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            stdout = result.stdout
            stderr = result.stderr
            # Extract processing time from stdout
            found_time = False
            for line in stdout.splitlines():
                if "Filter processing completed in" in line and "(measured similarly as for other filters)" in line:
                    time_str = line.split("in")[-1].split("seconds")[0].strip()
                    elapsed_time = float(time_str)
                    times.append(elapsed_time)
                    found_time = True
                    break
            if not found_time:
                print(f"MAG1C stdout (no time found): {stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running MAG1C for {name}:")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            continue

    if not times:
        print(f"WARNING: No successful MAG1C runs for {name}")
        return {"min": 0, "max": 0, "mean": 0, "median": 0}

    stats = {
        "min": min(times),
        "max": max(times),
        "mean": mean(times),
        "median": median(times)
    }

    print(f"{name} Computation Done! Min: {stats['min']:.4f}s, Max: {stats['max']:.4f}s, Mean: {stats['mean']:.4f}s, Median: {stats['median']:.4f}s")
    return stats

# Save statistics to CSV
def save_stats_to_csv(stats, filename, channels_list):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Channels", "Min", "Max", "Mean", "Median"])
        for key, method_stats in stats.items():
            # Extract method name and channels from the key (e.g., "ACE_optimized_50")
            parts = key.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                method_name = parts[0]
                channels = int(parts[1])
            else:
                method_name = key
                channels = "N/A"
            writer.writerow([method_name, channels, method_stats['min'], method_stats['max'], method_stats['mean'], method_stats['median']])

def test_differences(original_results, optimized_results, test=True):
    # Calculate the absolute differences between the original and optimized results
    if np.any(original_results) and np.any(optimized_results):
        diff = np.abs(original_results - optimized_results)

        # Get the maximal and average differences
        max_diff = diff.max()
        avg_diff = diff.mean()

        # Print the formatted differences with labels
        print(f"Maximal difference between original and optimized (sped-up) version: {max_diff:.32f}")
        print(f"Average difference between original and optimized (sped-up) version: {avg_diff:.32f}")

        # Assert that the results are close within a specified tolerance
        if test:
            np.testing.assert_allclose(optimized_results, original_results, atol=0.001, rtol=1)
    else:
        print("One or both arrays are invalid, no similarity testing is done.")


def str_to_precision(value):
    """Convert an integer string to a numpy dtype based on precision."""
    try:
        value = int(value)
        if value == 16:
            return np.float16
        elif value == 32:
            return np.float32
        elif value == 64:
            return np.float64
        else:
            raise argparse.ArgumentTypeError("Precision must be one of 16, 32, or 64.")
    except ValueError:
        raise argparse.ArgumentTypeError("Precision must be an integer (16, 32, or 64).")
    
def check_equivalence(calculated_array, saved_file_path, test_name="Model"):
        """
        Loads a saved reference array and compares it against the calculated array.
        """
        print(f"\n--- Equivalence Check: {test_name} ---")
        
        saved_array = np.load(saved_file_path).reshape(calculated_array.shape)
        diff = np.abs(saved_array - calculated_array)

        print(f"Max absolute difference: {diff.max():.6f}")
        print(f"Mean absolute difference: {diff.mean():.6f}")
        print(f"RMSE: {np.sqrt(np.mean(diff**2)):.6f}")

        non_matching = np.sum(diff > 1e-3)
        total_elements = diff.size
        print(f"Non-matching elements: {non_matching}/{total_elements} ({100*non_matching/total_elements:.4f}%)")

        try:
            np.testing.assert_allclose(saved_array, calculated_array, rtol=1e-1, atol=1e-3)
            print(f"✓ SUCCESS: {test_name} outputs match within tolerance! (Inside Mag1c the output scores are scaled by 10000 at the end, so the real computational difference is by 1e-4 smaller)")
        except AssertionError as e:
            print(f"✗ FAILED: {test_name} outputs differ!")
            print(f"Error: {e}")


def run_pipeline_and_validate(test_name, process_func, img_reshaped, mask, spectrum, nodata_val, saved_path, is_sas=False, repetitions=1):
    """
    Handles masking, measuring, formatting, and equivalence checking for a given model.
    """
    # 1. Initialize output array dynamically based on input size
    total_pixels = img_reshaped.shape[0]
    output_array = np.zeros((total_pixels,), dtype=np.float64)
    output_array[mask] = nodata_val

    # 2. Extract valid pixels and format for the model
    valid_pixels = img_reshaped[~mask, :]
    model_input = np.expand_dims(valid_pixels, axis=0)

    # 3. Measure process
    kwargs = {"SAS": True} if is_sas else {}
    out, stat = measure_process(test_name, process_func, model_input, spectrum, repetitions=repetitions, **kwargs)

    # 4. Format output and map back to unmasked pixels
    out = np.squeeze(out).reshape(-1)
    output_array[~mask] = out

    # 5. Check equivalence against the saved reference file
    check_equivalence(output_array, saved_path, test_name=test_name)

    return output_array, stat


def main():
    #Real data split into parts due to github limits are stitched first.
    output_path = "./resources/test_tile_512_512_125.img"
    split_parts = sorted([f for f in os.listdir('./resources') if f.startswith('test_tile_512_512_125_part')])
    # Only run this once to reconstruct
    if not os.path.exists(output_path) and split_parts:
        with open(output_path, 'wb') as outfile:
            for part in split_parts:
                with open(os.path.join("./resources/", part), 'rb') as infile:
                    outfile.write(infile.read())

        print(f"Reconstructed {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    #Use all 4 cores, wherever they can be used
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6


    parser = argparse.ArgumentParser(description="Compute ACE for a given hyperspectral image.")
    # Shape argument
    parser.add_argument("--precision", type=str_to_precision, default=np.float64,
                        help="Specify the precision type for floating point numbers. Options are 16, 32, or 64 (default is 64).")
    # Make hdr_path and methane_spectrum optional
    parser.add_argument("--hdr-path", type=str, nargs="?", default="resources/test_tile_512_512_125.hdr", help="Path to the hyperspectral HDR file.")
    parser.add_argument('--compute-original-filters', action='store_true', default=False, help='Set this flag to True (default is False) if you want to compute the unoptimized versions of the filters and see equivalence to the results of the optimized versions.')
    parser.add_argument('--repetitions', type=int, default=100, help='Number of repetitions for each benchmark measurement (default: 100).')
    parser.add_argument('--no-sleep', action='store_true', default=False, help='Disable sleep between measurements (default: sleep is enabled).')
    parser.add_argument('--channels', type=int, nargs='+', default=[10, 25, 35, 50, 72, 90, 100, 110, 125], help='List of channel counts to benchmark (default: 10 25 35 50 72 90 100 110 125).')
    parser.add_argument('--measure-morphological-baseline', action='store_true', default=False, help='Measure morphological baseline runtime (requires torch and kornia, default: False).')
    args = parser.parse_args()
    do_sleep = not args.no_sleep
    
    hyperspectral_img,  wavelengths, fwhm = load_hyperspectral_image(args.hdr_path)
    print("Initial_shape: ", hyperspectral_img.shape)
    
    
    mag1c_types = ["Original", "Tile-wise", "Tile-wise and Sampled"] 

    channels_list = args.channels
    stats = {}

    for channels in channels_list:
        print(f"Processing with {channels} channels...")
        hyperspectral_img_channel = hyperspectral_img[:, :, :channels]
        H, W, C = hyperspectral_img_channel.shape

        print(f"\n{'='*60}")
        print(f"  MAG1C (PyTorch) — {channels} channels")
        print(f"{'='*60}")
        for mag1c_type in mag1c_types:
            if do_sleep:
                sleep(500)
            print(f"Computing {mag1c_type} Mag1c (PyTorch) for {channels} channels...")
            output_metadata = {
                    "wavelength units": "nm",
                    "wavelength": wavelengths[:channels],
                    "fwhm": fwhm[:channels],
                }
            name = f"mag1c_t_{channels}"
            to_process_image = hyperspectral_img_channel if mag1c_type == "Original" else hyperspectral_img_channel.reshape(-1,1,channels)
            envi.save_image(
                f"{name}.hdr",
                to_process_image,
                shape=to_process_image.shape,
                interleave="bil",
                metadata=output_metadata,
                force=True,
            )
            mag1c_args = [sys.executable, "benchmark/mag1c_fork/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(300), str(2600), "--save-target-spectrum-centers", "--quiet", "--use-all-bands"]
            if mag1c_type == "Tile-wise and Sampled":
                mag1c_args += ["--sample", str(0.01)]
            if args.precision == np.float32:
                mag1c_args += ["--single"]

            mag1c_stats = measure_mag1c(mag1c_type, mag1c_args, repetitions=args.repetitions)
            stats[f"{mag1c_type}_{channels}"] = mag1c_stats

            output_hdr = f"{name}_ch4_cmfr.hdr"
            print(f"Looking for MAG1C output: {output_hdr}")
            if os.path.exists(output_hdr):
                ch4_img = spy.open_image(output_hdr).load()
                ch4_img = ch4_img.reshape((H, W, -1))[:,:,3]
                np.save(f"{name}_{mag1c_type}.npy", ch4_img)
                print(f"Saved {mag1c_type} output as {name}_{mag1c_type}.npy")
            else:
                print(f"MAG1C output {output_hdr} not found.")
                # List files that match the pattern to debug
                matching_files = [f for f in os.listdir("./") if f.startswith(name) and "ch4_cmfr" in f]
                print(f"Matching files found: {matching_files}")

        # Copy the methane spectrum for the current channel
        spectrum_copy_name = f"mag1c_spectrum_channel-{channels}.npy"
        if os.path.exists("mag1c_spectrum.npy"):
            os.rename("mag1c_spectrum.npy", spectrum_copy_name)
            print(f"Saved methane spectrum for {channels} channels as {spectrum_copy_name}")
        else:
            print("mag1c_spectrum.npy not found after MAG1C processing.")

        # Load the methane spectrum for the current channel
        methane_spectrum = np.load(f"mag1c_spectrum_channel-{channels}.npy").astype(args.precision)
        hyperspectral_img_reshaped = hyperspectral_img_channel.reshape(-1, channels)
        hyperspectral_img_reshaped = np.ascontiguousarray(hyperspectral_img_reshaped, dtype=args.precision)
        methane_spectrum = np.ascontiguousarray(methane_spectrum, dtype=args.precision)
        del to_process_image, output_metadata

        if do_sleep:
            sleep(500)
        ACE_optimized_results, stats[f"ACE_optimized_{channels}"] = measure_process("ACE_optimized", ACE_optimized, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=False)

        if do_sleep:
            sleep(500)
        ACE_optimized_addition_results, stats[f"ACE_optimized_addition_{channels}"] = measure_process("ACE_optimized_addition", ACE_optimized, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=True)

        if args.compute_original_filters:
            ACE_original_results, stats[f"ACE_original_{channels}"] = measure_process("ACE_original", ACE_original, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions)
            test_differences(ACE_original_results, ACE_optimized_addition_results)

        if do_sleep:
            sleep(500)
        MatchedFilterOptimized_results, stats[f"MatchedFilterOptimized_{channels}"] = measure_process("MatchedFilterOptimized", MatchedFilterOptimized, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=False)

        if do_sleep:
            sleep(500)
        MatchedFilterOptimized_addition_results, stats[f"MatchedFilterOptimized_addition_{channels}"] = measure_process("MatchedFilterOptimized_addition", MatchedFilterOptimized, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=True)
        if args.compute_original_filters:
            MatchedFilterOriginal_results, stats[f"MatchedFilterOriginal_{channels}"] = measure_process("MatchedFilterOriginal", MatchedFilterOriginal, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions)
            test_differences(MatchedFilterOriginal_results, MatchedFilterOptimized_addition_results)

        if do_sleep:
            sleep(500)
        CEM_optimized_results, stats[f"CEM_optimized_{channels}"] = measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions)

        if args.compute_original_filters:
            CEM_original_results, stats[f"CEM_original_{channels}"] = measure_process("CEM_original", CEM_original, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions)
            test_differences(CEM_original_results, CEM_optimized_results)

        """
        # Numpy versions of ACE, MatchedFilter, and CEM — kept for reference but not used

        from sped_up_filters import ACE_optimized_numpy, CEM_optimized_numpy, MatchedFilterOptimized_numpy

        if do_sleep:
            sleep(500)
        ACE_numpy_results, stats[f"ACE_numpy_{channels}"] = measure_process("ACE_numpy", ACE_optimized_numpy, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=False)

        if do_sleep:
            sleep(500)
        ACE_numpy_addition_results, stats[f"ACE_numpy_addition_{channels}"] = measure_process("ACE_numpy_addition", ACE_optimized_numpy, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=True)
        if args.compute_original_filters:
            test_differences(ACE_original_results, ACE_numpy_addition_results)

        if do_sleep:
            sleep(500)
        MatchedFilterNumpy_results, stats[f"MatchedFilterNumpy_{channels}"] = measure_process("MatchedFilterNumpy", MatchedFilterOptimized_numpy, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=False)

        if do_sleep:
            sleep(500)
        MatchedFilterNumpy_addition_results, stats[f"MatchedFilterNumpy_addition_{channels}"] = measure_process("MatchedFilterNumpy_addition", MatchedFilterOptimized_numpy, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions, use_addition=True)
        if args.compute_original_filters:
            test_differences(MatchedFilterOriginal_results, MatchedFilterNumpy_addition_results)

        if do_sleep:
            sleep(500)
        CEM_numpy_results, stats[f"CEM_numpy_{channels}"] = measure_process("CEM_numpy", CEM_optimized_numpy, hyperspectral_img_reshaped, methane_spectrum, repetitions=args.repetitions)
        if args.compute_original_filters:
            test_differences(CEM_original_results, CEM_numpy_results)
        """

        # MAG1C SAS measurement (Pure NumPy)
        print(f"\n{'='*60}")
        print(f"  MAG1C SAS (Pure NumPy) — {channels} channels")
        print(f"{'='*60}")
        if do_sleep:
            sleep(500)
        _, stats[f"mag1c_SAS_{channels}"] = measure_process("mag1c_SAS", mag1c_SAS, np.expand_dims(hyperspectral_img_reshaped, axis=0), methane_spectrum, SAS=True, repetitions=args.repetitions)

        # MAG1C Tile measurement (Pure NumPy)
        print(f"\n{'='*60}")
        print(f"  MAG1C Tile (Pure NumPy) — {channels} channels")
        print(f"{'='*60}")
        if do_sleep:
            sleep(500)
        _, stats[f"mag1c_tile_{channels}"] = measure_process("mag1c_tile", mag1c_tile, np.expand_dims(hyperspectral_img_reshaped, axis=0), methane_spectrum, repetitions=args.repetitions)

        # Morphological Baseline measurement
        if args.measure_morphological_baseline:
            import torch
            tensor = torch.tensor(CEM_optimized_results.reshape((1, 1, H, W)))
            threshold = 0.004
            if do_sleep:
                sleep(500)
            morphological_baseline_stats = measure_morphological_baseline(tensor, threshold, repetitions=args.repetitions)
            stats[f"Morphological_Baseline_{channels}"] = morphological_baseline_stats

    save_stats_to_csv(stats, "processing_times.csv", channels_list)

    # Use the last channel count for validation
    C = channels_list[-1]
    hyperspectral_img_channel = hyperspectral_img[:, :, :C]
    H, W = hyperspectral_img_channel.shape[0], hyperspectral_img_channel.shape[1]
    hyperspectral_img_reshaped = hyperspectral_img_channel.reshape(-1, C).astype(args.precision)
    methane_spectrum = np.load(f"mag1c_spectrum_channel-{C}.npy").astype(args.precision)

    # 1. Setup and Censor Mask
    censor_mask = get_censor_mask(hyperspectral_img_channel.reshape(-1, 1, C).astype(np.float64))
    print(f"{censor_mask.sum()} censored pixels detected.")

    # 2. Run mag1c_SAS
    print(f"\n{'='*60}")
    print(f"  Validating: NumPy vs PyTorch MAG1C outputs are equivalent")
    print(f"{'='*60}")
    saved_mag1c_sas_path = f"mag1c_t_{C}_Tile-wise and Sampled.npy"
    mag1c_sas_out, stats["mag1c_SAS"] = run_pipeline_and_validate(
        test_name="mag1c_SAS",
        process_func=mag1c_SAS,
        img_reshaped=hyperspectral_img_reshaped,
        mask=censor_mask,
        spectrum=methane_spectrum,
        nodata_val=NODATA,
        saved_path=saved_mag1c_sas_path,
        is_sas=True,
    )

    # 3. Run mag1c_tile
    saved_mag1c_tile_path = f"mag1c_t_{C}_Tile-wise.npy" 
    mag1c_tile_out, stats["mag1c_tile"] = run_pipeline_and_validate(
        test_name="mag1c_tile",
        process_func=mag1c_tile,
        img_reshaped=hyperspectral_img_reshaped,
        mask=censor_mask,
        spectrum=methane_spectrum,
        nodata_val=NODATA,
        saved_path=saved_mag1c_tile_path,
        is_sas=False,
    )

    for f in [f for f in os.listdir("./") if "mag1c_t_" in f]:
        os.remove(f)

if __name__ == "__main__":
    main()
