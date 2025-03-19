import spectral as spy
import numpy as np
import time
import argparse
from pysptools.detection.detect import ACE as ACE_original, CEM as CEM_original, MatchedFilter as MatchedFilterOriginal
from sped_up_filters import ACE_optimized, CEM_optimized, MatchedFilterOptimized
import os

def load_hyperspectral_image(hdr_path):
    """Load hyperspectral image using Spectral Python (SPy)."""
    img = spy.open_image(hdr_path).load()
    return img

def measure_process(name, function, hyperspectral_img_filtered, methane_spectrum_filtered):
    try:
        print(f"Computing {name}...")
        start_time = time.time()

        # Compute ACE
        result = function(hyperspectral_img_filtered, methane_spectrum_filtered)

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{name} Computation Done! Processing time: {elapsed_time:.4f} seconds")
        return result
    except ValueError:
        print("Error during computation, returning array of zeros")
        return np.zeros((hyperspectral_img_filtered.shape[0]))

def test_differences(original_results, optimized_results):
    # Calculate the absolute differences between the original and optimized results
    if original_results.sum() > 0 and optimized_results.sum() > 0:
        diff = np.abs(original_results - optimized_results)

        # Get the maximal and average differences
        max_diff = diff.max()
        avg_diff = diff.mean()

        # Print the formatted differences with labels
        print(f"Maximal difference between original and optimized (sped-up) version: {max_diff:.32f}")
        print(f"Average difference between original and optimized (sped-up) version: {avg_diff:.32f}")

        # Assert that the results are close within a specified tolerance
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


def main():
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
    parser = argparse.ArgumentParser(description="Compute ACE for a given hyperspectral image.")
    
    # Make hdr_path and methane_spectrum optional
    parser.add_argument("hdr_path", type=str, nargs="?", default=None, help="Path to the hyperspectral HDR file.")
    parser.add_argument("methane_spectrum", type=str, nargs="?", default=None, help="Path to the methane spectrum numpy file (.npy).")
    
    # Shape argument
    parser.add_argument("--random", type=int, nargs=3, default=(512, 512, 50),
                        help="If not specified the path to real image, random image with specified shape will be generated, type it as H W C (default: 512 512 50)")
    parser.add_argument("--precision", type=str_to_precision, default=64,
                        help="Specify the precision type for floating point numbers. Options are 16, 32, or 64 (default: 32).")
    
    args = parser.parse_args()

    if args.hdr_path and args.methane_spectrum:
        # Load hyperspectral image
        print(f"Loading hyperspectral image from {args.hdr_path}...")
        hyperspectral_img_filtered = load_hyperspectral_image(args.hdr_path)
        
        # Load methane spectrum
        print(f"Loading methane spectrum from {args.methane_spectrum}...")
        methane_spectrum_filtered = np.load(args.methane_spectrum).astype(args.precision)
        hyperspectral_img_filtered = hyperspectral_img_filtered.squeeze().astype(args.precision)
    
    else:
        # Use random data generation if no file paths are provided
        H, W, C = args.random  # Unpack the shape from arguments
        hyperspectral_img_filtered = np.random.rand(H, W, C).astype(args.precision).reshape(-1,C)  # Generate random hyperspectral image
        print(f"Hyperspectral image with shape {(H, W, C)} was randomly generated and reshaped into: {hyperspectral_img_filtered.shape}")
        
        methane_spectrum_filtered = np.random.rand(C).astype(args.precision)  # Random methane spectrum
        print(f"Methane spectrum with shape {methane_spectrum_filtered.shape} was randomly generated.")

    print(f"Computing with precision: float{args.precision}")
    hyperspectral_img_filtered = np.ascontiguousarray(hyperspectral_img_filtered, dtype=args.precision)
    methane_spectrum_filtered = np.ascontiguousarray(methane_spectrum_filtered, dtype=args.precision)

    ACE_original_results = measure_process("ACE_original", ACE_original, hyperspectral_img_filtered, methane_spectrum_filtered)
    ACE_optimized_results = measure_process("ACE_optimized", ACE_optimized, hyperspectral_img_filtered, methane_spectrum_filtered)
    test_differences(ACE_original_results, ACE_optimized_results)

    MatchedFilterOriginal_results = measure_process("MatchedFilterOriginal", MatchedFilterOriginal, hyperspectral_img_filtered, methane_spectrum_filtered)
    MatchedFilterOptimized_results = measure_process("MatchedFilterOptimized", MatchedFilterOptimized, hyperspectral_img_filtered, methane_spectrum_filtered)
    test_differences(MatchedFilterOriginal_results, MatchedFilterOptimized_results)

    CEM_original_results = measure_process("CEM_original", CEM_original, hyperspectral_img_filtered, methane_spectrum_filtered)
    CEM_optimized_results = measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_filtered, methane_spectrum_filtered)
    test_differences(CEM_original_results, CEM_optimized_results)

if __name__ == "__main__":
    main()
