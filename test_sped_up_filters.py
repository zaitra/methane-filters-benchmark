import spectral as spy
import numpy as np
import time
import argparse
from pysptools.detection.detect import ACE as ACE_original, CEM as CEM_original, MatchedFilter as MatchedFilterOriginal
from sped_up_filters import ACE_optimized, CEM_optimized, MatchedFilterOptimized
import os
import spectral.io.envi as envi
import subprocess


def load_hyperspectral_image(hdr_path):
    """Load hyperspectral image using Spectral Python (SPy)."""
    img = spy.open_image(hdr_path).load()
    return img

def measure_process(name, function, hyperspectral_img, methane_spectrum):
    try:
        print(f"Computing {name}...")
        start_time = time.time()

        # Compute ACE
        result = function(hyperspectral_img, methane_spectrum)

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{name} Computation Done! Processing time: {elapsed_time:.4f} seconds")
        return result
    except ValueError:
        print("Error during computation, returning array of zeros")
        return np.zeros((hyperspectral_img.shape[0]))

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
    # Shape argument
    parser.add_argument("--shape", type=int, nargs=3, default=(512, 512, 50),
                        help="The shape of the image that will be benchmarked, type it as H W C (default: 512 512 50)")
    parser.add_argument("--precision", type=str_to_precision, default=np.float64,
                        help="Specify the precision type for floating point numbers. Options are 16, 32, or 64 (default is 64).")
    
    args = parser.parse_args()

    # Use random data generation if no file paths are provided
    H, W, C = args.shape  # Unpack the shape from arguments
    hyperspectral_img = np.random.rand(H, W, C).astype(args.precision)
    hyperspectral_img_reshaped = hyperspectral_img.reshape(-1,C)  # Generate random hyperspectral image
    print(f"Hyperspectral image with shape {(H, W, C)} was randomly generated and reshaped into: {hyperspectral_img_reshaped.shape}")
    
    methane_spectrum = np.random.rand(C).astype(args.precision)  # Random methane spectrum
    print(f"Methane spectrum with shape {methane_spectrum.shape} was randomly generated.")

    print(f"Computing with precision: float{args.precision}")
    hyperspectral_img_reshaped = np.ascontiguousarray(hyperspectral_img_reshaped, dtype=args.precision)
    methane_spectrum = np.ascontiguousarray(methane_spectrum, dtype=args.precision)

    ACE_original_results = measure_process("ACE_original", ACE_original, hyperspectral_img_reshaped, methane_spectrum)
    ACE_optimized_results = measure_process("ACE_optimized", ACE_optimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(ACE_original_results, ACE_optimized_results)

    MatchedFilterOriginal_results = measure_process("MatchedFilterOriginal", MatchedFilterOriginal, hyperspectral_img_reshaped, methane_spectrum)
    MatchedFilterOptimized_results = measure_process("MatchedFilterOptimized", MatchedFilterOptimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(MatchedFilterOriginal_results, MatchedFilterOptimized_results)

    CEM_original_results = measure_process("CEM_original", CEM_original, hyperspectral_img_reshaped, methane_spectrum)
    CEM_optimized_results = measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(CEM_original_results, CEM_optimized_results)

    #To have kinda similar testing conditions, we have altered the mag1c time measurement to include only the sole filter function not preprocessing.
    for mag1c_type in ["Original", "Tile-wise", "Tile-wise and Sampled"]:
        print(f"Computing {mag1c_type} Mag1c...")
        output_metadata = {
                "wavelength units": "nm",
                "wavelength": np.unique(np.random.uniform(400, 2500, C * 10))[:C].astype(args.precision).tolist(),
                "fwhm": np.unique(np.random.uniform(1, 8, C * 10))[:C].astype(args.precision).tolist(),
            }
        name = "mag1c_test_tile"
        to_process_image = hyperspectral_img if mag1c_type == "Original" else hyperspectral_img.reshape(-1,1,C)
        envi.save_image(
            f"{name}.hdr",
            to_process_image,
            shape=to_process_image.shape,
            interleave="bil",
            metadata=output_metadata,
            force=True,
        )
        #The bands number selection is done in this scipr
        args = ["python", "mag1c_fork/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(300), str(2600)]
        if mag1c_type == "Tile-wise and Sampled":
            args += ["--sample", str(0.01)]
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            print("MAG1C Output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error running MAG1C:")
            print(e.stderr)
    for f in [f for f in os.listdir("./") if name in f]:
        os.remove(f)

if __name__ == "__main__":
    main()
