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

def test_differences(original_results, optimized_results, test=True):
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


def main():
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
    parser = argparse.ArgumentParser(description="Compute ACE for a given hyperspectral image.")
    # Shape argument
    parser.add_argument("--channels", type=int, default=50,
                        help="The number of bands used - default is 50.")
    parser.add_argument("--precision", type=str_to_precision, default=np.float64,
                        help="Specify the precision type for floating point numbers. Options are 16, 32, or 64 (default is 64).")
    # Make hdr_path and methane_spectrum optional
    parser.add_argument("--hdr-path", type=str, nargs="?", default=None, help="Path to the hyperspectral HDR file.")
    parser.add_argument('--compute-original-mag1c', action='store_true', default=False, help='Set this flag to True (default is False) if you want to compute the original column-wise mag1c.')

    args = parser.parse_args()
    
    C = args.channels  # Unpack the shape from arguments
    print(C)
    print(f"Loading hyperspectral image from {args.hdr_path}..., selected channels N: {C}")
    hyperspectral_img,  wavelengths, fwhm = load_hyperspectral_image(args.hdr_path)
    print("Initial_shape: ", hyperspectral_img.shape)
    hyperspectral_img = hyperspectral_img[:,:,:C]
    print("After channel selection: ", hyperspectral_img.shape)
    #To have kinda similar testing conditions, we have altered the mag1c time measurement to include only the sole filter function not preprocessing.
    mag1c_results = dict()
    print("Compute Original Mag1c: ", args.compute_original_mag1c)
    mag1c_types = ["Original"] if args.compute_original_mag1c else []
    mag1c_types += ["Tile-wise", "Tile-wise and Sampled"]
    for mag1c_type in mag1c_types:
        print(f"Computing {mag1c_type} Mag1c...")
        output_metadata = {
                "wavelength units": "nm",
                "wavelength": wavelengths[:C],
                "fwhm": fwhm[:C],
            }
        name = "mag1c_test_tile"
        to_process_image = hyperspectral_img if mag1c_type == "Original" else hyperspectral_img.reshape(-1,1,C)
        print(to_process_image.shape)
        envi.save_image(
            f"{name}.hdr",
            to_process_image,
            shape=to_process_image.shape,
            interleave="bil",
            metadata=output_metadata,
            force=True,
        )
        #The bands number selection is done in this scipr
        arg = ["python", "mag1c_fork/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(300), str(2600), "--save-target-spectrum-centers"]
        if mag1c_type == "Tile-wise and Sampled":
            arg += ["--sample", str(0.01)]
        if args.precision == np.float32:
            arg += ["--single"]
        try:
            result = subprocess.run(arg, capture_output=True, text=True, check=True)
            print("MAG1C Output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error running MAG1C:")
            print(e.stderr)
            continue
        mag1c_out = envi.open(f"{name}_ch4_cmfr.hdr", f"{name}_ch4_cmfr").load()[..., 3].squeeze()
        mag1c_out = np.clip(mag1c_out, 0, None)
        if mag1c_type != "Original":
            mag1c_out = mag1c_out.reshape((hyperspectral_img.shape[0], hyperspectral_img.shape[1]))
        mag1c_results[mag1c_type] = mag1c_out
    for f in [f for f in os.listdir("./") if name in f]:
        os.remove(f)
    if args.compute_original_mag1c:
        print("Original mag1c vs Tile-based mag1c:")
        test_differences(mag1c_results["Original"], mag1c_results["Tile-wise"])
        print("Original mag1c vs Sampled mag1c:")
        test_differences(mag1c_results["Original"], mag1c_results["Tile-wise and Sampled"])
    print("Tile-based mag1c vs Sampled mag1c:")
    test_differences(mag1c_results["Tile-wise"], mag1c_results["Tile-wise and Sampled"])
    

    # Load methane spectrum
    print(f"Loading methane spectrum for generated file from mag1c: mag1c_spectrum.npy...")
    methane_spectrum = np.load("mag1c_spectrum.npy").astype(args.precision)
    print(methane_spectrum.shape)
    hyperspectral_img = hyperspectral_img.squeeze().astype(args.precision)
    # Use random data generation if no file paths are provided
    hyperspectral_img_reshaped = hyperspectral_img.reshape(-1,methane_spectrum.shape[0]) 
    print(f"Computing with precision: float{args.precision}")
    #hyperspectral_img_reshaped = np.ascontiguousarray(hyperspectral_img_reshaped, dtype=args.precision)
    #methane_spectrum = np.ascontiguousarray(methane_spectrum, dtype=args.precision)
    del hyperspectral_img, mag1c_results, mag1c_out, wavelengths, fwhm

    ACE_original_results = measure_process("ACE_original", ACE_original, hyperspectral_img_reshaped, methane_spectrum)
    ACE_optimized_results = measure_process("ACE_optimized", ACE_optimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(ACE_original_results, ACE_optimized_results)

    MatchedFilterOriginal_results = measure_process("MatchedFilterOriginal", MatchedFilterOriginal, hyperspectral_img_reshaped, methane_spectrum)
    MatchedFilterOptimized_results = measure_process("MatchedFilterOptimized", MatchedFilterOptimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(MatchedFilterOriginal_results, MatchedFilterOptimized_results)

    CEM_original_results = measure_process("CEM_original", CEM_original, hyperspectral_img_reshaped, methane_spectrum)
    CEM_optimized_results = measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(CEM_original_results, CEM_optimized_results)
    

if __name__ == "__main__":
    main()

