import spectral as spy
import numpy as np
import time
import argparse
from pysptools.detection.detect import ACE as ACE_original, CEM as CEM_original, MatchedFilter as MatchedFilterOriginal
from sped_up_filters import ACE_optimized, CEM_optimized, MatchedFilterOptimized
import os
import spectral.io.envi as envi
import subprocess
import sys


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
    parser.add_argument("--channels", type=int, default=50,
                        help="The number of bands used - default is 50.")
    parser.add_argument("--precision", type=str_to_precision, default=np.float64,
                        help="Specify the precision type for floating point numbers. Options are 16, 32, or 64 (default is 64).")
    # Make hdr_path and methane_spectrum optional
    parser.add_argument("--hdr-path", type=str, nargs="?", default="resources/test_tile_512_512_125.hdr", help="Path to the hyperspectral HDR file.")
    parser.add_argument('--compute-original-mag1c', action='store_true', default=False, help='Set this flag to True (default is False) if you want to compute the original column-wise mag1c.')
    parser.add_argument('--compute-original-filters', action='store_true', default=False, help='Set this flag to True (default is False) if you want to compute the unoptimized versions of the filters.')
    args = parser.parse_args()
    
    C = args.channels  # Unpack the shape from arguments
    print(f"Loading hyperspectral image from {args.hdr_path}..., selected channels N: {C}")
    hyperspectral_img,  wavelengths, fwhm = load_hyperspectral_image(args.hdr_path)
    print("Initial_shape: ", hyperspectral_img.shape)
    hyperspectral_img = hyperspectral_img[:,:,:C]
    print("After channel selection: ", hyperspectral_img.shape)
    
    
    #To have kinda similar testing conditions, we have altered the mag1c time measurement to include only the sole filter function not preprocessing.
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
        envi.save_image(
            f"{name}.hdr",
            to_process_image,
            shape=to_process_image.shape,
            interleave="bil",
            metadata=output_metadata,
            force=True,
        )
        if args.compute_original_mag1c:
            del to_process_image, hyperspectral_img, wavelengths, fwhm, output_metadata
        #The bands number selection is done in this script, so we command mag1c to use all available 
        arg = [sys.executable, "benchmark/mag1c_fork/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(300), str(2600), "--save-target-spectrum-centers", "--quiet"]
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
        if args.compute_original_mag1c:
            print("Original Mag1c was computed, other filters are not due to memory constraint.")
            for f in [f for f in os.listdir("./") if name in f]:
                os.remove(f)
            return

    # Load methane spectrum
    print(f"Loading methane spectrum for generated file from mag1c: mag1c_spectrum.npy...")
    methane_spectrum = np.load("mag1c_spectrum.npy").astype(args.precision)
    hyperspectral_img = hyperspectral_img.squeeze().astype(args.precision)
    hyperspectral_img_reshaped = hyperspectral_img.reshape(-1,methane_spectrum.shape[0]) 

    print(f"Computing with precision: float{args.precision}")
    hyperspectral_img_reshaped = np.ascontiguousarray(hyperspectral_img_reshaped, dtype=args.precision)
    methane_spectrum = np.ascontiguousarray(methane_spectrum, dtype=args.precision)
    del hyperspectral_img, wavelengths, fwhm, to_process_image, output_metadata

    ACE_optimized_results = measure_process("ACE_optimized", ACE_optimized, hyperspectral_img_reshaped, methane_spectrum)
    if args.compute_original_filters:
        ACE_original_results = measure_process("ACE_original", ACE_original, hyperspectral_img_reshaped, methane_spectrum)
        test_differences(ACE_original_results, ACE_optimized_results)

    MatchedFilterOptimized_results = measure_process("MatchedFilterOptimized", MatchedFilterOptimized, hyperspectral_img_reshaped, methane_spectrum)
    if args.compute_original_filters:
        MatchedFilterOriginal_results = measure_process("MatchedFilterOriginal", MatchedFilterOriginal, hyperspectral_img_reshaped, methane_spectrum)
        test_differences(MatchedFilterOriginal_results, MatchedFilterOptimized_results)

    CEM_optimized_results = measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_reshaped, methane_spectrum)
    if args.compute_original_filters:
        CEM_original_results = measure_process("CEM_original", CEM_original, hyperspectral_img_reshaped, methane_spectrum)
        test_differences(CEM_original_results, CEM_optimized_results)
    for f in [f for f in os.listdir("./") if "mag1c" in f]:
        os.remove(f)

    import torch
    from kornia.morphology import dilation as kornia_dilation
    from kornia.morphology import erosion as kornia_erosion

    kernel_torch = torch.nn.Parameter(torch.from_numpy(np.array([[0, 1, 0],
                                                                [1, 1, 1],
                                                                [0, 1, 0]])).float(), requires_grad=False)

    def binary_opening(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        eroded = torch.clamp(kornia_erosion(x.float(), kernel), 0, 1) > 0
        return torch.clamp(kornia_dilation(eroded.float(), kernel), 0, 1) > 0

    def apply_threshold(pred: torch.Tensor, threshold) -> torch.Tensor:
        mag1c_thresholded = (pred > threshold)

        # https://programtalk.com/python-more-examples/kornia.morphology.dilation.bool/
        return binary_opening(mag1c_thresholded, kernel_torch).long()

    tensor = torch.tensor(CEM_optimized_results.reshape((1,1,512,512)))
    start_time = time.time()

    # Compute threshold
    result = apply_threshold(tensor, 0.004)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Computation of Morphological Baseline Done! Processing time: {elapsed_time:.4f} seconds")
    
    

if __name__ == "__main__":
    main()

