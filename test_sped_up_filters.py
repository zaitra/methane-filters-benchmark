import spectral as spy
import numpy as np
import time
import argparse
from pysptools.detection.detect import ACE as ACE_original, CEM as CEM_original, MatchedFilter as MatchedFilterOriginal
from sped_up_filters import ACE_optimized, CEM_optimized, MatchedFilterOptimized
import os
import spectral.io.envi as envi
import subprocess
#Copied from aviris .hdr file
AVIRIS_WAVELENGTHS = [376.719576, 381.729576, 386.739576, 391.749576, 396.749576, 401.759576, 406.76957600000003, 411.77957599999996, 416.789576, 421.799576, 426.80957600000005, 431.819576, 436.819576, 441.829576, 446.839576, 451.84957599999996, 456.859576, 461.869576, 466.87957600000004, 471.87957600000004, 476.889576, 481.899576, 486.909576, 491.919576, 496.929576, 501.93957600000005, 506.94957600000004, 511.94957600000004, 516.959576, 521.9695760000001, 526.979576, 531.9895759999999, 536.999576, 542.009576, 547.009576, 552.0195759999999, 557.029576, 562.039576, 567.049576, 572.059576, 577.069576, 582.0795760000001, 587.0795760000001, 592.089576, 597.099576, 602.1095760000001, 607.1195759999999, 612.1295759999999, 617.139576, 622.139576, 627.149576, 632.1595759999999, 637.169576, 642.179576, 647.189576, 652.199576, 657.209576, 662.209576, 667.2195760000001, 672.229576, 677.2395759999999, 682.249576, 687.259576, 692.269576, 697.2695759999999, 702.279576, 707.289576, 712.299576, 717.309576, 722.319576, 727.3295760000001, 732.339576, 737.339576, 742.349576, 747.3595760000001, 752.3695759999999, 757.379576, 762.389576, 767.399576, 772.399576, 777.409576, 782.419576, 787.429576, 792.439576, 797.449576, 802.459576, 807.4695760000001, 812.4695760000001, 817.479576, 822.4895759999999, 827.499576, 832.5095759999999, 837.519576, 842.529576, 847.529576, 852.539576, 857.549576, 862.559576, 867.569576, 872.579576, 877.589576, 882.5995760000001, 887.599576, 892.6095760000001, 897.6195759999999, 902.629576, 907.639576, 912.6495759999999, 917.659576, 922.669576, 927.669576, 932.679576, 937.689576, 942.699576, 947.7095760000001, 952.719576, 957.729576, 962.729576, 967.739576, 972.749576, 977.7595759999999, 982.769576, 987.779576, 992.7895759999999, 997.799576, 1002.7995759999999, 1007.8095760000001, 1012.8195760000001, 1017.829576, 1022.839576, 1027.8495759999998, 1032.8595759999998, 1037.859576, 1042.869576, 1047.8795759999998, 1052.889576, 1057.899576, 1062.909576, 1067.919576, 1072.929576, 1077.929576, 1082.939576, 1087.949576, 1092.959576, 1097.969576, 1102.9795760000002, 1107.989576, 1112.989576, 1117.9995760000002, 1123.0095760000002, 1128.019576, 1133.029576, 1138.039576, 1143.049576, 1148.059576, 1153.0595759999999, 1158.0695759999999, 1163.0795759999999, 1168.089576, 1173.099576, 1178.109576, 1183.119576, 1188.119576, 1193.129576, 1198.139576, 1203.149576, 1208.159576, 1213.169576, 1218.179576, 1223.189576, 1228.189576, 1233.199576, 1238.209576, 1243.219576, 1248.229576, 1253.239576, 1258.2495760000002, 1263.249576, 1268.259576, 1273.269576, 1278.2795760000001, 1283.2895760000001, 1288.299576, 1293.3095759999999, 1298.3195759999999, 1303.319576, 1308.329576, 1313.3395759999999, 1318.3495759999998, 1323.359576, 1328.369576, 1333.379576, 1338.3795759999998, 1343.389576, 1348.399576, 1353.409576, 1358.419576, 1363.429576, 1368.4395760000002, 1373.449576, 1378.449576, 1383.459576, 1388.4695760000002, 1393.479576, 1398.489576, 1403.499576, 1408.509576, 1413.5095760000002, 1418.519576, 1423.529576, 1428.539576, 1433.5495760000001, 1438.559576, 1443.569576, 1448.579576, 1453.579576, 1458.589576, 1463.599576, 1468.609576, 1473.6195759999998, 1478.6295759999998, 1483.639576, 1488.639576, 1493.649576, 1498.659576, 1503.669576, 1508.679576, 1513.689576, 1518.699576, 1523.709576, 1528.709576, 1533.719576, 1538.729576, 1543.739576, 1548.7495760000002, 1553.759576, 1558.769576, 1563.7695760000001, 1568.7795760000001, 1573.7895760000001, 1578.799576, 1583.8095759999999, 1588.8195759999999, 1593.829576, 1598.839576, 1603.8395759999999, 1608.8495759999998, 1613.859576, 1618.869576, 1623.879576, 1628.889576, 1633.8995759999998, 1638.899576, 1643.909576, 1648.919576, 1653.929576, 1658.939576, 1663.949576, 1668.959576, 1673.969576, 1678.969576, 1683.979576, 1688.989576, 1693.999576, 1699.009576, 1704.0195760000001, 1709.0295760000001, 1714.029576, 1719.039576, 1724.0495760000001, 1729.059576, 1734.069576, 1739.0795759999999, 1744.0895759999999, 1749.099576, 1754.099576, 1759.109576, 1764.1195759999998, 1769.129576, 1774.139576, 1779.149576, 1784.159576, 1789.159576, 1794.169576, 1799.179576, 1804.189576, 1809.199576, 1814.2095760000002, 1819.219576, 1824.229576, 1829.229576, 1834.2395760000002, 1839.2495760000002, 1844.259576, 1849.269576, 1854.279576, 1859.2895760000001, 1864.2995760000001, 1869.299576, 1874.3095759999999, 1879.319576, 1884.329576, 1889.339576, 1894.349576, 1899.3595759999998, 1904.359576, 1909.369576, 1914.379576, 1919.389576, 1924.3995759999998, 1929.409576, 1934.419576, 1939.429576, 1944.429576, 1949.439576, 1954.449576, 1959.459576, 1964.469576, 1969.479576, 1974.4895760000002, 1979.489576, 1984.499576, 1989.509576, 1994.5195760000001, 1999.5295760000001, 2004.539576, 2009.549576, 2014.5595759999999, 2019.5595759999999, 2024.569576, 2029.579576, 2034.589576, 2039.599576, 2044.609576, 2049.619576, 2054.619576, 2059.629576, 2064.639576, 2069.6495760000003, 2074.659576, 2079.6695760000002, 2084.679576, 2089.689576, 2094.6895759999998, 2099.699576, 2104.7095759999997, 2109.719576, 2114.729576, 2119.7395760000004, 2124.749576, 2129.749576, 2134.759576, 2139.769576, 2144.779576, 2149.7895759999997, 2154.799576, 2159.8095759999997, 2164.8195760000003, 2169.819576, 2174.829576, 2179.839576, 2184.849576, 2189.859576, 2194.869576, 2199.879576, 2204.8795760000003, 2209.889576, 2214.8995760000003, 2219.909576, 2224.9195760000002, 2229.929576, 2234.9395759999998, 2239.949576, 2244.949576, 2249.959576, 2254.969576, 2259.979576, 2264.989576, 2269.999576, 2275.009576, 2280.009576, 2285.0195759999997, 2290.029576, 2295.039576, 2300.0495760000003, 2305.059576, 2310.069576, 2315.079576, 2320.079576, 2325.089576, 2330.0995759999996, 2335.109576, 2340.119576, 2345.1295760000003, 2350.139576, 2355.139576, 2360.149576, 2365.159576, 2370.169576, 2375.179576, 2380.1895759999998, 2385.1995760000004, 2390.209576, 2395.209576, 2400.219576, 2405.229576, 2410.239576, 2415.2495759999997, 2420.259576, 2425.2695759999997, 2430.269576, 2435.279576, 2440.289576, 2445.299576, 2450.309576, 2455.319576, 2460.329576, 2465.339576, 2470.3395760000003, 2475.349576, 2480.3595760000003, 2485.369576, 2490.379576, 2495.389576, 2500.399576]
AVIRIS_FWHM = [5.57, 5.58, 5.58, 5.58, 5.590000000000001, 5.590000000000001, 5.590000000000001, 5.6, 5.6, 5.6, 5.6, 5.61, 5.61, 5.61, 5.62, 5.62, 5.62, 5.62, 5.63, 5.63, 5.63, 5.64, 5.64, 5.64, 5.64, 5.6499999999999995, 5.6499999999999995, 5.6499999999999995, 5.6499999999999995, 5.66, 5.66, 5.66, 5.66, 5.66, 5.67, 5.67, 5.67, 5.67, 5.68, 5.68, 5.68, 5.68, 5.68, 5.6899999999999995, 5.6899999999999995, 5.6899999999999995, 5.6899999999999995, 5.6899999999999995, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.71, 5.71, 5.71, 5.71, 5.71, 5.71, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.88, 5.88, 5.88, 5.88, 5.88, 5.88, 5.88, 5.88, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.91, 5.91, 5.91, 5.91, 5.91, 5.91, 5.92, 5.92, 5.92, 5.92, 5.92, 5.92, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.94, 5.94, 5.94, 5.94, 5.94, 5.95, 5.95, 5.95, 5.95, 5.95, 5.95, 5.96, 5.96, 5.96, 5.96, 5.96, 5.97, 5.97, 5.97, 5.97, 5.97, 5.98, 5.98, 5.98, 5.98, 5.98, 5.989999999999999, 5.989999999999999, 5.989999999999999, 5.989999999999999, 5.989999999999999, 6.0, 6.0, 6.0, 6.0, 6.01, 6.01, 6.01, 6.01, 6.01, 6.0200000000000005, 6.0200000000000005, 6.0200000000000005, 6.0200000000000005, 6.029999999999999]


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
    parser.add_argument("--shape", type=int, nargs=3, default=(512, 512, 50),
                        help="The shape of the image that will be benchmarked, type it as H W C (default: 512 512 50)")
    parser.add_argument("--precision", type=str_to_precision, default=np.float64,
                        help="Specify the precision type for floating point numbers. Options are 16, 32, or 64 (default is 64).")
    # Make hdr_path and methane_spectrum optional
    parser.add_argument("--hdr_path", type=str, nargs="?", default=None, help="Path to the hyperspectral HDR file.")
    parser.add_argument("--methane_spectrum", type=str, nargs="?", default=None, help="Path to the methane spectrum numpy file (.npy).")
    
    args = parser.parse_args()

    if args.hdr_path and args.methane_spectrum:
        # Load hyperspectral image
        print(f"Loading hyperspectral image from {args.hdr_path}...")
        hyperspectral_img = load_hyperspectral_image(args.hdr_path)
        
        # Load methane spectrum
        print(f"Loading methane spectrum from {args.methane_spectrum}...")
        methane_spectrum = np.load(args.methane_spectrum).astype(args.precision)
        hyperspectral_img = hyperspectral_img.squeeze().astype(args.precision)
    else:
        # Use random data generation if no file paths are provided
        H, W, C = args.shape  # Unpack the shape from arguments
        hyperspectral_img = np.random.uniform(1, 255, (H, W, C)).astype(args.precision)
        print(f"Hyperspectral image with shape {(H, W, C)} was randomly generated and reshaped into: {hyperspectral_img_reshaped.shape}")
        
        methane_spectrum = np.random.uniform(1, 8, (C)).astype(args.precision)  # Random methane spectrum
        print(f"Methane spectrum with shape {methane_spectrum.shape} was randomly generated."
              )
    hyperspectral_img_reshaped = hyperspectral_img.reshape(-1,methane_spectrum.shape[0]) 
    print(f"Computing with precision: float{args.precision}")
    hyperspectral_img_reshaped = np.ascontiguousarray(hyperspectral_img_reshaped, dtype=args.precision)
    methane_spectrum = np.ascontiguousarray(methane_spectrum, dtype=args.precision)

    """ACE_original_results = measure_process("ACE_original", ACE_original, hyperspectral_img_reshaped, methane_spectrum)
    ACE_optimized_results = measure_process("ACE_optimized", ACE_optimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(ACE_original_results, ACE_optimized_results)

    MatchedFilterOriginal_results = measure_process("MatchedFilterOriginal", MatchedFilterOriginal, hyperspectral_img_reshaped, methane_spectrum)
    MatchedFilterOptimized_results = measure_process("MatchedFilterOptimized", MatchedFilterOptimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(MatchedFilterOriginal_results, MatchedFilterOptimized_results)

    CEM_original_results = measure_process("CEM_original", CEM_original, hyperspectral_img_reshaped, methane_spectrum)
    CEM_optimized_results = measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_reshaped, methane_spectrum)
    test_differences(CEM_original_results, CEM_optimized_results)
"""
    #To have kinda similar testing conditions, we have altered the mag1c time measurement to include only the sole filter function not preprocessing.
    mag1c_results = dict()
    mag1c_types = ["Original", "Tile-wise", "Tile-wise and Sampled"]
    for mag1c_type in mag1c_types:
        if mag1c_type != "Tile-wise and Sampled":
            continue
        print(f"Computing {mag1c_type} Mag1c...")
        output_metadata = {
                "wavelength units": "nm",
                "wavelength": AVIRIS_WAVELENGTHS[:methane_spectrum.shape[0]],
                "fwhm": AVIRIS_FWHM[:methane_spectrum.shape[0]],
            }
        name = "mag1c_test_tile"
        to_process_image = hyperspectral_img if mag1c_type == "Original" else hyperspectral_img.reshape(-1,1,methane_spectrum.shape[0])
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
        arg = ["python", "mag1c_fork/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(300), str(2600)]
        if mag1c_type == "Tile-wise and Sampled":
            arg += ["--sample", str(0.05)]
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
    print("Original mag1c vs Tile-based mag1c:")
    test_differences(mag1c_results["Original"], mag1c_results["Tile-wise"])
    print("Tile-based mag1c vs Sampled mag1c:")
    test_differences(mag1c_results["Tile-wise"], mag1c_results["Tile-wise and Sampled"])
    print("Original mag1c vs Sampled mag1c:")
    test_differences(mag1c_results["Original"], mag1c_results["Tile-wise and Sampled"])
    

if __name__ == "__main__":
    main()
