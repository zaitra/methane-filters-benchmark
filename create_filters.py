import os
import pandas as pd
import numpy as np
import tifffile as tiff
from pysptools.detection.detect import ACE as ACE_original, MatchedFilter as MatchedFilterOriginal, CEM as CEM_original
from sped_up_filters import ACE_optimized, MatchedFilterOptimized, CEM_optimized
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import subprocess
import spectral.io.envi as envi
import hashlib
import matplotlib.pyplot as plt
import rasterio

#Copied from aviris .hdr file
AVIRIS_WAVELENGTHS = [376.719576, 381.729576, 386.739576, 391.749576, 396.749576, 401.759576, 406.76957600000003, 411.77957599999996, 416.789576, 421.799576, 426.80957600000005, 431.819576, 436.819576, 441.829576, 446.839576, 451.84957599999996, 456.859576, 461.869576, 466.87957600000004, 471.87957600000004, 476.889576, 481.899576, 486.909576, 491.919576, 496.929576, 501.93957600000005, 506.94957600000004, 511.94957600000004, 516.959576, 521.9695760000001, 526.979576, 531.9895759999999, 536.999576, 542.009576, 547.009576, 552.0195759999999, 557.029576, 562.039576, 567.049576, 572.059576, 577.069576, 582.0795760000001, 587.0795760000001, 592.089576, 597.099576, 602.1095760000001, 607.1195759999999, 612.1295759999999, 617.139576, 622.139576, 627.149576, 632.1595759999999, 637.169576, 642.179576, 647.189576, 652.199576, 657.209576, 662.209576, 667.2195760000001, 672.229576, 677.2395759999999, 682.249576, 687.259576, 692.269576, 697.2695759999999, 702.279576, 707.289576, 712.299576, 717.309576, 722.319576, 727.3295760000001, 732.339576, 737.339576, 742.349576, 747.3595760000001, 752.3695759999999, 757.379576, 762.389576, 767.399576, 772.399576, 777.409576, 782.419576, 787.429576, 792.439576, 797.449576, 802.459576, 807.4695760000001, 812.4695760000001, 817.479576, 822.4895759999999, 827.499576, 832.5095759999999, 837.519576, 842.529576, 847.529576, 852.539576, 857.549576, 862.559576, 867.569576, 872.579576, 877.589576, 882.5995760000001, 887.599576, 892.6095760000001, 897.6195759999999, 902.629576, 907.639576, 912.6495759999999, 917.659576, 922.669576, 927.669576, 932.679576, 937.689576, 942.699576, 947.7095760000001, 952.719576, 957.729576, 962.729576, 967.739576, 972.749576, 977.7595759999999, 982.769576, 987.779576, 992.7895759999999, 997.799576, 1002.7995759999999, 1007.8095760000001, 1012.8195760000001, 1017.829576, 1022.839576, 1027.8495759999998, 1032.8595759999998, 1037.859576, 1042.869576, 1047.8795759999998, 1052.889576, 1057.899576, 1062.909576, 1067.919576, 1072.929576, 1077.929576, 1082.939576, 1087.949576, 1092.959576, 1097.969576, 1102.9795760000002, 1107.989576, 1112.989576, 1117.9995760000002, 1123.0095760000002, 1128.019576, 1133.029576, 1138.039576, 1143.049576, 1148.059576, 1153.0595759999999, 1158.0695759999999, 1163.0795759999999, 1168.089576, 1173.099576, 1178.109576, 1183.119576, 1188.119576, 1193.129576, 1198.139576, 1203.149576, 1208.159576, 1213.169576, 1218.179576, 1223.189576, 1228.189576, 1233.199576, 1238.209576, 1243.219576, 1248.229576, 1253.239576, 1258.2495760000002, 1263.249576, 1268.259576, 1273.269576, 1278.2795760000001, 1283.2895760000001, 1288.299576, 1293.3095759999999, 1298.3195759999999, 1303.319576, 1308.329576, 1313.3395759999999, 1318.3495759999998, 1323.359576, 1328.369576, 1333.379576, 1338.3795759999998, 1343.389576, 1348.399576, 1353.409576, 1358.419576, 1363.429576, 1368.4395760000002, 1373.449576, 1378.449576, 1383.459576, 1388.4695760000002, 1393.479576, 1398.489576, 1403.499576, 1408.509576, 1413.5095760000002, 1418.519576, 1423.529576, 1428.539576, 1433.5495760000001, 1438.559576, 1443.569576, 1448.579576, 1453.579576, 1458.589576, 1463.599576, 1468.609576, 1473.6195759999998, 1478.6295759999998, 1483.639576, 1488.639576, 1493.649576, 1498.659576, 1503.669576, 1508.679576, 1513.689576, 1518.699576, 1523.709576, 1528.709576, 1533.719576, 1538.729576, 1543.739576, 1548.7495760000002, 1553.759576, 1558.769576, 1563.7695760000001, 1568.7795760000001, 1573.7895760000001, 1578.799576, 1583.8095759999999, 1588.8195759999999, 1593.829576, 1598.839576, 1603.8395759999999, 1608.8495759999998, 1613.859576, 1618.869576, 1623.879576, 1628.889576, 1633.8995759999998, 1638.899576, 1643.909576, 1648.919576, 1653.929576, 1658.939576, 1663.949576, 1668.959576, 1673.969576, 1678.969576, 1683.979576, 1688.989576, 1693.999576, 1699.009576, 1704.0195760000001, 1709.0295760000001, 1714.029576, 1719.039576, 1724.0495760000001, 1729.059576, 1734.069576, 1739.0795759999999, 1744.0895759999999, 1749.099576, 1754.099576, 1759.109576, 1764.1195759999998, 1769.129576, 1774.139576, 1779.149576, 1784.159576, 1789.159576, 1794.169576, 1799.179576, 1804.189576, 1809.199576, 1814.2095760000002, 1819.219576, 1824.229576, 1829.229576, 1834.2395760000002, 1839.2495760000002, 1844.259576, 1849.269576, 1854.279576, 1859.2895760000001, 1864.2995760000001, 1869.299576, 1874.3095759999999, 1879.319576, 1884.329576, 1889.339576, 1894.349576, 1899.3595759999998, 1904.359576, 1909.369576, 1914.379576, 1919.389576, 1924.3995759999998, 1929.409576, 1934.419576, 1939.429576, 1944.429576, 1949.439576, 1954.449576, 1959.459576, 1964.469576, 1969.479576, 1974.4895760000002, 1979.489576, 1984.499576, 1989.509576, 1994.5195760000001, 1999.5295760000001, 2004.539576, 2009.549576, 2014.5595759999999, 2019.5595759999999, 2024.569576, 2029.579576, 2034.589576, 2039.599576, 2044.609576, 2049.619576, 2054.619576, 2059.629576, 2064.639576, 2069.6495760000003, 2074.659576, 2079.6695760000002, 2084.679576, 2089.689576, 2094.6895759999998, 2099.699576, 2104.7095759999997, 2109.719576, 2114.729576, 2119.7395760000004, 2124.749576, 2129.749576, 2134.759576, 2139.769576, 2144.779576, 2149.7895759999997, 2154.799576, 2159.8095759999997, 2164.8195760000003, 2169.819576, 2174.829576, 2179.839576, 2184.849576, 2189.859576, 2194.869576, 2199.879576, 2204.8795760000003, 2209.889576, 2214.8995760000003, 2219.909576, 2224.9195760000002, 2229.929576, 2234.9395759999998, 2239.949576, 2244.949576, 2249.959576, 2254.969576, 2259.979576, 2264.989576, 2269.999576, 2275.009576, 2280.009576, 2285.0195759999997, 2290.029576, 2295.039576, 2300.0495760000003, 2305.059576, 2310.069576, 2315.079576, 2320.079576, 2325.089576, 2330.0995759999996, 2335.109576, 2340.119576, 2345.1295760000003, 2350.139576, 2355.139576, 2360.149576, 2365.159576, 2370.169576, 2375.179576, 2380.1895759999998, 2385.1995760000004, 2390.209576, 2395.209576, 2400.219576, 2405.229576, 2410.239576, 2415.2495759999997, 2420.259576, 2425.2695759999997, 2430.269576, 2435.279576, 2440.289576, 2445.299576, 2450.309576, 2455.319576, 2460.329576, 2465.339576, 2470.3395760000003, 2475.349576, 2480.3595760000003, 2485.369576, 2490.379576, 2495.389576, 2500.399576]
AVIRIS_FWHM = [5.57, 5.58, 5.58, 5.58, 5.590000000000001, 5.590000000000001, 5.590000000000001, 5.6, 5.6, 5.6, 5.6, 5.61, 5.61, 5.61, 5.62, 5.62, 5.62, 5.62, 5.63, 5.63, 5.63, 5.64, 5.64, 5.64, 5.64, 5.6499999999999995, 5.6499999999999995, 5.6499999999999995, 5.6499999999999995, 5.66, 5.66, 5.66, 5.66, 5.66, 5.67, 5.67, 5.67, 5.67, 5.68, 5.68, 5.68, 5.68, 5.68, 5.6899999999999995, 5.6899999999999995, 5.6899999999999995, 5.6899999999999995, 5.6899999999999995, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.71, 5.71, 5.71, 5.71, 5.71, 5.71, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.720000000000001, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.7299999999999995, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.760000000000001, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.77, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.78, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.8100000000000005, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.819999999999999, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.84, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.8500000000000005, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.859999999999999, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.87, 5.88, 5.88, 5.88, 5.88, 5.88, 5.88, 5.88, 5.88, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.890000000000001, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.8999999999999995, 5.91, 5.91, 5.91, 5.91, 5.91, 5.91, 5.92, 5.92, 5.92, 5.92, 5.92, 5.92, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.930000000000001, 5.94, 5.94, 5.94, 5.94, 5.94, 5.95, 5.95, 5.95, 5.95, 5.95, 5.95, 5.96, 5.96, 5.96, 5.96, 5.96, 5.97, 5.97, 5.97, 5.97, 5.97, 5.98, 5.98, 5.98, 5.98, 5.98, 5.989999999999999, 5.989999999999999, 5.989999999999999, 5.989999999999999, 5.989999999999999, 6.0, 6.0, 6.0, 6.0, 6.01, 6.01, 6.01, 6.01, 6.01, 6.0200000000000005, 6.0200000000000005, 6.0200000000000005, 6.0200000000000005, 6.029999999999999]

def hash_string_to_short(input_string):
    # Use SHA-256 to hash the string (returns a 64-character hex string)
    hashed = hashlib.sha256(input_string.encode()).hexdigest()
    # Return the first 8 characters of the hash as a short version
    return hashed[:8]

def save_tiff(array, metadata, path_to_save):
    # Update metadata for saving label as TIFF
    """metadata.update({
        "count": 1,  # Single-band
        "dtype": array.dtype,
        "compress": "lzw"  # Optional compression
    })"""

    # Save label as GeoTIFF
    with rasterio.open(path_to_save, "w", **metadata) as dst:
        dst.write(array, 1)  # Write to the first (and only) band

def save_all_reshaped_files(files, filepaths, filetypes, metadata, shape):
    for file, filepath, filetype in zip(files, filepaths, filetypes):
        save_tiff(file.reshape(shape).astype(filetype), metadata, filepath)

def create_empty_filter(hyperspectral_image, dtype):
    return np.zeros((hyperspectral_image.shape[0], hyperspectral_image.shape[1]), dtype=dtype)

def data_sanity_check(df,input_data_path):
    """Sanity check if all data was downloaded correctly"""
    STARCOP_BANDS_TEMP = []
    for idx, row in tqdm(df.iterrows()):
        tile_id = row["id"]
        tile_input_folder = os.path.join(input_data_path, tile_id)
        STARCOP_BANDS = [f for f in sorted(os.listdir(tile_input_folder)) if f.startswith("TOA_AVIRIS_")]
        if STARCOP_BANDS_TEMP:
            assert STARCOP_BANDS == STARCOP_BANDS_TEMP
        else:
            STARCOP_BANDS_TEMP = STARCOP_BANDS
    print("The wavelengths data are same across all tiles.")
    return STARCOP_BANDS

COLUMN = False
CREATE_TILE_MAG1C = False
CREATE_OTHER_FILTERS = True
RESUME = False
PRECISION = 64 # 64, 32, 16 However for some reason, float64 is the fastest (probably default dtype in numpy and scipy operations), most precise and stable (CEM does not handle float16) etc.
USE_SPED_UP_VERSIONS_OF_FILTERS = True
wavelengths_range = (1573, 2481)  # (2122, 2488) - MAG1C range used in original STARCOP data.
#Starcop data has range from 1573-2481 plus rgb 460, 550, 640 +-

# Define paths
note = "BY_COLUMNS_" if COLUMN else "WHOLE_IMAGE_"
note += "GENERATED-MAG1C_" if CREATE_TILE_MAG1C else "STARCOP-MAG1C_"
note += "SPED_UP_" if USE_SPED_UP_VERSIONS_OF_FILTERS else "ORIGINAL_"
note += str(wavelengths_range[0]) + "-" + str(wavelengths_range[1])
note += "_PRECISION-" + str(PRECISION)
note += "" #optional to generate more types of same dataset


csv_path = "../starcop_big/STARCOP_allbands/test.csv"
input_data_path = "../starcop_big/STARCOP_allbands"
output_data_path = os.path.join("data", note)

if USE_SPED_UP_VERSIONS_OF_FILTERS:
    ace = ACE_optimized
    cem = CEM_optimized
    mf = MatchedFilterOptimized
else:
    ace = ACE_original
    cem = CEM_original
    mf = MatchedFilterOriginal
# Load the CSV file
df = pd.read_csv(csv_path)
STARCOP_BANDS = data_sanity_check(df, input_data_path)

# Load CH4 transmittance data
ch4_data = []
centers = np.load("aviris_mag1c_centers.npy")
spectrum_mag1c = np.load("aviris_mag1c_spectrum.npy")
for i in range(len(centers)):
    ch4_data.append([i,centers[i], spectrum_mag1c[i]])

# Create a DataFrame
ch4_df = pd.DataFrame(ch4_data, columns=['id', 'wavelength', 'ch4_transmittance'])
# convert columns to appropriate data types
ch4_df['id'] = ch4_df['id'].astype(int)
ch4_df['wavelength'] = ch4_df['wavelength'].astype(float)
ch4_df['ch4_transmittance'] = ch4_df['ch4_transmittance'].astype(float)

#Create target spectrum
STARCOP_BANDS = [int(f.replace("TOA_AVIRIS_", "").replace("nm.tif", "")) for f in STARCOP_BANDS]
STARCOP_BANDS_FILTERED = [f for f in STARCOP_BANDS if f >= wavelengths_range[0] and f <= wavelengths_range[1]]
AVIRIS_WAVELENGTHS_FILTERED = []
AVIRIS_FWHM_FILTERED = []
temp_wavelengths = np.array(AVIRIS_WAVELENGTHS)  # Convert to NumPy for fast processing
temp_fwhm = np.array(AVIRIS_FWHM)

for curr_wv in STARCOP_BANDS_FILTERED:
    # Find the index of the closest wavelength in temp_wavelengths
    closest_idx = np.argmin(np.abs(temp_wavelengths - curr_wv))

    # Store the closest wavelength and corresponding FWHM
    AVIRIS_WAVELENGTHS_FILTERED.append(temp_wavelengths[closest_idx])
    AVIRIS_FWHM_FILTERED.append(temp_fwhm[closest_idx])

# Initialize an empty list to hold the transmittance values
transmittance_values = []

# For each wavelength in STARCOP_BANDS, find the closest wavelength in ch4_df
for band in STARCOP_BANDS_FILTERED:
    # Find the row in ch4_df with the closest wavelength
    closest_row = ch4_df.iloc[(ch4_df['wavelength'] - band).abs().argmin()]
    if band not in [460, 550, 640]: #RGB are little bit further < 3
        assert abs(band-closest_row["wavelength"]) < 1
    # Append the corresponding transmittance value
    transmittance_values.append(closest_row['ch4_transmittance'])
print("The wavelengths data from the spectrum file matches the loaded bands.")

DEFAULT_DTYPE = np.float64
if PRECISION == 32:
    DEFAULT_DTYPE = np.float32
if PRECISION == 16:
    DEFAULT_DTYPE = np.float16

# Convert the list of transmittance values into a numpy array
transmittance_array = np.array(transmittance_values, dtype = DEFAULT_DTYPE)
np.save("invalid_else.npy", 0)
np.save("invalid_mag1c.npy", 0)
hyperspectral_image = None
valid_mask = None


def init_worker(shared_dict):
    """Initialize global variables inside worker processes."""
    global hyperspectral_image, valid_mask
    hyperspectral_image = shared_dict["hyperspectral"]
    valid_mask = shared_dict["valid_mask"]

# Independent filter processing functions
def process_mf(idx):
    """Process a single column for Matched Filter."""
    return idx, mf(hyperspectral_image[:, idx, :][valid_mask[:,idx]], transmittance_array)

def process_ace(idx):
    """Process a single column for ACE."""
    return idx, ace(hyperspectral_image[:, idx, :][valid_mask[:,idx]], transmittance_array)

def process_cem(idx):
    """Process a single column for CEM."""
    return idx, cem(hyperspectral_image[:, idx, :][valid_mask[:,idx]], transmittance_array)

def process_tile(row):
    global hyperspectral_image, valid_mask
    tile_id = row["id"]
    tile_input_folder = os.path.join(input_data_path, tile_id)
    tile_output_folder = os.path.join(output_data_path, tile_id)
    os.makedirs(tile_output_folder, exist_ok=True)
    if RESUME and set(os.listdir(tile_output_folder)) >= {"ace.npy", "cem.npy", "label.npy", "mag1c.npy", "mf.npy"}:
        return

    # Load hyperspectral images based on wavelength range
    image_bands = []
    wavelengths = []
    sample_metadata = None  # To store metadata

    for file in sorted(os.listdir(tile_input_folder)):
        if file.startswith("TOA_AVIRIS_") and file.endswith(".tif"):
            wavelength = int(file.split("_")[-1].replace("nm.tif", ""))
            if wavelengths_range[0] <= wavelength <= wavelengths_range[1]:
                file_path = os.path.join(tile_input_folder, file)
                image_bands.append(tiff.imread(file_path))
                wavelengths.append(wavelength)

                # Capture metadata from the first valid TIFF
                if sample_metadata is None:
                    with rasterio.open(file_path) as src:
                        sample_metadata = src.meta.copy()

    if not image_bands:
        print(f"Skipping {tile_id}, no valid bands found.")
        return

    # Stack bands to create hyperspectral image
    hyperspectral_image = np.stack(image_bands, axis=-1).astype(DEFAULT_DTYPE)
    H,W,C = hyperspectral_image.shape
    valid_mask = ~np.all(hyperspectral_image == 0, axis=-1)
    files = []
    filepaths = []
    # Apply Matched Filter (MF), ACE, and CEM
    if CREATE_OTHER_FILTERS:
        mf_result = create_empty_filter(hyperspectral_image, DEFAULT_DTYPE)
        ace_result = create_empty_filter(hyperspectral_image, DEFAULT_DTYPE)
        cem_result = create_empty_filter(hyperspectral_image, DEFAULT_DTYPE)
        if COLUMN:
            non_zero_columns = np.where(np.any(valid_mask, axis=0))[0]
            num_workers = max(1, cpu_count() - 1)  # Use all available cores except one
            
            with Manager() as manager:
                shared_dict = manager.dict()
                shared_dict["hyperspectral"] = hyperspectral_image  # Shared memory for HSI data
                shared_dict["valid_mask"] = valid_mask  # Shared memory for HSI data
                print(hyperspectral_image.shape)
                # Create independent pools for MF, ACE, and CEM
                with Pool(num_workers, initializer=init_worker, initargs=(shared_dict,)) as pool:
                    mf_results = list(tqdm(pool.imap(process_mf, non_zero_columns), total=len(non_zero_columns)))
                
                with Pool(num_workers, initializer=init_worker, initargs=(shared_dict,)) as pool:
                    ace_results = list(tqdm(pool.imap(process_ace, non_zero_columns), total=len(non_zero_columns)))
                
                with Pool(num_workers, initializer=init_worker, initargs=(shared_dict,)) as pool:
                    cem_results = list(tqdm(pool.imap(process_cem, non_zero_columns), total=len(non_zero_columns)))

            # Store results in respective arrays
            for idx, mf_res in mf_results:
                mf_result[:, idx][valid_mask[:,idx]] = mf_res

            for idx, ace_res in ace_results:
                ace_result[:, idx][valid_mask[:,idx]] = ace_res

            for idx, cem_res in cem_results:
                cem_result[:, idx][valid_mask[:,idx]] = cem_res
        else:
            mf_result = mf_result.reshape((-1))
            ace_result = ace_result.reshape((-1))
            cem_result = cem_result.reshape((-1))
            hyperspectral_image = hyperspectral_image.reshape((-1,C))
            valid_mask = valid_mask.reshape((-1))
            hyperspectral_image_valid = hyperspectral_image[valid_mask,:]
            mf_result[valid_mask] = mf(hyperspectral_image_valid, transmittance_array)
            ace_result[valid_mask] = ace(hyperspectral_image_valid, transmittance_array)
            cem_result[valid_mask] = cem(hyperspectral_image_valid, transmittance_array)
        files += [mf_result, ace_result, cem_result, valid_mask]
        filepaths += ["mf", "ace", "cem", "valid_mask"]
        filepaths = [os.path.join(tile_output_folder,f"{x}.tif") for x in filepaths]
        filetypes = [np.float64 for x in range(len(files)-1)] + [np.uint8]
        save_all_reshaped_files(files, filepaths, filetypes, sample_metadata, (H,W))
        del mf_result, ace_result, cem_result, valid_mask
    
    label = tiff.imread(os.path.join(tile_input_folder, "labelbinary.tif"))
    mag1c = tiff.imread(os.path.join(tile_input_folder, "mag1c.tif"))
    if CREATE_TILE_MAG1C:
        output_metadata = {
            "wavelength units": "nm",
            "wavelength": AVIRIS_WAVELENGTHS_FILTERED,
            "fwhm": AVIRIS_FWHM_FILTERED,
        }
        name = hash_string_to_short(tile_id)
        to_process_image = hyperspectral_image if COLUMN else np.reshape(hyperspectral_image, (-1, 1, C))
        envi.save_image(
            f"{name}.hdr",
            to_process_image,
            shape=to_process_image.shape,
            interleave="bil",
            metadata=output_metadata,
            force=True,
        )
        try:
            result = subprocess.run(["python", "mag1c_zaitra/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(wavelengths_range[0]), str(wavelengths_range[1])], capture_output=True, text=True, check=True)
            print("MAG1C Output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            invalid_mag1c = np.load("invalid_mag1c.npy")
            invalid_mag1c += 1
            np.save("invalid_mag1c.npy", invalid_mag1c)
            print("Error running MAG1C:")
            print(e.stderr)
            shutil.rmtree(tile_output_folder)
            return
        mag1c_out = envi.open(f"{name}_ch4_cmfr.hdr", f"{name}_ch4_cmfr").load()[..., 3].squeeze(-1)
        for f in [f for f in os.listdir("./") if name in f]:
            os.remove(f)
        mag1c_out = np.clip(mag1c_out, 0, None)
        mag1c = mag1c_out if COLUMN else np.reshape(mag1c_out, (H, W))
    # Save filter output and label
    shutil.copy(os.path.join(tile_input_folder, "labelbinary.tif"), os.path.join(tile_output_folder, "labelbinary.tif"))
    save_tiff(mag1c, sample_metadata, os.path.join(tile_output_folder, "mag1c.tif"))
    del label, mag1c
    print(f"Processed tile {tile_id}")

if __name__ == "__main__":
    #Faster but unstable with mag1c and slow when computing by columns
    if CREATE_OTHER_FILTERS and not CREATE_TILE_MAG1C and not COLUMN:
        num_workers = max(1, cpu_count() - 1)  # Use all available cores except one
        with Pool(num_workers) as pool:
            list(tqdm(pool.imap_unordered(process_tile, [row for _, row in df.iterrows()]), total=len(df)))
    else:
        for _, row in tqdm(df.iterrows(),total=len(df)):
            process_tile(row)
    invalid_mag1c = np.load("invalid_mag1c.npy")
    invalid_else = np.load("invalid_else.npy")
    print(f"Processing complete. Excluded tiles because of mag1c failure: {invalid_mag1c}. Excluded tiles because of too much missing data: {invalid_else}")