import os
import pandas as pd
import numpy as np
import rasterio.errors
import tifffile as tiff
from pysptools.detection.detect import ACE as ACE_original, MatchedFilter as MatchedFilterOriginal, CEM as CEM_original
from sped_up_filters import ACE_optimized, MatchedFilterOptimized, CEM_optimized
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import subprocess
import spectral.io.envi as envi
import rasterio
import imagecodecs
import itertools
from utils import AVIRIS_FWHM, AVIRIS_WAVELENGTHS, hash_string_to_short, save_all_reshaped_files, save_tiff, starcop_data_sanity_check, select_the_bands_by_transmittance
hyperspectral_image = None
transmittance_array = None
valid_mask = None
config = None
ace = None
cem = None
mf = None
def define_all():
    global config, ace, cem, mf, transmittance_array
    ####################### SETTINGS ###############################
    COLUMN = config["COLUMN"]
    CREATE_TILE_MAG1C = config["CREATE_TILE_MAG1C"]
    if CREATE_TILE_MAG1C:
        CREATE_SAMPLED_MAG1C = config["CREATE_SAMPLED_MAG1C"]
        if CREATE_SAMPLED_MAG1C:
            SAMPLE_PERCENT = config["SAMPLE_PERCENT"]
    SELECT_BANDS = config["SELECT_BANDS"]
    if SELECT_BANDS:
        BANDS_N = config["BANDS_N"]
        STRATEGY = config["STRATEGY"]
        
    CREATE_OTHER_FILTERS = config["CREATE_OTHER_FILTERS"]
    RESUME = config["RESUME"]
    PRECISION = config["PRECISION"]
    USE_SPED_UP_VERSIONS_OF_FILTERS = config["USE_SPED_UP_VERSIONS_OF_FILTERS"]
    wavelengths_range = config["wavelengths_range"]
    
    csv_path = config["csv_path"]
    input_data_path = config["input_data_path"]
    output_data_path = config["output_data_path"]
    ###############################################################

    # Define paths
    note = "BY-COLUMNS_" if COLUMN else "WHOLE-IMAGE_"
    note += "TILE-" if CREATE_TILE_MAG1C else "STARCOP-"
    note += "AND-SAMPLED-MAG1C-" + str(SAMPLE_PERCENT) if CREATE_SAMPLED_MAG1C else "MAG1C"
    note += "_SPED-UP_" if USE_SPED_UP_VERSIONS_OF_FILTERS else "_ORIGINAL_"
    note += "PRECISION-" + str(PRECISION) + "_"
    note += str(wavelengths_range[0]) + "-" + str(wavelengths_range[1]) + "_"
    note += "SELECT-"+ STRATEGY.upper() + "_" if SELECT_BANDS else "SELECT-ALL_"

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
    STARCOP_BANDS = starcop_data_sanity_check(df, input_data_path)

    # Load CH4 transmittance data (created from mag1c)
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

    #SELECT channels inside the defined wavelengths interval from STARCOP bands
    STARCOP_BANDS = [int(f.replace("TOA_AVIRIS_", "").replace("nm.tif", "")) for f in STARCOP_BANDS]
    STARCOP_BANDS_FILTERED = [f for f in STARCOP_BANDS if f >= wavelengths_range[0] and f <= wavelengths_range[1]]
    STARCOP_BANDS_FILTERED.sort()

    # Create the target spectrum from presaved mag1c spectrum
    transmittance_values = []
    # For each wavelength in STARCOP_BANDS, find the closest wavelength in the mag1c generated spectrum.
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
    config["DEFAULT_DTYPE"] = DEFAULT_DTYPE
    # Convert the list of transmittance values into a numpy array
    transmittance_array = np.array(transmittance_values, dtype = DEFAULT_DTYPE)
    if SELECT_BANDS:
        STARCOP_BANDS_FILTERED, transmittance_array = select_the_bands_by_transmittance(np.array(STARCOP_BANDS_FILTERED), transmittance_array, BANDS_N, STRATEGY)
    #np.save("starcop_spectrum.npy", transmittance_array)
    #np.save("starcop_centers.npy", STARCOP_BANDS_FILTERED)
    note += "CHANNEL-N-" + str(len(transmittance_array))
    np.save("invalid_else.npy", 0)
    np.save("invalid_mag1c.npy", 0)
    config["output_data_path"] = os.path.join(output_data_path, note)
    config["STARCOP_BANDS_FILTERED"] = STARCOP_BANDS_FILTERED

    #The STARCOP channels are rounded to integers, so load the true float-like band info from AVIRIS-NG .hdr file.
    #This is only used for later, when creating mag1c.
    AVIRIS_WAVELENGTHS_FILTERED = []
    AVIRIS_FWHM_FILTERED = []
    temp_wavelengths = np.array(AVIRIS_WAVELENGTHS) # Convert to NumPy for fast processing
    temp_fwhm = np.array(AVIRIS_FWHM)
    for curr_wv in STARCOP_BANDS_FILTERED:
        # Find the index of the closest wavelength in temp_wavelengths
        closest_idx = np.argmin(np.abs(temp_wavelengths - curr_wv))

        # Store the closest wavelength and corresponding FWHM
        AVIRIS_WAVELENGTHS_FILTERED.append(temp_wavelengths[closest_idx])
        AVIRIS_FWHM_FILTERED.append(temp_fwhm[closest_idx])
    config["AVIRIS_WAVELENGTHS_FILTERED"] = AVIRIS_WAVELENGTHS_FILTERED
    config["AVIRIS_FWHM_FILTERED"] = AVIRIS_FWHM_FILTERED

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
    global hyperspectral_image, transmittance_array, valid_mask, config, ace, cem, mf
    tile_id = row["id"]
    tile_input_folder = os.path.join(config["input_data_path"], tile_id)
    tile_output_folder = os.path.join(config["output_data_path"], tile_id)
    os.makedirs(tile_output_folder, exist_ok=True)
    if config["RESUME"] and set(os.listdir(tile_output_folder)) >= {"ace.tif", "cem.tif", "mf.tif"}:
        return

    # Load hyperspectral images based on wavelength range
    image_bands = []
    sample_metadata = None  # To store metadata
    invalid = 0
    for wavelength in config["STARCOP_BANDS_FILTERED"]:
        file = f"TOA_AVIRIS_{wavelength}nm.tif"
        file_path = os.path.join(tile_input_folder, file)
        try:
            image_bands.append(tiff.imread(file_path))
        #skip invalid bands
        except (tiff.tifffile.TiffFileError, imagecodecs._imcd.ImcdError) as e1:
            print(f"Skipping {tile_id} due to error: {e1}, the concrete band is {file}.")
            invalid += 1
        # Capture metadata from the first valid TIFF
        if sample_metadata is None:
            with rasterio.open(file_path) as src:
                sample_metadata = src.meta.copy()
    #skip tiles containing invalid bands
    if invalid > 0:
        return
    if not image_bands:
        print(f"Skipping {tile_id}, no valid bands found.")
        return

    # Stack bands to create hyperspectral image
    hyperspectral_image = np.stack(image_bands, axis=-1).astype(config["DEFAULT_DTYPE"])
    H,W,C = hyperspectral_image.shape

    #Missing values are coded as 0
    valid_mask = ~np.all(hyperspectral_image == 0, axis=-1)

    #flatten and mask tile
    reshaped_hyperspectral_image = hyperspectral_image.reshape((-1,C))
    reshaped_valid_mask = valid_mask.reshape((-1))
    reshaped_hyperspectral_image = reshaped_hyperspectral_image[reshaped_valid_mask,:]

    #FOR COLUMN WISE PROCESSING - Exclude missing values column-wise, it causes errors otherwise.
    non_zero_columns = np.where(np.all(valid_mask, axis=0))[0]

    files = []
    filepaths = []

    # Apply Matched Filter (MF), ACE, and CEM
    if config["CREATE_OTHER_FILTERS"]:
        mf_result = np.zeros((H, W), dtype=config["DEFAULT_DTYPE"])
        ace_result = np.zeros((H, W), dtype=config["DEFAULT_DTYPE"])
        cem_result = np.zeros((H, W), dtype=config["DEFAULT_DTYPE"])
        if config["COLUMN"]:
            num_workers = max(1, cpu_count() - 1)  # Use all available cores except one
            
            with Manager() as manager:
                shared_dict = manager.dict()
                shared_dict["hyperspectral"] = hyperspectral_image  # Shared memory for HSI data
                shared_dict["valid_mask"] = valid_mask  # Shared memory for HSI data
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
        #The main tile-wise processing
        else:
            mf_result = mf_result.reshape((-1))
            ace_result = ace_result.reshape((-1))
            cem_result = cem_result.reshape((-1))
            mf_result[reshaped_valid_mask] = mf(reshaped_hyperspectral_image, transmittance_array)
            ace_result[reshaped_valid_mask] = ace(reshaped_hyperspectral_image, transmittance_array)
            cem_result[reshaped_valid_mask] = cem(reshaped_hyperspectral_image, transmittance_array)
        files += [mf_result, ace_result, cem_result, reshaped_valid_mask]
        filepaths += ["mf", "ace", "cem", "valid_mask"]
        filepaths = [os.path.join(tile_output_folder,f"{x}.tif") for x in filepaths]
        #No difference in metrics when saving as float32 vs float64 only the precision during processing differs
        filetypes = [np.float32 for _ in range(len(files)-1)] + [np.uint8]
        save_all_reshaped_files(files, filepaths, filetypes, sample_metadata, (H,W))
        del mf_result, ace_result, cem_result, valid_mask
    
    label = tiff.imread(os.path.join(tile_input_folder, "labelbinary.tif"))
    mag1c = tiff.imread(os.path.join(tile_input_folder, "mag1c.tif"))
    #Create mag1c tile-wise
    if config["CREATE_TILE_MAG1C"]:
        output_metadata = {
            "wavelength units": "nm",
            "wavelength": config["AVIRIS_WAVELENGTHS_FILTERED"],
            "fwhm": config["AVIRIS_FWHM_FILTERED"],
        }
        name = hash_string_to_short(tile_id)
        to_process_image = hyperspectral_image[:,non_zero_columns,:] if config["COLUMN"] else reshaped_hyperspectral_image.reshape((-1,1,C))
        envi.save_image(
            f"{name}.hdr",
            to_process_image,
            shape=to_process_image.shape,
            interleave="bil",
            metadata=output_metadata,
            force=True,
        )
        arg_list = []
        arg = ["python", "mag1c_fork/mag1c/mag1c.py", f"{name}","-o", "--use-wavelength-range", str(config["wavelengths_range"][0]), str(config["wavelengths_range"][1])]
        arg_list.append(arg)
        if config["CREATE_SAMPLED_MAG1C"]:
            arg_2 = arg + ["--sample", str(config["SAMPLE_PERCENT"])]
            arg_list.append(arg_2)
        for args in arg_list:
            mag1c = np.zeros((H, W), dtype=config["DEFAULT_DTYPE"])
            if not config["COLUMN"]:
                mag1c = mag1c.reshape((-1))
            try:
                result = subprocess.run(args, capture_output=True, text=True, check=True)
                print("MAG1C Output:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                invalid_mag1c = np.load("invalid_mag1c.npy")
                invalid_mag1c += 1
                np.save("invalid_mag1c.npy", invalid_mag1c)
                print("Error running MAG1C:")
                print(e.stderr)
                shutil.rmtree(tile_output_folder)
                for f in [f for f in os.listdir("./") if name in f]:
                    os.remove(f)
                return
            mag1c_out = envi.open(f"{name}_ch4_cmfr.hdr", f"{name}_ch4_cmfr").load()[..., 3].squeeze()
            mag1c_out = np.clip(mag1c_out, 0, None)
            if config["COLUMN"]:
                mag1c[:,non_zero_columns] = mag1c_out
            else:
                mag1c[reshaped_valid_mask] = mag1c_out
                mag1c = mag1c.reshape((H, W))
            title = "mag1c_tile.tif"
            if "--sample" in args:
                title = f"mag1c_tile_sampled-{config["SAMPLE_PERCENT"]}.tif"
            save_tiff(mag1c, sample_metadata, os.path.join(tile_output_folder, title))
        for f in [f for f in os.listdir("./") if name in f]:
            os.remove(f)
    # Save filter output and label
    shutil.copy(os.path.join(tile_input_folder, "labelbinary.tif"), os.path.join(tile_output_folder, "labelbinary.tif"))
    del label, mag1c
    print(f"Processed tile {tile_id}")

if __name__ == "__main__":
    config_base = {
    "COLUMN": False,
    "CREATE_TILE_MAG1C": True,
    "CREATE_SAMPLED_MAG1C": True,
    "SAMPLE_PERCENT": 0.01,
    "SELECT_BANDS": True,
    "BANDS_N": 35,#[10, 25],  # Only used if SELECT_BANDS is True
    "STRATEGY": ['highest-transmittance', 'evenly-spaced'],  # Only used if SELECT_BANDS is True
    "CREATE_OTHER_FILTERS": True,
    "RESUME": False,
    "PRECISION": 64,
    "USE_SPED_UP_VERSIONS_OF_FILTERS": True,
    "wavelengths_range": (200, 2600),
    "csv_path": "../starcop_big/STARCOP_allbands/test.csv",
    "input_data_path": "../starcop_big/STARCOP_allbands",
    "output_data_path": "./data"
    }
    # Identify which keys have list values
    list_keys = [key for key, value in config_base.items() if isinstance(value, list)]
    df = pd.read_csv(config_base.get("csv_path"))
    # Prepare the iterables for product
    iterables = [config_base[key] for key in list_keys]

    # Generate all combinations using itertools.product
    combinations = itertools.product(*iterables)

    # Iterate through each combination and update the config
    for combination in combinations:
        # Create a copy of the config to avoid modifying the original one
        config = config_base.copy()

        # Update the config with the current combination values
        for idx, key in enumerate(list_keys):
            config[key] = combination[idx]
        define_all()

        #Faster but unstable with mag1c and slow when computing by columns
        if config["CREATE_OTHER_FILTERS"] and not config["CREATE_TILE_MAG1C"] and not config["COLUMN"]:
            num_workers = max(1, cpu_count() - 1)  # Use all available cores except one
            with Pool(num_workers) as pool:
                list(tqdm(pool.imap_unordered(process_tile, [row for _, row in df.iterrows()]), total=len(df)))
        else:
            for _, row in tqdm(df.iterrows(),total=len(df)):
                process_tile(row)
        invalid_mag1c = np.load("invalid_mag1c.npy")
        invalid_else = np.load("invalid_else.npy")
        print(f"Processing complete. Excluded tiles because of mag1c failure: {invalid_mag1c}. Excluded tiles because of too much missing data: {invalid_else}")