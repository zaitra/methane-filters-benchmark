import os
import pandas as pd
import numpy as np
import tifffile as tiff
from pysptools.detection import MatchedFilter, ACE, CEM
from pysptools.detection.detect import ACE as ace, MatchedFilter as matched_filter, CEM as cem
import shutil
from tqdm import tqdm
def create_empty_filter(hyperspectral_image):
    return np.zeros((hyperspectral_image.shape[0], hyperspectral_image.shape[1]), dtype=np.float32)
# Define paths
csv_path = "../starcop_big/STARCOP_allbands/test.csv"
input_data_path = "../starcop_big/STARCOP_allbands"
output_data_path = "data"
ch4_transmittance_file = "ang_ch4_unit_3col_425chan.txt"
wavelengths_range = (2122, 2488)  # MAG1C range used in original STARCOP data.
COLUMN = True
# Load the CSV file
df = pd.read_csv(csv_path)

AVIRIS_BANDS_TEMP = []
#First sanity check if all data was downloaded correctly
for idx, row in tqdm(df.iterrows()):
    tile_id = row["id"]
    tile_input_folder = os.path.join(input_data_path, tile_id)
    AVIRIS_BANDS = [f for f in os.listdir(tile_input_folder) if f.startswith("TOA_AVIRIS_")]
    if AVIRIS_BANDS_TEMP:
        assert AVIRIS_BANDS == AVIRIS_BANDS_TEMP
    else:
        AVIRIS_BANDS_TEMP = AVIRIS_BANDS
print("The wavelengths data are same across all tiles.")


# Load CH4 transmittance data
ch4_data = []
with open(ch4_transmittance_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        ch4_data.append(parts)

# Create a DataFrame
ch4_df = pd.DataFrame(ch4_data, columns=['id', 'wavelength', 'ch4_transmittance'])
# convert columns to appropriate data types
ch4_df['id'] = ch4_df['id'].astype(int)
ch4_df['wavelength'] = ch4_df['wavelength'].astype(float)
ch4_df['ch4_transmittance'] = ch4_df['ch4_transmittance'].astype(float)

#Create target spectrum
AVIRIS_BANDS = [int(f.replace("TOA_AVIRIS_", "").replace("nm.tif", "")) for f in AVIRIS_BANDS]
AVIRIS_BANDS = [f for f in AVIRIS_BANDS if f >= wavelengths_range[0] and f <= wavelengths_range[1]]
# Initialize an empty list to hold the transmittance values
transmittance_values = []

# For each wavelength in AVIRIS_BANDS, find the closest wavelength in ch4_df
for band in AVIRIS_BANDS:
    # Find the row in ch4_df with the closest wavelength
    closest_row = ch4_df.iloc[(ch4_df['wavelength'] - band).abs().argmin()]
    assert abs(band-closest_row["wavelength"]) < 1
    # Append the corresponding transmittance value
    transmittance_values.append(closest_row['ch4_transmittance'])
print("The wavelengths data from the spectrum file matches the loaded bands.")

# Convert the list of transmittance values into a numpy array
transmittance_array = np.array(transmittance_values)
# Process each row in the CSV
for idx, row in tqdm(df.iterrows(), total=len(df)):
    tile_id = row["id"]
    tile_input_folder = os.path.join(input_data_path, tile_id)
    tile_output_folder = os.path.join(output_data_path, tile_id)
    
    # Create output folder if it does not exist
    try:
        os.makedirs(tile_output_folder, exist_ok=False)
    except OSError:
        shutil.rmtree(tile_output_folder)
        os.makedirs(tile_output_folder, exist_ok=False)

    # Load hyperspectral images based on wavelength range
    image_bands = []
    wavelengths = []

    for file in sorted(os.listdir(tile_input_folder)):
        if file.startswith("TOA_AVIRIS_") and file.endswith(".tif"):
            wavelength = int(file.split("_")[-1].replace("nm.tif", ""))
            if wavelengths_range[0] <= wavelength <= wavelengths_range[1]:
                file_path = os.path.join(tile_input_folder, file)
                image_bands.append(tiff.imread(file_path))
                wavelengths.append(wavelength)

    if not image_bands:
        print(f"Skipping {tile_id}, no valid bands found.")
        continue

    # Stack bands to create hyperspectral image
    hyperspectral_image = np.stack(image_bands, axis=-1)

    # Apply Matched Filter (MF), ACE, and CEM
    if COLUMN:
        mf_result = create_empty_filter(hyperspectral_image)
        ace_result = create_empty_filter(hyperspectral_image)
        cem_result = create_empty_filter(hyperspectral_image)
        for idx,column in tqdm(enumerate(hyperspectral_image), total=hyperspectral_image.shape[0]):
            mf_result[idx,:] = matched_filter(column, transmittance_array)
            ace_result[idx,:] = ace(column, transmittance_array)
            cem_result[idx,:] = cem(column, transmittance_array)
    else:
        mf_result = MatchedFilter().detect(hyperspectral_image, transmittance_array)
        ace_result = ACE().detect(hyperspectral_image, transmittance_array)
        cem_result = CEM().detect(hyperspectral_image, transmittance_array)
    label = tiff.imread(os.path.join(tile_input_folder, "labelbinary.tif"))
    mag1c = tiff.imread(os.path.join(tile_input_folder, "mag1c.tif"))

    # Save filter outputs as NumPy arrays
    np.save(os.path.join(tile_output_folder, "mf.npy"), mf_result)
    np.save(os.path.join(tile_output_folder, "ace.npy"), ace_result)
    np.save(os.path.join(tile_output_folder, "cem.npy"), cem_result)
    np.save(os.path.join(tile_output_folder, "label.npy"), label)
    np.save(os.path.join(tile_output_folder, "mag1c.npy"), mag1c)

    print(f"Processed tile {tile_id}")

print("Processing complete.")