import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from einops import rearrange

# Paths
raw_dir = "/home/rishabh.mondal/bkdb/statewise/"
#show folders in the raw_dir
print(os.listdir(raw_dir))
#take user input for the state name
state_name = input("Enter the state name: ")
raw_dir = "/home/rishabh.mondal/bkdb/statewise/"+state_name
png_dir= "/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/png_data"

part= f"{state_name}"
png_path=f"{png_dir}/{part}"
print(png_path)
# Create the directory
os.makedirs(png_path, exist_ok=True)

# Function to convert Zarr to PNG
def zarr_to_png(zarr_file, temp_dir):
    # Load the Zarr file
    img = xr.open_zarr(zarr_file, consolidated=False)
    # Extract the data
    data = img['data'].values
    # Rearrange the data as needed
    data = rearrange(data, "row col high width channel -> (row high) (col width) channel")
    # Create the path for the PNG file
    png_path = os.path.join(temp_dir, os.path.basename(zarr_file).replace('.zarr', '.png'))
    # Save the data to a PNG file
    plt.imsave(png_path, data)

# Get the list of Zarr files as per the part name
zarr_files = glob.glob(os.path.join(raw_dir, "*.zarr"))


# Parallelize the conversion process
Parallel(n_jobs=42)(delayed(zarr_to_png)(zarr_file, png_path) for zarr_file in tqdm(zarr_files, desc="Converting to PNG"))

#count the no of files in the png_path
print(f"Total files in {png_path}: {len(os.listdir(png_path))}")
print(f"Total files in {raw_dir}: {len(os.listdir(raw_dir))}")

#match the no of files in the png_path with the no of files in the raw_dir
if len(os.listdir(png_path)) == len(os.listdir(raw_dir)):
    print("All files converted successfully")
    
