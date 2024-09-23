import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from einops import rearrange

# Paths
raw_dir = "/home/rishabh.mondal/bkdb/statewise/up"
png_dir= "/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/png_data"

state_name="up"
part_name="4"

part= f"{state_name}_part_{part_name}"
png_path=f"{png_dir}/{part}"
print(png_path)
#remove the existing directory
os.system(f"rm -rf {png_path}")
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
total_files = len(zarr_files)
print(f"Total files: {total_files}")
quarter_size = total_files // 4
print(f"Quarter size: {quarter_size}")
part_index = int(part_name) - 1 
print(f"Part index: {part_index}")
start_index = part_index * quarter_size
print(f"Start index: {start_index}")
end_index = (part_index + 1) * quarter_size if part_index < 3 else total_files
print(f"End index: {end_index}")
zarr_files = zarr_files[start_index:end_index]


# Parallelize the conversion process
Parallel(n_jobs=42)(delayed(zarr_to_png)(zarr_file, png_path) for zarr_file in tqdm(zarr_files, desc="Converting to PNG"))
