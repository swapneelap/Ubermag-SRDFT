import micromagneticdata as md
import numpy as np
import xarray as xr
# import time

# start_time = time.time()
data_path = '/home/swapneel/Git/Ubermag-SRDFT/data'
sim_data = md.Data(name="SkyStable", dirname=data_path)
time_drive = sim_data[1]
# print(time_drive.info)

total_steps = time_drive.n
time_steps = np.array(time_drive.table.data['t'])
time_drive_array = []

for step in range(total_steps):
    time_drive_array.append(time_drive[step].array)

time_drive_array = np.array(time_drive_array)
drive_xarray = xr.DataArray(time_drive_array,
                            dims=['t', 'x', 'y', 'z', 'm'],
                            coords={'t': time_steps, 'm': ['mx', 'my', 'mz']})

drive_xarray.attrs = time_drive.info

print(drive_xarray)

numpy_drive_array = drive_xarray.values
print(numpy_drive_array.shape)
