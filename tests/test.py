import micromagneticdata as md
import numpy as np
import xarray as xr
import time

start_time = time.time()

data_path = "/home/swapneel/Projects/Ubermag-SRDFT/data/"
sim_data = md.Data(name="SkyStable", dirname=data_path)
time_drive = sim_data[1]
total_steps = time_drive.n
time_steps = time_drive.table.data['t'].to_numpy()
step_size = time_steps[1]-time_steps[0]
sim_list = []

reading_start = time.time()
for index in range(total_steps):
    sim_list.append(time_drive[index].array)
reading_stop = time.time()

magnetization_time_array = np.array(sim_list)

fft_array = np.fft.rfftn(magnetization_time_array, axes=(0,))
fft_freq = np.fft.rfftfreq(total_steps, step_size)

final_array = xr.DataArray(fft_array, dims=['f', 'x', 'y', 'z', 'P'],
                           coords={'f': fft_freq})

print(f'-----File loading time = {reading_stop - reading_start} seconds-----')
print(f'--------Total time = {time.time() - start_time} seconds-------')
