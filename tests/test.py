import micromagneticdata as md
import numpy as np
import xarray as xr

data_path = "/home/swapneel/Projects/Ubermag-SRDFT/data/"
sim_data = md.Data(name="SkyStable", dirname=data_path)
time_drive = sim_data[1]
total_steps = time_drive.n
time_steps = time_drive.table.data['t'].to_numpy()
step_size = time_steps[1]-time_steps[0]
sim_list = []

# reading_start = time.time()
for index in range(total_steps):
    sim_list.append(time_drive[index].array)
# reading_stop = time.time()

magnetization_time_array = np.array(sim_list)

fft_array = np.fft.rfftn(magnetization_time_array, axes=(0,))
fft_freq = np.fft.rfftfreq(total_steps, step_size)

final_array = xr.DataArray(fft_array, dims=['f', 'x', 'y', 'z', 'ft'],
                           coords={'f': fft_freq, 'ft': ['ftx', 'fty', 'ftz']})

power_array = np.abs(final_array)
# phase_array = np.angle(final_array) # Apperently np.angle is not a ufunc!

test_array = power_array.sum(dim=['x', 'y', 'z'])
test_array = test_array.drop_sel(f=0.0)
