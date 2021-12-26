import xarray as xr
import numpy as np


def rfft(drive):
    """RFFT of Drive object and return transform as xarray.DataArray."""

    total_steps = drive.n
    time_steps = drive.table.data['t'].to_numpy()
    time_step_size = time_steps[1] - time_steps[0]
    sampling_rate = 1.0 / time_step_size
    mag_time_list = []

    for index in range(total_steps):  # Takes the most amount of time.
        mag_time_list.append(drive[index].array)

    mag_time_array = np.array(mag_time_list)
    rfft_array = np.fft.rfft(mag_time_array, axis=0)
    frequencies = np.fft.rfftfreq(total_steps, time_step_size)

    rfft_data_array = xr.DataArray(rfft_array,
                                   dims=['f', 'x', 'y', 'z', 'ft'],
                                   coords={'f': frequencies,
                                           'ft': ['ft_x', 'ft_y', 'ft_z']})

    rfft_data_array.attrs["max_frequency"] = f'{sampling_rate/2.0} Hz'
    rfft_data_array.attrs["frequency_resolution"] = f'{sampling_rate/total_steps} Hz'
    rfft_data_array.f.attrs["units"] = "Hz"

    return rfft_data_array
