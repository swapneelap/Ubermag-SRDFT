import micromagneticdata as md
import numpy as np
import xarray as xr

data_path = "/home/swapneel/Projects/Ubermag-SRDFT/data/"
sim_data = md.Data(name="SkyStable", dirname=data_path)
print(sim_data[1].info)


def to_xarray(drive):
    """Convert Drive object to xarray.DataArray"""
    assert drive.info['driver'] == 'TimeDriver',\
                                    f"obtained '{drive.info['driver']}' insted of 'TimeDriver'"

    total_steps = drive.n
    time_steps = drive.table.data['t'].to_numpy()
    x_points = np.fromiter(drive[0].mesh.axis_points('x'), float)
    y_points = np.fromiter(drive[0].mesh.axis_points('y'), float)
    z_points = np.fromiter(drive[0].mesh.axis_points('z'), float)
    array_dims = list(drive[0].array.shape)
    array_dims.insert(0, total_steps)
    mag_time_array = np.empty(tuple(array_dims))

    for step in range(total_steps):
        mag_time_array[step] = drive[step].array

    mag_time_xarray = xr.DataArray(mag_time_array,
                                   dims=['t', 'x', 'y', 'z', 'm'],
                                   coords={'t': time_steps,
                                           'x': x_points,
                                           'y': y_points,
                                           'z': z_points,
                                           'm': ['mx', 'my', 'mz']})
    mag_time_xarray.attrs = drive.info
    mag_time_xarray.t.attrs['units'] = 's'
    mag_time_xarray.x.attrs['units'] = 'm'
    mag_time_xarray.y.attrs['units'] = 'm'
    mag_time_xarray.z.attrs['units'] = 'm'
    mag_time_xarray.m.attrs['units'] = 'A/m'

    return mag_time_xarray


# drive_xarray = to_xarray(sim_data[0])
drive_xarray = to_xarray(sim_data[1])

print(drive_xarray)


def xarray_rfft(da):
    """RFFT of xarray.DataArray and return transform as xarray.DataArray."""

    total_steps = da.attrs['n']
    time_steps = da['t'].values
    x_points = da['x'].values
    y_points = da['y'].values
    z_points = da['z'].values
    time_step_size = time_steps[1] - time_steps[0]
    sampling_rate = 1.0 / time_step_size

    mag_time_array = da.values
    rfft_array = np.fft.rfft(mag_time_array, axis=0)
    frequencies = np.fft.rfftfreq(total_steps, time_step_size)

    rfft_data_array = xr.DataArray(rfft_array,
                                   dims=['f', 'x', 'y', 'z', 'ft'],
                                   coords={'f': frequencies,
                                           'x': x_points,
                                           'y': y_points,
                                           'z': z_points,
                                           'ft': ['ft_x', 'ft_y', 'ft_z']})

    rfft_data_array.attrs['max_frequency'] = f'{sampling_rate/2.0} Hz'
    rfft_data_array.attrs['frequency_resolution'] = f'{sampling_rate/total_steps} Hz'
    rfft_data_array.f.attrs['units'] = 'Hz'
    rfft_data_array.x.attrs['units'] = 'm'
    rfft_data_array.y.attrs['units'] = 'm'
    rfft_data_array.z.attrs['units'] = 'm'

    return rfft_data_array


rfft_drive = xarray_rfft(drive_xarray)
print(rfft_drive)
