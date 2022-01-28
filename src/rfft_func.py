import xarray as xr
import numpy as np


def rfft(drive):
    """RFFT of Drive object and return transform as xarray.DataArray."""

    total_steps = drive.n
    time_steps = drive.table.data['t'].to_numpy()
    time_step_size = time_steps[1] - time_steps[0]
    sampling_rate = 1.0 / time_step_size
    mag_time_list = []

    for step in range(total_steps):  # Takes the most amount of time.
        mag_time_list.append(drive[step].array)

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


def to_xarray_dataset(drive):
    """Convert Drive object to xarray.Dataset"""
    assert drive.info['driver'] == 'TimeDriver',\
                                    f"obtained '{drive.info['driver']}' insted of 'TimeDriver'"

    total_steps = drive.n
    time_steps = drive.table.data['t'].to_numpy()
    m_init = drive.m0
    array_shape = list(m_init.array.shape)
    x_points = np.fromiter(m_init.mesh.axis_points('x'), float)
    y_points = np.fromiter(m_init.mesh.axis_points('y'), float)
    z_points = np.fromiter(m_init.mesh.axis_points('z'), float)
    array_dims = [total_steps] + array_shape
    m_time_array = np.empty(tuple(array_dims))

    for step in range(total_steps):
        m_time_array[step] = drive[step].array

    m_time_dataset = xr.Dataset({'mx': (['t', 'x', 'y', 'z'], m_time_array[..., 0]),
                                 'my': (['t', 'x', 'y', 'z'], m_time_array[..., 1]),
                                 'mz': (['t', 'x', 'y', 'z'], m_time_array[..., 2])},
                                coords={'t': time_steps, 'x': x_points,
                                        'y': y_points, 'z': z_points}
                                )

    m_time_dataset.attrs = drive.info
    m_time_dataset.attrs['m0'] = m_init.array
    m_time_dataset.t.attrs['units'] = 's'
    m_time_dataset.x.attrs['units'] = 'm'
    m_time_dataset.y.attrs['units'] = 'm'
    m_time_dataset.z.attrs['units'] = 'm'
    m_time_dataset['mx'].attrs['units'] = 'A/m'
    m_time_dataset['my'].attrs['units'] = 'A/m'
    m_time_dataset['mz'].attrs['units'] = 'A/m'

    return m_time_dataset


def rfft_xarray_dataset(ds):
    """RFFT of xarray.Dataset and return transform as xarray.Dataset."""

    total_steps = ds.attrs['n']
    time_steps = ds.coords['t'].values
    x_points = ds.coords['x'].values
    y_points = ds.coords['y'].values
    z_points = ds.coords['z'].values
    time_step_size = time_steps[1] - time_steps[0]
    sampling_rate = 1.0 / time_step_size

    m_time_array = np.stack([ds['mx'].values, ds['my'].values, ds['mz'].values],
                            axis=len(ds.attrs['m0'].shape))
    dm_time_array = np.subtract(m_time_array, ds.attrs['m0'])  # broadcasting

    rfft_array = np.fft.rfft(dm_time_array, axis=0)
    frequencies = np.fft.rfftfreq(total_steps, time_step_size)

    rfft_dataset = xr.Dataset({'ft_x': (['f', 'x', 'y', 'z'], rfft_array[..., 0]),
                               'ft_y': (['f', 'x', 'y', 'z'], rfft_array[..., 1]),
                               'ft_z': (['f', 'x', 'y', 'z'], rfft_array[..., 2])},
                              coords={'f': frequencies, 'x': x_points,
                                      'y': y_points, 'z': z_points}
                              )

    rfft_dataset.attrs['max_frequency'] = f'{sampling_rate/2.0} Hz'
    rfft_dataset.attrs['frequency_resolution'] = f'{sampling_rate/total_steps} Hz'
    rfft_dataset.f.attrs['units'] = 'Hz'
    rfft_dataset.x.attrs['units'] = 'm'
    rfft_dataset.y.attrs['units'] = 'm'
    rfft_dataset.z.attrs['units'] = 'm'

    return rfft_dataset
