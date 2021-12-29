## Space resolved discrete Fourier transform for Ubermag drive object.
- `src/rfft_func.py` contains the `rfft` function that returns an `xarray.DataArray` object corresponding to the Fourier transform (FT) of the drive object. Further, the `to_xarray` function returns an `xarray.DataArray` containing the magnetization snapshots corresponding to the input drive object. Finally, `xarray_rfft` function returns an `xarray.DataArray` containing FT corresponding to `xarray.DataArray` represenation of drive object (*i.e.* output of `to_xarray` function).

- `Spatially Resolved FFT Ubermag.ipynb` contains the code description and use examples of the `rfft` function.

- `Drive Xarray And FFT Xarray.ipynb` contains the code description and use examples of `to_xarray` and `xarray_rfft` functions.

> **_Note:_** `data` folder contains simulation data and it is round about 1.3 GB. To download the data, use `git lfs pull`.
