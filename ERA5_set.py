"""
Mismo codigo que ENSO-IOD
"""
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
###############################################################################
dir_files = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
out_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
# Funciones ###################################################################
def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

################################################################################
variables =['SLP']
name_variables = ['msl']

for v in variables:

    print(v)
    data = xr.open_dataset(dir_files + f'ERA5_{v}_40-20.nc')
    var_name = list(data.data_vars)[0]

    if v == 'SLP':
        data = data.drop_vars(['expver', 'number'])
        data = data.rename({'valid_time':'time'})

    data = data.rename({var_name: 'var'})
    data = data.rename({'longitude': 'lon'})
    data = data.rename({'latitude': 'lat'})

    data = Weights(data)
    data = data.rolling(time=3, center=True).mean()

    for mm, s_name in zip([10], ['SON']):
        aux = data.sel(time=data.time.dt.month.isin(mm))
        aux = Detrend(aux, 'time')

        print('to_netcdf...')
        aux.to_netcdf(f'{out_dir}{v}_{s_name}_mer_d_w.nc')

###############################################################################