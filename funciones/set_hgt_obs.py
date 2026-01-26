"""
Computo de hgt para complementar con data obs
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/aux/'

# ---------------------------------------------------------------------------- #
import os
import xarray as xr
from funciones.general_utils import xrFieldTimeDetrend, Weights
from funciones.general_utils import init_logger
year_start = 1959 # eof
year_end = 2020

data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'

# ---------------------------------------------------------------------------- #
logger = init_logger('funciones.set_hgt_obs.py')

# ---------------------------------------------------------------------------- #
output_exists = os.path.isfile(f'{out_dir}hgt_jja_1959-2020.nc')

if output_exists is False:
    logger.info('archivo hgt no existe, computando...')
    hgt_or = xr.open_dataset(f'{data_dir}ERA5_HGT200_40-20.nc')
    if 'longitude' in list(hgt_or.dims):
        hgt_or = hgt_or.rename({'latitude': 'lat', 'longitude': 'lon'})

    hgt_or_name_var = list(hgt_or.data_vars)[0]
    hgt_or = Weights(hgt_or)
    hgt_or = hgt_or.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-01'))

    hgt_or = hgt_or.rename({hgt_or_name_var: 'var'})
    logger.info('Detrend..')
    hgt = xrFieldTimeDetrend(hgt_or, 'time')  # Detrend
    logger.info('Rolling..')
    hgt = hgt.rolling(time=3, center=True).mean()  # estacional
    hgt = hgt.sel(time=hgt.time.dt.month.isin([6, 7, 8])) # pesa menos
    hgt = ((hgt.groupby('time.month') - hgt.groupby('time.month').mean('time'))
           / hgt.groupby('time.month').std('time'))  # standarizacion

    logger.info('Saving..')
    hgt.to_netcdf(f'{out_dir}hgt_jja_1959-2020.nc')

    del hgt, hgt_or

hgt = xr.open_dataset(f'{out_dir}hgt_jja_1959-2020.nc')





