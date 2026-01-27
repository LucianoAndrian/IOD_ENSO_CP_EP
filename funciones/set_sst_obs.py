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

data_dir = '/pikachu/datos/luciano.andrian/verif_2019_2023/'

# ---------------------------------------------------------------------------- #
import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
output_exists = os.path.isfile(f'{out_dir}sst_jja_1959-2020.nc')

if output_exists is False:
    logger.info('archivo sst no existe, computando...')
    sst_or = xr.open_dataset(f'{data_dir}sst.mnmean.nc')
    sst_or = sst_or.drop_dims('nbnds')
    sst_or_name_var = list(sst_or.data_vars)[0]
    sst_or = Weights(sst_or)
    sst_or = sst_or.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-01'))
    sst_or = sst_or.rename({sst_or_name_var: 'var'})

    logger.info('Detrend..')
    sst = xrFieldTimeDetrend(sst_or, 'time')  # Detrend
    logger.info('Rolling..')
    sst = sst.rolling(time=3, center=True).mean()  # estacional
    sst = sst.sel(time=sst.time.dt.month.isin([6, 7, 8])) # pesa menos
    sst = ((sst.groupby('time.month') - sst.groupby('time.month').mean('time'))
           / sst.groupby('time.month').std('time'))  # standarizacion

    logger.info('Saving..')
    sst.to_netcdf(f'{out_dir}sst_jja_1959-2020.nc')

    del sst, sst_or

sst = xr.open_dataset(f'{out_dir}sst_jja_1959-2020.nc')
