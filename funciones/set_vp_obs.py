"""
Computo de VP a partir de UV
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/aux/'

# ---------------------------------------------------------------------------- #
import os
import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind
from windspharm.examples import example_data_path
from funciones.general_utils import xrFieldTimeDetrend, Weights
from funciones.general_utils import init_logger

raw_data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'

# ---------------------------------------------------------------------------- #
import logging
logger = logging.getLogger(__name__)
v = 'UV200'
# ---------------------------------------------------------------------------- #
output_exists = os.path.isfile(f'{out_dir}vp_jja_from_UV200_w.nc')

if output_exists is False:
    logger.info('Archivo VP no existe, computando...')

    name_variables = ['u', 'v']

    for n_v in name_variables:
        n_v_exist = os.path.isfile(f'{out_dir}{n_v}_JJA_{v}_w_detrend.nc')
        if n_v_exist is False:
            logger.info(f'Computando {n_v}...')
            data = xr.open_dataset(f'{raw_data_dir}ERA5_{v}_40-20.nc')
            if n_v == 'u':
                logger.info('Drop v')
                data = data.drop_vars('v')
            elif n_v == 'v':
                logger.info('Drop u')
                data = data.drop_vars('u')

            data = data.rename({n_v: 'var'})
            data = data.rename({'longitude': 'lon'})
            data = data.rename({'latitude': 'lat'})

            logger.info('Weights and Rolling...')
            data = Weights(data)
            data = data.rolling(time=3, center=True).mean()

            data_mm = data.sel(time=data.time.dt.month.isin(7))
            logger.info('Detrend...')
            data_mm = xrFieldTimeDetrend(data_mm, 'time')
            logger.info('to_netcdf...')
            data_mm = data_mm + data.mean('time')
            data_mm.to_netcdf(f'{out_dir}{n_v}_JJA_{v}_w_detrend.nc')

            del data_mm


    logger.info('Computando VP from UV')
    ds = xr.open_dataset(example_data_path('uwnd_mean.nc'))
    uwnd_aux = ds['uwnd']

    # al haber aplicado el rolling el primer y ultimo
    # mes no estan y falla Vectorwind
    uwnd = xr.open_dataset(f'{out_dir}u_JJA_{v}_w_detrend.nc')
    uwnd = uwnd.sel(time=slice('1959-02-01', '2020-11-01'))
    vwnd = xr.open_dataset(f'{out_dir}v_JJA_{v}_w_detrend.nc')
    vwnd = vwnd.sel(time=slice('1959-02-01', '2020-11-01'))

    uwnd = uwnd.interp(lat=np.linspace(uwnd_aux.latitude.values[0],
                                       uwnd_aux.latitude.values[-1], 179),
                       lon=np.linspace(uwnd_aux.longitude.values[0],
                                       uwnd_aux.longitude.values[-1], 359))
    vwnd = vwnd.interp(lat=np.linspace(uwnd_aux.latitude.values[0],
                                       uwnd_aux.latitude.values[-1], 179),
                       lon=np.linspace(uwnd_aux.longitude.values[0],
                                       uwnd_aux.longitude.values[-1], 359))

    uwnd = uwnd.to_array()
    vwnd = vwnd.to_array()

    w = VectorWind(uwnd, vwnd)

    vp = w.sfvp()[1]
    vp.to_netcdf(f'{out_dir}vp_jja_from_{v}_w.nc')
    del vp
    logger.info('Done')

vp = xr.open_dataset(f'{out_dir}vp_jja_from_{v}_w.nc')
