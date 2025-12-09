"""
Pre-procesamiento hgt200, prec, tref, tsigma095
Anomalías respecto a la climatologia del hindcast y detrend de las anomalias
(similar 2_fixCFSv2_DMI_N34.py)
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from funciones.SelectNMME import SelectNMMEFiles
from funciones.preselect_utils import TwoClim_Anom_Seasons, \
    Anom_SeasonRealTime, Detrend_Seasons, Anom_Detrend_SeasonRealTime, \
    SetDataCFSv2, SplitFilesByMonotonicity, purge_extra_dim
from funciones.general_utils import init_logger
import warnings
warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
save_nc = True

# ---------------------------------------------------------------------------- #
def SaveNC(data, dir):
    logger.info(f'Loading data..')
    data = data.load()
    logger.info(f'Saving data..')
    data.to_netcdf(dir)
    logger.info('Data saved')

# ---------------------------------------------------------------------------- #
logger = init_logger('0_preselect_cfsv2.log')

# ---------------------------------------------------------------------------- #
# Las que deberian:
#variables = ['sst', 'hgt', 'tref', 'prec', 'hgt750', 'vpot200']
# SST ya esta
#variables = ['hgt', 'tref', 'prec', 'hgt750', 'vpot200']
# Por ahora:
variables = ['hgt', 'vpot200']

# ---------------------------------------------------------------------------- #
for v in variables:
    logger.info(f'variable: {v}')

    if v == 'T0995sigma' or v == 'hgt' or v=='hgt750' or v=='vpot200':
        dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
        dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
    else:
        dir_hc = '/pikachu/datos/osman/nmme/monthly/hindcast/'
        dir_rt = '/pikachu/datos/osman/nmme/monthly/real_time/'

    if v=='prec' or v=='tref':
        sa=True
    else:
        sa=False
    # usando SelectNMMEFiles con All=True,
    # abre TODOS los archivos .nc de la ruta en dir

    # HINDCAST -----------------------------------------------------------------
    logger.info(f'Hindcast')
    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                            dir=dir_hc, All=True)
    files = sorted(files, key=lambda x: x.split()[0])
    files = [x for x in files if all(
        year not in x for year in
        ['_2021', '_2022', '_2023', '_2024', '_2025'])]

    # Open files --------------------------------------------------------- #
    logger.info('Hindcast open files:')
    try:
        data = xr.open_mfdataset(files, decode_times=False)
        data = SetDataCFSv2(data, sa)

    except Exception as e:
        logger.warning('Error en la monotonía de la dimensión S')
        logger.warning('Usando SplitFilesByMonotonicity...')
        logger.warning(f'Error: {e}')

        files0, files1 = SplitFilesByMonotonicity(files)

        data0 = xr.open_mfdataset(files0, decode_times=False)
        data1 = xr.open_mfdataset(files1, decode_times=False)

        data0 = SetDataCFSv2(data0, sa)
        data1 = SetDataCFSv2(data1, sa)

        data = xr.concat([data0, data1], dim='time')

    logger.info('Hindcast files opened successfully')

    # -------------------------------------------------------------------- #
    logger.info('Hindcast purge_extra_dim')
    data = purge_extra_dim(data, dims_to_keep=['L', 'lat', 'lon', 'r', 'time'])

    # -------------------------------------------------------------------- #
    # media movil de 3 meses para separar en estaciones
    data = data.rolling(time=3, center=True).mean()

    # 1982-1998, 1999-2011
    data_1982_1998 = \
        data.sel(time=data.time.dt.year.isin(np.linspace(1982, 1998, 17)))
    data_1999_2011 = \
        data.sel(time=data.time.dt.year.isin(np.linspace(1999, 2011, 13)))

    # - Climatologias y anomalias detrend por seasons --------------------------
    # --------------------------------------------------------------------------
    logger.info('Hindcast compute...')
    logger.info('TwoClim_Anom_Seasons')
    son_clim_82_98, son_clim_99_11, son_anom_82_98, son_anom_99_11 = \
        TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, 10)

    logger.info('Detrend_Seasons')
    son_anom_82_98_detrend, son_anom_99_11_detrend = \
        Detrend_Seasons(data_1982_1998, data_1999_2011, 10)

    son_hindcast = xr.concat([son_anom_82_98, son_anom_99_11], dim='time')

    son_hindcast_detrend = xr.concat(
        [son_anom_82_98_detrend, son_anom_99_11_detrend], dim='time')

    logger.info('Saving: no detrend...')
    SaveNC(son_hindcast, f'{out_dir}{v}_aux_hindcast_no_detrend_son.nc')

    logger.info('Saving: detrend...')
    SaveNC(son_hindcast_detrend, f'{out_dir}{v}_aux_hindcast_detrend_son.nc')

    logger.info('Saving: clim...')
    SaveNC(son_clim_99_11, f'{out_dir}{v}_aux_son_clim_99_11.nc')

    del son_hindcast, son_clim_82_98, son_clim_99_11

    # --------------------------------------------------------------------------
    # Real-time ----------------------------------------------------------------
    logger.info('Realtime')
    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                            dir=dir_rt, All=True)
    files = sorted(files, key=lambda x: x.split()[0])

    files = [x for x in files if all(
        year not in x for year in
        ['_2021', '_2022', '_2023', '_2024', '_2025'])]

    # Open files --------------------------------------------------------- #
    logger.info('realtime open files:')
    try:
        data = xr.open_mfdataset(files, decode_times=False)
        data = SetDataCFSv2(data, sa)

    except:
        logger.warning('Error en la monotonía de la dimensión S')
        logger.warning('Usando SplitFilesByMonotonicity...')
        logger.warning(f'Error: {e}')

        files0, files1 = SplitFilesByMonotonicity(files)

        if len(np.intersect1d(files0, files1)) == 0:

            # Sin embargo, S parece estar duplicada
            data0 = xr.open_mfdataset(files0, decode_times=False)
            data1 = xr.open_mfdataset(files1, decode_times=False)

            data0 = SetDataCFSv2(data0, sa)
            data1 = SetDataCFSv2(data1, sa)

            t_duplicados = np.intersect1d(data0.time.values, data1.time.values)
            no_duplicados = ~np.isin(data0.time.values, t_duplicados)
            data0 = data0.sel(time=no_duplicados)

            data = xr.concat([data0, data1], dim='time')
        else:
            logger.info('Archivos duplicados en la selecciones de RealTime')

    logger.info('Realtime files opened successfully')

    # ------------------------------------------------------------------------ #
    logger.info('Realtime purge_extra_dim')
    data = purge_extra_dim(data, dims_to_keep=['L', 'lat', 'lon', 'r', 'time'])

    # ------------------------------------------------------------------------ #
    data = data.rolling(time=3, center=True).mean()

    # - Anomalias detrend por seasons ------------------------------------------
    # --------------------------------------------------------------------------
    logger.info('Realtime compute...')
    son_clim_99_11 = xr.open_dataset(f"{out_dir}{v}_aux_son_clim_99_11.nc")
    son_hindcast_no_detrend = \
        xr.open_dataset(f"{out_dir}{v}_aux_hindcast_no_detrend_son.nc")
    son_hindcast_detrend = \
        xr.open_dataset(f"{out_dir}{v}_aux_hindcast_detrend_son.nc")

    son_realtime_no_detrend = Anom_SeasonRealTime(data, son_clim_99_11, 10)
    logger.info('realtime_no_detrend loading')
    son_realtime_no_detrend.load()

    son_realtime_detrend = Anom_Detrend_SeasonRealTime(
        data, son_clim_99_11, 10)
    son_realtime_detrend.load()

    logger.info('Concat hindcast + realtime')
    son_f = xr.concat(
        [son_hindcast_no_detrend, son_realtime_no_detrend], dim='time')
    son_f_detrend = xr.concat(
        [son_hindcast_detrend, son_realtime_detrend], dim='time')

    # save ---------------------------------------------------------------------
    if save_nc:
        logger.info('saving nc')
        logger.info('Saving: no detrend...')
        SaveNC(son_f, f"{out_dir}{v}_son_no_detrend.nc")

        logger.info('Saving: detrend...')
        SaveNC(son_f_detrend,f"{out_dir}{v}_son_detrend.nc")

    del son_realtime_no_detrend, son_hindcast_no_detrend, data, son_f, \
        son_realtime_detrend, son_hindcast_detrend, son_f_detrend

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #