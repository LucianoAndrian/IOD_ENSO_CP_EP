"""
Pre-procesamiento hgt200, prec, tref, tsigma095
Anomalías respecto a la climatologia del hindcast y detrend de las anomalias
(similar 2_fixCFSv2_DMI_N34.py)
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from funciones.SelectNMME import SelectNMMEFiles
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
save_nc = True

variables = ['hgt', 'tref', 'prec', 'hgt750', 'vpot200']
variables = ['vpot200']
# Funciones ------------------------------------------------------------------ #
def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, main_month_season):

    for l in [0,1,2,3]:
        season_1982_1998 = \
            data_1982_1998.sel(
                time=data_1982_1998.time.dt.month.isin(main_month_season-l),
                L=l)
        season_1999_2011 = \
            data_1999_2011.sel(
                time=data_1999_2011.time.dt.month.isin(main_month_season-l),
                L=l)

        if l==0:
            season_clim_1982_1998 = season_1982_1998.mean(['r', 'time'])
            season_clim_1999_2011 = season_1999_2011.mean(['r', 'time'])

            season_anom_1982_1998 = season_1982_1998 - season_clim_1982_1998
            season_anom_1999_2011 = season_1999_2011 - season_clim_1999_2011
        else:
            season_clim_1982_1998 = \
                xr.concat([season_clim_1982_1998,
                           season_1982_1998.mean(['r', 'time'])], dim='L')
            season_clim_1999_2011 = \
                xr.concat([season_clim_1999_2011,
                           season_1999_2011.mean(['r', 'time'])], dim='L')

            aux_1982_1998 = \
                season_1982_1998 - season_1982_1998.mean(['r', 'time'])
            aux_1999_2011 = \
                season_1999_2011 - season_1999_2011.mean(['r', 'time'])

            season_anom_1982_1998 = \
                xr.concat([season_anom_1982_1998, aux_1982_1998], dim='time')
            season_anom_1999_2011 =\
                xr.concat([season_anom_1999_2011, aux_1999_2011], dim='time')

    return season_clim_1982_1998, season_clim_1999_2011,\
           season_anom_1982_1998, season_anom_1999_2011

def Anom_SeasonRealTime(data_realtime, season_clim_1999_2011,
                                main_month_season):

    for l in [0,1,2,3]:
        season_data = data_realtime.sel(
            time=data_realtime.time.dt.month.isin(main_month_season-l), L=l)
        aux_season_clim_1999_2011 = season_clim_1999_2011.sel(L=l)

        #Anomalia
        season_anom = season_data - aux_season_clim_1999_2011

        if l==0:
            season_anom_f = season_anom
        else:
            season_anom_f = xr.concat([season_anom_f, season_anom], dim='time')

    return season_anom_f

def Detrend_Seasons(season_anom_1982_1998, season_anom_1999_2011,
                    main_month_season):

    for l in [0,1,2,3]:
        #1982-1998
        aux_season_anom_1982_1998 = season_anom_1982_1998.sel(
            time=season_anom_1982_1998.time.dt.month.isin(main_month_season-l),
        L=l)

        aux = aux_season_anom_1982_1998.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(
            aux_season_anom_1982_1998['time'], aux[list(aux.data_vars)[0]])

        if l == 0:
            season_anom_1982_1998_detrened = \
                aux_season_anom_1982_1998 - aux_trend
        else:
            aux_detrend = aux_season_anom_1982_1998 - aux_trend
            season_anom_1982_1998_detrened = \
                xr.concat([season_anom_1982_1998_detrened, aux_detrend],
                          dim='time')

    # 1999-2011
        aux_season_anom_1999_2011 = season_anom_1999_2011.sel(
            time=season_anom_1999_2011.time.dt.month.isin(main_month_season - l),
        L=l)

        aux = aux_season_anom_1999_2011.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(
            aux_season_anom_1999_2011['time'], aux[list(aux.data_vars)[0]])
        if l==0:
            season_anom_1999_2011_detrend = \
                aux_season_anom_1999_2011 - aux_trend
        else:
            aux_detrend = aux_season_anom_1999_2011 - aux_trend
            season_anom_1999_2011_detrend = xr.concat(
                [season_anom_1999_2011_detrend, aux_detrend], dim='time')

    return season_anom_1982_1998_detrened, season_anom_1999_2011_detrend

def Anom_Detrend_SeasonRealTime(data_realtime, season_clim_1999_2011,
                                main_month_season):

    for l in [0,1,2,3]:
        season_data = data_realtime.sel(
            time=data_realtime.time.dt.month.isin(main_month_season-l), L=l)
        aux_season_clim_1999_2011 = season_clim_1999_2011.sel(L=l)

        #Anomalia
        season_anom = season_data - aux_season_clim_1999_2011

        #Detrend
        aux = season_anom.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(
            season_anom['time'], aux[list(aux.data_vars)[0]])

        if l==0:
            season_anom_detrend = season_anom - aux_trend
        else:
            aux_detrend = season_anom - aux_trend
            season_anom_detrend = xr.concat(
                [season_anom_detrend, aux_detrend], dim='time')

    return season_anom_detrend

def SetDataCFSv2(data, sa=True):
    data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
    if sa:
        data = data.sel(L=[0.5, 1.5, 2.5, 3.5], r=slice(1, 24),
                        lon=slice(275, 331), lat=slice(-70, 20))
    else:
        data = data.sel(L=[0.5, 1.5, 2.5, 3.5], r=slice(1, 24))

    data['L'] = [0, 1, 2, 3]
    data = xr.decode_cf(fix_calendar(data))  # corrigiendo fechas

    return data

def SplitFilesByMonotonicity(files):
    """ Divide la lista de archivos en segmentos donde el índice S
    sea monotónico """
    def test_open(sublist):
        try:
            xr.open_mfdataset(sublist, decode_times=False)
            return False
        except ValueError as e:
            print(e)
            return True

        xr.backends.file_manager._FILE_CACHE.clear()

    low, high = 0, len(files)

    mid = (low + high) // 2

    while high - low > 1:
        mid = (low + high) // 2
        subset = files[low:mid]

        if test_open(subset):  # Falla
            high = mid
        else:
            low = mid

        print(mid)

    return files[:mid], files[mid:]

def purge_extra_dim(data, dims_to_keep):

    name_var = list(data.data_vars)[0]

    extra_dim = [d for d in data[name_var].dims if d not in dims_to_keep]

    for dim in extra_dim:
        first_val = data[dim].values[0]
        data = data.sel({dim: first_val}).drop(dim)
        print(f'drop_dim: {dim}')

    return data

# ---------------------------------------------------------------------------- #
for v in variables:
    print(f"{v} ------------------------------------------------------------ #")

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
    print('hindcast')
    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                            dir=dir_hc, All=True)
    files = sorted(files, key=lambda x: x.split()[0])
    files = [x for x in files if all(
        year not in x for year in
        ['_2021', '_2022', '_2023', '_2024', '_2025'])]

    # Open files --------------------------------------------------------- #
    try:
        data = xr.open_mfdataset(files, decode_times=False)
        data = SetDataCFSv2(data, sa)

    except:
        print('Error en la monotonia de la dimencion S')
        print('Usando SplitFilesByMonotonicity...')
        files0, files1 = SplitFilesByMonotonicity(files)

        data0 = xr.open_mfdataset(files0, decode_times=False)
        data1 = xr.open_mfdataset(files1, decode_times=False)

        data0 = SetDataCFSv2(data0, sa)
        data1 = SetDataCFSv2(data1, sa)

        data = xr.concat([data0, data1], dim='time')

    print('Open files done')

    # -------------------------------------------------------------------- #
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
    son_clim_82_98, son_clim_99_11, son_anom_82_98, son_anom_99_11 = \
        TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, 10)

    son_anom_82_98_detrend, son_anom_99_11_detrend = \
        Detrend_Seasons(data_1982_1998, data_1999_2011, 10)

    son_hindcast = xr.concat([son_anom_82_98, son_anom_99_11], dim='time')

    son_hindcast_detrend = xr.concat(
        [son_anom_82_98_detrend, son_anom_99_11_detrend], dim='time')

    son_hindcast.to_netcdf(f"{out_dir}{v}_aux_hindcast_no_detrend_son.nc")
    son_hindcast_detrend.to_netcdf(f"{out_dir}{v}_aux_hindcast_detrend_son.nc")
    son_clim_99_11.to_netcdf(f"{out_dir}{v}_aux_son_clim_99_11.nc")
    del son_hindcast, son_clim_82_98, son_clim_99_11

    # --------------------------------------------------------------------------
    # Real-time ----------------------------------------------------------------
    print('real-time')
    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                            dir=dir_rt, All=True)
    files = sorted(files, key=lambda x: x.split()[0])

    files = [x for x in files if all(
        year not in x for year in
        ['_2021', '_2022', '_2023', '_2024', '_2025'])]

    # Open files --------------------------------------------------------- #
    try:
        data = xr.open_mfdataset(files, decode_times=False)
        data = SetDataCFSv2(data, sa)

    except:
        print('Error en la monotonia de la dimencion S')
        print('Usando SplitFilesByMonotonicity...')
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
            print('Archivos duplicados en la selecciones de RealTime')

    print('Open files done')

    # ------------------------------------------------------------------------ #
    data = purge_extra_dim(data, dims_to_keep=['L', 'lat', 'lon', 'r', 'time'])

    # ------------------------------------------------------------------------ #
    data = data.rolling(time=3, center=True).mean()

    # - Anomalias detrend por seasons ------------------------------------------
    # --------------------------------------------------------------------------
    son_clim_99_11 = xr.open_dataset(f"{out_dir}{v}_aux_son_clim_99_11.nc")
    son_hindcast_no_detrend = \
        xr.open_dataset(f"{out_dir}{v}_aux_hindcast_no_detrend_son.nc")
    son_hindcast_detrend = \
        xr.open_dataset(f"{out_dir}{v}_aux_hindcast_detrend_son.nc")

    son_realtime_no_detrend = Anom_SeasonRealTime(data, son_clim_99_11, 10)
    son_realtime_no_detrend.load()

    son_realtime_detrend = Anom_Detrend_SeasonRealTime(
        data, son_clim_99_11, 10)
    son_realtime_detrend.load()

    print('concat')
    son_f = xr.concat(
        [son_hindcast_no_detrend, son_realtime_no_detrend], dim='time')
    son_f_detrend = xr.concat(
        [son_hindcast_detrend, son_realtime_detrend], dim='time')

    # save ---------------------------------------------------------------------
    if save_nc:
        son_f.to_netcdf(f"{out_dir}{v}_son_no_detrend.nc")
        son_f_detrend.to_netcdf(f"{out_dir}{v}_son_detrend.nc")

    del son_realtime_no_detrend, son_hindcast_no_detrend, data, son_f, \
        son_realtime_detrend, son_hindcast_detrend, son_f_detrend

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')