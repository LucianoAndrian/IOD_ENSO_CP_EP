"""
Funciones para el la seleccion y preprocesamiento de pronosticos del cfsv2
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import warnings
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
# ---------------------------------------------------------------------------- #
def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, main_month_season):
    season_clim_1982_1998 = []
    season_clim_1999_2011 = []
    season_anom_1982_1998 = []
    season_anom_1999_2011 = []
    for l in [0,1,2,3]:
        season_1982_1998 = \
            data_1982_1998.sel(
                time=data_1982_1998.time.dt.month.isin(main_month_season-l),
                L=l)
        season_1999_2011 = \
            data_1999_2011.sel(
                time=data_1999_2011.time.dt.month.isin(main_month_season-l),
                L=l)

        season_clim_1982_1998_L = season_1982_1998.mean(['r', 'time'])
        season_clim_1999_2011_L = season_1999_2011.mean(['r', 'time'])

        season_anom_1982_1998_L = season_1982_1998 - season_clim_1982_1998_L
        season_anom_1999_2011_L = season_1999_2011 - season_clim_1999_2011_L

        season_clim_1982_1998.append(season_clim_1982_1998_L)
        season_clim_1999_2011.append(season_clim_1999_2011_L)
        season_anom_1982_1998.append(season_anom_1982_1998_L)
        season_anom_1999_2011.append(season_anom_1999_2011_L)

    season_clim_1982_1998 = xr.concat(season_clim_1982_1998, dim='L')
    season_clim_1999_2011 = xr.concat(season_clim_1999_2011, dim='L')
    season_anom_1982_1998 = xr.concat(season_anom_1982_1998, dim='time')
    season_anom_1999_2011 = xr.concat(season_anom_1999_2011, dim='time')

    return season_clim_1982_1998, season_clim_1999_2011,\
           season_anom_1982_1998, season_anom_1999_2011

def Anom_SeasonRealTime(data_realtime, season_clim_1999_2011,
                                main_month_season):
    season_anom = []
    for l in [0,1,2,3]:
        season_data = data_realtime.sel(
            time=data_realtime.time.dt.month.isin(main_month_season-l), L=l)
        aux_season_clim_1999_2011 = season_clim_1999_2011.sel(L=l)

        #Anomalia
        season_anom.append(season_data - aux_season_clim_1999_2011)

    season_anom = xr.concat(season_anom, dim='time')

    return season_anom

def Detrend_Seasons(season_anom_1982_1998, season_anom_1999_2011,
                    main_month_season):

    season_anom_1982_1998_detrend = []
    season_anom_1999_2011_detrend = []
    for l in [0,1,2,3]:
        #1982-1998
        aux_season_anom_1982_1998 = season_anom_1982_1998.sel(
            time=season_anom_1982_1998.time.dt.month.isin(main_month_season-l),
        L=l)

        aux = aux_season_anom_1982_1998.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(
            aux_season_anom_1982_1998['time'], aux[list(aux.data_vars)[0]])

        season_anom_1982_1998_detrend.append(
            aux_season_anom_1982_1998 - aux_trend)

        # 1999-2011
        aux_season_anom_1999_2011 = season_anom_1999_2011.sel(
            time=season_anom_1999_2011.time.dt.month.isin(main_month_season - l),
        L=l)

        aux = aux_season_anom_1999_2011.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(
            aux_season_anom_1999_2011['time'], aux[list(aux.data_vars)[0]])

        season_anom_1999_2011_detrend.append(
            aux_season_anom_1999_2011 - aux_trend)

    season_anom_1982_1998_detrend = xr.concat(season_anom_1982_1998_detrend,
                                              dim='time')
    season_anom_1999_2011_detrend = xr.concat(season_anom_1999_2011_detrend,
                                              dim='time')

    return season_anom_1982_1998_detrend, season_anom_1999_2011_detrend

def Anom_Detrend_SeasonRealTime(data_realtime, season_clim_1999_2011,
                                main_month_season):
    season_anom_detrend = []
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

        season_anom_detrend.append(season_anom - aux_trend)

    season_anom_detrend = xr.concat(season_anom_detrend, dim='time')

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