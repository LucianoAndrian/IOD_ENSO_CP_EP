"""
OBS Composites
"""
# ---------------------------------------------------------------------------- #
save = True
thr_sd = [0.5, 0.75, 1]
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'
cases_dates = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP_OBS/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import colors

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from Funciones import PlotFinal
if save:
    dpi = 300
else:
    dpi = 100

from test_indices import year_start, year_end
from SelectEvents_obs import Compute

# Funciones ------------------------------------------------------------------ #
def OpenSetCases(idx1, idx2, idx3, phase, dir):
    idx1 = idx1.lower()
    idx2 = idx2.lower()
    idx3 = idx3.lower()

    cases = {}
    cases['neutros'] = xr.open_dataset(f'{dir}OBS_neutros.nc')

    indices = [idx1, idx2, idx3]

    for i_num, i in enumerate(indices):

        idx, idx_aux2, idx_aux3 = Combinations(i, indices)

        cases[i] = {}

        # puro
        cases[i]['puros'] = xr.open_dataset(
            f'{dir}OBS_puros_{i}_{phase}.nc')

        # doble
        try:
            doble_2 = xr.open_dataset(
                f'{dir}OBS_simultaneos_dobles_{i}_{idx_aux2}_{phase}'
                f'.nc')

            doble_3 = xr.open_dataset(
                f'{dir}OBS_simultaneos_dobles_{i}_{idx_aux3}_{phase}'
                f'.nc')

            cases[i]['dobles'] = {}
            cases[i]['dobles'][idx_aux2] = doble_2
            cases[i]['dobles'][idx_aux3] = doble_3
        except:
            pass

        # triple
        try:
            # hay uno solo
            triple = xr.open_dataset(
                f'{dir}OBS_simultaneos_triples_{i}_{idx_aux2}_{idx_aux3}'
                f'_{phase}.nc')

            cases[i]['triples'] = {}
            cases[i]['triples'] = triple
        except:
            pass

    return cases

def Combinations(idx, indices):
    if idx == indices[0]:
        idx_aux2 = indices[1]
        idx_aux3 = indices[2]
    elif idx == indices[1]:
        idx_aux2 = indices[0]
        idx_aux3 = indices[2]
    elif idx == indices[2]:
        idx_aux2 = indices[0]
        idx_aux3 = indices[1]
    else:
        print('Error')
        idx, idx_aux2, idx_aux3 = None, None, None

    return idx, idx_aux2, idx_aux3

def Decider(i, k, indices, sk=None):
    if k == 'puros':
        if i == indices[0]:
            pos = 1
        elif i == indices[1]:
            pos = 3
        elif i == indices[2]:
            pos = 5
    elif k == 'triples':
        pos = 4
    elif k == 'dobles':
        if sk == indices[1]:
            pos = 0
        else:
            pos = 2

    return pos

def CompositeSimple(original_data, index):

    if len(index) != 0:
        comp_field = original_data.sel(
            time=original_data.time.isin([index]))
        if len(comp_field.time) != 0:
            comp_field = comp_field.mean(['time'], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field, len(index.time)
    else:
        print(' len index = 0')

def MakeComposite(data, cases):
    cases_ordenados = [None, None, None, None, None, None]
    titles = [None, None, None, None, None, None]

    neutros_comp = CompositeSimple(data, cases['neutros'].time)[0]

    indices = list(cases.keys())[1:]

    for i in indices:
        i_case = cases[i]
        i_case_k = list(i_case.keys())

        if len(i_case_k) > 1:
            for key in i_case_k:
                key_case = i_case[key]
                subkeys = list(key_case.keys())

                if len(subkeys) > 1:
                    for sk in subkeys:
                        pos = Decider(i, key, indices, sk)
                        aux_comp, len_comp = CompositeSimple(data, key_case[sk].time)
                        comp = aux_comp - neutros_comp
                        if comp is None:
                            comp = neutros_comp*0
                        cases_ordenados[pos] = comp
                        titles[pos] = f'{i.upper()} - {sk.upper()} - N:{len_comp}'

                else:
                    pos = Decider(i, key, indices)
                    aux_comp, len_comp = CompositeSimple(data, key_case.time)
                    comp = aux_comp - neutros_comp
                    if comp is None:
                        comp = neutros_comp * 0
                    cases_ordenados[pos] = comp
                    titles[pos] = f'{i.upper()} {key} - N:{len_comp}'

        else:
            key = i_case_k[0]
            pos = Decider(i, key, indices)
            aux_comp, len_comp = CompositeSimple(data, i_case[key].time)
            comp = aux_comp - neutros_comp
            if comp is None:
                comp = neutros_comp * np.nan
            cases_ordenados[pos] = comp
            titles[pos] = f'{i.upper()} - {key} - N:{len_comp}'

    for c_num, c in enumerate(cases_ordenados):
        if c is None:
            cases_ordenados[c_num] = neutros_comp*0

    cases_ordenados = xr.concat(cases_ordenados, dim='plots')
    var_name = list(data.data_vars)[0]
    if var_name != 'var':
        cases_ordenados = cases_ordenados.rename({var_name: 'var'})
    #cases_ordenados = Weights(cases_ordenados)

    return cases_ordenados, titles

def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def OpenSetData(data):

    if len(data.sel(lat=slice(20, -60)).lat) > 0:
        data = data.sel(lat=slice(20, -60), lon=slice(275, 330))
    else:
        data = data.sel(lat=slice(-60, 20), lon=slice(275, 330))
        data = data.sel(lat=slice(None, None, -1))


    time_index = data.time.to_index()

    mask_1016 = (time_index.month == 10) & (time_index.day == 16)
    if mask_1016.any():
        new_times = time_index.to_series().apply(
            lambda d: pd.Timestamp(
                f"{d.year}-10-01") if d.month == 10 and d.day == 16 else d
        )
        data = data.assign_coords(time=new_times.values)

    return data

# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'

dirs = [data_dir, data_dir, data_dir_t_pp, data_dir_t_pp]

variables = ['HGT200_SON_mer_d_w', 'HGT750_SON_mer_d_w',
             'tcru_w_c_d_0.25_SON', 'ppgpcc_w_c_d_1_SON']


cbar = colors.ListedColormap(['#641B00', '#892300', '#9B1C00', '#B9391B',
                              '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC', '#FFE6E6', 'white',
                              '#E6F2FF', '#B3DBFF',
                              '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF',
                              '#014A9B', '#013A75',
                              '#012A52'][::-1])

cbar.set_over('#4A1500')
cbar.set_under('#001F3F')
cbar.set_bad(color='white')

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                 '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07',
                                 '#543005', ][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

# ---------------------------------------------------------------------------- #
data_sst = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
data_sst = data_sst.rename({'sst':'var'})
data_sst = data_sst.drop('time_bnds')
data_sst = data_sst.rolling(time=3, center=True).mean()
data_sst = data_sst.sel(time=data_sst.time.dt.month.isin(10))
data_sst = Detrend(data_sst, 'time')

# ---------------------------------------------------------------------------- #
scale_pp = np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45])
scale_t = [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1]
aux_scale_hgt = [-100, -50, -30, -15, -5, 5, 15, 30, 50, 100]
aux_scale_hgt200 = [-150, -100, -50, -25, -10, 10, 25, 50, 100, 150]
variables = ['sst', 'tref', 'prec', 'hgt']
aux_scales = [scale_t, scale_t, scale_pp, aux_scale_hgt200]
aux_cbar = [cbar, cbar, cbar_pp, cbar]
dirs = [None, data_dir_t_pp, data_dir_t_pp, data_dir]
variables = [None, 'tcru_w_c_d_0.25_SON', 'ppgpcc_w_c_d_1_SON',
             'HGT200_SON_mer_d_w']

for sd in thr_sd:
    # ------------------------------------------------------------------------ #
    Compute(sd)
    # ------------------------------------------------------------------------ #
    for v, dir, scale, cbar in zip(variables, dirs, aux_scales, aux_cbar):

        if dir is None:
            data = data_sst
            v = 'sst'
        else:
            data = xr.open_dataset(f'{dir}{v}.nc')

        data = data.sel(time=slice(f'{year_start}-10-01', f'{year_end}-10-01'))

        if 'tcru' in v or 'ppgpcc' in v:
            use_hgt = True
            data = OpenSetData(data)
        else:
            use_hgt = False

        for f in ['pos', 'neg']:
            cases = OpenSetCases(idx1='dmi', idx2='ep', idx3='cp',
                                 phase=f, dir=cases_dates)

            cases_ordenados, titles = MakeComposite(data, cases)

            name = f'{v}_comp_IOD-EP-CP_{f}'
            if use_hgt:
                map = 'sa'
                cases_hgt750 = OpenSetCases(idx1='dmi', idx2='ep', idx3='cp',
                                            phase=f, dir=cases_dates)
                data_hgt = xr.open_dataset(f'{data_dir}HGT750_SON_mer_d_w.nc')
                cases_ordenados_hgt, _ = MakeComposite(data_hgt, cases_hgt750)

                data_ctn = cases_ordenados_hgt
                levels_ctn = aux_scale_hgt
                data_ctn_no_ocean_mask = True
                ocean_mask = True
                high = 3
            else:
                map = 'hs'
                data_ctn = cases_ordenados
                levels_ctn = scale
                ocean_mask = False
                high = 1.3

            name_fig = f'comp_{v}_{f}_{sd}SD_OBS'
            PlotFinal(data=cases_ordenados, levels=scale, cmap=cbar,
                      titles=titles, namefig=name_fig, map=map,
                      save=save, dpi=dpi, out_dir=out_dir,
                      data_ctn=data_ctn, color_ctn='k', high=high,
                      num_cases=None, num_cases_data=None, num_cols=3,
                      ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
                      data_ctn_no_ocean_mask=True)

    # ------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------ #
