"""
Composites
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'
cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'

plot_td = True
plot_n = True

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

# ---------------------------------------------------------------------------- #
def MakerMaskSig(data, r_crit):
    mask_sig = data.where((data < -1 * r_crit) | (data > r_crit))
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

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

def OpenSetCases(var, idx1, idx2, idx3, phase, dir):


    if 'prec' in var:
        fix = 30
    elif 'tref' in var or var == 'sst':
        fix = 1
    elif var:
        fix = 9.8

    var = var.lower()
    idx1 = idx1.lower()
    idx2 = idx2.lower()
    idx3 = idx3.lower()

    cases = {}
    # neutro
    cases['neutros'] = xr.open_dataset(f'{dir}{var}_neutros_SON.nc') * fix

    indices = [idx1, idx2, idx3]

    for i_num, i in enumerate(indices):

        idx, idx_aux2, idx_aux3 = Combinations(i, indices)

        cases[i] = {}

        # puro
        cases[i]['puros'] = xr.open_dataset(
            f'{dir}{var}_puros_{i}_{phase}_SON.nc') * fix

        # doble
        try:
            doble_2 = xr.open_dataset(
                f'{dir}{var}_simultaneos_dobles_{i}_{idx_aux2}_{phase}'
                f'_SON.nc') * fix

            doble_3 = xr.open_dataset(
                f'{dir}{var}_simultaneos_dobles_{i}_{idx_aux3}_{phase}'
                f'_SON.nc') * fix

            cases[i]['dobles'] = {}
            cases[i]['dobles'][idx_aux2] = doble_2
            cases[i]['dobles'][idx_aux3] = doble_3
        except:
            pass

        # triple
        try:
            # hay uno solo
            triple = xr.open_dataset(
                f'{dir}{var}_simultaneos_triples_{i}_{idx_aux2}_{idx_aux3}'
                f'_{phase}_SON.nc') * fix

            cases[i]['triples'] = {}
            cases[i]['triples'] = triple
        except:
            pass

    return cases

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

def MakeComposite(cases):
    cases_ordenados = [None, None, None, None, None, None]
    titles = [None, None, None, None, None, None]

    neutro = cases['neutros'].mean('time')

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
                        aux_comp = key_case[sk].mean('time')
                        len_comp = len(key_case[sk].time)
                        comp = aux_comp - neutro
                        cases_ordenados[pos] = comp
                        titles[pos] = f'{i.upper()} - {sk.upper()} - N:{len_comp}'

                else:
                    pos = Decider(i, key, indices)
                    aux_comp = key_case.mean('time')
                    len_comp = len(key_case.time)
                    comp = aux_comp - neutro
                    cases_ordenados[pos] = comp
                    titles[pos] = f'{i.upper()} {key} - N:{len_comp}'

        else:
            key = i_case_k[0]
            pos = Decider(i, key, indices)
            aux_comp = i_case[key].mean('time')
            len_comp = len(i_case[key].time)
            comp = aux_comp - neutro
            cases_ordenados[pos] = comp
            titles[pos] = f'{i.upper()} - {key} - N:{len_comp}'

    cases_ordenados = xr.concat(cases_ordenados, dim='plots')
    var_name = list(cases_ordenados.data_vars)[0]
    cases_ordenados = cases_ordenados.rename({var_name: 'var'})
    cases_ordenados = Weights(cases_ordenados)

    return cases_ordenados, titles

# ---------------------------------------------------------------------------- #
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

scale_pp = np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45])
scale_t = [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1]
aux_scale_hgt = [-100, -50, -30, -15, -5, 5, 15, 30, 50, 100]
aux_scale_hgt200 = [-150, -100, -50, -25, -10, 10, 25, 50, 100, 150]
variables = ['sst2', 'tref', 'prec', 'hgt']
aux_scales = [scale_t, scale_t, scale_pp, aux_scale_hgt200]
aux_cbar = [cbar, cbar, cbar_pp, cbar]

print('# CFSv2 Composite --------------------------------------------------- #')

for v, scale, cbar in zip(variables, aux_scales, aux_cbar):
    for f in ['pos', 'neg']:
        cases = OpenSetCases(var=v,
                             idx1='dmi', idx2='ep', idx3='cp',
                             phase=f,
                             dir=cases_fields)

        cases_ordenados, titles = MakeComposite(cases)

        name = f'{v}_comp_IOD-EP-CP_{f}'

        if v == 'tref' or v == 'prec':
            map = 'sa'

            cases_hgt750 = OpenSetCases(var='hgt',
                                        idx1='dmi', idx2='ep', idx3='cp',
                                        phase=f,
                                        dir=cases_fields)

            cases_ordenados_hgt, _ = MakeComposite(cases_hgt750)
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

        name_fig = f'comp_{v}_{f}'
        PlotFinal(data=cases_ordenados, levels=scale, cmap=cbar,
                  titles=titles, namefig=name_fig, map=map,
                  save=save, dpi=dpi, out_dir=out_dir,
                  data_ctn=data_ctn, color_ctn='k', high=high,
                  num_cases=None, num_cases_data=None, num_cols=3,
                  ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
                  data_ctn_no_ocean_mask=True)

print('# --------------------------------------------------------------------#')
print('# Td -----------------------------------------------------------------#')
if plot_td:
    out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/comp_td/'
    cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/' \
                    'aux_ep_cp_t/'

    variables = ['sst']
    aux_scales = [scale_t]
    aux_cbar = [cbar]

    for v, scale, cbar in zip(variables, aux_scales, aux_cbar):
        for f in ['pos', 'neg']:
            cases = OpenSetCases(var=v,
                                 idx1='dmi', idx2='ep', idx3='cp',
                                 phase=f,
                                 dir=cases_fields)

            cases_ordenados, titles = MakeComposite(cases)

            name = f'{v}_comp_IOD-EP-CP_{f}'

            if v == 'tref' or v == 'prec':
                map = 'sa'

                cases_hgt750 = OpenSetCases(var='hgt',
                                            idx1='dmi', idx2='ep', idx3='cp',
                                            phase=f,
                                            dir=cases_fields)

                cases_ordenados_hgt, _ = MakeComposite(cases_hgt750)
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

            name_fig = f'comp_Td_{v}_{f}'
            PlotFinal(data=cases_ordenados, levels=scale, cmap=cbar,
                      titles=titles, namefig=name_fig, map=map,
                      save=save, dpi=dpi, out_dir=out_dir,
                      data_ctn=data_ctn, color_ctn='k', high=high,
                      num_cases=None, num_cases_data=None, num_cols=3,
                      ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
                      data_ctn_no_ocean_mask=True)


print('# --------------------------------------------------------------------#')
print('# n ------------------------------------------------------------------#')
if plot_td:
    out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/comp_n/'
    cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/' \
                    'aux_ep_cp_n/'

    variables = ['sst']
    aux_scales = [scale_t]
    aux_cbar = [cbar]

    for v, scale, cbar in zip(variables, aux_scales, aux_cbar):
        for f in ['pos', 'neg']:
            cases = OpenSetCases(var=v,
                                 idx1='dmi', idx2='ep', idx3='cp',
                                 phase=f,
                                 dir=cases_fields)

            cases_ordenados, titles = MakeComposite(cases)

            name = f'{v}_comp_IOD-EP-CP_{f}'

            if v == 'tref' or v == 'prec':
                map = 'sa'

                cases_hgt750 = OpenSetCases(var='hgt',
                                            idx1='dmi', idx2='ep', idx3='cp',
                                            phase=f,
                                            dir=cases_fields)

                cases_ordenados_hgt, _ = MakeComposite(cases_hgt750)
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

            name_fig = f'comp_n_{v}_{f}'
            PlotFinal(data=cases_ordenados, levels=scale, cmap=cbar,
                      titles=titles, namefig=name_fig, map=map,
                      save=save, dpi=dpi, out_dir=out_dir,
                      data_ctn=data_ctn, color_ctn='k', high=high,
                      num_cases=None, num_cases_data=None, num_cols=3,
                      ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
                      data_ctn_no_ocean_mask=True)

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')
