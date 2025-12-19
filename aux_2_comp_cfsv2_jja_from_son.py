"""
Composites con el modelo CFSv2

Por ahora, se mantiene el orden de las figuras de 2 filas y 3 columnas
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'

cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'
sig_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/sig/'

# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from funciones.plots_utils import PlotFinal
from funciones.general_utils import Weights
from funciones.scales_and_cbars import get_scales, get_cbars
from funciones.general_utils import init_logger
import warnings
warnings.simplefilter("ignore")

if save:
    dpi = 300
else:
    dpi = 100
# ---------------------------------------------------------------------------- #
logger = init_logger('aux_2_comp_cfsv2_jja_from_son.log')

# aux funciones -------------------------------------------------------------- #
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

def OpenSetCases(var, idx1, idx2, idx3, phase, dir, sig=False,
                 trim='JJA_from_SON'):

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


    if sig is True:
        var = f'QT_{var}'

    cases = {}
    # neutro
    if sig is False:
        cases['neutros'] = xr.open_dataset(f'{dir}{var}_neutros_{trim}.nc') * fix

    indices = [idx1, idx2, idx3]

    for i_num, i in enumerate(indices):

        idx, idx_aux2, idx_aux3 = Combinations(i, indices)

        cases[i] = {}

        # puro
        cases[i]['puros'] = xr.open_dataset(
            f'{dir}{var}_puros_{i}_{phase}_{trim}.nc') * fix

        # doble
        try:
            doble_2 = xr.open_dataset(
                f'{dir}{var}_simultaneos_dobles_{i}_{idx_aux2}_{phase}'
                f'_{trim}.nc') * fix

            doble_3 = xr.open_dataset(
                f'{dir}{var}_simultaneos_dobles_{i}_{idx_aux3}_{phase}'
                f'_{trim}.nc') * fix

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
                f'_{phase}_{trim}.nc') * fix

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

def aux_ordenar_sig(sig):

    sig_ordenados = [None, None, None, None, None, None]
    indices = list(sig.keys())

    for i in indices:
        i_sig = sig[i]
        i_sig_k = list(i_sig.keys())

        if len(i_sig_k) > 1:
            for key in i_sig_k:
                key_sig = i_sig[key]
                subkeys = list(key_sig.keys())

                if len(subkeys) > 1:
                    for sk in subkeys:
                        pos = Decider(i, key, indices, sk)
                        sig_ordenados[pos] = key_sig[sk]

                else:
                    pos = Decider(i, key, indices)
                    sig_ordenados[pos] = key_sig

        else:
            key = i_sig_k[0]
            pos = Decider(i, key, indices)
            sig_ordenados[pos] = i_sig[key]

    sig_ordenados = xr.concat(sig_ordenados, dim='plots')
    var_name = list(sig_ordenados.data_vars)[0]
    sig_ordenados = sig_ordenados.rename({var_name: 'var'})

    return sig_ordenados

def SigMask(cases, sig):
    import numpy as np
    cases_sig = []
    var_name = list(cases.data_vars)[0]
    for p in cases.plots:
        aux_c = cases.sel(plots=p)[var_name]
        aux_s = sig.sel(plots=p)[var_name]

        aux = aux_c.where((aux_c < aux_s[0]) | (aux_c > aux_s[1]))
        aux_sig = aux.where(np.isnan(aux), 1)

        cases_sig.append(aux_sig)

    return xr.concat(cases_sig, dim='plots')

# ---------------------------------------------------------------------------- #
variables = ['sst', 'hgt']#, 'vpot200']
aux_scales = ['t_comp_cfsv2',  'hgt_comp_cfsv2']#, 'vpot200_cfsv2']
aux_cbar = ['cbar_rdbu', 'cbar_rdbu']#, 'cbar_rdbu']

for i in ['tk']:
    logger.info(f'Indice {i}')
    i_dir = f'{cases_fields}{i}/'
    s_dir = f'{sig_dir}{i}/'

    for v, sc, cb in zip(variables, aux_scales, aux_cbar):
        logger.info(f'Variable {v}')
        scale = get_scales(sc)
        cbar = get_cbars(cb)

        for f in ['pos', 'neg']:
            logger.info(f'fase {f}')
            logger.info(f'Cases JJA {f}')
            cases = OpenSetCases(var=v,
                                 idx1='dmi', idx2='ep', idx3='cp',
                                 phase=f,
                                 dir=i_dir)

            cases_ordenados, titles = MakeComposite(cases)

            # sig = OpenSetCases(var=v,
            #                    idx1='dmi', idx2='ep', idx3='cp',
            #                    phase=f,
            #                    dir=s_dir,
            #                    sig=True)
            #
            # sig_ordenados = aux_ordenar_sig(sig)
            #
            # cases_sig = SigMask(cases_ordenados, sig_ordenados)
            cases_sig = None

            # ---------------------------------------------------------------- #
            # logger.info(f'Cases SON to select OBS {f}')
            # cases_son = OpenSetCases(var=v,
            #                      idx1='dmi', idx2='ep', idx3='cp',
            #                      phase=f,
            #                      dir=i_dir,
            #                      trim='SON')

            if v == 'tref' or v == 'prec':
                map = 'sa'
                aux_cases = OpenSetCases(var='hgt',
                                            idx1='dmi', idx2='ep', idx3='cp',
                                            phase=f,
                                            dir=i_dir)

                aux_cases_ordenados, _ = MakeComposite(aux_cases)
                data_ctn = aux_cases_ordenados
                levels_ctn = get_scales('hgt_comp_cfsv2')
                data_ctn_no_ocean_mask = True
                ocean_mask = True
                high = 3

            elif v == 's2st':
                aux_cases = OpenSetCases(var='vpot200',
                                            idx1='dmi', idx2='ep', idx3='cp',
                                            phase=f,
                                            dir=i_dir)
                aux_cases_ordenados, _ = MakeComposite(aux_cases)
                data_ctn = aux_cases_ordenados
                map = 'hs'
                levels_ctn = get_scales('vpot200_cfsv2')
                ocean_mask = False
                high = 1.3

            else:
                map = 'hs'
                data_ctn = cases_ordenados
                levels_ctn = scale
                ocean_mask = False
                high = 1.3

            name_fig = f'comp_{i}/comp_{i}_jja_{v}_{f}'
            PlotFinal(data=cases_ordenados, levels=scale, cmap=cbar,
                      titles=titles, namefig=name_fig, map=map,
                      save=save, dpi=dpi, out_dir=out_dir,
                      data_ctn=data_ctn, color_ctn='k', high=high,
                      num_cases=None, num_cases_data=None, num_cols=3,
                      ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
                      data_ctn_no_ocean_mask=True,
                      sig_points=cases_sig, hatches='......')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #