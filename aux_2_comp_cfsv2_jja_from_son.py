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

import numpy as np
def c_diff(arr, h, dim, cyclic=False):
    # compute derivate of array variable respect to h associated to dim
    # adapted from kuchaale script
    ndim = arr.ndim
    lst = [i for i in range(ndim)]

    lst[dim], lst[0] = lst[0], lst[dim]
    rank = lst
    arr = np.transpose(arr, tuple(rank))

    if ndim == 3:
        shp = (arr.shape[0] - 2, 1, 1)
    elif ndim == 4:
        shp = (arr.shape[0] - 2, 1, 1, 1)

    d_arr = np.copy(arr)
    if not cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[0, ...]) / (h[1] - h[0])
        d_arr[-1, ...] = (arr[-1, ...] - arr[-2, ...]) / (h[-1] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    elif cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[-1, ...]) / (h[1] - h[-1])
        d_arr[-1, ...] = (arr[0, ...] - arr[-2, ...]) / (h[0] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    d_arr = np.transpose(d_arr, tuple(rank))

    return d_arr

def WAF(psiclm, psiaa, lon, lat,reshape=True, variable='var', hpalevel=200):
    #agregar xr=True

    if reshape:
        if len(psiclm['time']) > 1:
            psiclm = psiclm.mean('time')
        psiclm=psiclm[variable].values.reshape(1,len(psiclm.lat),len(psiclm.lon))


        variable_psiaa = list(psiaa.data_vars)[0]
        psiaa = psiaa[variable_psiaa].values.reshape(
            1, len(psiaa.lat), len(psiaa.lon))

    lon=lon.values
    lat=lat.values

    [xxx, nlats, nlons] = psiaa.shape  # get dimensions
    a = 6400000
    coslat = np.cos(lat * np.pi / 180)

    # climatological wind at psi level
    dpsiclmdlon = c_diff(psiclm, lon, 2)
    dpsiclmdlat = c_diff(psiclm, lat, 1)

    uclm = -1 * dpsiclmdlat
    vclm = dpsiclmdlon
    magU = np.sqrt(np.add(np.power(uclm, 2), np.power(vclm, 2)))

    dpsidlon = c_diff(psiaa, lon, 2)
    ddpsidlonlon = c_diff(dpsidlon, lon, 2)
    dpsidlat = c_diff(psiaa, lat, 1)
    ddpsidlatlat = c_diff(dpsidlat, lat, 1)
    ddpsidlatlon = c_diff(dpsidlat, lon, 2)

    termxu = dpsidlon * dpsidlon - psiaa * ddpsidlonlon
    termxv = dpsidlon * dpsidlat - ddpsidlatlon * psiaa
    termyv = dpsidlat * dpsidlat - psiaa * ddpsidlatlat

    # 0.2101 is the scale of p
    if hpalevel==200:
        coef = 0.2101
    elif hpalevel==750:
        coef = 0.74

    coeff1 = np.transpose(np.tile(coslat, (nlons, 1))) * (coef) / (2 * magU)
    # x-component
    px = coeff1 / (a * a * np.transpose(np.tile(coslat, (nlons, 1)))) * (
            uclm * termxu / np.transpose(np.tile(coslat, (nlons, 1))) + (vclm * termxv))
    # y-component
    py = coeff1 / (a * a) * (uclm / np.transpose(np.tile(coslat, (nlons, 1))) * termxv + (vclm * termyv))

    return px, py

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

            elif v == 'sst':
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

                wafx = None
                wafy = None
                data_waf = None

            elif v == 'hgt':
                aux_cases = OpenSetCases(var='sf',
                                         idx1='dmi', idx2='ep', idx3='cp',
                                         phase=f,
                                         dir=i_dir)

                aux_cases_ordenados, _ = MakeComposite(aux_cases)


                px_ordenados = []
                py_ordenados = []
                for p in aux_cases_ordenados.plots:

                    c = aux_cases_ordenados.sel(plots=p).drop_dims('Z')

                    px, py = WAF(aux_cases['neutros'], c, c.lon, c.lat,
                                 reshape=True, variable='hgt', hpalevel=200)
                    weights = np.transpose(np.tile(-2 * np.cos(
                        c.lat.values * 1 * np.pi / 180) + 2.1, (360, 1)))

                    weights_arr = np.zeros_like(px)
                    weights_arr[0, :, :] = weights
                    px *= weights_arr
                    py *= weights_arr

                    import xarray as xr

                    lat, lon = px[0].shape
                    px_da = xr.DataArray(
                        px[0],
                        dims=("lat", "lon"),
                        coords={
                            "lat": aux_cases['neutros'].lat.values,
                            "lon": aux_cases['neutros'].lon.values,
                        },
                        name="px"
                    )

                    py_da = xr.DataArray(
                        py[0],
                        dims=("lat", "lon"),
                        coords={
                            "lat": aux_cases['neutros'].lat.values,
                            "lon": aux_cases['neutros'].lon.values,
                        },
                        name="py"
                    )

                    px_ordenados.append(px_da)
                    py_ordenados.append(py_da)

                px_ordenados = xr.concat(px_ordenados, dim='plots')
                py_ordenados = xr.concat(py_ordenados, dim='plots')

                map = 'hs'
                data_ctn = cases_ordenados
                levels_ctn = scale
                ocean_mask = False
                high = 1.3

                wafx = px_ordenados
                wafy = py_ordenados
                waf_scale = None
                waf_label = 10e-5
                waf_step = 6,
                data_waf = aux_cases['neutros']

            else:
                map = 'hs'
                data_ctn = cases_ordenados
                levels_ctn = scale
                ocean_mask = False
                high = 1.3

            name_fig = f'comp_{i}/comp_{i}_jja_{v}_{f}_single_obs'
            PlotFinal(data=cases_ordenados, levels=scale, cmap=cbar,
                      titles=titles, namefig=name_fig, map=map,
                      save=save, dpi=dpi, out_dir=out_dir,
                      data_ctn=data_ctn, color_ctn='k', high=high,
                      num_cases=None, num_cases_data=None, num_cols=3,
                      ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
                      data_ctn_no_ocean_mask=True,
                      sig_points=cases_sig, hatches='......',
                      wafx=wafx, wafy=wafy,
                      waf_scale=None, waf_label=10e-5, waf_step=6,
                      data_waf=data_waf)

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #