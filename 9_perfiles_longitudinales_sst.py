"""
Perfiles longitudinales de sst en  EP, CP

Para no tener que hace un nuevo selectevents...
Usar lo ya usado:
EP: EP puros, EP-DMI
CP: CP puros, CP-DMI
EP-CP: dobles EP_CP, triples
"""

# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/proflon/'

# ---------------------------------------------------------------------------- #
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funciones.indices_utils import DMI
from funciones.general_utils import open_and_load, xrFieldTimeDetrend
from funciones.select_events_obs_utils import Compute
from funciones.plots_utils import DarkenColor

# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'

cfsv2_cases_dates = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/'
cfsv2_cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'

# ---------------------------------------------------------------------------- #
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

def SelectCase(files, index, index_out):
    files_select = [f for f in files if index in f and index_out not in f]
    files_index_pos = [f for f in files_select if f'{index}_pos' in f]
    files_index_neg = [f for f in files_select if f'{index}_neg' in f]

    return files_index_pos, files_index_neg

def OpenSetCases(cases, dir):
    cases_select = []
    for c in cases:
        cases_select.append(open_and_load(f'{dir}{c}'))

    if len(cases_select) > 1:
        cases_select = xr.concat(cases_select, dim='time')
    else:
        cases_select = cases_select[0]
    return cases_select.time

def OpenSetCasesFields(cases, dir, fix):
    cases_select = []
    nums = []
    for c in cases:
        data = open_and_load(f'{dir}{c}')*fix
        nums.append(len(data.time))
        cases_select.append(data)

    if len(cases_select) > 1:
        cases_select = xr.concat(cases_select, dim='time')
    else:
        cases_select = cases_select[0]
    return cases_select.mean('time'), np.sum(nums)

def OpenSetCasesFields_not_comp(cases, dir, fix):
    cases_select = []
    nums = []
    for c in cases:
        data = open_and_load(f'{dir}{c}')*fix
        nums.append(len(data.time))
        cases_select.append(data)

    if len(cases_select) > 1:
        cases_select = xr.concat(cases_select, dim='time')
    else:
        cases_select = cases_select[0]
    return cases_select, np.sum(nums)

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

def CompositeSimple_only_select(original_data, index):

    if len(index) != 0:
        comp_field = original_data.sel(
            time=original_data.time.isin([index]))
        if len(comp_field.time) != 0:
            comp_field = comp_field
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field, len(index.time)
    else:
        print(' len index = 0')

def tmp_PlotSST_lon(serie, color, title, name_fig, save,
                    lwd1=0.5, alpha=1, lwd2=2, dc=0.2,
                    out_dir=out_dir):
    serie_var = list(case_events.data_vars)[0]

    if save:
        dpi = 300
    else:
        dpi = 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(serie[serie_var].T, color=color, linewidth=lwd1, alpha=alpha)

    color2 = DarkenColor(color, dc=dc)
    ax.plot(serie[serie_var].T.mean('time'), color=color2, linewidth=lwd2)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('ºC', fontsize=12)

    xticks = np.arange(0, len(serie[serie_var].T), 20)

    ax.set_xticks(xticks)
    ax.set_xticklabels(serie.lon.values[xticks])
    ax.set_ylim([-3, 3])
    ax.grid(True, alpha=0.7)
    plt.tight_layout()

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', dpi=dpi)
        plt.close()
        print(f'Plot save: {out_dir}{name_fig}.png')
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
titles = ['EP positivos', 'CP positivos', 'EP negativos', 'CP negativos']
name_figs = ['EP_positivos', 'CP_positivos', 'EP_negativos', 'CP_negativos']
colors = ['firebrick', 'darkorange', 'mediumblue', 'dodgerblue']

# Obs ------------------------------------------------------------------------ #
from aux_set_obs_indices import cp_tk, ep_tk, cp_td, ep_td, cp_n, ep_n, \
    year_end, year_start

dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
dmi = dmi.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-31'))
dmi = dmi.sel(time=dmi.time.dt.month.isin(10))
variables = ['dmi', 'ep', 'cp']
temporal_out_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP_OBS/'

# ---------------------------------------------------------------------------- #
data_sst = open_and_load(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
data_sst = data_sst.rename({'sst':'var'})
data_sst = data_sst.drop('time_bnds')
data_sst = data_sst.rolling(time=3, center=True).mean()
data_sst = data_sst.sel(time=data_sst.time.dt.month.isin(10))
data_sst = xrFieldTimeDetrend(data_sst, 'time')

for cp, ep, t in zip([cp_tk, cp_td, cp_n],
                     [ep_tk, ep_td, ep_n],
                     ['Tk', 'Td', 'n']):

    Compute(variables, ds1=dmi, ds2=ep, ds3=cp, out_dir=temporal_out_dir,
            prefix='OBS', save=True, thr=0.5)

    # ------------------------------------------------------------------------ #
    # Cases select
    files = os.listdir(temporal_out_dir)
    cases_neutros = ['OBS_neutros.nc', 'OBS_puros_dmi_pos.nc',
                     'OBS_puros_dmi_neg.nc']

    # Neutros
    neutros = OpenSetCases(cases_neutros, dir=temporal_out_dir)

    # EP
    cases_pos, cases_neg = SelectCase(files, index='ep', index_out='cp')
    ep_pos_cases = OpenSetCases(cases_pos, dir=temporal_out_dir)
    ep_neg_cases = OpenSetCases(cases_neg, dir=temporal_out_dir)

    # CP
    cases_pos, cases_neg = SelectCase(files, index='cp', index_out='ep')
    cp_pos_cases = OpenSetCases(cases_pos, dir=temporal_out_dir)
    cp_neg_cases = OpenSetCases(cases_neg, dir=temporal_out_dir)

    # Composites ------------------------------------------------------------- #
    cases_ordenados = [ep_pos_cases, cp_pos_cases, ep_neg_cases, cp_neg_cases]

    data = data_sst.sel(time=slice(f'{year_start}-10-01', f'{year_end}-10-01'))
    data = data.sel(lon=slice(140, 280), lat=slice(5, -5)).mean('lat')

    neutro_comp, _ = CompositeSimple(data, neutros)

    for c, t2, fn, color in zip(cases_ordenados, titles, name_figs, colors):
        case_events, num = CompositeSimple_only_select(data, c)

        case_events = case_events - neutro_comp

        title_fig = f'{t}: {t2}'
        name_fig = f'sst_proflon_{t}_{fn}'

        tmp_PlotSST_lon(case_events, color, title_fig, name_fig, save)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
variables = ['sst']
v = 'sst'
for t in ['Tk', 'Td', 'n']:
    dir = f'{cfsv2_cases_fields}/{t.lower()}/'
    files = os.listdir(dir)
    files = [f for f in files if '.nc' in f]

    fix = 1
    cases_neutros = [f'{v}_neutros_SON.nc',
                     f'{v}_puros_dmi_pos_SON.nc',
                     f'{v}_puros_dmi_neg_SON.nc']
    # Neutros
    neutros, _ = OpenSetCasesFields(cases_neutros, dir, fix)
    neutros = neutros.sel(lon=slice(140, 280), lat=slice(-5, 5)).mean('lat')

    files_v = [f for f in files if v in f]
    files_v = [f for f in files_v if 'CFSv2' not in f]
    files_v = [f for f in files_v if 'todo' not in f]

    # EP
    cases_pos, cases_neg = SelectCase(files_v, index='ep', index_out='cp')
    ep_pos_cases, num_ep_pos = OpenSetCasesFields_not_comp(cases_pos, dir, fix)
    ep_neg_cases, num_ep_neg = OpenSetCasesFields_not_comp(cases_neg, dir, fix)

    # CP
    cases_pos, cases_neg = SelectCase(files_v, index='cp', index_out='ep')
    cp_pos_cases, num_cp_pos = OpenSetCasesFields_not_comp(cases_pos, dir, fix)
    cp_neg_cases, num_cp_neg = OpenSetCasesFields_not_comp(cases_neg, dir, fix)

    cases_ordenados = [ep_pos_cases, cp_pos_cases, ep_neg_cases, cp_neg_cases]

    for c, t2, fn, color in zip(cases_ordenados, titles, name_figs, colors):
        c = c.sel(lon=slice(140, 280), lat=slice(-5, 5)).mean('lat')
        case_events = c - neutros

        title_fig = f'CFSv2 - {t}: {t2}'
        name_fig = f'sst_proflon_cfsv2_{t}_{fn}'

        tmp_PlotSST_lon(case_events, color, title_fig, name_fig, save,
                    lwd1=0.1, alpha=0.5, dc=0.2)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #