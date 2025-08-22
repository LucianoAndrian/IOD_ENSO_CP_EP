"""
Composites EP, CP

Para no tener que hace un nuevo selectevents...
Usar lo ya usado:
EP: EP puros, EP-DMI
CP: CP puros, CP-DMI
EP-CP: dobles EP_CP, triples
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'

# ---------------------------------------------------------------------------- #
import os
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import colors
from Funciones import DMI, PlotFinal
from SelectEvents_obs import Compute

# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'

cfsv2_cases_dates = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/'
cfsv2_cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'

if save:
    dpi=300
else:
    dpi=100

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
        cases_select.append(xr.open_dataset(f'{dir}{c}'))

    if len(cases_select) > 1:
        cases_select = xr.concat(cases_select, dim='time')
    else:
        cases_select = cases_select[0]
    return cases_select.time

def OpenSetCasesFields(cases, dir, fix):
    cases_select = []
    nums = []
    for c in cases:
        data = xr.open_dataset(f'{dir}{c}')*fix
        nums.append(len(data.time))
        cases_select.append(data)

    if len(cases_select) > 1:
        cases_select = xr.concat(cases_select, dim='time')
    else:
        cases_select = cases_select[0]
    return cases_select.mean('time'), np.sum(nums)

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

def composite_to_plot(v, fix, cfsv2_cases_fields=cfsv2_cases_fields):
    cases_neutros = [f'{v}_neutros_SON.nc', f'{v}_puros_dmi_pos_SON.nc',
                     f'{v}_puros_dmi_neg_SON.nc']

    # Neutros
    neutros, _ = OpenSetCasesFields(cases_neutros, cfsv2_cases_fields, fix)

    if v == 'hgt':
        v_aux = 'hgt_'
    else:
        v_aux = v
    files_v = [f for f in files if v_aux in f]

    # EP
    cases_pos, cases_neg = SelectCase(files_v, index='ep', index_out='cp')
    ep_pos_cases, num_ep_pos = OpenSetCasesFields(cases_pos,
                                                  cfsv2_cases_fields, fix)
    ep_neg_cases, num_ep_neg = OpenSetCasesFields(cases_neg,
                                                  cfsv2_cases_fields, fix)

    # CP
    cases_pos, cases_neg = SelectCase(files_v, index='cp', index_out='ep')
    cp_pos_cases, num_cp_pos = OpenSetCasesFields(cases_pos,
                                                  cfsv2_cases_fields, fix)
    cp_neg_cases, num_cp_neg = OpenSetCasesFields(cases_neg,
                                                  cfsv2_cases_fields, fix)

    comps = []
    nums = []
    comps.append(ep_pos_cases - neutros)
    nums.append(num_ep_pos)

    comps.append(cp_pos_cases - neutros)
    nums.append(num_cp_pos)

    comps.append(cp_neg_cases - neutros)
    nums.append(num_cp_neg)

    comps.append(ep_neg_cases - neutros)
    nums.append(num_ep_neg)

    comps = xr.concat(comps, dim='plots')
    comps = comps.rename({v: 'var'})

    return comps

def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def OrdenarNC_wTime_fromW(data):
    newdata = xr.Dataset(
        data_vars=dict(
            var=(['time', 'lat', 'lon'], data['var'][0, :, :, :].values)

        ),
        coords=dict(
            lon=(['lon'], data.lon),
            lat=(['lat'], data.lat),
            time=(['time'], data.time)
        )
    )
    return newdata

# Obs ------------------------------------------------------------------------ #
from test_indices import cp_tk, ep_tk, cp_td, ep_td, cp_n, ep_n, \
        year_start, year_end

dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
dmi = dmi.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-31'))
dmi = dmi.sel(time=dmi.time.dt.month.isin(10))
variables = ['dmi', 'ep', 'cp']
temporal_out_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP_OBS/'

Compute(variables, ds1=dmi, ds2=ep_tk, ds3=cp_tk, out_dir=temporal_out_dir,
        prefix='OBS', save=True, thr=0.5)

# ---------------------------------------------------------------------------- #
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

# Pre Composites sets -------------------------------------------------------- #

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

scale_pp = np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45])
scale_t = [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1]
aux_scale_hgt = [-100, -50, -30, -15, -5, 5, 15, 30, 50, 100]
aux_scale_hgt200 = [-150, -100, -50, -25, -10, 10, 25, 50, 100, 150]

# Por ahora solo t y pp
variables = ['tref', 'prec', 'hgt']
aux_scales = [scale_t, scale_pp, aux_scale_hgt200]
aux_cbar = [ cbar, cbar_pp, cbar]
dirs = [data_dir_t_pp, data_dir_t_pp, data_dir]
variables = ['tcru_w_c_d_0.25_SON', 'ppgpcc_w_c_d_1_SON', 'HGT200_SON_mer_d_w']
map = 'sa'

# Composites ----------------------------------------------------------------- #
cases_ordenados = [ep_pos_cases, cp_pos_cases, ep_neg_cases, cp_neg_cases]
titles = ['EP positivos', 'CP positivos', 'EP negativos', 'CP negativos']

for v, dir, scale, cbar in zip(variables, dirs, aux_scales, aux_cbar):

    data = xr.open_dataset(f'{dir}{v}.nc')
    data = data.sel(time=slice(f'{year_start}-10-01', f'{year_end}-10-01'))

    if 'tcru' in v or 'ppgpcc' in v:
        data = OpenSetData(data)
        map = 'sa'
        ocean_mask = True
        levels_ctn = None
        data_ctn_no_ocean_mask = True
        high = 3
        width = 4
        data_hgt = xr.open_dataset(f'{data_dir}HGT200_SON_mer_d_w.nc')
        data_ctn = data_hgt.sel(lon=slice(275, 330), lat=slice(20, -65))
        levels_ctn = aux_scale_hgt200

    else:
        data_ctn = data
        map='hs'
        ocean_mask = False
        data_ctn_no_ocean_mask = False
        levels_ctn = scale
        high = 1.2
        width = 6

    neutro_comp, _ = CompositeSimple(data, neutros)
    neutro_comp_ctn, _ = CompositeSimple(data_ctn, neutros)

    comps = []
    comps_ctn = []
    nums = []
    for c in cases_ordenados:
        case_comp, num = CompositeSimple(data, c)
        comps.append(case_comp - neutro_comp)
        nums.append(num)

        case_comp_ctn, num = CompositeSimple(data_ctn, c)
        comps_ctn.append(case_comp_ctn - neutro_comp_ctn)

    comps = xr.concat(comps, dim='plots')
    comps_ctn = xr.concat(comps_ctn, dim='plots')

    name_fig = f'EP_CP_comp_{v.split("_")[0]}'
    PlotFinal(data=comps, levels=scale, cmap=cbar,
              titles=titles, namefig=name_fig, map=map,
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=comps_ctn, color_ctn='k', high=high, width=width,
              num_cases=True, num_cases_data=nums, num_cols=2,
              ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
              data_ctn_no_ocean_mask=data_ctn_no_ocean_mask, step=1)

# ---------------------------------------------------------------------------- #
v_from_w = ['div_UV200', 'vp_from_UV200_w'] # creadas a partir de windphsere

data1 = xr.open_dataset(data_dir + v_from_w[0] + '.nc')
data1 = Detrend(
    OrdenarNC_wTime_fromW(data1.rename({'divergence':'var'})), 'time')

data2 = xr.open_dataset(data_dir+ v_from_w[1] + '.nc')
data2 = Detrend(
    OrdenarNC_wTime_fromW(data2.rename({'velocity_potential':'var'})), 'time')

data3 = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                        "sst.mnmean.nc")
data3 = data3.sel(time=data3.time.dt.year.isin(range(1940,2021)))
data3 = data3.rename({'sst':'var'})
data3 = Detrend(data3, 'time')
data3 = data3.drop_dims('nbnds')

neutro_sst, _ = CompositeSimple(data3, neutros)
neutro_vp, _ = CompositeSimple(data2, neutros)
neutro_div, _ = CompositeSimple(data1, neutros)

comps_sst = []
comps_vp = []
comps_div = []
for c in cases_ordenados:
    comps_sst.append(CompositeSimple(data3, c)[0] - neutro_sst)
    comps_vp.append(CompositeSimple(data2, c)[0] - neutro_vp)
    comps_div.append(CompositeSimple(data1, c)[0] - neutro_div)

comps_sst = xr.concat(comps_sst, dim='plots')
comps_vp = xr.concat(comps_vp, dim='plots')
comps_div = xr.concat(comps_div, dim='plots')

scale_vp_comp = np.linspace(-4.5e6, 4.5e6, 13)
scale_div_comp = [-1.6e-06, 1.6e-06]
scale_sst_comp = [-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5]
cbar_sst = colors.ListedColormap(['#B98200', '#CD9E46', '#E2B361', '#E2BD5A',
                                  '#FFF1C6', 'white', '#B1FFD0', '#7CEB9F',
                                  '#52D770', '#32C355', '#1EAF3D'][::-1])
cbar_sst.set_over('#9B6500')
cbar_sst.set_under('#009B2E')
cbar_sst.set_bad(color='white')

PlotFinal(data=comps_sst, levels=scale_sst_comp, cmap=cbar_sst,
          titles=titles, namefig='EP_CP_comp_sst_vp_div', map='hs',
          save=save, dpi=dpi, out_dir=out_dir,
          data_ctn=comps_vp, levels_ctn=scale_vp_comp, color_ctn='k',
          data_ctn2=comps_div, levels_ctn2=scale_div_comp,
          color_ctn2=['#FF0002', '#0003FF'], high=1.3)

# CFSv2 ---------------------------------------------------------------------- #
files = os.listdir(cfsv2_cases_fields)
files = [f for f in files if '.nc' in f]

titles = ['EP positivos', 'CP positivos', 'EP negativos', 'CP negativos']
variables = ['tref', 'prec', 'hgt']
for v, scale, cbar in zip(variables, aux_scales, aux_cbar):

    if 'prec' in v or 'tref' in v:
        map = 'sa'
        high = 3
        width = 4
        ocean_mask = True
        data_ctn_no_ocean_mask = True
        levels_ctn = aux_scale_hgt200

        if 'prec' in v:
            fix=30
        else:
            fix=1

        comps = composite_to_plot(v, fix)
        data_ctn = composite_to_plot('hgt', 9.8)

    elif 'hgt' in v:
        map='hs'
        fix=9.8
        high = 1.2
        width = 6
        ocean_mask = False
        data_ctn_no_ocean_mask = False
        levels_ctn = scale

        comps = composite_to_plot(v, fix)
        data_ctn = comps

    # Plots
    name_fig = f'EP_CP_comp_{v.split("_")[0]}_CFSv2'
    PlotFinal(data=comps, levels=scale, cmap=cbar,
              titles=titles, namefig=name_fig, map=map,
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=data_ctn, color_ctn='k', high=high, width=width,
              num_cases=False, num_cases_data=None, num_cols=2,
              ocean_mask=ocean_mask, pdf=False, levels_ctn=levels_ctn,
              data_ctn_no_ocean_mask=data_ctn_no_ocean_mask, step=1)

print(' --------------------------------------------------------------------- ')
print('Done')
print(' --------------------------------------------------------------------- ')