"""
Comparacion de % de casos en cada categorias para OBS y CFSv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'
save = True
# ---------------------------------------------------------------------------- #
from SelectEvents_obs import Compute
from Funciones import DMI
import os
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib import colors

# Funciones ------------------------------------------------------------------ #
def open_and_load(path):
    ds = xr.open_dataset(path, engine='netcdf4')  # backend explícito
    ds_loaded = ds.load()  # carga a memoria
    ds.close()  # cierra archivo en disco
    return ds_loaded

def Porcentaje(total, dir, replace_str, column_name='columna'):

    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.nc')]

    porcentaje = {}

    for f in files:
        data = open_and_load(f'{dir}{f}')
        per = np.round((len(data.time.values) / total) * 100, 2)

        f = f.replace('.nc', '')
        f = f.replace(replace_str, '')
        f = f.replace('_SON', '')
        f = f.replace('simultaneos', 'sim')
        f = f.replace('dobles', 'do')
        f = f.replace('pos', 'p')
        f = f.replace('neg', 'n')
        f = f.replace('opuestos', 'op')
        f = f.replace('triples', 'tr')

        porcentaje[f] = per

    porcentaje = pd.DataFrame(porcentaje.values(), porcentaje.keys() )
    porcentaje = porcentaje.sort_index()
    porcentaje.columns = [column_name]
    print(porcentaje)

    return porcentaje

def Concat_df(df, df_to_concat, column_to_concat, new_column):
    df[new_column] = df.index.map(df_to_concat[column_to_concat]).fillna(0)
    return df

# de ENSO_IOD_SA
def PlotPDFTable(df, cmap, levels, title, name_fig='fig',
                 save=False, out_dir='~/', color_thr=0.4,
                 figsize=(6, 6)):

    if save:
        dpi = 300
    else:
        dpi = 100

    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(111)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    im = ax.imshow(df, cmap=cmap, norm=norm, aspect='auto')

    data_array = df.values
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            if np.abs(data_array[i, j]) > color_thr:
                color_num = 'white'
            else:
                color_num = 'k'
                ax.text(j, i, f"{data_array[i, j]:.2f}", ha='center',
                        va='center',
                        color=color_num)

    # Ticks principales en el centro de las celdas (para los labels)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=0, ha='center')

    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)

    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    fig.suptitle(title, size=12)

    ax.margins(0)
    plt.tight_layout()

    if save:
        plt.savefig(f"{out_dir}{name_fig}.png", dpi=dpi,
                        bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------- #
# CFSv2 ---------------------------------------------------------------------- #
total_cfsv2 = 3744

# Tk
dir_cfsv2_tk =  '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/'
porcentaje_cfsv2_tk = Porcentaje(total_cfsv2, dir_cfsv2_tk, 'CFSv2_',
                                 'tk_cfsv2')

# Td
dir_cfsv2_td =  '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/aux_ep_cp_t/'
porcentaje_cfsv2_td = Porcentaje(total_cfsv2, dir_cfsv2_td, 'CFSv2_Td_',
                                 'td_cfsv2')

# n
dir_cfsv2_n =  '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/aux_ep_cp_n/'
porcentaje_cfsv2_n = Porcentaje(total_cfsv2, dir_cfsv2_n, 'CFSv2_n_',
                                'n_cfsv2')

# ---------------------------------------------------------------------------- #
# OBS ------------------------------------------------------------------------ #
temporal_out_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP_OBS/'

from test_indices import cp_tk, ep_tk, cp_td, ep_td, cp_n, ep_n, \
        year_start, year_end

dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
dmi = dmi.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-31'))
dmi = dmi.sel(time=dmi.time.dt.month.isin(10))

total_obs = year_end-1 - year_start + 1
variables = ['dmi', 'ep', 'cp']

# Tk
Compute(variables, ds1=dmi, ds2=ep_tk, ds3=cp_tk, out_dir=temporal_out_dir,
        prefix='OBS', save=True, thr=0.5)
porcentaje_tk = Porcentaje(total_obs, temporal_out_dir, 'OBS_', 'tk_obs')

# Td
Compute(variables, ds1=dmi, ds2=ep_td, ds3=cp_td, out_dir=temporal_out_dir,
        prefix='OBS', save=True, thr=0.5)
porcentaje_td = Porcentaje(total_obs, temporal_out_dir, 'OBS_', 'td_obs')

# n
Compute(variables, ds1=dmi, ds2=ep_n, ds3=cp_n, out_dir=temporal_out_dir,
        prefix='OBS', save=True, thr=0.5)
porcentaje_n = Porcentaje(total_obs, temporal_out_dir, 'OBS_', 'n_obs')

# ---------------------------------------------------------------------------- #
# Concat --------------------------------------------------------------------- #
colmuns = ['tk_obs', 'td_cfsv2', 'td_obs', 'n_cfsv2', 'n_obs']
dfs = [porcentaje_tk, porcentaje_cfsv2_td, porcentaje_td, porcentaje_cfsv2_n, \
       porcentaje_n]

porcentajes = porcentaje_cfsv2_tk

for d, c in zip(dfs, colmuns):
    porcentajes = Concat_df(porcentajes, d,
                            column_to_concat=c,
                            new_column=c)

# ---------------------------------------------------------------------------- #
# Plot ----------------------------------------------------------------------- #
cbar_colors = [
    '#ffffff',  '#fef0ef',  '#fde0dd','#fccfc5','#fcbba1','#fca98d',
    '#fc9272', '#fb7c5a','#fb6a4a','#f4503c','#ef3b2c','#cb181d',
    '#a50f15']

cbar = colors.ListedColormap(cbar_colors)
cbar.set_over('#9C000E')
cbar.set_under('#ffffff')
cbar.set_bad(color='white')

PlotPDFTable(df=porcentajes, cmap=cbar,
             levels=[0, 1, 2, 4, 6, 8 ,10, 12, 14, 18],
             title='porcentajes de casos', name_fig='porcentajes',
             save=save, out_dir=out_dir, color_thr=100)

# ---------------------------------------------------------------------------- #
# Diferencias
cbar_bins2d = colors.ListedColormap(['#9B1C00', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white', 'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#014A9B'][::-1])
cbar_bins2d.set_over('#641B00')
cbar_bins2d.set_under('#012A52')
cbar_bins2d.set_bad(color='white')


cols = porcentajes.columns
# Restar columnas 0-1, 2-3, 4-5...
diffs = {}
for i in range(0, len(cols), 2):
    col_a = cols[i]
    col_b = cols[i+1]
    print(col_a, col_b)
    new_col = f"{col_a.split('_')[0]}_diff"
    diffs[new_col] = porcentajes[col_a] - porcentajes[col_b]

porcentajes_diff = pd.DataFrame(diffs)

PlotPDFTable(df=porcentajes_diff, cmap=cbar_bins2d,
             levels=np.linspace(-6, 6, 11),
             title='diff % cfsv2-obs', name_fig='porcentajes_diff',
             save=save, out_dir=out_dir, color_thr=100,
             figsize=(4.5,6))

omit = [13,14,15,16,17,18,21, 22, 23, 24, 25, 26]
filas = [[1,7],[7,13],[13,19],[19,21], [21,100], [0,100], omit]

sum_abs = []
porcentajes_abs = np.abs(porcentajes_diff)
for cf, f in enumerate(filas):
    if cf < 6:
        sum_abs.append(porcentajes_abs[f[0]:f[1]].sum())
    else:
        sum_abs.append(porcentajes_abs.drop(porcentajes_abs.index[omit]).sum())

porcentajes_sum_abs = pd.DataFrame(sum_abs)
porcentajes_sum_abs.index = ['Puros', 'dobles', 'dobles_op', 'triples', \
                             'triples_op', 'todo', 'todo_sin_op']

PlotPDFTable(df=porcentajes_sum_abs, cmap=cbar,
             levels=np.linspace(0, 60, 11),
             title='Suma de diferencias absolutas',
             name_fig='porcentajes_diff_abs_por_categorias',
             save=save, out_dir=out_dir,
             color_thr=100, figsize=(6,6))

# SD ------------------------------------------------------------------------- #
sds = []
for cf, f in enumerate(filas):
    if cf < 6:
        sds.append(porcentajes[f[0]:f[1]].std())
    else:
        sds.append(porcentajes.drop(porcentajes.index[omit]).std())

porcentajes_sd = pd.DataFrame(sds)
porcentajes_sd.index = ['Puros', 'dobles', 'dobles_op', 'triples', \
                        'triples_op', 'todo', 'todo_sin_op']

PlotPDFTable(df=porcentajes_sd, cmap=cbar,
             levels=np.linspace(0, 5, 11),
             title='SD %', name_fig='porcentajes_sd_por_categorias',
             save=save, out_dir=out_dir,
             color_thr=100, figsize=(6,6))

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #