"""
Comparacion de % de casos en cada categorias para OBS y CFSv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'

# ---------------------------------------------------------------------------- #
from SelectEvents_obs import Compute
from Funciones import DMI
import os
import xarray as xr
import numpy as np
import pandas as pd

# Funciones ------------------------------------------------------------------ #
def open_and_load(path):
        ds = xr.open_dataset(path, engine='netcdf4')  # backend explícito
        ds_loaded = ds.load()  # carga a memoria
        ds.close()  # cierra archivo en disco
        return ds_loaded

def Porcentaje(total, dir, replace_str):

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
                f = f.replace('dobles', 'sim')
                f = f.replace('pos', 'p')
                f = f.replace('neg', 'n')
                f = f.replace('opuestos', 'op')
                f = f.replace('triples', 'tr')

                porcentaje[f] = per

        porcentaje = pd.DataFrame(porcentaje.values(), porcentaje.keys() )
        porcentaje = porcentaje.sort_index()
        print(porcentaje)

        return porcentaje

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
porcentaje_tk = Porcentaje(total_obs, temporal_out_dir, 'OBS_')
porcentaje_tk.to_csv(f'{out_dir}porcentaje_tk.txt', sep='\t', index=True)

# Td
Compute(variables, ds1=dmi, ds2=ep_td, ds3=cp_td, out_dir=temporal_out_dir,
        prefix='OBS', save=True, thr=0.5)
porcentaje_td = Porcentaje(total_obs, temporal_out_dir, 'OBS_')
porcentaje_td.to_csv(f'{out_dir}porcentaje_td.txt', sep='\t', index=True)

# n
Compute(variables, ds1=dmi, ds2=ep_n, ds3=cp_n, out_dir=temporal_out_dir,
        prefix='OBS', save=True, thr=0.5)
porcentaje_n = Porcentaje(total_obs, temporal_out_dir, 'OBS_')
porcentaje_n.to_csv(f'{out_dir}porcentaje_n.txt', sep='\t', index=True)

# CFSv2 ---------------------------------------------------------------------- #
total_cfsv2 = 3744

# Tk
dir_cfsv2_tk =  '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/'
porcentaje_cfsv2_tk = Porcentaje(total_cfsv2, dir_cfsv2_tk, 'CFSv2_')
porcentaje_cfsv2_tk.to_csv(f'{out_dir}porcentaje_cfsv2_tk.txt',
                           sep='\t', index=True)

# Td
dir_cfsv2_td =  '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/aux_ep_cp_t/'
porcentaje_cfsv2_td = Porcentaje(total_cfsv2, dir_cfsv2_td, 'CFSv2_')
porcentaje_cfsv2_td.to_csv(f'{out_dir}porcentaje_cfsv2_td.txt',
                           sep='\t', index=True)
# n
dir_cfsv2_n =  '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/aux_ep_cp_n/'
porcentaje_cfsv2_n = Porcentaje(total_cfsv2, dir_cfsv2_n, 'CFSv2_')
porcentaje_cfsv2_n.to_csv(f'{out_dir}porcentaje_cfsv2_n.txt',
                          sep='\t', index=True)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #