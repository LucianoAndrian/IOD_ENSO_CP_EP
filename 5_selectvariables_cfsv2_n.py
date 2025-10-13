"""
Seleccion de los campos de las variables para cada caso de eventos IOD y ENSO
A partir de los sst_* salida de 2_CFSv2_DMI_N34.py ara asegurar
correspondencia entre los eventos de los índices y los campos seleccionados.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
from multiprocessing import Process
from funciones.select_variables_cfsv2 import SelectVariables
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/aux_ep_cp_n/'

cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/aux_ep_cp_n/'
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/aux_ep_cp_n/'

# Funcion -------------------------------------------------------------------- #
def Aux_SelectEvents(f, var_file, cases_dir, data_dir, out_dir,
                     replace_name):

    aux_cases = xr.open_dataset(f'{cases_dir}{f}')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]:'index'})

    data_var = xr.open_dataset(f'{data_dir}{var_file}')
    case_events = SelectVariables(aux_cases, data_var)

    f_name = f.replace(replace_name, "")
    var_name = var_file.split('_')[0]
    case_events.to_netcdf(f'{out_dir}{var_name}_{f_name}')

def Run(files, var_file, div, cases_dir=cases_date_dir, data_dir=data_dir,
        out_dir=out_dir, replace_name='CFSv2_'):
    for i in range(0, len(files), div):
        batch = files[i:i + div]
        processes = [Process(target=Aux_SelectEvents,
                             args=(f, var_file, cases_dir, data_dir, out_dir,
                                   replace_name))
                     for f in batch]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

# ---------------------------------------------------------------------------- #
files = os.listdir(cases_date_dir)
files = [f for f in files if f.endswith('.nc')]

# EP n ----------------------------------------------------------------------- #
var_file = 'EP_n_SON_Leads_r_CFSv2.nc'
div = len(files) // 2
Run(files, var_file, div, data_dir=data_dir_indices, cases_dir=cases_date_dir,
    out_dir=out_dir, replace_name='CFSv2_n_')

# CP n ----------------------------------------------------------------------- #
var_file = 'CP_n_SON_Leads_r_CFSv2.nc'

div = len(files) // 2
Run(files, var_file, div, data_dir=data_dir_indices, cases_dir=cases_date_dir,
    out_dir=out_dir, replace_name='CFSv2_n_')

# DMI con n ----------------------------------------------------------------- #
var_file = 'DMI_SON_Leads_r_CFSv2.nc'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

div = len(files) // 2
Run(files, var_file, div, data_dir=data_dir_indices, cases_dir=cases_date_dir,
    out_dir=out_dir, replace_name='CFSv2_n_')

# SST ------------------------------------------------------------------------ #
var_file = 'sst_son.nc'
div = len(files) // 2
Run(files, var_file, div, data_dir=data_dir, cases_dir=cases_date_dir,
    out_dir=out_dir, replace_name='CFSv2_n_')

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')
