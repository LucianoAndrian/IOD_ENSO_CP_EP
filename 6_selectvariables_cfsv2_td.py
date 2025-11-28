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
from funciones.general_utils import init_logger
import warnings
warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/td/'

cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/td/'
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/td/'

# ---------------------------------------------------------------------------- #
logger = init_logger('6_selectvariables_cfsv2_td.py')

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
div = len(files) // 2

# EP Td ---------------------------------------------------------------------- #
logger.info('EP')
var_file = 'EP_Td_SON_Leads_r_CFSv2.nc'
Run(files, var_file, div, data_dir=data_dir_indices, replace_name='CFSv2_Td_')

# CP Td ---------------------------------------------------------------------- #
logger.info('CP')
var_file = 'CP_Td_SON_Leads_r_CFSv2.nc'
Run(files, var_file, div, data_dir=data_dir_indices, replace_name='CFSv2_Td_')

# DMI con Td ----------------------------------------------------------------- #
logger.info('DMI')
var_file = 'DMI_SON_Leads_r_CFSv2.nc'
aux_data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
Run(files, var_file, div, data_dir=aux_data_dir_indices,
    replace_name='CFSv2_Td_')

# SST ------------------------------------------------------------------------ #
logger.info('SST')
var_file = 'sst_son.nc'
Run(files, var_file, div, data_dir=data_dir, replace_name='CFSv2_Td_')

# HGT ------------------------------------------------------------------------ #
logger.info('HGT')
var_file = 'hgt_son.nc'
Run(files, var_file, div, data_dir=data_dir, replace_name='CFSv2_Td_')

# HGT750 --------------------------------------------------------------------- #
logger.info('HGT750')
var_file = 'hgt750_son_detrend.nc'
Run(files, var_file, div, data_dir=data_dir, replace_name='CFSv2_Td_')

# vpot ----------------------------------------------------------------------- #
logger.info('VPOT200')
var_file = 'vpot200_son_detrend.nc'
Run(files, var_file, div, data_dir=data_dir, replace_name='CFSv2_Td_')

# Tref ----------------------------------------------------------------------- #
logger.info('tref')
var_file = 'tref_son.nc'
Run(files, var_file, div, data_dir=data_dir, replace_name='CFSv2_Td_')

# Prec ----------------------------------------------------------------------- #
logger.info('prec')
var_file = 'prec_son.nc'
Run(files, var_file, div, data_dir=data_dir, replace_name='CFSv2_Td_')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #