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
out_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/tk/'

cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/tk/'
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/tk/'

# ---------------------------------------------------------------------------- #
logger = init_logger('aux_1_selectvariables_cfsv2_mam_tk.log')

# Funcion -------------------------------------------------------------------- #
def Aux_SelectEvents(f, var_file, cases_dir, data_dir, out_dir,
                     replace_name):

    logger.info(f'{cases_dir}{f}')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]:'index'})

    data_var = xr.open_dataset(f'{data_dir}{var_file}')
    case_events = SelectVariables(aux_cases, data_var, new_month=4)

    f_name = f.replace(replace_name, "")
    f_name = f_name.replace('SON', 'MAM_from_SON')
    var_name = var_file.split('_')[0]
    logger.info(f'saving {out_dir}{var_name}_{f_name}')
    case_events.to_netcdf(f'{out_dir}{var_name}_{f_name}')

def Run(files, var_file, div, cases_dir=cases_date_dir, data_dir=data_dir,
        out_dir=out_dir, replace_name='CFSv2_'):
    logger.info(f'Run()...')
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

# EP ------------------------------------------------------------------------- #
logger.info('EP')
var_file = 'EP_SON_Leads_r_CFSv2.nc'
Run(files, var_file, div, data_dir=data_dir_indices)

# CP ------------------------------------------------------------------------- #
logger.info('CP')
var_file = 'CP_SON_Leads_r_CFSv2.nc'
Run(files, var_file, div, data_dir=data_dir_indices)

# DMI ------------------------------------------------------------------------ #
logger.info('DMI')
var_file = 'DMI_SON_Leads_r_CFSv2.nc'
aux_data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
Run(files, var_file, div, data_dir=aux_data_dir_indices)

# SST ------------------------------------------------------------------------ #
logger.info('SST')
var_file = 'sst_mam_detrend.nc'
Run(files, var_file, div, data_dir=data_dir)

# HGT ------------------------------------------------------------------------ #
logger.info('HGT')
var_file = 'hgt_mam_detrend.nc'
Run(files, var_file, div, data_dir=data_dir)

# vpot ----------------------------------------------------------------------- #
logger.info('VPOT200')
var_file = 'vpot200_mam_detrend.nc'
Run(files, var_file, div, data_dir=data_dir)

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #