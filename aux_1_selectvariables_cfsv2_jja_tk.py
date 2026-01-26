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

from funciones.set_sst_obs import sst
from funciones.set_hgt_obs import hgt
import cftime

# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/tk/'

cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/tk/'
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/tk/'
dir_data_obs='/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
# ---------------------------------------------------------------------------- #
logger = init_logger('aux_1_selectvariables_cfsv2_jja_tk.log')

# Funcion -------------------------------------------------------------------- #
def Aux_SelectEvents(f, var_file, cases_dir, data_dir, out_dir,
                     replace_name, new_month, new_L=0, mode_single_obs=False,
                     obs_var=None):

    aux_cases = xr.open_dataset(f'{cases_dir}{f}')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]:'index'})
    aux_cases_selected = aux_cases.sel(
        time=aux_cases.time.dt.month.isin(new_month))
    len_L = len(aux_cases_selected.L)
    aux_cases_selected['L'] = [new_L]*len_L
    aux_cases_selected = aux_cases_selected.assign_coords(
        L=("time", aux_cases_selected.L.values)
    )

    data_var = xr.open_dataset(f'{data_dir}{var_file}')
    case_events = SelectVariables(aux_cases_selected, data_var)

    data_var_name = list(data_var.data_vars)[0]

    # ---- OBS ---- #
    aux_cases_no_new_month = aux_cases.sel(
        time=aux_cases.time.dt.month != new_month)
    cfsv2_years_no_new_month = aux_cases_no_new_month.time.dt.year

    if obs_var is not None:

        try:
            obs_selected = obs_var.sel(
                time=obs_var.time.dt.month.isin(new_month), month=new_month)
        except:
            obs_selected = obs_var.sel(
                time=obs_var.time.dt.month.isin(new_month))

        obs_var = list(obs_selected.data_vars)[0]

        if mode_single_obs:
            obs_selected = obs_selected.sel(
                time=obs_selected.time.dt.year.isin(cfsv2_years_no_new_month))
        else:
            aux_obs_selected = []
            for y in cfsv2_years_no_new_month:
                aux_obs_selected.append(
                    obs_selected.sel(time=obs_selected.time.dt.year.isin(y)))

            obs_selected = xr.concat(aux_obs_selected, dim='time')

        obs_selected = obs_selected.interp(lon=case_events.lon.values,
                                           lat=case_events.lat.values)

        obs_selected = obs_selected.assign_coords(
            time=("time", [
                cftime.Datetime360Day(
                    t.year, t.month, t.day,
                    t.hour, t.minute, t.second
                )
                for t in obs_selected.time.to_index()
            ])
        )

        case_events = xr.concat([case_events.drop(['r', 'L']),
                                 obs_selected.rename({obs_var: data_var_name})],
                                 dim='time')

    else:
        case_events = case_events.drop(['r', 'L'])


    f_name = f.replace(replace_name, "")
    f_name = f_name.replace('SON', 'JJA_from_SON')
    var_name = var_file.split('_')[0]
    logger.info(f'saving {out_dir}{var_name}_{f_name}')
    case_events.to_netcdf(f'{out_dir}{var_name}_{f_name}')

def Run(files, var_file, div, new_month, new_L, cases_dir=cases_date_dir,
        data_dir=data_dir, out_dir=out_dir, replace_name='CFSv2_',
        mode_single_obs=False,
        obs_var=None):
    logger.info(f'Run()...')
    for i in range(0, len(files), div):
        batch = files[i:i + div]
        processes = [Process(target=Aux_SelectEvents,
                             args=(f, var_file, cases_dir, data_dir, out_dir,
                                   replace_name, new_month, new_L,
                                   mode_single_obs,
                                   obs_var))
                     for f in batch]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

# ---------------------------------------------------------------------------- #
files = os.listdir(cases_date_dir)
files = [f for f in files if f.endswith('.nc')]
div = len(files) // 2

logger.info('Computo sobre indices desabilitados')
# EP ------------------------------------------------------------------------- #
# logger.info('EP')
# var_file = 'EP_SON_Leads_r_CFSv2.nc'
# Run(files, var_file, div, data_dir=data_dir_indices)

# CP ------------------------------------------------------------------------- #
# logger.info('CP')
# var_file = 'CP_SON_Leads_r_CFSv2.nc'
# Run(files, var_file, div, data_dir=data_dir_indices)

# DMI ------------------------------------------------------------------------ #
# logger.info('DMI')
# var_file = 'DMI_SON_Leads_r_CFSv2.nc'
# aux_data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# Run(files, var_file, div, data_dir=aux_data_dir_indices)

logger.info('Computo sobre variables...')
new_month=7
new_L=0
# SST ------------------------------------------------------------------------ #
logger.info('SST')
var_file = 'sst_jja_detrend.nc'
Run(files, var_file, div, new_month, new_L, data_dir=data_dir, obs_var=sst)

# HGT ------------------------------------------------------------------------ #
hgt = xr.open_dataset(f'{dir_data_obs}HGT200_JJA_mer_d_w.nc')
logger.info('HGT')
var_file = 'hgt_jja_detrend.nc'
Run(files, var_file, div, new_month, new_L, data_dir=data_dir, obs_var=hgt)

# vpot ----------------------------------------------------------------------- #
# logger.info('VPOT200')
# var_file = 'vpot200_jja_detrend.nc'
# Run(files, var_file, div, new_month, new_L, data_dir=data_dir)

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #