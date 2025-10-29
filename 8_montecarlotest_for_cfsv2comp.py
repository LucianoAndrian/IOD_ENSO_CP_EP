"""
Test de Monte Carlo para las composiciones en CFSv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/sig/'

# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
import glob
import math
from datetime import datetime
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import gc
import multiprocessing as mp

# ---------------------------------------------------------------------------- #
dir_events = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'
mc_tmp_dir = '/pikachu/datos/luciano.andrian/aux_mc/'
# Functions ------------------------------------------------------------------ #
def CompositesSimple_CFSv2(data, index):

    index_array = np.array(index)
    if len(index_array) != len(np.unique(index_array)):
        raise ValueError("Valores duplicados en index")

    data_sel = data.sel(position=index_array, drop=True)
    data_sel = data_sel.mean('position')

    return data_sel

def NumberPerts(data_to_concat, neutro, num = 0):
    #si las permutaciones posibles:
    # > 10000 --> permutaciones = 10000
    # < 1000 ---> permutaciones = todas las posibles
    #
    # si num != 0 permutaciones = num

    total = len(data_to_concat.position) + len(neutro.position)
    len1 = len(neutro.position)
    len2 = len(data_to_concat.position)

    try:
        total_perts = math.factorial(total) / (math.factorial(len2) * \
                                           math.factorial(len1))
    except OverflowError:
        total_perts = 10000

    if num == 0:
        if total_perts >= 10000:
            tot = 10000
            print('M = 10000')
        else:
            tot = total_perts
            print('M = ' + str(total_perts))

    else:
        tot = num

    jump = 9 #10 por .nc que guarde
    M = []
    n = 0

    while n < tot:
        aux = list(np.linspace((0 + n), (n + jump), (jump + 1)))
        M.append(aux)
        n = n + jump + 1

    return M

def selectcases(v, dir):
    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.nc')]
    files = [f for f in files if f'{v}_' in f]
    files = [f for f in files if 'neutro' not in f]
    return files

# ---------------------------------------------------------------------------- #
indices = ['tk', 'td', 'n']
variables = ['sst', 'hgt']

for i in indices:
    print(i)

    for v in variables:
        print(v)

        neutro = xr.open_dataset(f'{dir_events}{i}/{v}_neutros_SON.nc')

        # if len(neutro.sel(lat=slice(-60, 20)).lat.values) > 0:
        #     neutro = neutro.sel(lat=slice(-60, 20), lon=slice(275, 330))
        # else:
        #     neutro = neutro.sel(lat=slice(20, -60), lon=slice(275, 330))

        len_neutro = len(neutro.time)
        neutro = neutro.rename({'time': 'position'})
        neutro = neutro.drop_vars(['r', 'L'])

        cases = selectcases(v, f'{dir_events}{i}')
        # se podrian seleccionar mejor los cases para no hacer todos
        # pero todavia no tenemos definido cuales vamos a usar
        # mas allá del tiempo de procesamiento los archivos finales pesan muy poco

        for c in cases:
            print(c)
            # Borrar archivos temporales ------------------------------------- #
            files = glob.glob(f'{mc_tmp_dir}*.nc')

            if len(files) != 0:
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        print('Error: ' + f)

            # ---------------------------------------------------------------- #
            event = xr.open_dataset(f'{dir_events}{i}/{c}')

            # if len(event.sel(lat=slice(-60, 20)).lat.values) > 0:
            #     event = event.sel(lat=slice(-60, 20), lon=slice(275, 330))
            # else:
            #     event = event.sel(lat=slice(20, -60), lon=slice(275, 330))

            event = event.rename({'time': 'position'})
            event = event.drop_vars(['r', 'L'])

            concat = xr.concat([neutro, event], dim='position')

            concat['position'] = np.arange(len(concat.position))

            # temporal hasta definir que cases usar, para acelerar el computo
            try:
                concat = concat.interp(lon=np.arange(concat.lon[0], concat.lon[-1], 2),
                                       lat=np.arange(concat.lat[0], concat.lat[-1], 2))
            except:
                concat = concat.interp(lon=np.arange(concat.lon[0], concat.lon[-1], 2),
                                       lat=np.arange(concat.lat[0], concat.lat[-1], -2))

            def PermuDatesComposite(n, data=concat, len_neutro=len_neutro):

                for a in n:
                    rn = np.random.RandomState(616 + int(a))
                    dates_rn = rn.permutation(data.position)
                    neutro_new = dates_rn[0:len_neutro]
                    data_new = dates_rn[len_neutro:]

                    neutro_comp = CompositesSimple_CFSv2(data, neutro_new)
                    event_comp = CompositesSimple_CFSv2(data, data_new)

                    comp = event_comp - neutro_comp

                    if a == n[0]:
                        comp_concat = comp
                    else:
                        comp_concat = xr.concat([comp_concat, comp],
                                                dim='position')

                comp_concat.to_netcdf(f'{mc_tmp_dir}Comps_{int(a)}.nc')

                del neutro_comp, event_comp, comp, comp_concat
                gc.collect()

            M = NumberPerts(concat, neutro, 0)

            hour = datetime.now().hour

            if (hour > 19) | (hour < 8):
                n_proc = 30
            else:
                n_proc = 16

            print(f'Procesos: {n_proc}')

            with mp.Pool(processes=n_proc) as pool:
                pool.map(PermuDatesComposite, [n for n in M])
            print('pool ok')
            del event, concat
            gc.collect()

            # NO chunks! acumula RAM en cada ciclo. ~14gb en 3 ciclos...
            aux = xr.open_mfdataset(f'{mc_tmp_dir}Comps_*.nc', parallel=True,
                                    combine='nested', concat_dim="position",
                                    coords="different",
                                    compat="broadcast_equals")

            print(f'Quantiles ------------------------------------------- ')
            aux = aux.chunk({'position': -1})
            qt = aux.quantile([.05, .95], dim='position',
                              interpolation='linear')
            qt.to_netcdf(f'{out_dir}{i}/QT_{c}',
                         compute=True)
            aux.close()
            del qt, aux
            gc.collect()

print('# ------------------------------------------------------------------- #')
print('# ------------------------------------------------------------------- #')
print('done')
print('# ------------------------------------------------------------------- #')