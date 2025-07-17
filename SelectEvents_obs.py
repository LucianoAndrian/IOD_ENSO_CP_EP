"""
Selección y clasificación de los eventos IOD y ENSO EP y CP
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from Funciones import DMI
# ---------------------------------------------------------------------------- #
save_nc = False
out_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP_OBS/'

## Funciones ----------------------------------------------------------------- #
def xrClassifierEvents(index):
    try:
        var_name = list(index.data_vars)[0]
    except:
        var_name = None
    try:
        index_pos = index[var_name].time[index[var_name] > 0]
        index_neg = index[var_name].time[index[var_name] < 0]
    except:
        index_pos = index.time[index > 0]
        index_neg = index.time[index < 0]
    try:
        aux_index_f = index[var_name][np.where(~np.isnan(index[var_name]))]
    except:
        aux_index_f = index[np.where(~np.isnan(index))]

    return index_pos, index_neg, aux_index_f

def ConcatEvent(xr_original, xr_to_concat, dim='time'):
    if (len(xr_to_concat.time) != 0) and (len(xr_original.time) != 0):
        xr_concat = xr.concat([xr_original, xr_to_concat], dim=dim)
    elif (len(xr_to_concat.time) == 0) and (len(xr_original.time) != 0):
        xr_concat = xr_original
    elif (len(xr_to_concat.time) != 0) and (len(xr_original.time) == 0):
        xr_concat = xr_to_concat
    elif (len(xr_to_concat.time) == 0) and (len(xr_original.time) == 0):
        return xr_original

    return xr_concat

def UniqueValues(target, extract):
    target = set(target)
    extract = set(extract)

    target_uniques = target - extract
    return np.array(list(target_uniques))

def SameDatesAs(data, dates):
    return data.sel(time=data.time.isin(dates))

def SetDates(*args):
    if len(args) == 2:
        dates = np.intersect1d(args[0], args[1])
    elif len(args) == 3:
        dates = np.intersect1d(args[0], np.intersect1d(args[1], args[2]))
    else:
        raise ValueError("Error: número de argumentos")

    return SameDatesAs(args[0], dates)

def InitEvent_dict(variables):
    def empty(): return xr.Dataset(coords={'time': []})

    index1 = variables[0]
    index2 = variables[1]
    index3 = variables[2]

    signs = ['pos', 'neg']
    dobles = [f'{index1}_{index2}', f'{index1}_{index3}', f'{index2}_{index3}']

    dobles_op = [f'{index1}_pos_{index2}_neg', f'{index1}_neg_{index2}_pos',
                 f'{index1}_pos_{index3}_neg', f'{index1}_neg_{index3}_pos',
                 f'{index2}_pos_{index3}_neg', f'{index2}_neg_{index3}_pos']

    triples = [f'{index1}_{index2}_{index3}']

    triples_op = [f'{index1}_pos_{index2}_pos_{index3}_neg',
                  f'{index1}_pos_{index2}_neg_{index3}_neg',
                  f'{index1}_pos_{index2}_neg_{index3}_pos',
                  f'{index1}_neg_{index2}_pos_{index3}_pos',
                  f'{index1}_neg_{index2}_pos_{index3}_neg',
                  f'{index1}_neg_{index2}_neg_{index3}_pos']

    data = {
        'todo': {i: {s: empty() for s in signs} for i in variables},
        'puros': {i: {s: empty() for s in signs} for i in variables},
        'simultaneos': {
            'dobles': {i: {s: empty() for s in signs} for i in dobles},
            'dobles_op': {i: empty() for i in dobles_op},
            'triples': {i: {s: empty() for s in signs} for i in triples},
            'triples_opuestos': {k: empty() for k in triples_op}
        },
        'neutros': empty()
    }
    return data

def ClassifyEvents(indices, idx1, idx2, idx3, data_ref, data_ref2=None):

    aux_events = InitEvent_dict(indices)

    idx1_name, idx2_name, idx3_name = list(aux_events['todo'].keys())

    # Clasificacion general
    r = None
    idx1_pos, idx1_neg, idx1_f = xrClassifierEvents(idx1)
    idx2_pos, idx2_neg, idx2_f = xrClassifierEvents(idx2)
    idx3_pos, idx3_neg, idx3_f = xrClassifierEvents(idx3)

    # Fechas ----------------------------------------------------------------- #
    # Se usan para seleccionar los eventos
    # Doble simultaneo
    idx1_idx2_fechas = np.intersect1d(idx1_f.time, idx2_f.time)
    idx1_idx3_fechas = np.intersect1d(idx1_f.time, idx3_f.time)
    idx2_idx3_fechas = np.intersect1d(idx2_f.time, idx3_f.time)

    # Triple simultaneo
    triple_fechas = np.intersect1d(idx1_idx2_fechas, idx3_f.time)

    # Dobles puros
    idx1_idx2_fechas = UniqueValues(idx1_idx2_fechas, triple_fechas)
    idx1_idx3_fechas = UniqueValues(idx1_idx3_fechas, triple_fechas)
    idx2_idx3_fechas = UniqueValues(idx2_idx3_fechas, triple_fechas)

    # Eventos ---------------------------------------------------------------- #
    # Simultaneos triples
    idx1_triple_values = idx1_f.sel(time=idx1_f.time.isin(triple_fechas))
    idx2_triple_values = idx2_f.sel(time=idx2_f.time.isin(triple_fechas))
    idx3_triple_values = idx3_f.sel(time=idx3_f.time.isin(triple_fechas))

    # Simultaneos dobles
    idx1_idx2_values = idx1_f.sel(time=idx1_f.time.isin(idx1_idx2_fechas))
    idx1_idx3_values = idx1_f.sel(time=idx1_f.time.isin(idx1_idx3_fechas))
    idx2_idx1_values = idx2_f.sel(time=idx2_f.time.isin(idx1_idx2_fechas))
    idx2_idx3_values = idx2_f.sel(time=idx2_f.time.isin(idx2_idx3_fechas))
    idx3_idx1_values = idx3_f.sel(time=idx3_f.time.isin(idx1_idx3_fechas))
    idx3_idx2_values = idx3_f.sel(time=idx3_f.time.isin(idx2_idx3_fechas))

    # Clasificancion de eventos ---------------------------------------------- #
    # Simultaneos triples
    idx1_triples_pos, idx1_triples_neg = xrClassifierEvents(
        idx1_triple_values)[0:2]
    idx2_triples_pos, idx2_triples_neg = xrClassifierEvents(
        idx2_triple_values)[0:2]
    idx3_triples_pos, idx3_triples_neg = xrClassifierEvents(
        idx3_triple_values)[0:2]

    # Simultaneos dobles
    idx1_idx2_pos, idx1_idx2_neg = xrClassifierEvents(
        idx1_idx2_values)[0:2]
    idx1_idx3_pos, idx1_idx3_neg = xrClassifierEvents(
        idx1_idx3_values)[0:2]
    idx2_idx1_pos, idx2_idx1_neg = xrClassifierEvents(
        idx2_idx1_values)[0:2]
    idx2_idx3_pos, idx2_idx3_neg = xrClassifierEvents(
        idx2_idx3_values)[0:2]
    idx3_idx1_pos, idx3_idx1_neg = xrClassifierEvents(
        idx3_idx1_values)[0:2]
    idx3_idx2_pos, idx3_idx2_neg = xrClassifierEvents(
        idx3_idx2_values)[0:2]

    # ------------------------------------------------------------------------ #
    # Checkeo en caso de trabajar con indices que provienen de variables
    # diferentes, e.j. hgt y sst. En caso de haber faltantes solo quedan los
    # eventos comunes

    # Triples
    sim_pos = SetDates(idx1_triples_pos, idx2_triples_pos, idx3_triples_pos)
    sim_neg = SetDates(idx1_triples_neg, idx2_triples_neg, idx3_triples_neg)

    # Dobles
    idx1_idx2_pos = SetDates(idx1_idx2_pos, idx2_idx1_pos)
    idx1_idx2_neg = SetDates(idx1_idx2_neg, idx2_idx1_neg)
    idx1_idx3_pos = SetDates(idx1_idx3_pos, idx3_idx1_pos)
    idx1_idx3_neg = SetDates(idx1_idx3_neg, idx3_idx1_neg)
    idx2_idx3_pos = SetDates(idx2_idx3_pos, idx3_idx2_pos)
    idx2_idx3_neg = SetDates(idx2_idx3_neg, idx3_idx2_neg)

    # Dobles op
    idx1_pos_idx2_neg = SetDates(idx1_idx2_pos, idx2_idx1_neg)
    idx1_neg_idx2_pos = SetDates(idx1_idx2_neg, idx2_idx1_pos)
    idx1_pos_idx3_neg = SetDates(idx1_idx3_pos, idx3_idx1_neg)
    idx1_neg_idx3_pos = SetDates(idx1_idx3_neg, idx3_idx1_pos)
    idx2_pos_idx3_neg = SetDates(idx2_idx3_pos, idx3_idx2_neg)
    idx2_neg_idx3_pos = SetDates(idx2_idx3_neg, idx3_idx2_pos)
    for t in [idx1_pos_idx2_neg, idx1_neg_idx2_pos, idx1_pos_idx3_neg,
              idx1_neg_idx3_pos, idx2_pos_idx3_neg, idx2_neg_idx3_pos]:
        if len(t) > 0:
            print('# ------ #')
            print(r)
            print('# ------ #')

    # Triples op
    idx1_pos_idx2_pos_idx3_neg = SetDates(idx1_triples_pos, idx2_triples_pos,
                                          idx3_triples_neg)
    idx1_pos_idx2_neg_idx3_pos = SetDates(idx1_triples_pos, idx2_triples_neg,
                                          idx3_triples_pos)
    idx1_pos_idx2_neg_idx3_neg = SetDates(idx1_triples_pos, idx2_triples_neg,
                                          idx3_triples_neg)
    idx1_neg_idx2_neg_idx3_pos = SetDates(idx1_triples_neg, idx2_triples_neg,
                                          idx3_triples_pos)
    idx1_neg_idx2_pos_idx3_pos = SetDates(idx1_triples_neg, idx2_triples_pos,
                                          idx3_triples_pos)
    idx1_neg_idx2_pos_idx3_neg = SetDates(idx1_triples_neg, idx2_triples_pos,
                                          idx3_triples_neg)

    # ------------------------------------------------------------------------ #

    # Eventos puros
    idx1_puros = idx1_f.sel(time=~idx1_f.time.isin(triple_fechas))
    try:
        idx1_puros = idx1_puros.sel(
            time=~idx1_puros.time.isin(idx1_idx2_fechas))
        idx1_puros = idx1_puros.sel(
            time=~idx1_puros.time.isin(idx1_idx3_fechas))
    except:
        pass


    idx2_puros = idx2_f.sel(time=~idx2_f.time.isin(triple_fechas))
    try:
        idx2_puros = idx2_puros.sel(
            time=~idx2_puros.time.isin(idx1_idx2_fechas))
        idx2_puros = idx2_puros.sel(
            time=~idx2_puros.time.isin(idx2_idx3_fechas))
    except:
        pass

    idx3_puros = idx3_f.sel(time=~idx3_f.time.isin(triple_fechas))
    try:
        idx3_puros = idx3_puros.sel(
            time=~idx3_puros.time.isin(idx1_idx3_fechas))
        idx3_puros = idx3_puros.sel(
            time=~idx3_puros.time.isin(idx2_idx3_fechas))
    except:
        pass

    idx1_puros_pos, idx1_puros_neg = xrClassifierEvents(idx1_puros)[0:2]
    idx2_puros_pos, idx2_puros_neg = xrClassifierEvents(idx2_puros)[0:2]
    idx3_puros_pos, idx3_puros_neg = xrClassifierEvents(idx3_puros)[0:2]

    # Neutros ---------------------------------------------------------------- #
    # Es necesario checkear los posibles datos faltantes
    aux_idx1 = data_ref#.sel(r=r)
    var_name = list(aux_idx1.data_vars)[0]
    dates_ref = aux_idx1.time[np.where(~np.isnan(aux_idx1[var_name]))]

    # en caso de usar indices de otra variables seria necesario abrir uno de esos
    # indices tambien
    if data_ref2 is not None:
        aux_idx2 = data_ref2#.sel(r=r)
        dates_ref_idx2 = aux_idx2.time[np.where(~np.isnan(aux_idx2.sam))]
        dates_ref = np.intersect1d(dates_ref, dates_ref_idx2)

    # Cuales de esas fechas no fueron idx1
    mask = np.in1d(dates_ref, idx1_f.time, invert=True)
    neutros = data_ref.sel(time=data_ref.time.isin(dates_ref[mask]))

    # Cuales de esas fechas no fueron idx2
    mask = np.in1d(neutros.time, idx2_f.time, invert=True)
    neutros = neutros.time[mask]

    # Cuales de esas fechas no fueron idx3
    mask = np.in1d(neutros.time, idx3_f.time, invert=True)
    neutros = neutros.time[mask]

    # ------------------------------------------------------------------------ #

    # Tdo: 6
    aux_events['todo'][idx1_name]['pos'] = idx1_pos
    aux_events['todo'][idx1_name]['neg'] = idx1_neg
    aux_events['todo'][idx2_name]['pos'] = idx2_pos
    aux_events['todo'][idx2_name]['neg'] = idx2_neg
    aux_events['todo'][idx3_name]['pos'] = idx3_pos
    aux_events['todo'][idx3_name]['neg'] = idx3_neg

    # Puros 6:
    aux_events['puros'][idx1_name]['pos'] = idx1_puros_pos
    aux_events['puros'][idx1_name]['neg'] = idx1_puros_neg
    aux_events['puros'][idx2_name]['pos'] = idx2_puros_pos
    aux_events['puros'][idx2_name]['neg'] = idx2_puros_neg
    aux_events['puros'][idx3_name]['pos'] = idx3_puros_pos
    aux_events['puros'][idx3_name]['neg'] = idx3_puros_neg

    # Simultaneos dobles 6:
    aux_events['simultaneos']['dobles'][f'{idx1_name}_{idx2_name}']['pos'] = idx1_idx2_pos
    aux_events['simultaneos']['dobles'][f'{idx1_name}_{idx2_name}']['neg'] = idx1_idx2_neg
    aux_events['simultaneos']['dobles'][f'{idx1_name}_{idx3_name}']['pos'] = idx1_idx3_pos
    aux_events['simultaneos']['dobles'][f'{idx1_name}_{idx3_name}']['neg'] = idx1_idx3_neg
    aux_events['simultaneos']['dobles'][f'{idx2_name}_{idx3_name}']['pos'] = idx2_idx3_pos
    aux_events['simultaneos']['dobles'][f'{idx2_name}_{idx3_name}']['neg'] = idx2_idx3_neg

    # Simultaneos dobles op 6:
    aux_events['simultaneos']['dobles_op']\
        [f'{idx1_name}_pos_{idx2_name}_neg'] = idx1_pos_idx2_neg
    aux_events['simultaneos']['dobles_op']\
        [f'{idx1_name}_neg_{idx2_name}_pos'] = idx1_neg_idx2_pos
    aux_events['simultaneos']['dobles_op']\
        [f'{idx1_name}_pos_{idx3_name}_neg'] = idx1_pos_idx3_neg
    aux_events['simultaneos']['dobles_op']\
        [f'{idx1_name}_neg_{idx3_name}_pos'] = idx1_neg_idx3_pos
    aux_events['simultaneos']['dobles_op']\
        [f'{idx2_name}_pos_{idx3_name}_neg'] = idx2_pos_idx3_neg
    aux_events['simultaneos']['dobles_op']\
        [f'{idx2_name}_neg_{idx3_name}_pos'] = idx2_neg_idx3_pos

    # Simultaneos triples 2:
    aux_events['simultaneos']['triples']\
        [f'{idx1_name}_{idx2_name}_{idx3_name}']['pos'] = sim_pos
    aux_events['simultaneos']['triples']\
        [f'{idx1_name}_{idx2_name}_{idx3_name}']['neg'] = sim_neg

    # Simultaneos triples  op 6:
    aux_events['simultaneos']['triples_opuestos']\
        [f'{idx1_name}_pos_{idx2_name}_pos_{idx3_name}_neg'] = idx1_pos_idx2_pos_idx3_neg
    aux_events['simultaneos']['triples_opuestos']\
        [f'{idx1_name}_pos_{idx2_name}_neg_{idx3_name}_neg'] = idx1_pos_idx2_neg_idx3_neg
    aux_events['simultaneos']['triples_opuestos']\
        [f'{idx1_name}_pos_{idx2_name}_neg_{idx3_name}_pos'] = idx1_pos_idx2_neg_idx3_pos
    aux_events['simultaneos']['triples_opuestos']\
        [f'{idx1_name}_neg_{idx2_name}_pos_{idx3_name}_pos'] = idx1_neg_idx2_pos_idx3_pos
    aux_events['simultaneos']['triples_opuestos']\
        [f'{idx1_name}_neg_{idx2_name}_neg_{idx3_name}_pos'] = idx1_neg_idx2_neg_idx3_pos
    aux_events['simultaneos']['triples_opuestos']\
        [f'{idx1_name}_neg_{idx2_name}_pos_{idx3_name}_neg'] = idx1_neg_idx2_pos_idx3_neg

    aux_events['neutros'] = neutros

    return aux_events

def save_event_dict_to_netcdf(event_dict, out_dir, season='', prefix=''):
    """
    Guarda cada elemento del dict anidado de eventos como archivo NetCDF,
    usando la clave como nombre de archivo.

    Args:
        event_dict (dict): Diccionario de eventos anidado.
        out_dir (str): Carpeta de salida.
        season (str): Opcional, nombre de la estación, e.g. 'SON'.
        prefix (str): Prefijo opcional para los nombres de archivo.
    """
    import os

    def recursive_save(d, name_parts):
        for k, v in d.items():
            new_name_parts = name_parts + [k]
            if isinstance(v, (xr.Dataset, xr.DataArray)):
                # Verificar si tiene dimensión 'time' y si está vacío
                if 'time' in v.dims and v.sizes.get('time', 0) == 0:
                    print(f"Skipping save for {'_'.join(new_name_parts)}: Dataset is empty.")
                else:
                    # Generar nombre de archivo
                    filename = '_'.join([prefix] + new_name_parts + [season]).strip('_') + '.nc'
                    filepath = os.path.join(out_dir, filename)
                    print(f"Saving: {filepath}")
                    v.to_netcdf(filepath)
            elif isinstance(v, dict):
                recursive_save(v, new_name_parts)
            else:
                print(f"Skipping unknown type at: {'_'.join(new_name_parts)}")

    recursive_save(event_dict, [])

def Compute(trh_sd, save_nc=False):
    from test_indices import cp_tk, ep_tk, year_start, year_end
    dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
    dmi = dmi.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-31'))
    dmi_or = dmi.sel(time=dmi.time.dt.month.isin(10))

    # ---------------------------------------------------------------------------- #
    # Criterio simple
    data_dmi = dmi_or.where(np.abs(dmi_or) > trh_sd * dmi_or.std())
    data_ep = ep_tk.where(np.abs(ep_tk) > trh_sd * ep_tk.std())
    data_cp = cp_tk.where(np.abs(cp_tk) > trh_sd * cp_tk.std())

    data_dmi = data_dmi.to_dataset(name='sst')
    data_ep = data_ep.to_dataset(name='sst')
    data_cp = data_cp.to_dataset(name='sst')
    dmi_ds = dmi_or.to_dataset(name='sst')

    indices = ['dmi', 'ep', 'cp']
    events = ClassifyEvents(indices, data_dmi, data_ep, data_cp, dmi_ds)

    len_neutros = len(events['neutros'].time)
    print(f'Neutros: {len_neutros}')
    if save_nc:
        save_event_dict_to_netcdf(events, out_dir, prefix='OBS')
    else:
        return events

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #