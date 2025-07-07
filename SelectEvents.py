"""
Selección y clasificación de los eventos IOD y ENSO EP y CP en CFSv2 a partir de
las salidas de 2_CFSv2_DMI_N34.py (IOD) y Enso_CP_EP_CFSv2.py
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ---------------------------------------------------------------------------- #
save_nc = True
out_dir = '/pikachu/datos/luciano.andrian/cases_dates_EP_CP/'

dates_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

## Funciones ----------------------------------------------------------------- #
def xrClassifierEvents(index, r, by_r=True):
    var_name = list(index.data_vars)[0]
    if by_r:
        index_r = index.sel(r=r)
        aux_index_r = index_r.time[np.where(~np.isnan(index_r[var_name]))]
        index_r_f = index_r.sel(time=index_r.time.isin(aux_index_r))

        index_pos = index_r_f[var_name].time[index_r_f[var_name] > 0]
        index_neg = index_r_f[var_name].time[index_r_f[var_name] < 0]

        return index_pos, index_neg, index_r_f
    else:
        index_pos = index[var_name].time[index[var_name] > 0]
        index_neg = index[var_name].time[index[var_name] < 0]

        return index_pos, index_neg

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
    dobles_op = [f'{index1}_pos_{index2}_neg', f'{index1}_neg_{index3}_pos',
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

def ClassifyEventsPerMember(r, indices, idx1, idx2, idx3, data_ref,
                               data_ref2=None):
    aux_events = InitEvent_dict(indices)

    idx1_name, idx2_name, idx3_name = list(aux_events['todo'].keys())

    # Clasificacion general
    idx1_pos, idx1_neg, idx1_f = xrClassifierEvents(idx1, r)
    idx2_pos, idx2_neg, idx2_f = xrClassifierEvents(idx2, r)
    idx3_pos, idx3_neg, idx3_f = xrClassifierEvents(idx3, r)

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
        idx1_triple_values, r=666, by_r=False)
    idx2_triples_pos, idx2_triples_neg = xrClassifierEvents(
        idx2_triple_values, r=666, by_r=False)
    idx3_triples_pos, idx3_triples_neg = xrClassifierEvents(
        idx3_triple_values, r=666, by_r=False)

    # Simultaneos dobles
    idx1_idx2_pos, idx1_idx2_neg = xrClassifierEvents(
        idx1_idx2_values, r=666, by_r=False)
    idx1_idx3_pos, idx1_idx3_neg = xrClassifierEvents(
        idx1_idx3_values, r=666, by_r=False)
    idx2_idx1_pos, idx2_idx1_neg = xrClassifierEvents(
        idx2_idx1_values, r=666, by_r=False)
    idx2_idx3_pos, idx2_idx3_neg = xrClassifierEvents(
        idx2_idx3_values, r=666, by_r=False)
    idx3_idx1_pos, idx3_idx1_neg = xrClassifierEvents(
        idx3_idx1_values, r=666, by_r=False)
    idx3_idx2_pos, idx3_idx2_neg = xrClassifierEvents(
        idx3_idx2_values, r=666, by_r=False)

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
    idx1_puros = idx1_puros.sel(time=~idx1_puros.time.isin(idx1_idx2_fechas))
    idx1_puros = idx1_puros.sel(time=~idx1_puros.time.isin(idx1_idx3_fechas))

    idx2_puros = idx2_f.sel(time=~idx2_f.time.isin(triple_fechas))
    idx2_puros = idx2_puros.sel(time=~idx2_puros.time.isin(idx1_idx2_fechas))
    idx2_puros = idx2_puros.sel(time=~idx2_puros.time.isin(idx2_idx3_fechas))

    idx3_puros = idx3_f.sel(time=~idx3_f.time.isin(triple_fechas))
    idx3_puros = idx3_puros.sel(time=~idx3_puros.time.isin(idx1_idx3_fechas))
    idx3_puros = idx3_puros.sel(time=~idx3_puros.time.isin(idx2_idx3_fechas))

    idx1_puros_pos, idx1_puros_neg = xrClassifierEvents(idx1_puros, r=666,
                                                        by_r=False)
    idx2_puros_pos, idx2_puros_neg = xrClassifierEvents(idx2_puros, r=666,
                                                        by_r=False)
    idx3_puros_pos, idx3_puros_neg = xrClassifierEvents(idx3_puros, r=666,
                                                        by_r=False)


    # Neutros ---------------------------------------------------------------- #
    # Es necesario checkear los posibles datos faltantes
    aux_idx1 = data_ref.sel(r=r)
    var_name = list(aux_idx1.data_vars)[0]
    dates_ref = aux_idx1.time[np.where(~np.isnan(aux_idx1[var_name]))]

    # en caso de usar indices de otra variables seria necesario abrir uno de esos
    # indices tambien
    if data_ref2 is not None:
        aux_idx2 = data_ref2.sel(r=r)
        dates_ref_idx2 = aux_idx2.time[np.where(~np.isnan(aux_idx2.sam))]
        dates_ref = np.intersect1d(dates_ref, dates_ref_idx2)

    # Cuales de esas fechas no fueron idx1
    mask = np.in1d(dates_ref, idx1_f.time, invert=True)
    neutros = data_ref.sel(time=data_ref.time.isin(dates_ref[mask]), r=r)

    # Cuales de esas fechas no fueron idx2
    mask = np.in1d(neutros.time, idx2_f.time, invert=True)
    neutros = neutros.time[mask]

    # Cuales de esas fechas no fueron idx3
    mask = np.in1d(neutros.time, idx3_f.time, invert=True)
    neutros = neutros.time[mask]

    # ------------------------------------------------------------------------ #

    # Todo: 6
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

# Funciones chatgpt
def merge_event_dicts(dict1, dict2):
    merged = {}

    def merge_ds(ds1, ds2):
        if not hasattr(ds1, 'time') or not hasattr(ds2, 'time'):
            raise TypeError(f"Uno de los elementos no tiene atributo `.time`: {type(ds1)}, {type(ds2)}")
        if ds1.time.size == 0:
            return ds2
        if ds2.time.size == 0:
            return ds1
        return xr.concat([ds1, ds2], dim='time')

    for key in dict1:
        if isinstance(dict1[key], dict):
            merged[key] = {}
            for subkey in dict1[key]:
                if isinstance(dict1[key][subkey], dict):
                    merged[key][subkey] = {}
                    for subsubkey in dict1[key][subkey]:
                        val1 = dict1[key][subkey][subsubkey]
                        val2 = dict2[key][subkey][subsubkey]
                        if isinstance(val1, (xr.Dataset, xr.DataArray)):
                            merged[key][subkey][subsubkey] = merge_ds(val1, val2)
                        else:
                            # Nivel más profundo
                            merged[key][subkey][subsubkey] = {}
                            for subsubsubkey in val1:
                                merged[key][subkey][subsubkey][subsubsubkey] = merge_ds(
                                    val1[subsubsubkey], dict2[key][subkey][subsubkey][subsubsubkey]
                                )
                else:
                    merged[key][subkey] = merge_ds(dict1[key][subkey], dict2[key][subkey])
        else:
            merged[key] = merge_ds(dict1[key], dict2[key])
    return merged

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

# ---------------------------------------------------------------------------- #

dmi = xr.open_dataset(dates_dir + 'DMI_SON_Leads_r_CFSv2.nc')
ep = xr.open_dataset(dates_dir + 'EP_SON_Leads_r_CFSv2.nc')
cp = xr.open_dataset(dates_dir + 'CP_SON_Leads_r_CFSv2.nc')

# Criterios:
data_dmi = dmi.where(np.abs(dmi) > 0.5*dmi.mean('r').std())
data_ep = ep.where(np.abs(ep) > 0.5*ep.mean('r').std())
data_cp = cp.where(np.abs(cp) > 0.5*cp.mean('r').std())

# ---------------------------------------------------------------------------- #
indices = ['dmi', 'ep', 'cp']
events = InitEvent_dict(indices)
for r in range(1, 25):

    events_r = ClassifyEventsPerMember(
        r, indices, data_dmi, data_ep, data_cp, dmi)

    events = merge_event_dicts(events, events_r)

if save_nc is True:
    save_event_dict_to_netcdf(events, out_dir, season='SON', prefix='CFSv2')

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')