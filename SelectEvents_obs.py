"""
Selección y clasificación de los eventos IOD y ENSO EP y CP
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

## Funciones ----------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
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

def ClasificarEventos_OBS(ds1, ds2, ds3, variables, thr=0.5):
    """
    Clasifica eventos con claves compatibles con InitEvent_dict.

    Args:
        ds1, ds2, ds3 (xr.Dataset): indices con variable 'sst' y dimensiones (time, r).
        variables (list of str): lista con los nombres de los índices [i1, i2, i3].
        thr (float): Umbral absoluto.

    Returns:
        dict: Diccionario anidado de eventos con fechas por categoría.
    """
    index1, index2, index3 = variables

    i1 = ds1['sst']
    i2 = ds2['sst']
    i3 = ds3['sst']

    # Máscaras absolutas
    i1_abs, i2_abs, i3_abs = np.abs(i1) > thr, np.abs(i2) > thr, np.abs(i3) > thr

    # Máscaras por signo
    i1_pos, i1_neg = i1 > thr, i1 < -thr
    i2_pos, i2_neg = i2 > thr, i2 < -thr
    i3_pos, i3_neg = i3 > thr, i3 < -thr

    # Diccionario de salida
    eventos = InitEvent_dict(variables)

    # Neutros
    eventos['neutros'] = ds1.time.where(~(i1_abs | i2_abs | i3_abs)).dropna('time')

    # Puros
    eventos['puros'][index1]['pos'] = \
        ds1.time.where(i1_pos & ~i2_abs & ~i3_abs).dropna('time')
    eventos['puros'][index1]['neg'] = \
        ds1.time.where(i1_neg & ~i2_abs & ~i3_abs).dropna('time')
    eventos['puros'][index2]['pos'] = \
        ds1.time.where(i2_pos & ~i1_abs & ~i3_abs).dropna('time')
    eventos['puros'][index2]['neg'] = \
        ds1.time.where(i2_neg & ~i1_abs & ~i3_abs).dropna('time')
    eventos['puros'][index3]['pos'] = \
        ds1.time.where(i3_pos & ~i1_abs & ~i2_abs).dropna('time')
    eventos['puros'][index3]['neg'] = \
        ds1.time.where(i3_neg & ~i1_abs & ~i2_abs).dropna('time')

    # Dobles simultáneos (mismo signo)
    eventos['simultaneos']['dobles'][f'{index1}_{index2}']['pos'] = \
        ds1.time.where(i1_pos & i2_pos & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index1}_{index2}']['neg'] = \
        ds1.time.where(i1_neg & i2_neg & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index1}_{index3}']['pos'] = \
        ds1.time.where(i1_pos & i3_pos & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index1}_{index3}']['neg'] = \
        ds1.time.where(i1_neg & i3_neg & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index2}_{index3}']['pos'] = \
        ds1.time.where(i2_pos & i3_pos & ~i1_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index2}_{index3}']['neg'] = \
        ds1.time.where(i2_neg & i3_neg & ~i1_abs).dropna('time')

    # Dobles de signo opuesto
    eventos['simultaneos']['dobles_op'][f'{index1}_pos_{index2}_neg'] = \
        ds1.time.where(i1_pos & i2_neg & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index1}_neg_{index2}_pos'] = \
        ds1.time.where(i1_neg & i2_pos & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index1}_pos_{index3}_neg'] = \
        ds1.time.where(i1_pos & i3_neg & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index1}_neg_{index3}_pos'] = \
        ds1.time.where(i1_neg & i3_pos & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index2}_pos_{index3}_neg'] = \
        ds1.time.where(i2_pos & i3_neg & ~i1_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index2}_neg_{index3}_pos'] = \
        ds1.time.where(i2_neg & i3_pos & ~i1_abs).dropna('time')

    # Triples mismo signo
    eventos['simultaneos']['triples'][f'{index1}_{index2}_{index3}']['pos'] = \
        ds1.time.where(i1_pos & i2_pos & i3_pos).dropna('time')
    eventos['simultaneos']['triples'][f'{index1}_{index2}_{index3}']['neg'] = \
        ds1.time.where(i1_neg & i2_neg & i3_neg).dropna('time')

    # Triples signo opuesto
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_pos_{index2}_pos_{index3}_neg'] = \
        ds1.time.where(i1_pos & i2_pos & i3_neg).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_pos_{index2}_neg_{index3}_neg'] = \
        ds1.time.where(i1_pos & i2_neg & i3_neg).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_pos_{index2}_neg_{index3}_pos'] = \
        ds1.time.where(i1_pos & i2_neg & i3_pos).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_neg_{index2}_pos_{index3}_pos'] = \
        ds1.time.where(i1_neg & i2_pos & i3_pos).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_neg_{index2}_pos_{index3}_neg'] = \
        ds1.time.where(i1_neg & i2_pos & i3_neg).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_neg_{index2}_neg_{index3}_pos'] = \
        ds1.time.where(i1_neg & i2_neg & i3_pos).dropna('time')

    return eventos

def OpenAndSetIndice(dates_dir, name_file):
    indice = xr.open_dataset(dates_dir + name_file)
    indice_std = indice.std('time')
    indice = indice/indice_std
    return indice

def SetIndice(indice):
    indice_std = indice.std('time')
    indice = indice/indice_std
    return indice

def Compute(variables, ds1, ds2, ds3, thr, save, out_dir, prefix):

    ds1 = ds1.to_dataset(name='sst')
    ds2 = ds2.to_dataset(name='sst')
    ds3 = ds3.to_dataset(name='sst')

    ds1 = SetIndice(ds1)
    ds2 = SetIndice(ds2)
    ds3 = SetIndice(ds3)

    eventos = ClasificarEventos_OBS(ds1=ds1, ds2=ds2, ds3=ds3,
                                    variables=variables,
                                    thr=thr)

    if save:
        save_event_dict_to_netcdf(eventos, out_dir, season='', prefix=prefix)
        print('done')
    else:
        return eventos

# ---------------------------------------------------------------------------- #