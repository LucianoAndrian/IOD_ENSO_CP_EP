"""
Diferencias en el area de la PDF cases vs neutros - CFSv2
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'
cases_fields = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'

# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import colors
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
from Funciones import PDF_cases, AreaBetween, PlotPDFTable

# Funciones ------------------------------------------------------------------ #
def SelectFiles(files, str_to_select):
    return [f for f in files if str_to_select in f and 'neutros' not in f]

def SetCases(files, variable):
    cases = [f.replace(f'{variable}_', '') for f in files]
    cases = [f.replace(f'_SON.nc', '') for f in cases]
    return cases

def OmitCases(files, str):
    return [f for f in files if str not in f]

def SetNameIndex(df):
    df = df.sort_index()
    new_index=[]
    for i in df.index:
        i = i.replace('simultaneos_', '')
        i = i.replace('opuestos_', '')
        i = i.replace('op_', '')
        i = i.replace('triples_', '')
        i = i.replace('puros_', '')
        i = i.replace('dobles_', '')

        i = i.replace('dmi', 'DMI')
        i = i.replace('ep', 'EP')
        i = i.replace('cp', 'CP')
        i = i.replace('_pos', '-pos')
        i = i.replace('_neg', '-neg')
        i = i.replace('_', ' ')
        new_index.append(i)

    df.index = new_index
    return df
# ---------------------------------------------------------------------------- #
if save:
    dpi = 300
else:
    dpi = 100

cbar_bins2d = colors.ListedColormap(['#9B1C00', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white', 'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#014A9B'][::-1])
cbar_bins2d.set_over('#641B00')
cbar_bins2d.set_under('#012A52')
cbar_bins2d.set_bad(color='white')

cbar_pp_bins2d = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white', 'white',
                                '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07',
                                 '#543005', ][::-1])
cbar_pp_bins2d.set_under('#3F2404')
cbar_pp_bins2d.set_over('#00221A')
cbar_pp_bins2d.set_bad(color='white')

print(' PDFs CFSv2 ---------------------------------------------------------- ')
box_name = ['Am', 'NeB', 'N-SESA', 'S-SESA', 'Chile-Cuyo', 'Patagonia']
box_lats = [[-13, 2], [-15, 2], [-29, -17], [-39, -25], [-40,-30], [-56, -40]]
box_lons = [[291, 304], [311, 325], [303, 315], [296, 306], [285,293], [287, 295]]

files = os.listdir(cases_fields)


for v, v_cbar in zip(['tref', 'prec'], [cbar_bins2d, cbar_pp_bins2d]):

    files_select = SelectFiles(files, v)
    cases = SetCases(files_select, v)
    cases = OmitCases(cases, 'op')

    result = PDF_cases(variable=v,
                       box_lons=box_lons, box_lats=box_lats, box_name=box_name,
                       cases=cases, cases_dir=cases_fields)

    regiones_areas = {}
    for k in result.keys():
        aux = result[k]

        areas = {}
        for c in cases:
            areas[c] = AreaBetween(aux['clim'], aux[c])

        regiones_areas[k] = areas

    df = pd.DataFrame(regiones_areas)
    df.to_csv(f'{out_dir}area_entre_pdfs_{v}.csv')
    df = SetNameIndex(df)

    PlotPDFTable(np.round(df, 2), cmap=v_cbar,
                 levels=[-1, -0.8, -0.6, -0.4, -0.2, -0.1,
                         0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
                 title='',
                 save=save, name_fig=f'pdf_table_{v}',
                 out_dir=out_dir, dpi=dpi,
                 color_thr=0.5,
                 figsize=(8,6),
                 pdf=False)

print(' --------------------------------------------------------------------- ')
print('Done')
print(' --------------------------------------------------------------------- ')