"""
Figuras ENSO-IOD-SA
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'
plot_bins = False
plot_bins_2d = True
plot_pdf = False
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import colors
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from Funciones import SetDataToPlotFinal, PlotFinal, CaseComp, RenameDataset, \
    BinsByCases, PlotFinal_CompositeByMagnitude, PDF_cases, PlotPdfs, \
    SelectBins2D, SelectDatesBins, PlotBars, MakeMask, PlotBins2D, \
    SetBinsByCases, AreaBetween, PlotPDFTable, PlotFinalTwoVariables, \
    PlotBins2DTwoVariables

# ---------------------------------------------------------------------------- #
if save:
    dpi = 300
else:
    dpi = 100





# ---------------------------------------------------------------------------- #
def MakerMaskSig(data, r_crit):
    mask_sig = data.where((data < -1 * r_crit) | (data > r_crit))
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

# ---------------------------------------------------------------------------- #
data_dir_proc = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/'

nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'

sig_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/nc_quantiles/' \
          'DMIbase/' # resultados de MC
cfsv2_sig_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'CFSv2_nc_quantiles/' # resultados de MC

data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

cases_dir = "/pikachu/datos/luciano.andrian/cases_fields/"
dates_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# Funciones ------------------------------------------------------------------ #
def OpenObsDataSet(name, sa=True, dir='/pikachu/datos/luciano.andrian/'
                                      'observado/ncfiles/data_obs_d_w_c/'):

    aux = xr.open_dataset(dir + name + '.nc')

    if sa:
        aux2 = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
        if len(aux2.lat) > 0:
            return aux2
        else:
            aux2 = aux.sel(lon=slice(270, 330), lat=slice(-60, 15))
            return aux2
    else:
        return aux


def OpenAndSetRegre(v, data_dir_proc=data_dir_proc):
    print(
        f'{v} ------------------------------------------------------------- ')
    regre_n34 = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_n34.nc')
    regre_corr_n34 = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_n34_corr.nc')
    regre_dmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi.nc')
    regre_corr_dmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi_corr.nc')
    regre_n34_wodmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_n34_wodmi.nc')
    regre_corr_n34_wodmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_n34_wodmi_corr.nc')
    regre_dmi_won34 = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi_won34.nc')
    regre_corr_dmi_won34 = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi_won34_corr.nc')

    regre_n34, regre_corr_n34, regre_dmi, regre_corr_dmi, regre_n34_wodmi, \
        regre_corr_n34_wodmi, regre_dmi_won34, regre_corr_dmi_won34 = \
        RenameDataset('var', regre_n34, regre_corr_n34, regre_dmi,
                      regre_corr_dmi, regre_n34_wodmi, regre_corr_n34_wodmi,
                      regre_dmi_won34, regre_corr_dmi_won34)

    return regre_n34, regre_corr_n34, regre_dmi, regre_corr_dmi, \
        regre_n34_wodmi, regre_corr_n34_wodmi, regre_dmi_won34, \
        regre_corr_dmi_won34
# Scales and colorbars ------------------------------------------------------- #
# Regresion ------------------------------------------------------------------ #
scale_hgt = [-300, -200, -100, -50, -25, 0, 25, 50, 100, 200, 300]
scale_hgt=[-300, -270, -240, -210, -180, -150, -120, -90, -60,
 -30, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

scale_hgt_750 = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0,
                 10, 20, 30, 40, 50, 60 ,70 ,80 ,90, 100]
scale_pp = np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45])
scale_t = [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1]

scale_pp_val = [-60, -30, -10, -5, 0, 5, 10, 30, 60]
scale_t_val = [-2,-.8,-.4,-.1, 0, .1, .4, .8, 2]

# Pensando en x mensual.
scale_t_val = [-3, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 3]
scale_pp_val = [-60, -30, -15, -5, 0, 5, 15, 30, 60] # -2 -> 2 mm/day
# Composite ------------------------------------------------------------------ #
scale_hgt_comp = [-500, -300, -200, -100, -50, 0, 50, 100, 200, 300, 500]
scale_t_comp = [-1.5, -1, -.5, -.25, -.1, 0, .1, .25, .5, 1, 1.5]
scale_pp_comp = [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40]
scale_hgt_comp_cfsv2 = [-375, -275, -175, -75, -25, 0, 25, 75, 175, 275, 375]

# CFSv2
scale_snr = [-1, -.8, -.6, -.5, -.1, 0, 0.1, 0.5, 0.6, 0.8, 1]
# iods ind
scale_hgt_ind = [-575,-475, -375, -275, -175, -75,0,
                 75, 175, 275, 375, 475, 575]

# ---------------------------------------------------------------------------- #
cbar_sst = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55', '#F28C89',
                                  '#FFCECC', 'white', '#B3DBFF', '#83B9EB',
                                  '#5E9AD7', '#3C7DC3', '#2064AF'][::-1])
cbar_sst.set_over('#9B1C00')
cbar_sst.set_under('#014A9B')
cbar_sst.set_bad(color='white')

cbar_sst = colors.ListedColormap(['#B98200', '#CD9E46', '#E2B361', '#E2BD5A',
                                  '#FFF1C6', 'white', '#B1FFD0', '#7CEB9F',
                                  '#52D770', '#32C355', '#1EAF3D'][::-1])
cbar_sst.set_over('#9B6500')
cbar_sst.set_under('#009B2E')
cbar_sst.set_bad(color='white')


cbar = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#2064AF', '#014A9B'][::-1])
cbar.set_over('#641B00')
cbar.set_under('#012A52')
cbar.set_bad(color='white')


cbar = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#2064AF', '#014A9B'][::-1])
cbar.set_over('#641B00')
cbar.set_under('#012A52')
cbar.set_bad(color='white')

cbar = colors.ListedColormap([
                                 '#641B00', '#892300', '#9B1C00', '#B9391B',
                                 '#CD4838', '#E25E55',
                                 '#F28C89', '#FFCECC', '#FFE6E6', 'white',
                                 '#E6F2FF', '#B3DBFF',
                                 '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF',
                                 '#014A9B', '#013A75',
                                 '#012A52'
                             ][::-1])

cbar.set_over('#4A1500')
cbar.set_under('#001F3F')
cbar.set_bad(color='white')

cbar_snr = colors.ListedColormap(['#070B4F','#2E07AC', '#387AE4' ,'#6FFE9B',
                                  '#FFFFFF',
                                  '#FFFFFF', '#FFFFFF',
                                  '#FEB77E','#CA3E72','#782281','#251255'])
cbar_snr.set_over('#251255')
cbar_snr.set_under('#070B4F')
cbar_snr.set_bad(color='white')

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07',
                                 '#543005', ][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

cbar_pp_11 = colors.ListedColormap(['#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                '#F1DFB3', '#DCBC75', '#995D13',
                                    '#6A3D07'][::-1])
cbar_pp_11.set_under('#6A3D07')
cbar_pp_11.set_over('#004C42')
cbar_pp_11.set_bad(color='white')

cbar_pp_19 = colors.ListedColormap([
    '#001912',  '#003C30',  '#004C42',  '#0C7169',
    '#3FA293',  '#79C8BC',  '#A1D7CD', '#CFE9E5', '#E8F4F2',
    'white',
    '#F6E7C8', '#F1DFB3', '#DCBC75', '#C89D4F',
    '#B17B2C', '#995D13', '#7F470E', '#472705', '#2F1803'][::-1])

cbar_pp_19.set_under('#230F02')
cbar_pp_19.set_over('#0D2D25')
cbar_pp_19.set_bad('white')

cbar_ks = colors.ListedColormap(['#C2FAB6', '#FAC78F'])

cbar_ks.set_under('#5DCCBF')
cbar_ks.set_over('#FA987C')
cbar_ks.set_bad(color='white')


cbar_snr_t = colors.ListedColormap(['#00876c', '#439981', '#6aaa96', '#8cbcac',
                                    '#aecdc2', '#cfdfd9', '#FFFFFF',
                                    '#f1d4d4', '#f0b8b8', '#ec9c9d',
                                    '#e67f83', '#de6069', '#d43d51'])

cbar_snr_t.set_under('#006D53')
cbar_snr_t.set_over('#AB183F')
cbar_snr_t.set_bad(color='white')


cbar_snr_pp = colors.ListedColormap(['#a97300', '#bb8938', '#cc9f5f',
                                     '#dbb686','#e9cead',
                                     '#f5e6d6', '#ffffff', '#dce9eb',
                                     '#b8d4d8', '#95bfc5','#70aab2',
                                     '#48959f', '#00818d'])

cbar_snr_pp.set_under('#6A3D07')
cbar_snr_pp.set_over('#1E6D5A')
cbar_snr_pp.set_bad(color='white')

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

# Variables comunes ---------------------------------------------------------- #
#sig = True
periodos = [[1940, 2020]]
y1 = 0
p = periodos[0]
s = 'SON'
s_count = 0

# scatter
in_label_size = 18
label_legend_size = 18
tick_label_size = 15
scatter_size_fix = 3

# Titles --------------------------------------------------------------------- #
subtitulos_regre = [r"$ONI$",  r"$DMI$",
                    r"$ONI|_{DMI}$",
                    r"$DMI|_{ONI}$"]

# En orden de ploteo
cases = ['N34_un_pos', 'N34_un_neg', 'DMI_un_pos', 'DMI_un_neg',
         'DMI_sim_pos', 'DMI_sim_neg']

cases_cfsv2 =['n34_puros_pos', 'dmi_puros_pos', 'sim_pos',
              'n34_puros_neg', 'dmi_puros_neg', 'sim_neg']

title_case = ['Pure positive IOD ', 'Pure El Niño', 'El Niño - positive IOD',
               'Pure negative IOD', 'Pure La Niña', 'La Niña - negative IOD']

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
print('# CFSv2 Composite by magnitude -------------------------------------- #')
print('Set -----------------------------------------------------------------')
cases = ['dmi_puros_pos', 'dmi_puros_neg',
        'n34_puros_pos', 'n34_puros_neg',
        'sim_pos', 'sim_neg',
        #'dmi_neg_n34_pos', 'dmi_pos_n34_neg',
        'neutros']

row_titles = [None, None, 'Strong EN', None, None, None,
              None, 'Moderate EN', None, None, 'Neutro ENSO',
              None, None, None, None, 'Moderate LN', None, None,
              None, None, 'Strong LN']

col_titles = [None, None, 'Neutro IOD', 'Moderate IOD+',
              'Strong IOD+', None, None, None, None, None,
              'Strong IOD-', 'Moderate IOD-']

# Orden de ploteo ------------------------------------------------------------ #
bin_limits = [[-4.5,-1], #0 s
              [-1, -0.5], #1 m
              [-0.5, 0.5], #2 -
              [0.5, 1], #3  m
              [1, 4.5]] #4 s

indices = ['n34', 'dmi']
magnitudes = ['s', 'm']
cases_names, cases_magnitude, bins_by_cases_n34, bins_by_cases_dmi = \
    SetBinsByCases(indices, magnitudes, bin_limits, cases)

aux_scale_hgt = [-100, -50, -30, -15, -5, 5, 15, 30, 50, 100]
aux_scale_hgt200 = [-150, -100, -50, -25, -10, 10, 25, 50, 100, 150]

variables = ['tref', 'prec', 'hgt750', 'hgt']
aux_scales = [scale_t, scale_pp, aux_scale_hgt, aux_scale_hgt200]
aux_cbar = [cbar, cbar_pp, cbar, cbar]
aux_cbar_snr = [cbar_snr_t, cbar_snr_pp, cbar_snr, cbar_snr]
aux_scales_clim = [np.linspace(0,30,11), np.linspace(0, 300, 11),
                   np.linspace(1.1e5,1.2e5,11), np.linspace(1.1e5,1.2e5,11)]
aux_cbar_clim = ['OrRd', 'PuBu', 'OrRd', 'OrRd']


def CasesOrdenados_Suma(cases_ordenados):
    cases_ordenados.loc[dict(plots=3)] = \
        cases_ordenados.sel(plots=[2, 13]).sum(dim='plots')
    cases_ordenados.loc[dict(plots=4)] = \
        cases_ordenados.sel(plots=[2, 14]).sum(dim='plots')
    cases_ordenados.loc[dict(plots=8)] = \
        cases_ordenados.sel(plots=[7, 13]).sum(dim='plots')
    cases_ordenados.loc[dict(plots=9)] = \
        cases_ordenados.sel(plots=[7, 14]).sum(dim='plots')

    cases_ordenados.loc[dict(plots=15)] = \
        cases_ordenados.sel(plots=[10, 17]).sum(dim='plots')
    cases_ordenados.loc[dict(plots=16)] = \
        cases_ordenados.sel(plots=[11, 17]).sum(dim='plots')
    cases_ordenados.loc[dict(plots=20)] = \
        cases_ordenados.sel(plots=[10, 22]).sum(dim='plots')
    cases_ordenados.loc[dict(plots=21)] = \
        cases_ordenados.sel(plots=[11, 22]).sum(dim='plots')

    return cases_ordenados


print('Plot ----------------------------------------------------------------- ')
for v, v_scale, v_cbar, v_scale_clim, v_cbar_clim, v_cbar_snr in zip(
        variables, aux_scales, aux_cbar, aux_scales_clim, aux_cbar_clim,
        aux_cbar_snr):
    print(v)

    if v == 'prec':
        fix = 30
        fix_clim = 0
        v_in_name = v
        scale_in_hgt = aux_scale_hgt
    elif v == 'tref':
        fix = 1
        fix_clim = 273
        v_in_name = v
        scale_in_hgt = aux_scale_hgt
    else:
        fix = 9.8
        fix_clim = 0
        if v == 'hgt750':
            v_in_name = 'HGT'
            scale_in_hgt = aux_scale_hgt
        else:
            v_in_name = 'hgt'
            scale_in_hgt = aux_scale_hgt200

    aux_comps = {}
    aux_num_comps = {}
    aux_comps_hgt = {}
    aux_num_comps_hgt = {}
    aux_comps_snr = {}
    aux_num_comps_snr = {}
    n_count = 0
    n_count_hgt = 0
    n_count_snr = 0

    for c_count, c in enumerate(cases):

        # BinsByCases -------------------------------------------------------- #
        print('comp --------------------------------------------------------- ')
        cases_bin, num_bin, auxx = BinsByCases(
            v=v, v_name=v_in_name, fix_factor=fix, s='SON', mm=10, c=c,
            c_count=c_count,
            bin_limits=bin_limits,
            bins_by_cases_dmi=bins_by_cases_dmi,
            bins_by_cases_n34=bins_by_cases_n34,
            snr=False, cases_dir=cases_dir, dates_dir=dates_dir,
            neutro_clim=True)

        bins_aux_dmi = bins_by_cases_dmi[c_count]
        bins_aux_n34 = bins_by_cases_n34[c_count]

        for b_n34 in range(0, len(bins_aux_n34)):
            for b_dmi in range(0, len(bins_aux_dmi)):
                aux_comps[cases_names[n_count]] = cases_bin[b_dmi][b_n34]
                aux_num_comps[cases_names[n_count]] = num_bin[b_dmi][b_n34]
                n_count += 1

        print('snr ---------------------------------------------------------- ')
        cases_bin_snr, num_bin_snr, auxx_snr = BinsByCases(
            v=v, v_name=v_in_name, fix_factor=fix, s='SON', mm=10, c=c,
            c_count=c_count,
            bin_limits=bin_limits,
            bins_by_cases_dmi=bins_by_cases_dmi,
            bins_by_cases_n34=bins_by_cases_n34,
            snr=True, cases_dir=cases_dir, dates_dir=dates_dir,
            neutro_clim=True)

        bins_aux_dmi_snr = bins_by_cases_dmi[c_count]
        bins_aux_n34_snr = bins_by_cases_n34[c_count]

        for b_n34 in range(0, len(bins_aux_n34_snr)):
            for b_dmi in range(0, len(bins_aux_dmi_snr)):
                aux_comps_snr[cases_names[n_count_snr]] = \
                    cases_bin_snr[b_dmi][b_n34]
                aux_num_comps_snr[cases_names[n_count_snr]] = \
                    num_bin_snr[b_dmi][b_n34]
                n_count_snr += 1

        # hgt750 contornos --------------------------------------------------- #
        if v == 'prec' or v == 'tref':
            check_t_pp = True

            cases_bin_hgt, num_bin_hgt, auxx_hgt = BinsByCases(
                v='hgt750', v_name='HGT', fix_factor=9.8, s='SON', mm=10,
                c=c,
                c_count=c_count,
                bin_limits=bin_limits,
                bins_by_cases_dmi=bins_by_cases_dmi,
                bins_by_cases_n34=bins_by_cases_n34,
                snr=False, cases_dir=cases_dir, dates_dir=dates_dir,
                neutro_clim=True)

            bins_aux_dmi_hgt = bins_by_cases_dmi[c_count]
            bins_aux_n34_hgt = bins_by_cases_n34[c_count]

            for b_n34 in range(0, len(bins_aux_n34_hgt)):
                for b_dmi in range(0, len(bins_aux_dmi_hgt)):
                    aux_comps_hgt[cases_names[n_count_hgt]] = \
                        cases_bin_hgt[b_dmi][b_n34]
                    aux_num_comps_hgt[cases_names[n_count_hgt]] = \
                        num_bin_hgt[b_dmi][b_n34]
                    n_count_hgt += 1

            lat_hgt = auxx_hgt.lat.values
            lon_hgt = auxx_hgt.lon.values

        else:
            check_t_pp = False

    lat = np.arange(-60, 20 + 1)
    lon = np.arange(275, 330 + 1)

    # Clim ------------------------------------------------------------------- #
    if check_t_pp is True:
        clim = xr.open_dataset(f'/pikachu/datos/luciano.andrian/'
                               f'val_clim_cfsv2/hindcast_{v}_cfsv2_'
                               f'mc_no-norm_son.nc') * fix
        clim = clim.rename({list(clim.data_vars)[0]: 'var'})

        clim = clim.sel(lon=lon, lat=lat)
        clim = clim - fix_clim

    clim_hgt = xr.open_dataset(f'/pikachu/datos/luciano.andrian/'
                           f'val_clim_cfsv2/hindcast_cfsv2_meanclim_son.nc'
                           ) * 9.8
    clim_hgt = clim_hgt.rename({list(clim_hgt.data_vars)[0]: 'var'})
    clim_hgt = clim_hgt.sel(lon=lon, lat=lat)

    # ordenando por categorias ----------------------------------------------- #
    cases_ordenados = []
    cases_ordenados_sig = []
    cases_ordenados_hgt = []
    cases_ordenados_snr = []

    aux_num = []
    aux_num_hgt = []
    aux_num_snr = []
    for c in cases_magnitude:
        try:
            aux = aux_comps[c]
            aux_num.append(aux_num_comps[c])

            aux_snr = aux_comps_snr[c]
            aux_num_snr.append(aux_num_comps_snr[c])

            if check_t_pp is True:
                aux_hgt = aux_comps_hgt[c]
                aux_num_hgt.append(aux_num_comps_hgt[c])

        except:
            aux = aux_comps[cases_magnitude[2]] * 0
            aux_num.append('')

            aux_snr = aux_comps_snr[cases_magnitude[2]] * 0
            aux_num_snr.append('')

            if check_t_pp is True:
                aux_hgt = aux_comps_hgt[cases_magnitude[2]] * 0
                aux_num_hgt.append('')

        if check_t_pp is True:
            lat_aux = lat
            lon_aux = lon
        else:
            lat_aux = lat_hgt[::-1]
            lon_aux = lon_hgt
            if v == 'hgt750':
                lat_aux = lat_hgt

        da = xr.DataArray(aux, dims=["lat", "lon"],
                          coords={"lat": lat_aux,
                                  "lon": lon_aux},
                          name="var")

        da_snr = xr.DataArray(aux_snr['var'], dims=["lat", "lon"],
                              coords={"lat": lat_aux,
                                      "lon": lon_aux},
                              name="var")

        if len(da.sel(lat=slice(-60, 20)).lat.values)>0:
            da = da.sel(lat=slice(-60, 20), lon=slice(275, 330))
            da_snr = da_snr.sel(lat=slice(-60, 20), lon=slice(275, 330))
        else:
            da = da.sel(lat=slice(20, -60), lon=slice(275, 330))
            da_snr = da_snr.sel(lat=slice(20, -60), lon=slice(275, 330))


        try:
            if v == 'hgt':
                aux_sig = xr.open_dataset(
                    f'{cfsv2_sig_dir}{v}_QT_{c}_CFSv2_detrend_05.nc')
            else:
                aux_sig = xr.open_dataset(
                    f'{cfsv2_sig_dir}{v}_QT_{c}_CFSv2_detrend_05.nc') * fix

            aux_sig['lat'] = aux_sig['lat'][::-1]

            # if len(aux_sig.sel(lat=slice(-60, 20)).lat.values) > 0:
            #     aux_sig = aux_sig.sel(lat=slice(-60, 20), lon=slice(275, 330))
            # else:
            #     aux_sig = aux_sig.sel(lat=slice(20, -60), lon=slice(275, 330))
            #
            # aux_sig = aux_sig.sel(lat=lat, lon=lon)
            da_sig = da.where((da < aux_sig[v][0]) |
                              (da > aux_sig[v][1]))
            da_sig = da_sig.where(np.isnan(da_sig), 1)
        except:
            da_sig = da * 0

        if check_t_pp is True:
            da_hgt = xr.DataArray(aux_hgt, dims=["lat", "lon"],
                                  coords={"lat": lat_hgt, "lon": lon_hgt},
                                  name="var")
            da_hgt = da_hgt.sel(lat=slice(None, None, -1))

            cases_ordenados_hgt.append(da_hgt)

        cases_ordenados.append(da)
        cases_ordenados_snr.append(da_snr)
        cases_ordenados_sig.append(da_sig)

    cases_ordenados = xr.concat(cases_ordenados, dim='plots')
    cases_ordenados_snr = xr.concat(cases_ordenados_snr, dim='plots')
    cases_ordenados_sig = xr.concat(cases_ordenados_sig, dim='plots')*0

    if check_t_pp is True:
        cases_ordenados_hgt = xr.concat(cases_ordenados_hgt, dim='plots')
        ocean_mask = True
    else:
        cases_ordenados_hgt = xr.concat(cases_ordenados, dim='plots')
        clim = clim_hgt
        ocean_mask = False

    cases_ordenados = CasesOrdenados_Suma(cases_ordenados)
    cases_ordenados_hgt = CasesOrdenados_Suma(cases_ordenados_hgt)

    PlotFinal_CompositeByMagnitude(data=cases_ordenados, levels=v_scale,
                                   cmap=v_cbar, titles=aux_num,
                                   namefig=f'{v}_cfsv2_comp_by_magnitude_SUMA',
                                   map='sa', save=save, dpi=dpi,
                                   out_dir=out_dir,
                                   data_ctn=cases_ordenados_hgt,
                                   levels_ctn = scale_in_hgt,
                                   color_ctn='k', row_titles=row_titles,
                                   col_titles=col_titles, clim_plot=clim,
                                   clim_cbar=v_cbar_clim,
                                   clim_levels=v_scale_clim,
                                   high=1.5, width=5.5,
                                   clim_plot_ctn=clim_hgt,
                                   clim_levels_ctn=np.linspace(1.1e5,1.2e5,11),
                                   ocean_mask=ocean_mask,
                                   data_ctn_no_ocean_mask=True,
                                   plot_step=1,
                                   cbar_pos='V',
                                   sig_data=cases_ordenados_sig,
                                   hatches='...')

    # # SNR plot
    # if check_t_pp is True:
    #     data_ctn_snr = None
    # else:
    #     data_ctn_snr = cases_ordenados_snr
    # PlotFinal_CompositeByMagnitude(data=cases_ordenados_snr, levels=scale_snr,
    #                                cmap=v_cbar_snr, titles=aux_num,
    #                                namefig=f'snr_{v}_cfsv2_comp_by_magnitude',
    #                                map='sa', save=save, dpi=dpi,
    #                                out_dir=out_dir,
    #                                data_ctn=data_ctn_snr,
    #                                levels_ctn = scale_snr,
    #                                color_ctn='k', row_titles=row_titles,
    #                                col_titles=col_titles, clim_plot=clim,
    #                                clim_cbar=v_cbar_clim,
    #                                clim_levels=v_scale_clim,
    #                                high=1.5, width=5.5,
    #                                clim_plot_ctn=None,
    #                                clim_levels_ctn=np.linspace(1.1e5,1.2e5,11),
    #                                ocean_mask=ocean_mask,
    #                                data_ctn_no_ocean_mask=True,
    #                                plot_step=1,
    #                                cbar_pos='V')
