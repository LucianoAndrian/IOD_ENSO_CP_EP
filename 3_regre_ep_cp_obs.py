"""
Regresion total y parcial de EP y CP observado
No se tiene en cuenta el IOD

FIGURAS TEMPORALES
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/regre_obs/'

# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
from matplotlib import colors
import matplotlib.pyplot as plt
from funciones.plots_utils import SetDataToPlotFinal, PlotFinalTwoVariables, \
    PlotFinal
from funciones.regre_utils import ComputeWithEffect, ComputeWithoutEffect
from funciones.general_utils import xrFieldTimeDetrend, SameDateAs

if save:
    dpi = 300
else:
    dpi = 100

# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

# ---------------------------------------------------------------------------- #
def set_data(data, lats=None, lons=None, year_start=1940, year_end=2020):
    if lats is not None and len(data.sel(lat=slice(lats[0], lats[-1])).lat) > 0:
        data = data.sel(lat=slice(lats[0], lats[-1]))
    else:
        data = data.sel(lat=slice(lats[-1], lats[0]))

    if lons is not None:
        data = data.sel(lon=slice(lons[0], lons[-1]))

    if year_start is not None and year_end is not None:
        data = data.sel(time=data.time.dt.year.isin(np.arange(year_start, year_end+1)))

    return data, data.time

def set_and_rename(*args):
    dataset = []
    for arg in args:
        arg = arg.to_dataset(name='var')
        dataset.append(arg)

    return tuple(dataset)

def compute_partial_regre(data,index1, index2):
    data_index1, data_corr_index1, data_index2, data_corr_index2, _, _, _, _ = \
        ComputeWithEffect(data=data, data2=None,
                          n34=index1.sel(time=index1.time.dt.month.isin(10)),
                          dmi=index2.sel(time=index2.time.dt.month.isin(10)),
                          two_variables=False, m=10, full_season=False,
                          time_original=time_original)

    data_index1_woindex2, data_corr_index1_woindex2, data_index2_woindex1, \
        data_corr_index2_woindex1 = ComputeWithoutEffect(
        data=data,
        n34=index1.sel(time=index1.time.dt.month.isin(10)),
        dmi= index2.sel(time=index2.time.dt.month.isin(10)),
        m=10, time_original=time_original)

    data_index1, data_corr_index1, data_index2, data_corr_index2, \
        data_index1_woindex2, data_corr_index1_woindex2, \
        data_index2_woindex1, data_corr_index2_woindex1 = set_and_rename(
        data_index1, data_corr_index1, data_index2, data_corr_index2,
        data_index1_woindex2, data_corr_index1_woindex2,
        data_index2_woindex1, data_corr_index2_woindex1)

    return (data_index1, data_corr_index1, data_index2, data_corr_index2,
            data_index1_woindex2, data_corr_index1_woindex2,
            data_index2_woindex1, data_corr_index2_woindex1)

def MakerMaskSig(data, r_crit):
    mask_sig = data.where((data < -1 * r_crit) | (data > r_crit))
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

def compute_regre(var1=None, var2=None, ep=None, cp=None, r_crit=1,
                  var1_is_var3=False):
    var1_ep, var1_corr_ep, var1_cp, var1_corr_cp, var1_ep_wocp, \
        var1_corr_ep_wocp, var1_cp_woep, var1_corr_cp_woep = \
        compute_partial_regre(var1, ep, cp)

    if var2 is not None:
        var2_ep, var2_corr_ep, var2_cp, var2_corr_cp, var2_ep_wocp, \
            var2_corr_ep_wocp, var2_cp_woep, var2_corr_cp_woep = \
            compute_partial_regre(var2, ep, cp)

        aux_v = SetDataToPlotFinal(var1_ep, var1_ep_wocp,
                                   var1_cp, var1_cp_woep,
                                   var2_ep, var2_ep_wocp,
                                   var2_cp, var2_cp_woep)

        aux_sig_v = SetDataToPlotFinal(
            var1_ep * MakerMaskSig(var1_corr_ep, r_crit),
            var1_ep_wocp * MakerMaskSig(var1_corr_ep_wocp, r_crit),
            var1_cp * MakerMaskSig(var1_corr_cp, r_crit),
            var1_cp_woep * MakerMaskSig(var1_corr_cp_woep, r_crit),
            var2_ep * MakerMaskSig(var2_corr_ep, r_crit),
            var2_ep_wocp * MakerMaskSig(var2_corr_ep_wocp, r_crit),
            var2_cp * MakerMaskSig(var2_corr_cp, r_crit),
            var2_cp_woep * MakerMaskSig(var2_corr_cp_woep, r_crit))
    else:
        if var1_is_var3:
            # Va doble para poder sumarlo al plot
            aux_v = SetDataToPlotFinal(var1_ep, var1_ep_wocp,
                                       var1_cp, var1_cp_woep,
                                       var1_ep, var1_ep_wocp,
                                       var1_cp, var1_cp_woep)

            aux_sig_v = None

        else:
            aux_v = SetDataToPlotFinal(var1_ep, var1_ep_wocp,
                                       var1_cp, var1_cp_woep)

            aux_sig_v = SetDataToPlotFinal(
                var1_ep * MakerMaskSig(var1_corr_ep, r_crit),
                var1_ep_wocp * MakerMaskSig(var1_corr_ep_wocp, r_crit),
                var1_cp * MakerMaskSig(var1_corr_cp, r_crit),
                var1_cp_woep * MakerMaskSig(var1_corr_cp_woep, r_crit))

    return aux_v, aux_sig_v

def OrdenarNC_wTime_fromW(data):
    newdata = xr.Dataset(
        data_vars=dict(
            var=(['time', 'lat', 'lon'], data['var'][0, :, :, :].values)

        ),
        coords=dict(
            lon=(['lon'], data.lon),
            lat=(['lat'], data.lat),
            time=(['time'], data.time)
        )
    )
    return newdata

# ---------------------------------------------------------------------------- #
from aux_set_obs_indices import cp_tk, ep_tk, cp_td, ep_td, cp_n, ep_n, \
    year_start, year_end

scale_hgt=[-300, -270, -240, -210, -180, -150, -120, -90, -60,
 -30, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
scale_pp = np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45])
scale_t = [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1]

cbar = colors.ListedColormap(['#641B00', '#892300', '#9B1C00', '#B9391B',
                              '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC', '#FFE6E6', 'white',
                              '#E6F2FF', '#B3DBFF',
                              '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF',
                              '#014A9B', '#013A75','#012A52'][::-1])

cbar.set_over('#4A1500')
cbar.set_under('#001F3F')
cbar.set_bad(color='white')

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07',
                                 '#543005', ][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

print('Prec y Tref --------------------------------------------------------- #')
# pp
prec = xr.open_dataset(f'{data_dir_t_pp}ppgpcc_w_c_d_1_SON.nc')
prec, time_original = set_data(prec, lats=[20,-60], lons=[275, 330],
                               year_start=year_start, year_end=year_end)
# temp
temp = xr.open_dataset(f'{data_dir_t_pp}tcru_w_c_d_0.25_SON.nc')
temp, _ = set_data(temp, lats=[20,-60], lons=[275, 330],
                               year_start=year_start, year_end=year_end)

# hgt200
data_hgt = xr.open_dataset(f'{data_dir}HGT200_SON_mer_d_w.nc')
data_hgt = data_hgt.sel(time=data_hgt.time.dt.year.isin(
    np.arange(year_start, year_end+1)))

v_from_w = ['div_UV200', 'vp_from_UV200_w'] # creadas a partir de windphsere

data_div = xr.open_dataset(data_dir + v_from_w[0] + '.nc')
data_div = xrFieldTimeDetrend(
    OrdenarNC_wTime_fromW(data_div.rename({'divergence':'var'})), 'time')

data_vp = xr.open_dataset(data_dir+ v_from_w[1] + '.nc')
data_vp = xrFieldTimeDetrend(
    OrdenarNC_wTime_fromW(data_vp.rename({'velocity_potential':'var'})), 'time')

data_sst = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                        "sst.mnmean.nc")
data_sst = data_sst.sel(time=data_sst.time.dt.year.isin(range(1940,2021)))
data_sst = data_sst.rename({'sst':'var'})
data_sst = xrFieldTimeDetrend(data_sst, 'time')
data_sst = data_sst.drop_dims('nbnds')

data_div = SameDateAs(data_div, data_hgt)
data_vp =  SameDateAs(data_vp, data_hgt)
data_sst = SameDateAs(data_sst, data_hgt)

print('# Regresion --------------------------------------------------------- #')
t_critic = 1.66  # es MUY similar (2 digitos) para ambos períodos
r_crit = np.sqrt(1 / (((np.sqrt((year_end - year_start) - 2) / t_critic) ** 2) + 1))

for ep, cp, name in zip([ep_tk, ep_td, ep_n], [cp_tk, cp_td, cp_n],
                        ['tk', 'td', 'n']):

    aux_v, aux_sig_v = compute_regre(var1=prec, var2=temp, ep=ep, cp=cp,
                                     r_crit=r_crit, var1_is_var3=False)

    aux_ctn, _ = compute_regre(var1=data_hgt, var2=None, ep=ep, cp=cp,
                               r_crit=r_crit,
                               var1_is_var3=True)

    aux_hgt, aux_sig_hgt = compute_regre(var1=data_hgt, var2=None, ep=ep, cp=cp,
                                         r_crit=r_crit,
                                         var1_is_var3=False)


    subtitulos_regre = [r"$EP$", r"$EP|_{CP}$", r"$CP$", r"$CP|_{EP}$",
                        r"$EP$", r"$EP|_{CP}$", r"$CP$", r"$CP|_{EP}$"]
    plt.rcParams['hatch.linewidth'] = 1
    PlotFinalTwoVariables(data=aux_v, num_cols=4,
                          levels_r1=scale_pp, cmap_r1=cbar_pp,
                          levels_r2=scale_t, cmap_r2=cbar,
                          data_ctn=aux_ctn, levels_ctn_r1=scale_hgt,
                          levels_ctn_r2=scale_hgt, color_ctn='k',
                          titles=subtitulos_regre, namefig=f'regre_pp_t_{name}',
                          save=save, dpi=dpi,
                          out_dir=out_dir, pdf=False,
                          high=2.5, width=7.7, step=1,
                          ocean_mask=False, num_cases=False,
                          num_cases_data=None,
                          sig_points=aux_sig_v, hatches='...',
                          data_ctn_no_ocean_mask=False)

    plt.rcParams['hatch.linewidth'] = 0.5
    PlotFinal(data=aux_hgt, levels=scale_hgt, cmap=cbar,
              titles=subtitulos_regre, namefig=f'regre_hgt_{name}', map='hs',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=aux_hgt, color_ctn='k', high=1.2, width=6,
              num_cases=False, num_cases_data=None, num_cols=2,
              ocean_mask=False, pdf=False, levels_ctn=scale_hgt,
              data_ctn_no_ocean_mask=False, step=1,
              sig_points=aux_sig_hgt, hatches='...')

    scale_sst = [-1, -.5, -.1, 0, .1, .5, 1]
    scale_div = [-4.33e-07, 4.33e-07]
    scale_vp = [-3e6, -2.5e6, -2e6, -1.5e6, -1e6, -0.5e6, 0, 0.5e6, 1e6, 1.5e6,
                2e6, 2.5e6, 3e6]

    cbar_sst = colors.ListedColormap(
        ['#B98200', '#CD9E46', '#E2B361', '#E2BD5A',
         '#FFF1C6', 'white', '#B1FFD0', '#7CEB9F',
         '#52D770', '#32C355', '#1EAF3D'][::-1])
    cbar_sst.set_over('#9B6500')
    cbar_sst.set_under('#009B2E')
    cbar_sst.set_bad(color='white')

    aux_sst, _ = compute_regre(var1=data_sst, var2=None, ep=ep, cp=cp,
                               r_crit=r_crit, var1_is_var3=False)

    aux_vp, _ = compute_regre(var1=data_vp, var2=None, ep=ep, cp=cp,
                              r_crit=r_crit, var1_is_var3=False)

    aux_div, _ = compute_regre(var1=data_div, var2=None, ep=ep, cp=cp,
                               r_crit=r_crit, var1_is_var3=False)

    PlotFinal(data=aux_sst, levels=scale_sst, cmap=cbar_sst,
              titles=subtitulos_regre, namefig=f'regre_sst_vp_div_{name}', map='hs',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=aux_vp, levels_ctn=scale_vp, color_ctn='k',
              data_ctn2=aux_div, levels_ctn2=scale_div,
              color_ctn2=['#FF0002', '#0003FF'], high=1.3, pdf=False)

print(' --------------------------------------------------------------------- ')
print('Done')
print(' --------------------------------------------------------------------- ')