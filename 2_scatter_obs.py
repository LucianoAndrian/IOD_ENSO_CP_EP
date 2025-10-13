"""
Scatter plots comparando indices EP CP entre
Takahayi et al. 2011
Tedeschi et alo. 2014
Sulivan et al. 2016

FUNCION DE FIGURAS TEMPORAL, NO DEFINIVITIVA
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'
# ---------------------------------------------------------------------------- #

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from funciones.indices_utils import DMI


# ---------------------------------------------------------------------------- #
def aux_tmp_PlotScatter(idx1, idx2, idx1_name, idx2_name, save=save,
                out_dir=out_dir, name_fig='fig',
                idx3=None, idx3_name=None, title=''):
    if save:
        dpi = 300
    else:
        dpi = 300
    
    in_label_size = 13
    label_legend_size = 12
    tick_label_size = 11
    scatter_size_fix = 3
    fig, ax = plt.subplots(figsize=(7.08661, 7.08661))

    try:
        idx2_name_2 = idx2_name.split('-')[0]
    except:
        idx2_name_2 = idx2_name

    # todos
    ax.scatter(x=idx1, y=idx2, marker='o',
               s=15 * scatter_size_fix, edgecolor='k', color='r', alpha=1,
               label=f'{idx1_name} vs {idx2_name_2}')

    if idx3 is not None:
        ax.scatter(x=idx1, y=idx3, marker='D',
                   s=15 * scatter_size_fix, edgecolor='k', color='g', alpha=1,
                   label=f'{idx1_name} vs {idx3_name}')

    ax.legend(loc=(.01, .85), fontsize=label_legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, pad=1)
    ax.set_ylim((-5, 5))
    ax.set_xlim((-5, 5))

    ax.axhspan(-.5 , .5,#/ idx1_sd.sst.values, .5 / idx1_sd.sst.values,
               alpha=0.2, color='black', zorder=0)
    ax.axvspan(-.5 , .5,#/ idx2_sd.sst.values, .5 / idx2_sd.sst.values,
               alpha=0.2, color='black', zorder=0)
    ax.set_xlabel(f'{idx1_name}', size=in_label_size)
    ax.set_ylabel(f'{idx2_name}', size=in_label_size)
    ax.text(-4.9, 4.6, f'{idx2_name}+/{idx1_name}-', dict(size=in_label_size))
    ax.text(-.2, 4.6,  f'{idx2_name}+', dict(size=in_label_size))
    ax.text(+3.7, 4.6, f'{idx2_name}+/{idx1_name}+', dict(size=in_label_size))
    ax.text(+4.2, -.1,  f'{idx1_name}+', dict(size=in_label_size))
    ax.text(+3.7, -4.9, f'{idx2_name}-/{idx1_name}+', dict(size=in_label_size))
    ax.text(-.2, -4.9, f'{idx2_name}-', dict(size=in_label_size))
    ax.text(-4.9, -4.9, f'{idx2_name}-/{idx1_name}-', dict(size=in_label_size))
    ax.text(-4.9, -.1, f'{idx1_name}-', dict(size=in_label_size))
    plt.tight_layout()
    plt.title(title)

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()
# ---------------------------------------------------------------------------- #
from aux_set_obs_indices import cp_tk, ep_tk, cp_td, ep_td, cp_n, ep_n, \ 
    year_end, year_start

dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
dmi = dmi.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-31'))
dmi = dmi.sel(time=dmi.time.dt.month.isin(10))

# Tk ------------------------------------------------------------------------- #
aux_tmp_PlotScatter(idx1=dmi, idx2=ep_tk, idx3=cp_tk,
                    idx1_name='DMI', idx2_name='EP-CP', idx3_name='CP',
                    save=save, out_dir=out_dir,  name_fig='DMI_vs_EP_CP_obs',
                    title='DMI vs EP_tk - CP_tk')

aux_tmp_PlotScatter(idx1=ep_tk, idx2=cp_tk, idx1_name='EP_tk', idx2_name='CP_tk',
                    save=save, out_dir=out_dir,  name_fig='EP_vs_CP_obs',
                    title='EP_tk vs CP_tk')

# td ------------------------------------------------------------------------- #
aux_tmp_PlotScatter(idx1=dmi, idx2=ep_td, idx3=cp_td,
                    idx1_name='DMI', idx2_name='EP-CP', idx3_name='CP',
                    save=save, out_dir=out_dir,  name_fig='DMI_vs_EP_CP_td_obs',
                    title='DMI vs EP_td - CP_td')

aux_tmp_PlotScatter(idx1=ep_td, idx2=cp_td, idx1_name='EP_td', idx2_name='CP_td',
                    save=save, out_dir=out_dir,  name_fig='EP_vs_CP_td_obs',
                    title='EP_td vs CP_td')

# n -------------------------------------------------------------------------- #
aux_tmp_PlotScatter(idx1=dmi, idx2=ep_n, idx3=cp_n,
                    idx1_name='DMI', idx2_name='EP/CP_n', idx3_name='CP_n',
                    save=save, out_dir=out_dir,  name_fig='DMI_vs_EP_CP_n_obs',
                    title='DMI vs EP_n - CP_n')

aux_tmp_PlotScatter(idx1=ep_n, idx2=cp_n, idx1_name='EP_n', idx2_name='CP_n',
                    save=save, out_dir=out_dir,  name_fig='EP_vs_CP_n_obs',
                    title='DMI vs EP_n - CP_n')

# Tk vs Td vs N -------------------------------------------------------------- #
aux_tmp_PlotScatter(idx1=ep_tk, idx2=ep_td, idx1_name='EP_tk', idx2_name='EP_td',
                    save=save, out_dir=out_dir,  name_fig='EP_tk_vs_EP_td_obs',
                    title='EP_tk vs EP_td')

aux_tmp_PlotScatter(idx1=cp_tk, idx2=cp_td, idx1_name='CP_tk', idx2_name='CP_td',
                    save=save, out_dir=out_dir,  name_fig='CP_tk_vs_CP_td_obs',
                    title='CP_tk vs CP_td')

# ---------------------------------------------------------------------------- #
aux_tmp_PlotScatter(idx1=ep_tk, idx2=ep_n, idx1_name='EP_tk', idx2_name='EP_n',
                    save=save, out_dir=out_dir,  name_fig='EP_tk_vs_EP_n_obs',
                    title='EP_tk vs EP_n')

aux_tmp_PlotScatter(idx1=cp_tk, idx2=cp_n, idx1_name='CP_tk', idx2_name='CP_n',
                    save=save, out_dir=out_dir,  name_fig='CP_tk_vs_CP_n_obs',
                    title='CP_tk vs CP_n')

# ---------------------------------------------------------------------------- #
aux_tmp_PlotScatter(idx1=ep_td, idx2=ep_n, idx1_name='EP_tk', idx2_name='EP_n',
            save=save, out_dir=out_dir,  name_fig='EP_td_vs_EP_n_obs',
            title='EP_td vs EP_n')

aux_tmp_PlotScatter(idx1=cp_td, idx2=cp_n, idx1_name='CP_td', idx2_name='CP_n',
            save=save, out_dir=out_dir,  name_fig='CP_td_vs_CP_n_obs',
            title='CP_td vs CP_n')
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #