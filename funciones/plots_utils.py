
from itertools import groupby
import xarray as xr
import numpy as np
import pandas as pd
from pandas.io.common import file_exists

pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import statsmodels.formula.api as smf
import cartopy.feature

import matplotlib as mpl
import matplotlib.path as mpath
from matplotlib.font_manager import FontProperties
import scipy.stats as st
import string
from numpy import ma
from matplotlib import colors
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from scipy.signal import convolve2d
import os
import matplotlib.patches as mpatches
from scipy.integrate import trapz
from matplotlib.colors import BoundaryNorm
from scipy.stats import spearmanr
import matplotlib.colors as mcolors

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob

from funciones.general_utils import MakeMask

def SetDataToPlotFinal(*args):
    data_arrays = []
    first = True
    for arg in args:
        if not isinstance(arg, xr.DataArray):
            try:
                arg = xr.DataArray(arg, dims=['lat', 'lon'])
            except:
                arg = arg.to_array()
                if 1 in arg.shape:
                    arg = arg.squeeze()
                arg = xr.DataArray(arg, dims=['lat', 'lon'])
        else:
            if 1 in arg.shape:
                arg = arg.squeeze()

        if first is False:
            if data_arrays[0].lon.values[-1] != arg.lon.values[-1]:
                arg = arg.interp(lon = data_arrays[0].lon.values,
                                 lat = data_arrays[0].lat.values)

        data_arrays.append(arg)
        first = False

    data = xr.concat(data_arrays, dim='plots')
    data = data.assign_coords(plots=range(data.shape[0]))

    return data

def PlotFinal(data, levels, cmap, titles, namefig, map, save, dpi, out_dir,
              data_ctn=None, levels_ctn=None, color_ctn=None,
              data_ctn2=None, levels_ctn2=None, color_ctn2=None,
              data_waf=None, wafx=None, wafy=None, waf_scale=None,
              waf_step=None, waf_label=None, sig_points=None, hatches=None,
              num_cols=None, high=2, width = 7.08661, step=2, cbar_pos = 'H',
              num_cases=False, num_cases_data=None, pdf=False, ocean_mask=False,
              data_ctn_no_ocean_mask=False, data_ctn2_no_ocean_mask=False,
              pcolormesh=False):

    # cantidad de filas necesarias
    if num_cols is None:
        num_cols = 2
    width = width
    plots = data.plots.values
    num_plots = len(plots)
    num_rows = np.ceil(num_plots / num_cols).astype(int)

    crs_latlon = ccrs.PlateCarree()

    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        step_lon = 60
        high = high
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        step_lon = 60
        high = high
    elif map.upper() == 'HS_EX':
        extent = [0, 359, -65, -20]
        step_lon = 60
        high = 2
    elif map.upper() == 'SA':
        extent = [275, 330, -60, 20]
        step_lon = 20
        high = high
    else:
        print(f"Mapa {map} no seteado")
        return

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    i = 0
    for ax, plot in zip(axes.flatten(), plots):
        no_plot = False

        # CONTOUR ------------------------------------------------------------ #
        if data_ctn is not None:
            if levels_ctn is None:
                levels_ctn = levels.copy()
            try:
                if isinstance(levels_ctn, np.ndarray):
                    levels_ctn = levels_ctn[levels_ctn != 0]
                else:
                    levels_ctn.remove(0)
            except:
                pass
            aux_ctn = data_ctn.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values

                ax.contour(data_ctn.lon.values[::step],
                           data_ctn.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.4,
                           levels=levels_ctn, transform=crs_latlon,
                           colors=color_ctn)

        # CONTOUR2 ----------------------------------------------------------- #
        if data_ctn2 is not None:
            if levels_ctn2 is None:
                levels_ctn2 = levels.copy()

            try:
                if isinstance(levels_ctn2, np.ndarray):
                    levels_ctn2 = levels_ctn2[levels_ctn != 0]
                else:
                    levels_ctn2.remove(0)
            except:
                pass
            aux_ctn = data_ctn2.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn2_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values
                ax.contour(data_ctn2.lon.values[::step],
                           data_ctn2.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.5,
                           levels=levels_ctn2, transform=crs_latlon,
                           colors=color_ctn2)

        # CONTOURF OR COLORMESH ---------------------------------------------- #
        aux = data.sel(plots=plot)
        if aux.mean().values != 0:

            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux = aux * mask_ocean.mask

            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            if pcolormesh is True:
                im = ax.pcolormesh(aux.lon.values[::step],
                                   aux.lat.values[::step],
                                   aux_var[::step, ::step],
                                   vmin=np.min(levels), vmax=np.max(levels),
                                   transform=crs_latlon, cmap=cmap)
            else:
                im = ax.contourf(aux.lon.values[::step], aux.lat.values[::step],
                                 aux_var[::step, ::step],
                                 levels=levels,
                                 transform=crs_latlon, cmap=cmap, extend='both')


        else:
            ax.axis('off')
            no_plot=True

        # WAF ---------------------------------------------------------------- #
        if data_waf is not None:
            wafx_aux = wafx.sel(plots=plot)
            wafy_aux = wafy.sel(plots=plot)

            if ocean_mask is True:
                mask_ocean = MakeMask(wafx_aux)
                wafx_aux = wafx_aux * mask_ocean.mask
                wafy_aux = wafy_aux * mask_ocean.mask


            Q60 = np.nanpercentile(np.sqrt(np.add(np.power(wafx_aux, 2),
                                                  np.power(wafy_aux, 2))), 60)
            M = np.sqrt(np.add(np.power(wafx_aux, 2),
                               np.power(wafy_aux, 2))) < Q60
            # mask array
            wafx_mask = ma.array(wafx_aux, mask=M)
            wafy_mask = ma.array(wafy_aux, mask=M)
            Q99 = np.nanpercentile(np.sqrt(np.add(np.power(wafx_aux, 2),
                                                  np.power(wafy_aux, 2))), 99)
            M = np.sqrt(np.add(np.power(wafx_aux, 2),
                               np.power(wafy_aux, 2))) > Q99
            # mask array
            wafx_mask = ma.array(wafx_mask, mask=M)
            wafy_mask = ma.array(wafy_mask, mask=M)

            # plot vectors
            lons, lats = np.meshgrid(data_waf.lon.values, data_waf.lat.values)
            Q = ax.quiver(lons[::waf_step, ::waf_step],
                          lats[::waf_step, ::waf_step],
                          wafx_mask[::waf_step, ::waf_step],
                          wafy_mask[::waf_step, ::waf_step],
                          transform=crs_latlon, pivot='tail',
                          width=1.7e-3, headwidth=4, alpha=1, headlength=2.5,
                          color='k', scale=waf_scale, angles='xy',
                          scale_units='xy')

            ax.quiverkey(Q, 0.85, 0.05, waf_label,
                         f'{waf_label:.1e} $m^2$ $s^{{-2}}$',
                         labelpos='E', coordinates='figure', labelsep=0.05,
                         fontproperties=FontProperties(size=6, weight='light'))

        # SIG ---------------------------------------------------------------- #
        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']
                try:
                    comp_sig_var = aux_sig_points['var']
                except:
                    comp_sig_var = aux_sig_points.values
                cs = ax.contourf(aux_sig_points.lon[::step],
                                 aux_sig_points.lat[::step],
                                 comp_sig_var[::step, ::step],
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower')

                for i2, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i2 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)

        # no plotear --------------------------------------------------------- #
        if no_plot is False:
            if num_cases:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]}), "
                                       f"$N={num_cases_data[plot]}$",
                        transform=ax.transAxes, size=6)
            else:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes, size=6)
            i = i + 1

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2,
                          resolution='110m')
            if map.upper() == 'SA':
                ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                               linestyle='-', linewidth=0.2, color='k')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                              zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            ax.set_xticks(np.arange(0, 360, step_lon), crs=crs_latlon)
            ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            ax.tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=4)
            ax.set_extent(extent, crs=crs_latlon)

            ax.set_aspect('equal')
            ax.set_title(titles[plot], fontsize=6, pad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    #cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.235, 0.03, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=5, pad=1)
        fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        aux_color = cmap.colors[2]
        patch = mpatches.Patch(color=aux_color, label='Ks < 0')

        legend = fig.legend(handles=[patch], loc='lower center', fontsize=8,
                            frameon=True, framealpha=1, fancybox=True)
        legend.set_bbox_to_anchor((0.5, 0.01), transform=fig.transFigure)
        legend.get_frame().set_linewidth(0.5)
        fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{namefig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()


def PlotFinalTwoVariables(data, num_cols,
                          levels_r1, cmap_r1,
                          levels_r2, cmap_r2,
                          data_ctn, levels_ctn_r1, levels_ctn_r2, color_ctn,
                          titles, namefig, save, dpi, out_dir, pdf=False,
                          high=2, width = 7.08661, step=1,
                          ocean_mask=False, num_cases=False,
                          num_cases_data=None,
                          sig_points=None, hatches=None,
                          data_ctn_no_ocean_mask=False):

    plots = data.plots.values
    num_plots = len(plots)
    num_rows = np.ceil(num_plots / num_cols).astype(int)

    crs_latlon = ccrs.PlateCarree()

    map == 'SA'
    extent = [275, 330, -60, 20]
    step_lon = 20
    high = high

    change_row = len(data.plots)/2

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    i = 0
    for ax, plot in zip(axes.flatten(), plots):
        no_plot = False

        if plot < change_row:
            levels = levels_r1
            levels_ctn = levels_ctn_r1
            cmap = cmap_r1
            row1=True
        else:
            levels = levels_r2
            levels_ctn = levels_ctn_r2
            cmap = cmap_r2
            row1 = False


        # CONTOUR ------------------------------------------------------------ #
        if data_ctn is not None:
            if levels_ctn is None:
                levels_ctn = levels.copy()
            try:
                if isinstance(levels_ctn, np.ndarray):
                    levels_ctn = levels_ctn[levels_ctn != 0]
                else:
                    levels_ctn.remove(0)
            except:
                pass
            aux_ctn = data_ctn.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values

                ax.contour(data_ctn.lon.values[::step],
                           data_ctn.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.4,
                           levels=levels_ctn, transform=crs_latlon,
                           colors=color_ctn)

        aux = data.sel(plots=plot)
        if aux.mean().values != 0:

            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux = aux * mask_ocean.mask

            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            if row1 is True:
                im_r1= ax.contourf(aux.lon.values[::step],
                                   aux.lat.values[::step],
                                   aux_var[::step, ::step],
                                   levels=levels,
                                   transform=crs_latlon, cmap=cmap,
                                   extend='both')
            else:
                im_r2 = ax.contourf(aux.lon.values[::step],
                                    aux.lat.values[::step],
                                    aux_var[::step, ::step],
                                    levels=levels,
                                    transform=crs_latlon, cmap=cmap,
                                    extend='both')
        else:
            ax.axis('off')
            no_plot=True


        # sig ---------------------------------------------------------------- #
        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']
                try:
                    comp_sig_var = aux_sig_points['var']
                except:
                    comp_sig_var = aux_sig_points.values
                cs = ax.contourf(aux_sig_points.lon[::step],
                                 aux_sig_points.lat[::step],
                                 comp_sig_var[::step, ::step],
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower')

                for i2, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i2 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)


        # no plotear --------------------------------------------------------- #
        if no_plot is False:
            if num_cases:
                ax.text(-0.01, 1.055, f"({string.ascii_lowercase[i]}) "
                                       f"$N={num_cases_data[plot]}$",
                        transform=ax.transAxes, size=6)
            else:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes, size=6)
            i = i + 1

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2,
                          resolution='110m')

            ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                               linestyle='-', linewidth=0.2, color='k')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                              zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            ax.set_xticks(np.arange(0, 360, step_lon), crs=crs_latlon)
            ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            ax.tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=4)
            ax.set_extent(extent, crs=crs_latlon)

            ax.set_aspect('equal')
            ax.set_title(titles[plot], fontsize=6, pad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    pos1 = fig.add_axes([0.87, 0.55, 0.015, 0.35])
    cb1 = fig.colorbar(im_r1, cax=pos1, orientation='vertical')
    cb1.ax.tick_params(labelsize=5, pad=1)

    pos2 = fig.add_axes([0.87, 0.1, 0.015, 0.35])
    cb2 = fig.colorbar(im_r2, cax=pos2, orientation='vertical')
    cb2.ax.tick_params(labelsize=5, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0.05, hspace=0.25, left=0.05,
                        right=0.85, top=0.95)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{namefig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()


    plt.show()


def DarkenColor(color, dc=0.2, as_hex=True):
    amount = 1 - dc
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    r, g, b = mcolors.to_rgb(c)
    new_rgb = (r * amount, g * amount, b * amount)
    return mcolors.to_hex(new_rgb) if as_hex else new_rgb