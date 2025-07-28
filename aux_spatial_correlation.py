"""
Correlacion espacial entre EOF observado y del modelo cfsv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/'
save = False
all_eof = True

# ---------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
from eofs.xarray import Eof
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------- #
def Compute(data, modeL):

    weights = np.sqrt(np.abs(np.cos(np.radians(data.lat))))
    try:
        name = list(data.data_vars)[0]
    except:
        name = None

    if modeL is True:
        aux = data*weights
        # r y time a a una misma variable para Eof
        aux_st = aux.rename({'time': 'time2'})
        aux_st = aux_st.stack(time=('r', 'time2'))
        aux_st = aux_st.transpose('time', 'lat', 'lon')

        # eof ------------------------------------#
        try:
            if name is None:
                solver = Eof(aux_st)
            else:
                solver = Eof(aux_st[name])

        except ValueError as ve:
            if str(ve) == 'all input data is missing':
                print('campos faltantes')
                aux_st = aux_st.where(~np.isnan(aux_st), drop=True)

                if name is None:
                    solver = Eof(aux_st)
                else:
                    solver = Eof(aux_st[name])

        eof_L_r = solver.eofsAsCovariance(neofs=2)
        pcs = solver.pcs(pcscaling=1)
        #var_per = np.around(solver.varianceFraction(neigs=2).values * 100, 1)

        #pcs = pcs.rename({'time2': 'time'})
        pc1_f_r = pcs[:,0]
        pc2_f_r = pcs[:,1]
        eof_f_r = eof_L_r

        pc1_f_r = pc1_f_r.unstack('time').rename( {'time2':'time'})
        pc2_f_r = pc2_f_r.unstack('time').rename( {'time2':'time'})

        pc1_f_em = None
        pc2_f_em = None
        eof_f_em = None

    else:

        for l in [0, 1, 2, 3]:
            print('L:' + str(l))
            # Todos las runs ------------------------------------------------- #

            aux = data.sel(time=data['L'] == l) * weights  # .mean('r')

            # r y time a a una misma variable para Eof
            aux_st = aux.rename({'time': 'time2'})
            aux_st = aux_st.stack(time=('r', 'time2'))
            aux_st = aux_st.transpose('time', 'lat', 'lon')

            # eof ------------------------------------#
            try:
                if name is None:
                    solver = Eof(aux_st)
                else:
                    solver = Eof(xr.DataArray(aux_st[name]))

            except ValueError as ve:
                if str(ve) == 'all input data is missing':
                    print('Lead ' + str(l) + ' con campos faltantes')
                    aux_st = aux_st.where(~np.isnan(aux_st), drop=True)

                    if name is None:
                        solver = Eof(aux_st)
                    else:
                        solver = Eof(aux_st[name])

            eof_l_r = solver.eofsAsCovariance(neofs=2)
            pcs = solver.pcs(pcscaling=1).unstack()

            pc1_l_r = pcs.sel(mode=0)
            pc2_l_r = pcs.sel(mode=1)

            print('Done EOF r')
            del aux_st
            del solver

            # EM ---
            aux = aux.mean('r')

            # eof ------------------------------------#
            try:
                if name is None:
                    solver = Eof(xr.DataArray(aux))
                else:
                    solver = Eof(xr.DataArray(aux[name]))

            except ValueError as ve:
                if str(ve) == 'all input data is missing':
                    print('Lead ' + str(l) + ' con campos faltantes')
                    aux = aux.where(~np.isnan(aux), drop=True)
                    if name is None:
                        solver = Eof(xr.DataArray(aux))
                    else:
                        solver = Eof(xr.DataArray(aux[name]))

            eof_l_em = solver.eofsAsCovariance(neofs=2)
            pcs = solver.pcs(pcscaling=1).unstack()

            pc1_l_em = pcs.sel(mode=0)
            pc2_l_em = pcs.sel(mode=1)

            # para guardar...
            if l == 0:
                pc1_f_r = pc1_l_r#.drop('L')
                pc2_f_r = pc2_l_r#.drop('L')
                eof_f_r = eof_l_r

                pc1_f_em = pc1_l_em#.drop('L')
                pc2_f_em = pc2_l_em#.drop('L')
                eof_f_em = eof_l_em

            else:
                # pc1_f_r = xr.concat([pc1_f_r, pc1_l_r.drop('L')], dim='time2')
                # pc2_f_r = xr.concat([pc2_f_r, pc2_l_r.drop('L')], dim='time2')
                # eof_f_r = xr.concat([eof_f_r, eof_l_r], dim='L')
                #
                # pc1_f_em = xr.concat([pc1_f_em, pc1_l_em.drop('L')], dim='time')
                # pc2_f_em = xr.concat([pc2_f_em, pc2_l_em.drop('L')], dim='time')
                # eof_f_em = xr.concat([eof_f_em, eof_l_em], dim='L')

                pc1_f_r = xr.concat([pc1_f_r, pc1_l_r], dim='time2')
                pc2_f_r = xr.concat([pc2_f_r, pc2_l_r], dim='time2')
                eof_f_r = xr.concat([eof_f_r, eof_l_r], dim='L')

                pc1_f_em = xr.concat([pc1_f_em, pc1_l_em], dim='time')
                pc2_f_em = xr.concat([pc2_f_em, pc2_l_em], dim='time')
                eof_f_em = xr.concat([eof_f_em, eof_l_em], dim='L')


            print('Done concat')

    try:
        pc1_f_r = pc1_f_r.rename({'time2':'time'})
        pc2_f_r = pc2_f_r.rename({'time2':'time'})
    except:
        pass

    return pc1_f_r, pc2_f_r, eof_f_r, pc1_f_em, pc2_f_em, eof_f_em

def PlotOne(field, levels = np.arange(-1,1.1,0.1), dpi=100, sa=False,
            extend=None, title=''):

    cbar = [
        # deep → pale blue              |  WHITE  |  pale → deep red
        '#014A9B', '#155AA8', '#276BB4', '#397AC1', '#4E8CCE',
        '#649FDA', '#7BB2E7', '#97C7F3', '#B7DDFF',
        '#FFFFFF',  # −0.1 ↔ 0.0
        '#FFFFFF',  # 0.0 ↔ 0.1
        '#FFD1CF', '#F7A8A5', '#EF827E', '#E5655D',
        '#D85447', '#CB4635', '#BE3D23', '#AE2E11', '#9B1C00'
    ]
    cbar = colors.ListedColormap(cbar, name="blue_white_red_20")
    cbar.set_over('#641B00')
    cbar.set_under('#012A52')
    cbar.set_bad(color='white')

    if sa is True:
        if extend is None:
            extend = [275, 330, -60, 20]
        fig_size = (4, 6)
        cbar = 'BrBG'
    else:
        fig_size = (8, 4)
        if extend is None:
            extend = [0, 359, -40, 40]

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent(extend, crs=crs_latlon)

    try:
        field_to_plot = field['var']
    except:
        field_to_plot = field

    im = ax.contourf(field.lon, field.lat, field_to_plot,
                 levels=levels, transform=crs_latlon, cmap=cbar,
                     extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(title)
    plt.tight_layout()

    plt.show()

# ---------------------------------------------------------------------------- #

sst = xr.open_dataset('/pikachu/datos/luciano.andrian/cases_fields/sst_son.nc')

# Takahashi et al. 2011 ------------------------------------------------------ #
sst_sel = sst.sel(lon=slice(110,290), lat=slice(-10,10))
_, _, eof_cfsv2, _, _, _ = Compute(sst_sel, True)

# ---------------------------------------------------------------------------- #
from test_indices import eof_tk
# ---------------------------------------------------------------------------- #
PlotOne(eof_tk[0])
PlotOne(eof_cfsv2.sel(mode=0))


a = eof_tk.sel(mode=0).sortby(['lat', 'lon'])
b = eof_cfsv2.sel(mode=0).sortby(['lat', 'lon'])
mask = np.isfinite(a) & np.isfinite(b)

a_vals = a.where(mask).values.flatten()
b_vals = b.where(mask).values.flatten()

