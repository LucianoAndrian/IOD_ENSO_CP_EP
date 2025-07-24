"""
Calculo de los indices EP y CP en CFSv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
our_dir_plots = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/index_regre/'
save_plots = True
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

if save_plots:
    dpi = 300
else:
    dpi = 100

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

def CheckSign(eof, pc1, pc2):

    pc1_f = []
    pc2_f = []

    if 'L' in eof.dims:

        for l in eof.L.values:
            sign_pc1 = np.sign(eof.sel(mode=0, L=l, lon=slice(210, 250)).mean()).values
            sign_pc2 = np.sign(eof.sel(mode=1, L=l, lon=slice(150, 180)).mean()).values

            if sign_pc1 < 0:
                pc1_L = pc1.sel(time=pc1.time.dt.month.isin(10 - l)) * sign_pc1
            else:
                pc1_L = pc1.sel(time=pc1.time.dt.month.isin(10 - l))

            if sign_pc2 < 0:
                pc2_L = pc2.sel(time=pc2.time.dt.month.isin(10 - l)) * sign_pc2
            else:
                pc2_L = pc2.sel(time=pc2.time.dt.month.isin(10 - l))

            pc1_f.append(pc1_L)
            pc2_f.append(pc2_L)

        pc1_f = xr.concat(pc1_f, 'time')
        pc2_f = xr.concat(pc2_f, 'time')

    else:
        sign_pc1 = np.sign(eof.sel(mode=0, lon=slice(210, 250)).mean())
        sign_pc2 = np.sign(eof.sel(mode=1, lon=slice(150, 180)).mean())

        if sign_pc1 < 0:
            pc1 = pc1 * sign_pc1
        if sign_pc2 < 0:
            pc2 = pc2 * sign_pc2
        pc1_f = pc1
        pc2_f = pc2

    return pc1_f, pc2_f

def CheckEvent(sst, cp, ep, year, lead, r):
    month = 10-lead
    if month < 10:
        month = '0'+str(month)

    cp_val = np.round(cp.sel(r=r, time=f'{year}-{month}-01').values, 3)
    ep_val = np.round(ep.sel(r=r, time=f'{year}-{month}-01').values, 3)

    plt.imshow(sst.sel(r=r, time=f'{year}-{month}-01')['sst'][0, :, :])
    plt.title(f'CP:{cp_val}, EP:{ep_val}')
    plt.show()

def PlotOne(field, levels = np.arange(-1,1.1,0.1), dpi=dpi, sa=False,
            extend=None, title='', name_fig='', out_dir=our_dir_plots,
            save=False):

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
    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', dpi=dpi)
        plt.close()
    else:
        plt.show()

def RegreField(field, index, return_coef=False):
    """
    Devuelve la parte del campo `field` explicada linealmente por `index`.
    """
    field = field.rename({'time':'time2'})
    field = field.stack(time=('time2', 'r'))
    try:
        name_var = list(field.data_vars)[0]
        da = field[name_var]
    except:
        da = field
        pass

    da = da.where(~np.isnan(da), 0)

    try:
        name_var_index = list(index.data_vars)[0]
        index = index[name_var_index]
    except:
        pass
    index = index.rename({'time':'time2'})
    index = index.stack(time=('time2', 'r'))
    index = index.where(~np.isnan(index), 0)

    # 2 usar el indice en "time" para usar esa dimencion para la regresion
    da_idx = da.copy()
    da_idx = da_idx.assign_coords(time=index)

    # 3 Regresión
    coef = da_idx.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients
    beta      = coef.sel(degree=1)   # pendiente
    intercept = coef.sel(degree=0)   # término independiente

    # 4 Reconstruir la parte explicada y restaurar las fechas reales
    fitted = beta * da_idx['time'] + intercept
    fitted = fitted.assign_coords(time=da['time'])

    if return_coef is True:
        result = beta
    else:
        result = fitted

    return result
# ---------------------------------------------------------------------------- #

sst = xr.open_dataset('/pikachu/datos/luciano.andrian/cases_fields/sst_son.nc')

# Takahashi et al. 2011 ------------------------------------------------------ #
sst_sel = sst.sel(lon=slice(110,290), lat=slice(-10,10))
sst_pac = sst.sel(lon=slice(110,290), lat=slice(-60,20))

if all_eof:
    pc1_f, pc2_f, eof_f, _, _, _ = Compute(sst_sel, True)
    PlotOne(eof_f.sel(mode=0))
    pc1_ch, pc2_ch = CheckSign(eof_f, pc1_f, pc2_f)
else:
    pc1_f_r, pc2_f_r, eof_f_r, _, _, _ = Compute(sst_sel, False)
    PlotOne(eof_f_r.sel(mode=0, L=1))
    # En algunos leads el oef puede tener signo cambiado
    # Acomodo los signos de cada pc en funcion del EOF
    pc1_ch, pc2_ch = CheckSign(eof_f_r, pc1_f_r, pc2_f_r)

cp = (pc1_ch + pc2_ch)/np.sqrt(2)
cp = cp.to_dataset(name='sst')
ep = (pc1_ch - pc2_ch)/np.sqrt(2)
ep = ep.to_dataset(name='sst')

pc1_reg_tk = RegreField(sst, pc1_ch, return_coef=True)
pc2_reg_tk = RegreField(sst, pc2_ch, return_coef=True)
cp_reg_tk = RegreField(sst, cp, return_coef=True)
ep_reg_tk = RegreField(sst, ep, return_coef=True)

levels = [-0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9]
PlotOne(pc1_reg_tk, levels=levels,
        title='Regression Coef. PC1 - Takahashi et al. 2011',
        name_fig='pc1_reg_tk', save=save_plots, out_dir=our_dir_plots)
PlotOne(pc2_reg_tk, levels=levels,
        title='Regression Coef. PC2 - Takahashi et al. 2011',
        name_fig='pc2_reg_tk', save=save_plots, out_dir=our_dir_plots)
PlotOne(cp_reg_tk, levels=levels,
        title='Regression Coef. CP ENSO - Takahashi et al. 2011',
        name_fig='cp_reg_tk', save=save_plots, out_dir=our_dir_plots)
PlotOne(ep_reg_tk, levels=levels,
        title='Regression Coef. EP ENSO - Takahashi et al. 2011',
        name_fig='ep_reg_tk', save=save_plots, out_dir=our_dir_plots)

# Tedeschi et al. 2014 ------------------------------------------------------- #
cp_td = sst.sel(lon=slice(160, 210), lat=slice(-5,5)).mean(['lon', 'lat'])
ep_td = sst.sel(lon=slice(220, 270), lat=slice(-5,5)).mean(['lon', 'lat'])

cp_reg_td = RegreField(sst, cp_td, return_coef=True)
ep_reg_td = RegreField(sst, ep_td, return_coef=True)
PlotOne(cp_reg_td, levels=levels,
        title='Regression Coef. CP ENSO - Tedeschi et al. 2014',
        name_fig='cp_reg_td', save=save_plots, out_dir=our_dir_plots)
PlotOne(ep_reg_td, levels=levels,
        title='Regression Coef. EP ENSO - Tedeschi et al. 2014',
        name_fig='ep_reg_td', save=save_plots, out_dir=our_dir_plots)

# Sulivan et al. 2016 -------------------------------------------------------- #
n3 = sst.sel(lon=slice(210, 270), lat=slice(-5,5)).mean(['lon', 'lat'])
n4 = sst.sel(lon=slice(200, 210), lat=slice(-5,5)).mean(['lon', 'lat'])

n3 = (n3 - n3.mean('time'))/n3.std('time')
n4 = (n4 - n4.mean('time'))/n4.std('time')

ep_n = n3 - 0.5*n4
cp_n = n4 - 0.5*n3

cp_reg_n = RegreField(sst, cp_n, return_coef=True)
ep_reg_n = RegreField(sst, ep_n, return_coef=True)
PlotOne(cp_reg_n, levels=levels,
        title='Regression Coef. CP ENSO - Sulivan et al. 2016',
        name_fig='cp_reg_n', save=save_plots, out_dir=our_dir_plots)
PlotOne(ep_reg_n, levels=levels,
        title='Regression Coef. EP ENSO - Sulivan et al. 2016',
        name_fig='ep_reg_n', save=save_plots, out_dir=our_dir_plots)

# ---------------------------------------------------------------------------- #
if save:
    print('Saving...')
    cp.to_netcdf(f'{out_dir}CP_SON_Leads_r_CFSv2.nc')
    ep.to_netcdf(f'{out_dir}EP_SON_Leads_r_CFSv2.nc')

    cp_td.to_netcdf(f'{out_dir}/aux_ep_cp_t/CP_Td_SON_Leads_r_CFSv2.nc')
    ep_td.to_netcdf(f'{out_dir}/aux_ep_cp_t/EP_Td_SON_Leads_r_CFSv2.nc')

    cp_n.to_netcdf(f'{out_dir}/aux_ep_cp_n/CP_n_SON_Leads_r_CFSv2.nc')
    ep_n.to_netcdf(f'{out_dir}/aux_ep_cp_n/EP_n_SON_Leads_r_CFSv2.nc')

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')