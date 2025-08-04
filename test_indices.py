"""
Test ENSO EP y CP
Takahayi et al. 2011
Tedeschi et alo. 2014
Sulivan et al. 2016
"""
# ---------------------------------------------------------------------------- #
plots = False
save = False
save_plots = False
our_dir =  '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/index_regre/'
# ---------------------------------------------------------------------------- #
from eofs.xarray import Eof
import xarray as xr
from Funciones import Nino34CPC
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

if save:
    dpi = 300
else:
    dpi = 100

# ---------------------------------------------------------------------------- #
def plotindex(index, indexwo, title='title', label='label'):
    plt.plot(index, color='k',
             label='original')
    plt.plot(indexwo, color='r',
             label=label)
    plt.grid()
    plt.ylim((-3,3))
    plt.legend()
    plt.title(title)
    plt.show()

def PlotOne(field, levels = np.arange(-1,1.1,0.1), dpi=dpi, sa=False,
            extend=None, title='', name_fig='', out_dir=our_dir,
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

def PlotTimeSeries(serie1, serie2, serie3, events_select=None,
                   label1='Serie 1', label2='Serie 2', label3='Serie 3',
                   shift_year=True, title=''):

    fig, ax = plt.subplots(figsize=(8, 3))

    if shift_year:
        time1 = pd.to_datetime(serie1.time.values) - pd.DateOffset(years=1)
        time2 = pd.to_datetime(serie2.time.values) - pd.DateOffset(years=1)
        time3 = pd.to_datetime(serie3.time.values) - pd.DateOffset(years=1)
    else:
        time1 = serie1.time.values
        time2 = serie2.time.values
        time3 = serie3.time.values

    ax.hlines(0, time1[0] , time1[-1], colors='k', linewidth=0.8)
    ax.plot(time1, serie1.values, color='black', label=label1)
    ax.plot(time2, serie2.values, color='dodgerblue', label=label2)
    ax.plot(time3, serie3.values, color='red', label=label3)

    if events_select is not None:
        if hasattr(events_select, 'values'):
            events_select = events_select.values

        events_set = set(pd.to_datetime(events_select))
        mask = pd.to_datetime(serie1.time.values).isin(events_set)

        ax.plot(time1[mask], serie1.values[mask], 'o', color='black')
        ax.plot(time2[mask], serie2.values[mask], 'o', color='dodgerblue')
        ax.plot(time3[mask], serie3.values[mask], 'o', color='red')

    ax.set_ylim((-3, 3))
    ax.set_xlabel('Año')
    ax.set_title(title)
    ax.legend(loc=2)
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt

def RegreField(field, index, return_coef=False):
    """
    Devuelve la parte del campo `field` explicada linealmente por `index`.
    """
    if isinstance(field, xr.Dataset):
        da = field['var']
    else:
        da = field

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

def find_top_events(index, num_of_events):
    pos = index.sortby(index, ascending=False).isel(time=slice(num_of_events))
    neg = index.sortby(index, ascending=True).isel(time=slice(num_of_events))

    return pos, neg
# ---------------------------------------------------------------------------- #
n34_or = Nino34CPC(xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc"),
    start=1920, end=2020)[0]

year_start = 1959
year_end = 2020

n34 = n34_or.sel(time=slice(f'{year_start}-08-01', f'{year_end}-04-01'))

n34_son = n34.sel(time=n34.time.dt.month.isin(10))

sst_or = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
sst_or = sst_or.drop_dims('nbnds')
sst_or = sst_or.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-01'))
sst_or = sst_or.rename({'sst':'var'})

sst = xrFieldTimeDetrend(sst_or, 'time') # Detrend
sst = sst.rolling(time=5, center=True).mean() #estacional
sst = ((sst.groupby('time.month') - sst.groupby('time.month').mean('time'))
       /sst.groupby('time.month').std('time')) # standarizacion
sst = sst.sel(time=sst.time.dt.month.isin(10)) # SON
sst = sst.sel(month=10)

pacequ = sst.sel(lon=slice(110,290), lat=slice(10,-10))['var']
pacequ = pacequ.transpose('time', 'lat', 'lon')

# Takahashi et al. 2011 - EOF ------------------------------------------------ #
solver = Eof(pacequ)
eof_tk = solver.eofsAsCovariance(neofs=2)
pcs = solver.pcs(pcscaling=1)
var_per = np.around(solver.varianceFraction(neigs=3).values * 100,1)

if plots:
    PlotOne(eof_tk[0])
    PlotOne(eof_tk[1])

pc1 = -pcs.sel(mode=0)
pc2 = -pcs.sel(mode=1)

if plots:
    print(f'var exp {var_per[0]}%')
    PlotTimeSeries(serie1=n34_son, serie2=pc1, serie3=pc2,
                   label1='ONI', label2='pc1', label3='pc2', title='')

cp_tk = (pc1 + pc2)/np.sqrt(2)
ep_tk = (pc1 - pc2)/np.sqrt(2)

if plots:
    pc1_reg_tk = RegreField(sst, pc1, return_coef=True)
    pc2_reg_tk = RegreField(sst, pc2, return_coef=True)
    cp_reg_tk = RegreField(sst, cp_tk, return_coef=True)
    ep_reg_tk = RegreField(sst, ep_tk, return_coef=True)

    levels = [-0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9]

    # PlotOne(pc1_reg_tk, levels=levels)
    # PlotOne(pc2_reg_tk, levels=levels)
    PlotOne(cp_reg_tk, levels=levels,
            title='Regression Coef. CP ENSO - Takahashi et al. 2011',
            name_fig='cp_tk_obs', out_dir=our_dir, save=save_plots)

    PlotOne(ep_reg_tk, levels=levels,
            title='Regression Coef. EP ENSO - Takahashi et al. 2011',
            name_fig='ep_tk_obs', out_dir=our_dir, save=save_plots)

    PlotTimeSeries(serie1=n34_son, serie2=cp_tk, serie3=ep_tk,
                   label1='ONI', label2='CP', label3='EP',
                   title='Takahashi et al. 2011')

# Tedeschi et al. 2014 ------------------------------------------------------- #
cp_td = sst.sel(lon=slice(160, 210), lat=slice(5,-5)).mean(['lon', 'lat'])['var']
ep_td = sst.sel(lon=slice(220, 270), lat=slice(5,-5)).mean(['lon', 'lat'])['var']

if plots:
    PlotTimeSeries(serie1=n34_son, serie2=cp_td, serie3=ep_td,
                   label1='ONI', label2='CP', label3='EP',
                   title='Tedeschi et al. 2014')

    cp_reg_td = RegreField(sst, cp_td, return_coef=True)
    ep_reg_td = RegreField(sst, ep_td, return_coef=True)
    PlotOne(cp_reg_td, levels=levels,
            title='Regression Coef. CP ENSO - Tedeschi et al. 2014',
            name_fig='cp_td_obs', out_dir=our_dir, save=save_plots)
    PlotOne(ep_reg_td, levels=levels,
            title='Regression Coef. EP ENSO - Tedeschi et al. 2014',
            name_fig='ep_td_obs', out_dir=our_dir, save=save_plots)

# Sulivan et al. 2016 -------------------------------------------------------- #
n3 = sst.sel(lon=slice(210, 270), lat=slice(5,-5)).mean(['lon', 'lat'])['var']
n4 = sst.sel(lon=slice(200, 210), lat=slice(5,-5)).mean(['lon', 'lat'])['var']
n3 = n3.sel(time=n3.time.dt.month.isin([10]))
n4 = n4.sel(time=n4.time.dt.month.isin([10]))

n3 = (n3 - n3.mean('time'))/n3.std('time')
n4 = (n4 - n4.mean('time'))/n4.std('time')

ep_n = n3 - 0.5*n4
cp_n = n4 - 0.5*n3

if plots:
    PlotTimeSeries(serie1=n34_son, serie2=cp_n, serie3=ep_n,
                   label1='ONI', label2='CP', label3='EP',
                   title='Sulivan et al. 2016')

    cp_reg_n = RegreField(sst, cp_n, return_coef=True)
    ep_reg_n = RegreField(sst, ep_n, return_coef=True)
    PlotOne(cp_reg_n, levels=levels,
            title='Regression Coef. CP ENSO - Sulivan et al. 2016',
            name_fig='cp_n_obs', out_dir=our_dir, save=save_plots)

    PlotOne(ep_reg_n, levels=levels,
            title='Regression Coef. EP ENSO - Sulivan et al. 2016',
            name_fig='ep_n_obs', out_dir=our_dir, save=save_plots)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #