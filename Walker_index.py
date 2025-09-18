"""
Indice Walker, cual?
"""
# ---------------------------------------------------------------------------- #
compare_n34 = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'

# ---------------------------------------------------------------------------- #
dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

# ---------------------------------------------------------------------------- #
import xarray as xr

# ---------------------------------------------------------------------------- #
def WalkerIndex(slp):

    if min(slp.lon.values)<0:
        from Funciones import ChangeLons
        slp = ChangeLons(slp)

    slp_eq = slp.sel(lat=slice(5, -5))
    slp_west = slp_eq.sel(lon=slice(80, 160)).mean(dim=["lat", "lon"])
    slp_east = slp_eq.sel(lon=slice(200, 280)).mean(dim=["lat", "lon"])

    walker_index = slp_east - slp_west

    walker_index = walker_index.rename(
        {list(walker_index.data_vars)[0]: 'walker_index'})

    return walker_index

# ---------------------------------------------------------------------------- #
data = xr.open_dataset(f'{dir}SLP_SON_mer_d_w.nc')

wi = WalkerIndex(data)

# ---------------------------------------------------------------------------- #
if compare_n34:
    name_fig = 'wi_n34'
    from Funciones import Nino34CPC
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    n34_or = Nino34CPC(xr.open_dataset(
        "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc"),
        start=1920, end=2020)[0]

    year_start = 1940
    year_end = 2020

    n34 = n34_or.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-01'))
    n34_son = n34.sel(time=n34.time.dt.month.isin(10))

    # ------------------------------------------------------------------------ #
    t = wi['time'].values
    fix, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, -n34_son/n34_son.std('time'),
             color='red', label='-ONI')
    ax.plot(t, wi.walker_index/wi.std('time').walker_index,
             color='blue', label='Walker Index')
    ax.axhline(0, color='black', linestyle='--')
    r = float(np.corrcoef(-n34_son.values, wi['walker_index'].values)[0, 1])
    ax.set_title(f'ONI y Walker Index normalizados - r = {r:.3f}')
    ax.legend()
    ax.grid(True, linestyle = '--', alpha = 0.6)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(f'{out_dir}{name_fig}.png')
    plt.show()

# ---------------------------------------------------------------------------- #
