"""
Indice Walker, cual?
"""
# ---------------------------------------------------------------------------- #
dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

# ---------------------------------------------------------------------------- #
import xarray as xr

# ---------------------------------------------------------------------------- #
def WalkerIndex(slp):

    slp_eq = slp.sel(lat=slice(5, -5))
    slp_west = slp_eq.sel(lon=slice(80, 160)).mean(dim=["lat", "lon"])
    slp_east = slp_eq.sel(lon=slice(200 - 360, 280 - 360)).mean(dim=["lat", "lon"])

    walker_index = slp_east - slp_west

    walker_index = walker_index.rename(
        {list(walker_index.data_vars)[0]: 'walker_index'})

    return walker_index

# ---------------------------------------------------------------------------- #
data = xr.open_dataset(f'{dir}SLP_SON_mer_d_w.nc')

wi = WalkerIndex(data)

# ---------------------------------------------------------------------------- #