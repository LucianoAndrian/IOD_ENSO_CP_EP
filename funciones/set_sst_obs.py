import xarray as xr
from funciones.general_utils import xrFieldTimeDetrend
year_start = 1959 # eof
year_end = 2020

sst_or = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
sst_or = sst_or.drop_dims('nbnds')
sst_or = sst_or.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-01'))
sst_or = sst_or.rename({'sst':'var'})

sst = xrFieldTimeDetrend(sst_or, 'time') # Detrend
sst = sst.rolling(time=3, center=True).mean() #estacional
sst = ((sst.groupby('time.month') - sst.groupby('time.month').mean('time'))
       /sst.groupby('time.month').std('time')) # standarizacion

