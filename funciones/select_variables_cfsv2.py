
import xarray as xr
import cftime
from datetime import timedelta
import numpy as np

def aux_subtract_months_no_360day(date, dif_month):

    y = date.time.dt.year.values
    m = date.time.dt.month.values
    d = date.time.dt.day.values

    # convertir a índice absoluto de meses
    total_m = y * 12 + (m - 1)
    # restar meses
    total_m -= dif_month

    # reconstruir año/mes
    new_y = total_m // 12
    new_m = total_m % 12 + 1

    return type(date.item())(new_y, new_m, d)

def subtract_months(date, new_month):

    if isinstance(date, xr.DataArray):
        dif_month = (date.time.dt.month.values - new_month)
        if isinstance(date.item(), cftime.Datetime360Day):
            new_date = date - timedelta(days=np.int(dif_month*30))

        else:
            new_date = aux_subtract_months_no_360day(date, dif_month)
    else:
        print('date debe ser xr.DataArray')
        new_date = None

    return new_date

def SelectVariables(dates, data, new_month=None):

    t_count=0
    t_count_aux = 0
    for t in dates.index:
        if new_month is not None:
            t = subtract_months(t, new_month)

        try:
            r_t = t.r.values
        except:
            r_t = dates.r[t_count_aux].values
        L_t = t.L.values
        t_t = t.values
        try: #q elegancia la de francia...
            t_t*1
            t_t = t.time.values
        except:
            pass

        if t_count == 0:
            aux = data.where(data.L == L_t).sel(r=r_t, time=t_t)
            t_count += 1
        else:
            aux = xr.concat([aux,
                             data.where(data.L == L_t).sel(r=r_t, time=t_t)],
                            dim='time')
    return aux