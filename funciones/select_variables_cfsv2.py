
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

def subtract_months(date, original_month, new_month):

    dif_month = original_month - new_month

    if isinstance(date, xr.DataArray):
        if isinstance(date.item(), cftime.Datetime360Day):
            new_date = date - timedelta(days=np.int(dif_month*30))
        else:
            new_date = aux_subtract_months_no_360day(date, dif_month)
    else:
        print('date debe ser xr.DataArray')
        new_date = None

    return new_date


def SelectVariables(dates, data):
    data_select = []

    for t in dates.index:

        try:
            r_t = t.r.values
        except:
            r_t = dates.r[0].values
        L_t = t.L.values
        t_t = t.values

        try:
            t_t * 1
            t_t = t.time.values
        except:
            pass

        mask = ((data.L == L_t) &
                (data.r == r_t) &
                (data.time == t_t))

        selected = data.where(mask, drop=True)
        data_select.append(selected.isel(r=0))

    try:
        out_put = xr.concat(data_select, dim='time')
    except:
        out_put = None

    return out_put
