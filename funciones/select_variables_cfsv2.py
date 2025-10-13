
import xarray as xr

def SelectVariables(dates, data):

    t_count=0
    t_count_aux = 0
    for t in dates.index:
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