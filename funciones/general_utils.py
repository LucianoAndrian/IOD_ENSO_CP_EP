"""
Funciones generales
"""
# ---------------------------------------------------------------------------- #
import xarray as xr

# ---------------------------------------------------------------------------- #
def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt

# ---------------------------------------------------------------------------- #
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
