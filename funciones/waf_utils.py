import numpy as np
from scipy.ndimage import gaussian_filter

def c_diff(arr, h, dim, cyclic=False):
    # compute derivate of array variable respect to h associated to dim
    # adapted from kuchaale script
    ndim = arr.ndim
    lst = [i for i in range(ndim)]

    lst[dim], lst[0] = lst[0], lst[dim]
    rank = lst
    arr = np.transpose(arr, tuple(rank))

    if ndim == 3:
        shp = (arr.shape[0] - 2, 1, 1)
    elif ndim == 4:
        shp = (arr.shape[0] - 2, 1, 1, 1)

    d_arr = np.copy(arr)
    if not cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[0, ...]) / (h[1] - h[0])
        d_arr[-1, ...] = (arr[-1, ...] - arr[-2, ...]) / (h[-1] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    elif cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[-1, ...]) / (h[1] - h[-1])
        d_arr[-1, ...] = (arr[0, ...] - arr[-2, ...]) / (h[0] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    d_arr = np.transpose(d_arr, tuple(rank))

    return d_arr

def WAF(psiclm, psiaa, lon, lat,reshape=True, variable='var', hpalevel=200):
    #agregar xr=True

    if reshape:
        if len(psiclm['time']) > 1:
            psiclm = psiclm.mean('time')
        psiclm=psiclm[variable].values.reshape(1,len(psiclm.lat),len(psiclm.lon))


        variable_psiaa = list(psiaa.data_vars)[0]
        psiaa = psiaa[variable_psiaa].values.reshape(
            1, len(psiaa.lat), len(psiaa.lon))

    lon=lon.values
    lat=lat.values

    [xxx, nlats, nlons] = psiaa.shape  # get dimensions
    a = 6400000
    coslat = np.cos(lat * np.pi / 180)

    # climatological wind at psi level
    dpsiclmdlon = c_diff(psiclm, lon, 2)
    dpsiclmdlat = c_diff(psiclm, lat, 1)

    uclm = -1 * dpsiclmdlat
    vclm = dpsiclmdlon
    magU = np.sqrt(np.add(np.power(uclm, 2), np.power(vclm, 2)))

    dpsidlon = c_diff(psiaa, lon, 2)
    ddpsidlonlon = c_diff(dpsidlon, lon, 2)
    dpsidlat = c_diff(psiaa, lat, 1)
    ddpsidlatlat = c_diff(dpsidlat, lat, 1)
    ddpsidlatlon = c_diff(dpsidlat, lon, 2)

    termxu = dpsidlon * dpsidlon - psiaa * ddpsidlonlon
    termxv = dpsidlon * dpsidlat - ddpsidlatlon * psiaa
    termyv = dpsidlat * dpsidlat - psiaa * ddpsidlatlat

    # 0.2101 is the scale of p
    if hpalevel==200:
        coef = 0.2101
    elif hpalevel==750:
        coef = 0.74

    coeff1 = np.transpose(np.tile(coslat, (nlons, 1))) * (coef) / (2 * magU)
    # x-component
    px = coeff1 / (a * a * np.transpose(np.tile(coslat, (nlons, 1)))) * (
            uclm * termxu / np.transpose(np.tile(coslat, (nlons, 1))) + (vclm * termxv))
    # y-component
    py = coeff1 / (a * a) * (uclm / np.transpose(np.tile(coslat, (nlons, 1))) * termxv + (vclm * termyv))

    return px, py


def WAF_from_geopot(hgtclim, hgta, lon, lat, hpalevel=200):

    try:
        lon = lon.values
        lat = lat.values
    except:
        pass

    # QG:
    a = 6400000 #rt
    coslat = np.cos(lat * np.pi / 180)
    coslat = coslat.reshape(1, len(lat), 1)

    sinlat = np.sin(lat * np.pi / 180)
    sinlat = sinlat.reshape(1, len(lat), 1)

    # Coriolis parameter
    #f = 2 * 7.292e-5  * sinlat
    f0 = 2 * 7.292e-5 * np.sin(-43 * np.pi / 180)

    # Reshape - necesario para c_diff
    variable_clim = list(hgtclim.data_vars)[0]
    hgtclim=hgtclim[variable_clim].values.reshape(
        1,len(hgtclim.lat), len(hgtclim.lon))
    variable_hgta = list(hgta.data_vars)[0]
    hgta = hgta[variable_hgta].values.reshape(
        1, len(hgta.lat), len(hgta.lon))

    # Climatology:
    u_clim = (-1/(a*f0)) * c_diff(hgtclim, lat, 1)
    v_clim = (1/(a*f0*coslat)) * c_diff(hgtclim, lon, 2)
    wind_mag = np.sqrt(u_clim**2 + v_clim**2)

    # Primer termino:
    coef1 = hpalevel*1000*coslat/(2*wind_mag)

    # Viento anomalo:
    u_anom = (-1/(a*f0)) * c_diff(hgta, lat, 1)
    v_anom = (1/(a*f0*coslat)) * c_diff(hgta, lon, 2)

    u_anom = gaussian_filter(u_anom, sigma=1)
    v_anom = gaussian_filter(v_anom, sigma=1)

    # 2das derivadas:
    duanom_dlat = c_diff(u_anom, lat, 1)
    dvanom_dlon = c_diff(v_anom, lon, 2)
    dvanom_dlat = c_diff(v_anom, lat, 1)

    # wx
    wx_aux = u_clim * (v_anom**2 - (hgta/(a*f0*coslat)) * dvanom_dlon) + \
             v_clim * (-u_anom*v_anom + (hgta/(a*f0)) *duanom_dlat)

    wx = coef1 * wx_aux

    # wy
    wy_aux = -u_clim * (u_anom*v_anom + (hgta/(a*f0)) * dvanom_dlat) + \
             v_clim * (u_anom**2 + (hgta/(a*f0)) * duanom_dlat)

    wy = coef1 * wy_aux

    return wx, wy

