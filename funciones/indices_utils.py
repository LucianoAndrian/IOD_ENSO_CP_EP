"""
Funciones de o por calcular indices observados y del modelo CFSv2
"""
# ---------------------------------------------------------------------------- #
from itertools import groupby
import xarray as xr
import numpy as np
import pandas as pd
from pandas.io.common import file_exists

pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import statsmodels.formula.api as smf
import cartopy.feature

import matplotlib as mpl
import matplotlib.path as mpath
from matplotlib.font_manager import FontProperties
import scipy.stats as st
import string
from numpy import ma
from matplotlib import colors
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from scipy.signal import convolve2d
import os
import matplotlib.patches as mpatches
from scipy.integrate import trapz
from matplotlib.colors import BoundaryNorm
from scipy.stats import spearmanr
import matplotlib.colors as mcolors

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob

# ---------------------------------------------------------------------------- #
def MovingBasePeriodAnomaly(data, start=1920, end=2020):
    # first five years
    start_num = start
    start = str(start)

    initial = data.sel(time=slice(start + '-01-01', str(start_num + 5) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 14) + '-01-01', str(start_num + 5 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')


    start_num = start_num + 6
    result = initial

    while (start_num != end-4) & (start_num < end-4):

        aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 15) + '-01-01', str(start_num + 4 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')

        start_num = start_num + 5

        result = xr.concat([result, aux], dim='time')

    if start_num > end - 4:
        start_num = start_num - 5

    aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
          data.sel(time=slice(str(end-29) + '-01-01', str(end) + '-12-31')).groupby('time.month').mean('time')

    result = xr.concat([result, aux], dim='time')

    return (result)

def Nino34CPC(data, start=1920, end=2020):

    # Calculates the Niño3.4 index using the CPC criteria.
    # Use ERSSTv5 to obtain exactly the same values as those reported.

    start_year = str(start-14)
    end_year = str(end)
    sst = data
    # N34
    ninio34 = sst.sel(lat=slice(4.0, -4.0), lon=slice(190, 240), time=slice(start_year+'-01-01', end_year + '-12-31'))
    ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)

    # compute monthly anomalies
    ninio34 = MovingBasePeriodAnomaly(data=ninio34, start=start, end=end)

    # compute 5-month running mean
    ninio34_filtered = np.convolve(ninio34, np.ones((3,)) / 3, mode='same')  #
    ninio34_f = xr.DataArray(ninio34_filtered, coords=[ninio34.time.values], dims=['time'])

    aux = abs(np.round(ninio34_f, 1)) >= 0.5
    results = []
    for k, g in groupby(enumerate(aux.values), key=lambda x: x[1]):
        if k:
            g = list(g)
            results.append([g[0][0], len(g)])

    n34 = []
    n34_df = pd.DataFrame(columns=['N34', 'Años', 'Mes'], dtype=float)
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 5 consecutive seasons
        if len_true >= 5:
            a = results[m][0]
            n34.append([np.arange(a, a + results[m][1]), ninio34_f[np.arange(a, a + results[m][1])].values])

            for l in range(0, len_true):
                if l < (len_true):
                    main_month_num = results[m][0] + l
                    if main_month_num != 1210:
                        n34_df = n34_df.append({'N34': np.around(ninio34_f[main_month_num].values, 1),
                                            'Años': np.around(ninio34_f[main_month_num]['time.year'].values),
                                            'Mes': np.around(ninio34_f[main_month_num]['time.month'].values)},
                                           ignore_index=True)

    return ninio34_f, n34, n34_df

