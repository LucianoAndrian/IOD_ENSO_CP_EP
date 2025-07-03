"""
Funciones generales para ENSO_IOD
"""
################################################################################
from itertools import groupby
import xarray as xr
import numpy as np
import pandas as pd
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

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob

out_dir = '~/'
################################################################################
# Niño3.4 & DMI ################################################################
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

def DMIndex(iodw, iode, sst_anom_sd=True, xsd=0.5, opposite_signs_criteria=True):

    limitsize = len(iodw) - 2

    # dipole mode index
    dmi = iodw - iode

    # criteria
    western_sign = np.sign(iodw)
    eastern_sign = np.sign(iode)
    opposite_signs = western_sign != eastern_sign



    sd = np.std(dmi) * xsd
    print(str(sd))
    sdw = np.std(iodw.values) * xsd
    sde = np.std(iode.values) * xsd

    valid_criteria = dmi.__abs__() > sd

    results = []
    if opposite_signs_criteria:
        for k, g in groupby(enumerate(opposite_signs.values), key=lambda x: x[1]):
            if k:
                g = list(g)
                results.append([g[0][0], len(g)])
    else:
        for k, g in groupby(enumerate(valid_criteria.values), key=lambda x: x[1]):
            if k:
                g = list(g)
                results.append([g[0][0], len(g)])


    iods = pd.DataFrame(columns=['DMI', 'Años', 'Mes'], dtype=float)
    dmi_raw = []
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 3 consecutive seasons
        if len_true >= 3:

            for l in range(0, len_true):

                if l < (len_true - 2):

                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != limitsize:
                        main_month_name = dmi[main_month_num]['time.month'].values  # "name" 1 2 3 4 5

                        main_season = dmi[main_month_num]
                        b_season = dmi[main_month_num - 1]
                        a_season = dmi[main_month_num + 1]

                        # abs(dmi) > sd....(0.5*sd)
                        aux = (abs(main_season.values) > sd) & \
                              (abs(b_season) > sd) & \
                              (abs(a_season) > sd)

                        if sst_anom_sd:
                            if aux:
                                sstw_main = iodw[main_month_num]
                                sstw_b = iodw[main_month_num - 1]
                                sstw_a = iodw[main_month_num + 1]
                                #
                                aux2 = (abs(sstw_main) > sdw) & \
                                       (abs(sstw_b) > sdw) & \
                                       (abs(sstw_a) > sdw)
                                #
                                sste_main = iode[main_month_num]
                                sste_b = iode[main_month_num - 1]
                                sste_a = iode[main_month_num + 1]

                                aux3 = (abs(sste_main) > sde) & \
                                       (abs(sste_b) > sde) & \
                                       (abs(sste_a) > sde)

                                if aux3 & aux2:
                                    iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                        'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                        'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                       ignore_index=True)

                                    a = results[m][0]
                                    dmi_raw.append([np.arange(a, a + results[m][1]),
                                                    dmi[np.arange(a, a + results[m][1])].values])


                        else:
                            if aux:
                                iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                    'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                    'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                   ignore_index=True)

    return iods, dmi_raw

def DMI(per = 0, filter_bwa = True, filter_harmonic = True,
        filter_all_harmonic=True, harmonics = [],
        start_per=1920, end_per=2020,
        sst_anom_sd=True, opposite_signs_criteria=True):

    western_io = slice(50, 70) # definicion tradicional

    start_per = str(start_per)
    end_per = str(end_per)

    if per == 2:
        movinganomaly = True
        start_year = '1906'
        end_year = '2020'
        change_baseline = False
        start_year2 = '1920'
        end_year2 = '2020_30r5'
        print('30r5')
    else:
        movinganomaly = False
        start_year = start_per
        end_year = end_per
        change_baseline = False
        start_year2 = '1920'
        end_year2 = end_per
        print('All')

    ##################################### DATA #####################################
    # ERSSTv5
    sst = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
    dataname = 'ERSST'
    ##################################### Pre-processing #####################################
    iodw = sst.sel(lat=slice(10.0, -10.0), lon=western_io,
                       time=slice(start_year + '-01-01', end_year + '-12-31'))
    iodw = iodw.sst.mean(['lon', 'lat'], skipna=True)
    iodw2 = iodw
    if per == 2:
        iodw2 = iodw2[168:]
    # -----------------------------------------------------------------------------------#
    iode = sst.sel(lat=slice(0, -10.0), lon=slice(90, 110),
                   time=slice(start_year + '-01-01', end_year + '-12-31'))
    iode = iode.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------#
    bwa = sst.sel(lat=slice(20.0, -20.0), lon=slice(40, 110),
                  time=slice(start_year + '-01-01', end_year + '-12-31'))
    bwa = bwa.sst.mean(['lon', 'lat'], skipna=True)
    # ----------------------------------------------------------------------------------#

    if movinganomaly:
        iodw = MovingBasePeriodAnomaly(iodw)
        iode = MovingBasePeriodAnomaly(iode)
        bwa = MovingBasePeriodAnomaly(bwa)
    else:
        # change baseline
        if change_baseline:
            iodw = iodw.groupby('time.month') - \
                   iodw.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            iode = iode.groupby('time.month') - \
                   iode.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            bwa = bwa.groupby('time.month') - \
                  bwa.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                      'time')
            print('baseline: ' + str(start_year2) + ' - ' + str(end_year2))
        else:
            print('baseline: All period')
            iodw = iodw.groupby('time.month') - iodw.groupby('time.month').mean('time', skipna=True)
            iode = iode.groupby('time.month') - iode.groupby('time.month').mean('time', skipna=True)
            bwa = bwa.groupby('time.month') - bwa.groupby('time.month').mean('time', skipna=True)

    # ----------------------------------------------------------------------------------#
    # Detrend
    iodw_trend = np.polyfit(range(0, len(iodw)), iodw, deg=1)
    iodw = iodw - (iodw_trend[0] * range(0, len(iodw)) + iodw_trend[1])
    # ----------------------------------------------------------------------------------#
    iode_trend = np.polyfit(range(0, len(iode)), iode, deg=1)
    iode = iode - (iode_trend[0] * range(0, len(iode)) + iode_trend[1])
    # ----------------------------------------------------------------------------------#
    bwa_trend = np.polyfit(range(0, len(bwa)), bwa, deg=1)
    bwa = bwa - (bwa_trend[0] * range(0, len(bwa)) + bwa_trend[1])
    # ----------------------------------------------------------------------------------#

    # 3-Month running mean
    iodw_filtered = np.convolve(iodw, np.ones((3,)) / 3, mode='same')
    iode_filtered = np.convolve(iode, np.ones((3,)) / 3, mode='same')
    bwa_filtered = np.convolve(bwa, np.ones((3,)) / 3, mode='same')

    # Common preprocessing, for DMIs other than SY2003a
    iode_3rm = iode_filtered
    iodw_3rm = iodw_filtered

    #################################### follow SY2003a #######################################

    # power spectrum
    # aux = FFT2(iodw_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)
    # aux2 = FFT2(iode_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)

    # filtering harmonic
    if filter_harmonic:
        if filter_all_harmonic:
            for harmonic in range(15):
                iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                iode_filtered = WaveFilter(iode_filtered, harmonic)
            else:
                for harmonic in harmonics:
                    iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                    iode_filtered = WaveFilter(iode_filtered, harmonic)

    ## max corr. lag +3 in IODW
    ## max corr. lag +6 in IODE

    # ----------------------------------------------------------------------------------#
    # ENSO influence
    # pre processing same as before
    if filter_bwa:
        ninio3 = sst.sel(lat=slice(5.0, -5.0), lon=slice(210, 270),
                         time=slice(start_year + '-01-01', end_year + '-12-31'))
        ninio3 = ninio3.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            ninio3 = MovingBasePeriodAnomaly(ninio3)
        else:
            if change_baseline:
                ninio3 = ninio3.groupby('time.month') - \
                         ninio3.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby(
                             'time.month').mean(
                             'time')

            else:

                ninio3 = ninio3.groupby('time.month') - ninio3.groupby('time.month').mean('time', skipna=True)

            trend = np.polyfit(range(0, len(ninio3)), ninio3, deg=1)
            ninio3 = ninio3 - (trend[0] * range(0, len(ninio3)) +trend[1])

        # 3-month running mean
        ninio3_filtered = np.convolve(ninio3, np.ones((3,)) / 3, mode='same')

        # ----------------------------------------------------------------------------------#
        # removing BWA effect
        # lag de maxima corr coincide para las dos bases de datos.
        lag = 3
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iodw_f = iodw_filtered - recta

        lag = 6
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iode_f = iode_filtered - recta
        print('BWA filtrado')
    else:
        iodw_f = iodw_filtered
        iode_f = iode_filtered
        print('BWA no filtrado')
    # ----------------------------------------------------------------------------------#

    # END processing
    if movinganomaly:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw.time.values], dims=['time'])
        start_year = '1920'
    else:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw2.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw2.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw2.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw2.time.values], dims=['time'])

    ####################################### compute DMI #######################################

    dmi_sy_full, dmi_raw = DMIndex(iodw_f, iode_f,
                                   sst_anom_sd=sst_anom_sd,
                                   opposite_signs_criteria=opposite_signs_criteria)

    return dmi_sy_full, dmi_raw, (iodw_f-iode_f)#, iodw_f - iode_f, iodw_f, iode_f

def DMI2(end_per=1920, start_per=2020, filter_harmonic=True, filter_bwa=False,
         sst_anom_sd=True, opposite_signs_criteria=True):

    # argumentos fijos ------------------------------------------------------------------------------------------------#
    movinganomaly = False
    change_baseline = False
    start_year2 = '6666'
    end_year2 = end_per
    #------------------------------------------------------------------------------------------------------------------#
    western_io = slice(50, 70)  # definicion tradicional
    start_per = str(start_per)
    end_per = str(end_per)

    start_year = start_per
    end_year = end_per
    ####################################################################################################################
    # DATA - ERSSTv5 --------------------------------------------------------------------------------------------------#
    sst = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")

    # Pre-processing --------------------------------------------------------------------------------------------------#
    iodw = sst.sel(lat=slice(10.0, -10.0), lon=western_io,
                       time=slice(start_year + '-01-01', end_year + '-12-31'))
    iodw = iodw.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------------------------------------#
    iode = sst.sel(lat=slice(0, -10.0), lon=slice(90, 110),
                   time=slice(start_year + '-01-01', end_year + '-12-31'))
    iode = iode.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------------------------------------#

    if movinganomaly:
        iodw = MovingBasePeriodAnomaly(iodw)
        iode = MovingBasePeriodAnomaly(iode)
    else:
        if change_baseline:
            iodw = iodw.groupby('time.month') - \
                   iodw.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            iode = iode.groupby('time.month') - \
                   iode.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            print('baseline: ' + str(start_year2) + ' - ' + str(end_year2))
        else:
            print('baseline: All period')
            iodw = iodw.groupby('time.month') - iodw.groupby('time.month').mean('time', skipna=True)
            iode = iode.groupby('time.month') - iode.groupby('time.month').mean('time', skipna=True)

    # Detrend ---------------------------------------------------------------------------------------------------------#
    iodw_trend = np.polyfit(range(0, len(iodw)), iodw, deg=1)
    iodw = iodw - (iodw_trend[0] * range(0, len(iodw)) + iodw_trend[1])
    #------------------------------------------------------------------------------------------------------------------#
    iode_trend = np.polyfit(range(0, len(iode)), iode, deg=1)
    iode = iode - (iode_trend[0] * range(0, len(iode)) + iode_trend[1])
    #------------------------------------------------------------------------------------------------------------------#
    # 3-Month running mean --------------------------------------------------------------------------------------------#
    iodw_filtered = np.convolve(iodw, np.ones((3,)) / 3, mode='same')
    iode_filtered = np.convolve(iode, np.ones((3,)) / 3, mode='same')

    # Filtering Harmonic ----------------------------------------------------------------------------------------------#
    if filter_harmonic:
        for harmonic in range(15):
            iodw_filtered = WaveFilter(iodw_filtered, harmonic)
            iode_filtered = WaveFilter(iode_filtered, harmonic)

    # Filter BWA #######################################################################################################
    if filter_bwa:
        bwa = sst.sel(lat=slice(20.0, -20.0), lon=slice(40, 110),
                      time=slice(start_year + '-01-01', end_year + '-12-31'))
        bwa = bwa.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            bwa = MovingBasePeriodAnomaly(bwa)
        else:
            bwa = bwa.groupby('time.month') - \
                  bwa.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')). \
                      groupby('time.month').mean('time')

        # Detrend -----------------------------------------̣̣------------------------------------------------------------#
        bwa_trend = np.polyfit(range(0, len(bwa)), bwa, deg=1)
        bwa = bwa - (bwa_trend[0] * range(0, len(bwa)) + bwa_trend[1])
        bwa_filtered = np.convolve(bwa, np.ones((3,)) / 3, mode='same')

        ninio3 = sst.sel(lat=slice(5.0, -5.0), lon=slice(210, 270),
                         time=slice(start_year + '-01-01', end_year + '-12-31'))
        ninio3 = ninio3.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            ninio3 = MovingBasePeriodAnomaly(ninio3)
        else:
            if change_baseline:
                ninio3 = ninio3.groupby('time.month') - \
                         ninio3.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31'))\
                             .groupby('time.month').mean('time')
            else:
                ninio3 = ninio3.groupby('time.month') - ninio3.groupby('time.month').mean('time', skipna=True)

            trend = np.polyfit(range(0, len(ninio3)), ninio3, deg=1)
            ninio3 = ninio3 - (trend[0] * range(0, len(ninio3)) + trend[1])

        # 3-month running mean
        ninio3_filtered = np.convolve(ninio3, np.ones((3,)) / 3, mode='same')

        # -------------------------------------------------------------------------------------------------------------#
        # removing BWA effect
        # lag de maxima corr coincide para las dos bases de datos.
        lag = 3
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iodw_f = iodw_filtered - recta

        lag = 6
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iode_f = iode_filtered - recta
        print('BWA filtrado')
    else:
        iodw_f = iodw_filtered
        iode_f = iode_filtered

    ####################################################################################################################
    # END processing --------------------------------------------------------------------------------------------------#
    iodw_f = xr.DataArray(iodw_f, coords=[iodw.time.values], dims=['time'])
    iode_f = xr.DataArray(iode_f, coords=[iodw.time.values], dims=['time'])

    # Compute DMI ######################################################################################################
    dmi_sy_full, dmi_raw = DMIndex(iodw_f, iode_f,
                                   sst_anom_sd=sst_anom_sd,
                                   opposite_signs_criteria=opposite_signs_criteria)
    return dmi_sy_full, dmi_raw, (iodw_f - iode_f)
    ####################################################################################################################

def PlotEnso_Iod(dmi, ninio, title, fig_name = 'fig_enso_iod', out_dir=out_dir,
                 save=False):

    fig, ax = plt.subplots()
    im = plt.scatter(x=dmi, y=ninio, marker='o', s=20, edgecolor='black', color='gray')

    plt.ylim((-4, 4));
    plt.xlim((-4, 4))
    plt.axhspan(-0.31, 0.31, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-0.5, 0.5, alpha=0.2, color='black', zorder=0)
    # ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('IOD')
    plt.ylabel('Niño 3.4')

    plt.text(-3.8, 3.4, '    EN/IOD-', dict(size=10))
    plt.text(-.1, 3.4, 'EN', dict(size=10))
    plt.text(+2.6, 3.4, ' EN/IOD+', dict(size=10))
    plt.text(+2.6, -.1, 'IOD+', dict(size=10))
    plt.text(+2.3, -3.4, '    LN/IOD+', dict(size=10))
    plt.text(-.1, -3.4, 'LN', dict(size=10))
    plt.text(-3.8, -3.4, ' LN/IOD-', dict(size=10))
    plt.text(-3.8, -.1, 'IOD-', dict(size=10))
    plt.title(title)
    if save:
        plt.savefig(out_dir + 'ENSO_IOD'+ fig_name + '.jpg')
    else:
        plt.show()

def SelectYears(df, name_var, main_month=1, full_season=False):

    if full_season:
        print('Full Season JJASON')
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([7, 8, 9, 10, 11]))[name_var],
                            'Años': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Años'],
                            'Mes': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Mes']})
        mmin, mmax = 6, 11

    else:
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([main_month]))[name_var],
                            'Años': df.where(df.Mes.isin([main_month]))['Años'],
                            'Mes': df.where(df.Mes.isin([main_month]))['Mes']})
        mmin, mmax = main_month - 1, main_month + 1

        if main_month == 1:
            mmin, mmax = 12, 2
        elif main_month == 12:
            mmin, mmax = 11, 1

    return aux.dropna(), mmin, mmax

def ClassifierEvents(df, full_season=False):
    if full_season:
        print('full season')
        df_pos = set(df.Años.values[np.where(df['Ind'] > 0)])
        df_neg = set(df.Años.values[np.where(df['Ind'] < 0)])
    else:
        df_pos = df.Años.values[np.where(df['Ind'] > 0)]
        df_neg = df.Años.values[np.where(df['Ind'] < 0)]

    return df_pos, df_neg

def NeutralEvents(df, mmin, mmax, start=1920, end = 2020, double=False,
                  df2=None, var_original=None):

    x = np.arange(start, end + 1, 1)

    start = str(start)
    end = str(end)

    mask = np.in1d(x, df.Años.values, invert=True)
    if mmax ==1: #NDJ
        print("NDJ Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]+1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=11, mmax=12))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(1))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    elif mmin == 12: #DJF
        print("DJF Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]-1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=1, mmax=2))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(12))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    else:
        mask = np.in1d(x, df.Años.values, invert=True)
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_years = list(set(neutro.time.dt.year.values))
        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=mmin, mmax=mmax))
        neutro = neutro.mean(['time'], skipna=True)

    return neutro, neutro_years

################################################################################
# Varias #######################################################################
def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)

def WaveFilter(serie, harmonic):

    sum = 0
    sam = 0
    N = np.size(serie)

    sum = 0
    sam = 0

    for j in range(N):
        sum = sum + serie[j] * np.sin(harmonic * 2 * np.pi * j / N)
        sam = sam + serie[j] * np.cos(harmonic * 2 * np.pi * j / N)

    A = 2*sum/N
    B = 2*sam/N

    xs = np.zeros(N)

    for j in range(N):
        xs[j] = A * np.sin(2 * np.pi * harmonic * j / N) + B * np.cos(2 * np.pi * harmonic * j / N)

    fil = serie - xs
    return(fil)

def Composite(original_data, index_pos, index_neg, mmin, mmax):
    comp_field_pos=0
    comp_field_neg=0

    if len(index_pos) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos+1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=11, mmax=12))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(1))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos - 1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=1, mmax=2))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(2))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        else:
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin([index_pos]))
            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=mmin, mmax=mmax))
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])


    if len(index_neg) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg + 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=11, mmax=12))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(1))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if (len(comp_field_neg.time) != 0):
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dmis(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg - 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=1, mmax=2))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(2))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

        else:
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin([index_neg]))
            comp_field_neg = comp_field_neg.sel(time=is_months(month=comp_field_neg['time.month'],
                                                               mmin=mmin, mmax=mmax))
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

    return comp_field_pos, comp_field_neg

def MultipleComposite(var, n34, dmi, season,start = 1920, full_season=False,
                      compute_composite=False):

    seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
               'JJA','JAS', 'ASO', 'SON', 'OND', 'NDJ']


    def check(x):
        if x is None:
            x = [0]
            return x
        else:
            if len(x) == 0:
                x = [0]
                return x
        return x

    if full_season:
        main_month_name = 'JJASON'
        main_month = None
    else:
        main_month, main_month_name = len(seasons[:season]) + 1, seasons[season]

    print(main_month_name)

    N34, N34_mmin, N34_mmax = SelectYears(df=n34, name_var='N34',
                                          main_month=main_month, full_season=full_season)
    DMI, DMI_mmin, DMI_mmax = SelectYears(df=dmi, name_var='DMI',
                                          main_month=main_month, full_season=full_season)
    DMI_sim_pos = [0,0]
    DMI_sim_neg = [0,0]
    DMI_un_pos = [0,0]
    DMI_un_neg = [0,0]
    DMI_pos = [0,0]
    DMI_neg = [0,0]
    N34_sim_pos = [0,0]
    N34_sim_neg = [0,0]
    N34_un_pos = [0,0]
    N34_un_neg = [0,0]
    N34_pos = [0,0]
    N34_neg = [0,0]
    All_neutral = [0, 0]

    if (len(DMI) != 0) & (len(N34) != 0):
        # All events
        DMI_pos, DMI_neg = ClassifierEvents(DMI, full_season=full_season)
        N34_pos, N34_neg = ClassifierEvents(N34, full_season=full_season)

        # both neutral, DMI and N34
        if compute_composite:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[0]

        else:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[1]


        # Simultaneous events
        sim_events = np.intersect1d(N34.Años.values, DMI.Años.values)

        try:
            # Simultaneos events
            DMI_sim = DMI.where(DMI.Años.isin(sim_events)).dropna()
            N34_sim = N34.where(N34.Años.isin(sim_events)).dropna()
            DMI_sim_pos_aux, DMI_sim_neg_aux = ClassifierEvents(DMI_sim)
            N34_sim_pos_aux, N34_sim_neg_aux = ClassifierEvents(N34_sim)


            # Existen eventos simultaneos de signo opuesto?
            # cuales?
            sim_pos = np.intersect1d(DMI_sim_pos_aux, N34_sim_pos_aux)
            sim_pos2 = np.intersect1d(sim_pos, DMI_sim_pos_aux)
            DMI_sim_pos = sim_pos2

            sim_neg = np.intersect1d(DMI_sim_neg_aux, N34_sim_neg_aux)
            sim_neg2 = np.intersect1d(DMI_sim_neg_aux, sim_neg)
            DMI_sim_neg = sim_neg2


            if (len(sim_events) != (len(sim_pos) + len(sim_neg))):
                dmi_pos_n34_neg = np.intersect1d(DMI_sim_pos_aux, N34_sim_neg_aux)
                dmi_neg_n34_pos = np.intersect1d(DMI_sim_neg_aux, N34_sim_pos_aux)
            else:
                dmi_pos_n34_neg = None
                dmi_neg_n34_pos = None

            # Unique events
            DMI_un = DMI.where(-DMI.Años.isin(sim_events)).dropna()
            N34_un = N34.where(-N34.Años.isin(sim_events)).dropna()

            DMI_un_pos, DMI_un_neg = ClassifierEvents(DMI_un)
            N34_un_pos, N34_un_neg = ClassifierEvents(N34_un)

            if compute_composite:
                print('Making composites...')
                # ------------------------------------ SIMULTANEUS ---------------------------------------------#
                DMI_sim = Composite(original_data=var, index_pos=DMI_sim_pos, index_neg=DMI_sim_neg,
                                    mmin=DMI_mmin, mmax=DMI_mmax)

                # ------------------------------------ UNIQUES -------------------------------------------------#
                DMI_un = Composite(original_data=var, index_pos=DMI_un_pos, index_neg=DMI_un_neg,
                                   mmin=DMI_mmin, mmax=DMI_mmax)

                N34_un = Composite(original_data=var, index_pos=N34_un_pos, index_neg=N34_un_neg,
                                   mmin=N34_mmin, mmax=N34_mmax)
            else:
                print('Only dates, no composites')
                DMI_sim = None
                DMI_un = None
                N34_un = None

        except:
            DMI_sim = None
            DMI_un = None
            N34_un = None
            DMI_sim_pos = None
            DMI_sim_neg = None
            DMI_un_pos = None
            DMI_un_neg = None
            print('Only uniques events[3][4]')

        if compute_composite:
            # ------------------------------------ ALL ---------------------------------------------#
            dmi_comp = Composite(original_data=var, index_pos=list(DMI_pos), index_neg=list(DMI_neg),
                                 mmin=DMI_mmin, mmax=DMI_mmax)
            N34_comp = Composite(original_data=var, index_pos=list(N34_pos), index_neg=list(N34_neg),
                                 mmin=N34_mmin, mmax=N34_mmax)
        else:
            dmi_comp=None
            N34_comp=None

    DMI_sim_pos = check(DMI_sim_pos)
    DMI_sim_neg = check(DMI_sim_neg)
    DMI_un_pos = check(DMI_un_pos)
    DMI_un_neg = check(DMI_un_neg)
    DMI_pos = check(DMI_pos)
    DMI_neg = check(DMI_neg)

    N34_sim_pos = check(N34_sim_pos)
    N34_sim_neg = check(N34_sim_neg)
    N34_un_pos = check(N34_un_pos)
    N34_un_neg = check(N34_un_neg)
    N34_pos = check(N34_pos)
    N34_neg = check(N34_neg)

    DMI_pos_N34_neg = check(dmi_pos_n34_neg)
    DMI_neg_N34_pos = check(dmi_neg_n34_pos)

    All_neutral = check(All_neutral)


    if compute_composite:
        print('test')
        return DMI_sim, DMI_un, N34_un, dmi_comp, N34_comp, All_neutral, DMI_sim_pos, DMI_sim_neg, \
               DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, DMI_neg, N34_pos, N34_neg
    else:
        return list(All_neutral),\
               list(set(DMI_sim_pos)), list(set(DMI_sim_neg)),\
               list(set(DMI_un_pos)), list(set(DMI_un_neg)),\
               list(set(N34_un_pos)), list(set(N34_un_neg)),\
               list(DMI_pos), list(DMI_neg), \
               list(N34_pos), list(N34_neg), \
               list(DMI_pos_N34_neg), list(DMI_neg_N34_pos)

def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt

def CompositeSimple(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(month=comp_field['time.month'], mmin=mmin, mmax=mmax))
        if len(comp_field.time) != 0:
            comp_field = comp_field.mean(['time'], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field
    else:
        print(' len index = 0')

def CaseComp(data, s, mmonth, c, two_variables=False, data2=None,
             return_neutro_comp=False, nc_date_dir='None'):
    """
    Las fechas se toman del periodo 1920-2020 basados en el DMI y N34 con ERSSTv5
    Cuando se toman los periodos 1920-1949 y 1950_2020 las fechas que no pertencen
    se excluyen de los composites en CompositeSimple()
    """
    mmin = mmonth[0]
    mmax = mmonth[-1]

    aux = xr.open_dataset(nc_date_dir + '1920_2020' + '_' + s + '.nc')
    neutro = aux.Neutral

    try:
        case = aux[c]
        case = case.where(case >= 1940)
        aux.close()

        case_num = len(case.values[np.where(~np.isnan(case.values))])
        case_num2 = case.values[np.where(~np.isnan(case.values))]

        neutro_comp = CompositeSimple(original_data=data, index=neutro, mmin=mmin, mmax=mmax)
        data_comp = CompositeSimple(original_data=data, index=case, mmin=mmin, mmax=mmax)

        comp = data_comp - neutro_comp

        if two_variables:
            neutro_comp2 = CompositeSimple(original_data=data2, index=neutro, mmin=mmin, mmax=mmax)
            data_comp2 = CompositeSimple(original_data=data2, index=case, mmin=mmin, mmax=mmax)

            comp2 = data_comp2 - neutro_comp2
        else:
            comp2 = None
    except:
        print('Error en ' + s + c)

    if two_variables:
        if return_neutro_comp:
            return comp, case_num, comp2, neutro_comp, neutro_comp2
        else:
            return comp, case_num, comp2
    else:
        if return_neutro_comp:
            return comp, case_num, neutro_comp
        else:
            return comp, case_num

def SelectCase(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(
            time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(
                month=comp_field['time.month'], mmin=mmin, mmax=mmax))

        return comp_field
    else:
        print('len index = 0')

def CaseSNR(data, s, mmonth, c, nc_date_dir='None'):
    mmin = mmonth[0]
    mmax = mmonth[-1]

    aux = xr.open_dataset(nc_date_dir + '1920_2020' + '_' + s + '.nc')
    neutro = aux.Neutral

    try:
        case = aux[c]
        case = case.where(case >= 1940)
        aux.close()

        case_num = len(case.values[np.where(~np.isnan(case.values))])

        neutro_comp = SelectCase(original_data=data, index=neutro,
                                     mmin=mmin, mmax=mmax)
        data_comp = SelectCase(original_data=data, index=case,
                                 mmin=mmin, mmax=mmax)

        comp = data_comp.mean(['time'], skipna=True) -\
               neutro_comp.mean(['time'], skipna=True)

        spread = data - comp
        spread = spread.std(['time'], skipna=True)

        snr = comp / spread

        return snr, case_num
    except:
        print('Error en ' + s + ' ' + c)

def ChangeLons(data, lon_name='lon'):
    data['_longitude_adjusted'] = xr.where(
        data[lon_name] < 0,
        data[lon_name] + 360,
        data[lon_name])

    data = (
        data
            .swap_dims({lon_name: '_longitude_adjusted'})
            .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
            .drop(lon_name))

    data = data.rename({'_longitude_adjusted': 'lon'})

    return data

def MakeMask(DataArray, dataname='mask'):
    import regionmask
    mask=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(DataArray)
    mask = xr.where(np.isnan(mask), mask, 1)
    mask = mask.to_dataset(name=dataname)
    return mask

def OpenDatasets(name, interp=False):
    pwd_datos = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_viejo/'
    def ChangeLons(data, lon_name='lon'):
        data['_longitude_adjusted'] = xr.where(
            data[lon_name] < 0,
            data[lon_name] + 360,
            data[lon_name])

        data = (
            data
                .swap_dims({lon_name: '_longitude_adjusted'})
                .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
                .drop(lon_name))

        data = data.rename({'_longitude_adjusted': 'lon'})

        return data


    def xrFieldTimeDetrend(xrda, dim, deg=1):
        # detrend along a single dimension
        aux = xrda.polyfit(dim=dim, deg=deg)
        try:
            trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
        except:
            trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

        dt = xrda - trend
        return dt

    aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
    pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))

    aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
    t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))

    aux = xr.open_dataset(pwd_datos + 't_cru.nc')
    t_cru = ChangeLons(aux)

    ### Precipitation ###
    if name == 'pp_20CR-V3':
        # NOAA20CR-V3
        aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
        pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        pp_20cr = pp_20cr.rename({'prate': 'var'})
        pp_20cr = pp_20cr.__mul__(86400 * (365 / 12))  # kg/m2/s -> mm/month
        pp_20cr = pp_20cr.drop('time_bnds')
        pp_20cr = xrFieldTimeDetrend(pp_20cr, 'time')

        return pp_20cr
    elif name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
        # interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_gpcc = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_gpcc = pp_gpcc.rename({'precip': 'var'})
        pp_gpcc = xrFieldTimeDetrend(pp_gpcc, 'time')

        return pp_gpcc
    elif name == 'pp_PREC':
        # PREC
        aux = xr.open_dataset(pwd_datos + 'pp_PREC.nc')
        pp_prec = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            pp_prec = pp_prec.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_prec = pp_prec.rename({'precip': 'var'})
        pp_prec = pp_prec.__mul__(365 / 12)  # mm/day -> mm/month
        pp_prec = xrFieldTimeDetrend(pp_prec, 'time')

        return pp_prec
    elif name == 'pp_chirps':
        # CHIRPS
        aux = xr.open_dataset(pwd_datos + 'pp_chirps.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.rename({'precip': 'var', 'latitude': 'lat'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            aux = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_ch = aux
        pp_ch = xrFieldTimeDetrend(pp_ch, 'time')

        return pp_ch
    elif name == 'pp_CMAP':
        # CMAP
        aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_cmap = aux.__mul__(365 / 12)  # mm/day -> mm/month
        pp_cmap = xrFieldTimeDetrend(pp_cmap, 'time')

        return pp_cmap
    elif name == 'pp_gpcp':
        # GPCP2.3
        aux = xr.open_dataset(pwd_datos + 'pp_gpcp.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            pp_gpcp = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        aux = aux.drop('lat_bnds')
        aux = aux.drop('lon_bnds')
        aux = aux.drop('time_bnds')
        pp_gpcp = aux.__mul__(365 / 12)  # mm/day -> mm/month
        pp_gpcp = xrFieldTimeDetrend(pp_gpcp, 'time')

        return pp_gpcp
    elif name == 't_20CR-V3':
        # 20CR-v3
        aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
        t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        t_20cr = t_20cr.rename({'air': 'var'})
        t_20cr = t_20cr - 273
        t_20cr = t_20cr.drop('time_bnds')
        t_20cr = xrFieldTimeDetrend(t_20cr, 'time')
        return t_20cr

    elif name == 't_cru':
        # CRU
        aux = xr.open_dataset(pwd_datos + 't_cru.nc')
        t_cru = ChangeLons(aux)
        t_cru = t_cru.sel(lon=slice(270, 330), lat=slice(-60, 20),
                          time=slice('1920-01-01', '2020-12-31'))
        # interpolado a 1x1
        if interp:
            t_cru = t_cru.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)
        t_cru = t_cru.rename({'tmp': 'var'})
        t_cru = t_cru.drop('stn')
        t_cru = xrFieldTimeDetrend(t_cru, 'time')
        return t_cru
    elif name == 't_BEIC': # que mierda pasaAAA!
        # Berkeley Earth etc
        aux = xr.open_dataset(pwd_datos + 't_BEIC.nc')
        aux = aux.rename({'longitude': 'lon', 'latitude': 'lat', 'temperature': 'var'})
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20), time=slice(1920, 2020.999))
        if interp:
            aux = aux.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)

        t_cru = t_cru.sel(time=slice('1920-01-01', '2020-12-31'))
        aux['time'] = t_cru.time.values
        aux['month_number'] = t_cru.time.values[-12:]
        t_beic_clim_months = aux.climatology
        t_beic = aux['var']
        # reconstruyendo?¿
        t_beic = t_beic.groupby('time.month') + t_beic_clim_months.groupby('month_number.month').mean()
        t_beic = t_beic.drop('month')
        t_beic = xr.Dataset(data_vars={'var': t_beic})
        t_beic = xrFieldTimeDetrend(t_beic, 'time')
        return t_beic

    elif name == 't_ghcn_cams':
        # GHCN

        aux = xr.open_dataset(pwd_datos + 't_ghcn_cams.nc')
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_ghcn = aux.rename({'air': 'var'})
        t_ghcn = t_ghcn - 273
        t_ghcn = xrFieldTimeDetrend(t_ghcn, 'time')
        return t_ghcn

    elif name == 't_hadcrut':
        # HadCRUT
        aux = xr.open_dataset(pwd_datos + 't_hadcrut_anom.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.sel(lon=slice(270, 330), latitude=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, latitude=t_20cr.lat.values)
        aux = aux.rename({'tas_mean': 'var', 'latitude': 'lat'})
        t_had = aux.sel(time=slice('1920-01-01', '2020-12-31'))

        aux = xr.open_dataset(pwd_datos + 't_hadcrut_mean.nc')
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_had_clim = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        aux = aux.rename({'tem': 'var'})
        aux['time'] = t_cru.time.values[-12:]
        # reconstruyendo?¿
        t_had = t_had.groupby('time.month') + aux.groupby('time.month').mean()
        t_had = t_had.drop('realization')
        t_had = t_had.drop('month')
        t_had = xrFieldTimeDetrend(t_had, 'time')

        return t_had

    elif name == 't_era20c':

        # ERA-20C
        aux = xr.open_dataset(pwd_datos + 't_era20c.nc')
        aux = aux.rename({'t2m': 'var', 'latitude': 'lat', 'longitude': 'lon'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_era20 = aux - 273
        t_era20 = xrFieldTimeDetrend(t_era20, 'time')

        return t_era20
    elif name == 'pp_lieb':
        aux = xr.open_dataset(pwd_datos + 'pp_liebmann.nc')
        aux = aux.sel(time=slice('1985-01-01', '2010-12-31'))
        aux = aux.resample(time='1M', skipna=True).mean()
        aux = ChangeLons(aux, 'lon')
        pp_lieb = aux.sel(lon=slice(275, 330), lat=slice(-50, 20))
        pp_lieb = pp_lieb.__mul__(365 / 12)
        pp_lieb = pp_lieb.drop('count')
        pp_lieb = pp_lieb.rename({'precip': 'var'})
        pp_lieb = xrFieldTimeDetrend(pp_lieb, 'time')
        return pp_lieb

def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

################################################################################
# WAF ##########################################################################
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
        psiclm=psiclm[variable].values.reshape(1,len(psiclm.lat),len(psiclm.lon))
        psiaa = psiaa[variable].values.reshape(1, len(psiaa.lat), len(psiaa.lon))

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

def PlotWAFCountours(comp, comp_var, title='Fig', name_fig='Fig',
                     save=False, dpi=200, levels=np.linspace(-1.5, 1.5, 13),
                     contour=False, cmap='RdBu_r', number_events='',
                     waf=False, px=None, py=None, text=True, waf_scale=None,
                     waf_units=None, two_variables = False, comp2=None, step=1,
                     step_waf=12, levels2=np.linspace(-1.5, 1.5, 13),
                     contour0=False, color_map='#4B4B4B', color_arrow='#400004'):

    fig = plt.figure(figsize=(9, 3.5), dpi=dpi)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([0, 359, -80, 20], crs=crs_latlon)

    im = ax.contourf(comp.lon[::step], comp.lat[::step], comp_var[::step,::step],
                     levels=levels,transform=crs_latlon, cmap=cmap, extend='both')
    if contour:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                            transform=crs_latlon, colors='k', linewidths=1)
        ax.clabel(values, inline=1, fontsize=5, fmt='%1.1f')

    if contour0:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=0,
                            transform=crs_latlon, colors='magenta', linewidths=1)
        ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

    if two_variables:
        print('Plot Two Variables')
        comp_var2 = comp2['var']
        levels_contour2 = levels2.copy()
        if isinstance(levels2, np.ndarray):
            levels_contour2 = levels2[levels2 != 0]
        else:
            levels_contour2.remove(0)
        values2 = ax.contour(comp2.lon, comp2.lat, comp_var2, levels=levels_contour2,
                            transform=crs_latlon, colors='k', linewidths=0.8, alpha=0.8)
        #ax.clabel(values2, inline=1, fontsize=5, fmt='%1.1f')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    #ax.add_feature(cartopy.feature.COASTLINE)
    ax.coastlines(color=color_map, linestyle='-', alpha=1, linewidth=0.6)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(0, 360, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-80, 20, 10), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    if waf:
        Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 0)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)
        # plot vectors
        lons, lats = np.meshgrid(comp.lon.values, comp.lat.values)
        ax.quiver(lons[::step_waf, ::step_waf],
                  lats[::step_waf, ::step_waf],
                  px_mask[0, ::step_waf, ::step_waf],
                  py_mask[0, ::step_waf, ::step_waf], transform=crs_latlon,pivot='tail',
                  width=0.0020,headwidth=4.1, alpha=1, color=color_arrow, scale=waf_scale, scale_units=waf_units)
                  #, scale=1/10)#, width=1.5e-3, headwidth=3.1,  # headwidht (default3)
                  #headlength=2.2)  # (default5))

    plt.title(title, fontsize=10)
    if text:
        plt.figtext(0.5, 0.01, 'Number of events: ' + str(number_events), ha="center", fontsize=10,
                bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    #plt.tight_layout()

    if save:
        plt.savefig(name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

################################################################################
# Regression ###################################################################
def LinearReg(xrda, dim, deg=1):
    # liner reg along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg, skipna=True)
    return aux

def LinearReg1_D(dmi, n34):

    df = pd.DataFrame({'dmi': dmi.values, 'n34': n34.values})

    result = smf.ols(formula='n34~dmi', data=df).fit()
    n34_pred_dmi = result.params[1] * dmi.values + result.params[0]

    result = smf.ols(formula='dmi~n34', data=df).fit()
    dmi_pred_n34 = result.params[1] * n34.values + result.params[0]

    return n34 - n34_pred_dmi, dmi - dmi_pred_n34

def RegWEffect(n34, dmi,data=None, data2=None, m=9,two_variables=False):
    var_reg_n34_2=0
    var_reg_dmi_2=1

    data['time'] = n34
     #print('Full Season')
    # try:
    #     aux = LinearReg(data.groupby('month')[m], 'time')
    # except:
    #     aux = LinearReg(data.groupby('time.month')[m], 'time')
    aux = LinearReg(data, 'time')
    # aux = xr.polyval(data.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
    #       aux.var_polyfit_coefficients[1]
    var_reg_n34 = aux.var_polyfit_coefficients[0]

    data['time'] = dmi
    # try:
    #     aux = LinearReg(data.groupby('month')[m], 'time')
    # except:
    #     aux = LinearReg(data.groupby('time.month')[m], 'time')
    aux = LinearReg(data, 'time')
    var_reg_dmi = aux.var_polyfit_coefficients[0]
    var_reg_dmi = aux.var_polyfit_coefficients[0]

    if two_variables:
        print('Two Variables')

        data2['time'] = n34
        #print('Full Season data2, m ignored')
        #aux = LinearReg(data2.groupby('month')[m], 'time')
        aux = LinearReg(data2, 'time')
        var_reg_n34_2 = aux.var_polyfit_coefficients[0]

        data2['time'] = dmi
        #aux = LinearReg(data2.groupby('month')[m], 'time')
        aux = LinearReg(data2, 'time')
        var_reg_dmi_2 = aux.var_polyfit_coefficients[0]

    return var_reg_n34, var_reg_dmi, var_reg_n34_2, var_reg_dmi_2

def RegWOEffect(n34, n34_wo_dmi, dmi, dmi_wo_n34, m=9, datos=None):

    datos['time'] = n34

    try:
        #aux = LinearReg(datos.groupby('month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) +\
        #       aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        #aux = LinearReg(datos.groupby('time.month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('time.month')[m].time, aux.var_polyfit_coefficients[0]) +\
        #       aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time, aux.var_polyfit_coefficients[0]) +\
              aux.var_polyfit_coefficients[1]
    #wo n34
    try:
        #var_regdmi_won34 = datos.groupby('month')[m]-aux
        var_regdmi_won34 = datos - aux

        #var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m] #index wo influence
        var_regdmi_won34['time'] = dmi_wo_n34
        var_dmi_won34 = LinearReg(var_regdmi_won34,'time')
    except:
        #var_regdmi_won34 = datos.groupby('time.month')[m] - aux
        var_regdmi_won34 = datos - aux

        #var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m]  # index wo influence
        var_regdmi_won34['time'] = dmi_wo_n34  # index wo influence
        var_dmi_won34 = LinearReg(var_regdmi_won34, 'time')

    #-----------------------------------------#

    datos['time'] = dmi
    try:
        #aux = LinearReg(datos.groupby('month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
        #   aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        aux = LinearReg(datos.groupby('time.month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('time.month')[m].time, aux.var_polyfit_coefficients[0]) + \
        #   aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    #wo
    try:
        # var_regn34_wodmi = datos.groupby('month')[m]-aux
        # var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
        var_regn34_wodmi = datos-aux
        var_regn34_wodmi['time'] = n34_wo_dmi #index wo influence
        var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    except:
        # var_regn34_wodmi = datos.groupby('time.month')[m]-aux
        # var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
        var_regn34_wodmi = datos - aux
        var_regn34_wodmi['time'] = n34_wo_dmi #index wo influence
        var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    return var_n34_wodmi.var_polyfit_coefficients[0],\
           var_dmi_won34.var_polyfit_coefficients[0],\
           var_regn34_wodmi,var_regdmi_won34

def Corr(datos, index, time_original, m=9):
    try:
        # aux_corr1 = xr.DataArray(datos.groupby('month')[m]['var'],
        #                      coords={'time': time_original.groupby('time.month')[m].values,
        #                              'lon': datos.lon.values, 'lat': datos.lat.values},
        #                      dims=['time', 'lat', 'lon'])

        aux_corr1 = xr.DataArray(datos['var'],
                             coords={'time': time_original.values,
                                     'lon': datos.lon.values, 'lat': datos.lat.values},
                             dims=['time', 'lat', 'lon'])
    except:
        # aux_corr1 = xr.DataArray(datos.groupby('time.month')[m]['var'],
        #                      coords={'time': time_original.groupby('time.month')[m].values,
        #                              'lon': datos.lon.values, 'lat': datos.lat.values},
        #                      dims=['time', 'lat', 'lon'])
        aux_corr1 = xr.DataArray(datos['var'],
                             coords={'time': time_original.values,
                                     'lon': datos.lon.values, 'lat': datos.lat.values},
                             dims=['time', 'lat', 'lon'])

    # aux_corr2 = xr.DataArray(index.groupby('time.month')[m],
    #                          coords={'time': time_original.groupby('time.month')[m]},
    #                          dims={'time'})
    aux_corr2 = xr.DataArray(index,
                             coords={'time': time_original},
                             dims={'time'})

    return xr.corr(aux_corr1, aux_corr2, 'time')

def PlotReg(data, data_cor, levels=np.linspace(-100,100,2), cmap='RdBu_r',
            dpi=100, save=False, title='\m/', name_fig='fig_PlotReg',
            sig=True, out_dir='', two_variables = False, data2=None,
            data_cor2=None, levels2 = np.linspace(-100,100,2), sig2=True,
            sig_point2=False, color_sig2='k', color_contour2='k', step=1,
            SA=False, color_map='#d9d9d9', color_sig='magenta',
            sig_point=False, r_crit=1, third_variable=False, data3=None,
            levels3=np.linspace(-1,1,11)):

    levels_contour = levels.copy()
    if isinstance(levels_contour, np.ndarray):
        levels_contour = levels_contour[levels_contour != 0]
    else:
        levels_contour.remove(0)

    crs_latlon = ccrs.PlateCarree()
    if SA:
        fig = plt.figure(figsize=(5, 6), dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([270,330, -60,20], crs=crs_latlon)
    else:
        fig = plt.figure(figsize=(9, 3.5), dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([0, 359, -80, 20], crs=crs_latlon)

    ax.contour(data.lon[::step], data.lat[::step], data[::step, ::step],
               linewidths=.5, alpha=0.5, levels=levels_contour,
               transform=crs_latlon, colors='black')

    im = ax.contourf(data.lon[::step], data.lat[::step], data[::step,::step],
                     levels=levels, transform=crs_latlon, cmap=cmap,
                     extend='both')
    if sig:
        if sig_point:
            colors_l = [color_sig, color_sig]
            cs = ax.contourf(data_cor.lon, data_cor.lat,
                             data_cor.where(np.abs(data_cor) > np.abs(r_crit)),
                             transform=crs_latlon, colors='none',
                             hatches=["...", "..."], extend='lower')

            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor(colors_l[i % len(colors_l)])

            for collection in cs.collections:
                collection.set_linewidth(0.)

        else:
            ax.contour(data_cor.lon[::step], data_cor.lat[::step],
                       data_cor[::step, ::step],
                       levels=np.linspace(-r_crit, r_crit, 2),
                       colors=color_sig, transform=crs_latlon, linewidths=1)


    if two_variables:
        ax.contour(data2.lon, data2.lat, data2, levels=levels2,
                   colors=color_contour2, transform=crs_latlon, linewidths=1)
        if sig2:
            if sig_point2:
                colors_l = [color_sig2, color_sig2]
                cs = ax.contourf(
                    data_cor2.lon, data_cor2.lat,
                    data_cor2.where(np.abs(data_cor2) > np.abs(r_crit)),
                    transform=crs_latlon, colors='none', hatches=["...", "..."],
                    extend='lower', alpha=0.5)

                for i, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)
                # para hgt200 queda mejor los dos juntos
                ax.contour(data_cor2.lon[::step], data_cor2.lat[::step],
                           data_cor2[::step, ::step],
                           levels=np.linspace(-r_crit, r_crit, 2),
                           colors=color_sig2, transform=crs_latlon,
                           linewidths=1)
            else:
                ax.contour(data_cor2.lon, data_cor2.lat, data_cor2,
                           levels=np.linspace(-r_crit, r_crit, 2),
                           colors=color_sig2, transform=crs_latlon,
                           linewidths=1)

                cbar = colors.ListedColormap([color_sig2, 'white', color_sig2])
                cbar.set_over(color_sig2)
                cbar.set_under(color_sig2)
                cbar.set_bad(color='white')
                ax.contourf(data_cor2.lon, data_cor2.lat, data_cor2,
                            levels=[-1,-r_crit, 0, r_crit,1], cmap=cbar,
                            transform=crs_latlon, linewidths=1, alpha=0.3)

    if third_variable:
        ax.contour(data3.lon[::2], data3.lat[::2], data3[::2,::2],
                   levels=levels3, colors=['#D300FF','#00FF5D'],
                   transform=crs_latlon, linewidths=1.5)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)

    if SA:
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, zorder=17)
        ocean = cartopy.feature.NaturalEarthFeature(
            'physical', 'ocean', scale='50m', facecolor='white', alpha=1)
        ax.add_feature(ocean, linewidth=0.2, zorder=15)
        ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
        ax.set_yticks(np.arange(-60, 20, 20), crs=crs_latlon)

        ax2 = ax.twinx()
        ax2.set_yticks([])
    else:

        ax.set_xticks(np.arange(0, 360, 30), crs=crs_latlon)
        ax.set_yticks(np.arange(-80, 20, 10), crs=crs_latlon)
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        ax.coastlines(color=color_map, linestyle='-', alpha=1)

    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', zorder=20)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        print('save: ' + out_dir + name_fig + '.jpg')
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()

    else:
        plt.show()

def ComputeWithEffect(data=None, data2=None, n34=None, dmi=None,
                     two_variables=False, full_season=False,
                     time_original=None,m=9):
    print('Reg...')
    print('#-- With influence --#')
    aux_n34, aux_dmi, aux_n34_2, aux_dmi_2 = \
        RegWEffect(data=data, data2=data2,
                   n34=n34.__mul__(1 / n34.std('time')),
                   dmi=dmi.__mul__(1 / dmi.std('time')),
                   m=m, two_variables=two_variables)

    if full_season:
        print('Full Season')
        n34 = n34.rolling(time=5, center=True).mean()
        dmi = dmi.rolling(time=5, center=True).mean()

    print('Corr...')
    aux_corr_n34 = Corr(datos=data, index=n34, time_original=time_original, m=m)
    aux_corr_dmi = Corr(datos=data, index=dmi, time_original=time_original, m=m)

    aux_corr_dmi_2 = 0
    aux_corr_n34_2 = 0
    if two_variables:
        print('Corr2..')
        aux_corr_n34_2 = Corr(datos=data2, index=n34,
                              time_original=time_original, m=m)
        aux_corr_dmi_2 = Corr(datos=data2, index=dmi,
                              time_original=time_original, m=m)

    return aux_n34, aux_corr_n34, aux_dmi, aux_corr_dmi, aux_n34_2, \
        aux_corr_n34_2, aux_dmi_2, aux_corr_dmi_2

def ComputeWithoutEffect(data, n34, dmi, m, time_original):
    # -- Without influence --#
    print('# -- Without influence --#')
    print('Reg...')
    # dmi wo n34 influence and n34 wo dmi influence
    dmi_wo_n34, n34_wo_dmi = LinearReg1_D(n34.__mul__(1 / n34.std('time')),
                                          dmi.__mul__(1 / dmi.std('time')))

    # Reg WO
    aux_n34_wodmi, aux_dmi_won34, data_n34_wodmi, data_dmi_won34 = \
        RegWOEffect(n34=n34.__mul__(1 / n34.std('time')),
                   n34_wo_dmi=n34_wo_dmi,
                   dmi=dmi.__mul__(1 / dmi.std('time')),
                   dmi_wo_n34=dmi_wo_n34,
                   m=m, datos=data)

    print('Corr...')
    aux_corr_n34 = Corr(datos=data_n34_wodmi, index=n34_wo_dmi,
                        time_original=time_original,m=m)
    aux_corr_dmi = Corr(datos=data_dmi_won34, index=dmi_wo_n34,
                        time_original=time_original,m=m)

    return aux_n34_wodmi, aux_corr_n34, aux_dmi_won34, aux_corr_dmi

################################################################################
# CFSv2 ########################################################################
def SelectNMMEFiles(model_name, variable, dir, anio='0', in_month='0',
                    by_r=False, r='0',  All=False):

    """
    Selecciona los archivos en funcion de del mes de entrada (in_month)
    o del miembro de ensamble (r)

    :param model_name: [str] nombre del modelo
    :param variable:[str] variable usada en el nombre del archivo
    :param dir:[str] directorio de los archivos a abrir
    :param anio:[str] anio de inicio del pronostico
    :param in_month:[str] mes de inicio del pronostico
    :param by_r: [bool] True para seleccionar todos los archivos de un mismo
     miembro de ensamble
    :param r: [str] solo si by_r = True, numero del miembro de ensamble que s
    e quiere abrir
    :return: lista con los nombres de los archivos seleccionados
    """

    if by_r==False:

        if ((isinstance(model_name, str) == False) | (isinstance(variable, str) == False) |
                (isinstance(dir, str) == False) | (isinstance(in_month, str) == False)
                | (isinstance(anio, str) == False)):
            print('ERROR: model_name, variable, dir and in_month must be a string')
            return

        if int(in_month) < 10:
            m_in = '0'
        else:
            m_in = ''

        if in_month == '1':
            y1 = 0
            m1 = -11
            m_en = ''
        elif int(in_month) > 10:
            y1 = 1
            m1 = 1
            m_en = ''
            print('Year in chagend')
            anio = str(int(anio) - 1)
        else:
            y1 = 1
            m1 = 1
            m_en = '0'

    if by_r:
        if (isinstance(r, str) == False):
            print('ERROR: r must be a string')
            return

        files = glob.glob(dir + variable + '_Amon_' + model_name + '_*'
                          '_r'+ r +'_*' + '-*.nc')

    elif All:
        print('All=True')
        files = glob.glob(dir + variable + '_Amon_' + model_name + '_*'
                          '_r*' +'_*' + '-*.nc')

    else:
        files = glob.glob(dir + variable + '_Amon_' + model_name + '_' +
                          anio + m_in + in_month +
                          '_r*_' + anio + m_in + in_month + '-' +
                          str(int(anio) + y1) + m_en +
                          str(int(in_month) - m1) + '.nc')

    return files

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

def SelectBins(data, min, max, sd=1):
    # sd opcional en caso de no estar escalado
    if np.abs(min) > np.abs(max):
        return (data >= min*sd) & (data < max*sd)
    elif np.abs(min) < np.abs(max):
        return (data > min*sd) & (data <= max*sd)
    elif np.abs(min) == np.abs(max):
        return (data >= min*sd) & (data <= max*sd)

def BinsByCases(v, v_name, fix_factor, s, mm, c, c_count,
                bin_limits, bins_by_cases_dmi, bins_by_cases_n34, dates_dir,
                cases_dir, snr=False, neutro_clim=False, box=False, box_lat=[],
                box_lon=[], ocean_mask=False):

    def Weights(data):
        weights = np.transpose(
            np.tile(np.cos(np.arange(
                data.lat.min().values, data.lat.max().values+1) * np.pi / 180),
                    (len(data.lon), 1)))
        try:
            data_w = data * weights
        except:
            data_w = data.transpose('time', 'lat', 'lon') * weights

        return data_w

    # 1. se abren los archivos de los índices (completos y se pesan por su SD)
    # tambien los archivos de la variable en cuestion pero para cada "case" = c

    data_dates_dmi_or = xr.open_dataset(dates_dir + 'DMI_' + s +
                                        '_Leads_r_CFSv2.nc')
    data_dates_dmi_or /=  data_dates_dmi_or.mean('r').std()

    data_dates_n34_or = xr.open_dataset(dates_dir + 'N34_' + s +
                                        '_Leads_r_CFSv2.nc')

    aux_n34_std =  data_dates_n34_or.mean('r').std()
    data_dates_n34_or /= aux_n34_std

    print('1.1 Climatología y case')
    if v != 'hgt':
        end_nc_file = '_detrend_05.nc'
        if v == 'hgt750': # que maravilla...
            end_nc_file = '__detrend_05.nc'
        #end_nc_file = '_no_detrend_05.nc'

    else:
        end_nc_file = '_05.nc'

    if neutro_clim:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_neutros' + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)
    else:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_' + s.lower() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)



    if ocean_mask is True:
        mask = MakeMask(clim, list(clim.data_vars)[0])
        clim = clim * mask

    try:
        case = Weights(
            xr.open_dataset(cases_dir + v + '_' + c + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)

        if ocean_mask is True:
            mask = MakeMask(case, list(case.data_vars)[0])
            case = case * mask

    except:
        print(f"case {c}, no encontrado para {v}")
        aux = clim.mean('time').__mul__(0)
        return aux, aux, aux

    try:
        clim = clim.sel(P=750)
        case = case.sel(P=750)
    except:
        pass

    if v == 'tref' or v == 'prec' or v == 'tsigma':
        lat = np.arange(-60, 20 + 1)
        lon = np.arange(275, 330 + 1)
    else:
        lat = np.linspace(-80, 20, 101)
        lon = np.linspace(0, 359, 360)

    if clim.lat[0]>clim.lat[-1]:
        lat_clim = lat[::-1]
    else:
        lat_clim = lat
    clim = clim.sel(lat=slice(lat_clim[0], lat_clim[-1]),
                    lon=slice(lon[0], lon[-1]))

    if case.lat[0] > case.lat[-1]:
        lat_case = lat[::-1]
    else:
        lat_case = lat
    case = case.sel(lat=slice(lat_case[0], lat_case[-1]),
                    lon=slice(lon[0], lon[-1]))

    print('Anomalía')
    for l in [0, 1, 2, 3]:
        try:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['r', 'time'])
        except:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['time'])

        if l==0:
            anom = case.sel(time=case.time.dt.month.isin(mm - l)) - clim_aux
        else:
            anom2 = case.sel(time=case.time.dt.month.isin(mm - l)) - clim_aux
            anom = xr.concat([anom, anom2], dim='time')

    print( '1.2')
    anom = anom.sortby(anom.time.dt.month)

    if box is True:
        anom = anom.sel(lat=slice(min(box_lat), max(box_lat)),
                        lon=slice(box_lon[0], box_lon[1]))
        anom = anom.mean(['lon', 'lat'], skipna=True)
        aux_set_ds = ['time']
    else:
        aux_set_ds = ['time', 'lat', 'lon']

    # 2. Vinculo fechas case -> índices DMI y N34 para poder clasificarlos
    # las fechas entre el case variable y el case indices COINCIDEN,
    # DE ESA FORMA SE ELIGIERON LOS CASES VARIABLE
    # pero diferen en orden. Para evitar complicar la selección usando r y L
    # con .sortby(..time.dt.month) en cada caso se simplifica el problema
    # y coinciden todos los eventos en fecha, r y L

    cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates/'

    aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]: 'index'})

    case_sel_dmi = SelectVariables(aux_cases, data_dates_dmi_or)
    case_sel_dmi = case_sel_dmi.sortby(case_sel_dmi.time.dt.month)
    case_sel_dmi_n34 = SelectVariables(aux_cases, data_dates_n34_or)
    case_sel_dmi_n34 = case_sel_dmi_n34.sortby(case_sel_dmi_n34.time.dt.month)

    print('2.1 uniendo var, dmi y n34')
    try:
        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    # No deberia suceder pero con tref hay fechas duplicadas en 2011 y los campos
    # de estas fechas no son iguales. son 4 datos en total.
    except:
        print('error de anios, revisando...')
        times_to_remove = []
        for t in anom.time.values:
            lista = anom.sel(time=t).r.values
            vistos = set()

            try:
                len(lista)
                for valor in lista:
                    if valor in vistos:
                        print(f'error en anio {t}, será removido')
                        times_to_remove.append(t)
                    else:
                        vistos.add(valor)
            except:
                pass

        print(times_to_remove)
        for t in np.unique(times_to_remove):
            anom = anom.sel(time=anom.time != t)
            case_sel_dmi = case_sel_dmi.sel(
                time=case_sel_dmi.time != t)
            case_sel_dmi_n34 = case_sel_dmi_n34.sel(
                time=case_sel_dmi_n34.time != t)

        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    bins_aux_dmi = bins_by_cases_dmi[c_count]
    bins_aux_n34 = bins_by_cases_n34[c_count]
    print("3. Seleccion en cada bin")
    anom_bin_main = list()
    num_bin_main = list()
    # loops en las bins para el dmi segun case
    for ba_dmi in range(0, len(bins_aux_dmi)):
        bins_aux = data_merged.where(
            SelectBins(data_merged.dmi,
                       bin_limits[bins_aux_dmi[ba_dmi]][0],
                       bin_limits[bins_aux_dmi[ba_dmi]][1]))

        anom_bin = list()
        num_bin = list()
        # loop en las correspondientes al n34 segun case
        for ba_n34 in range(0, len(bins_aux_n34)):
            bin_f = bins_aux.where(
                SelectBins(
                    bins_aux.n34,
                    bin_limits[bins_aux_n34[ba_n34]][0]/aux_n34_std.sst.values,
                    bin_limits[bins_aux_n34[ba_n34]][1]/aux_n34_std.sst.values))

            if snr:
                spread = bin_f - bin_f.mean(['time'])
                spread = spread.std('time')
                SNR = bin_f.mean(['time']) / spread
                anom_bin.append(SNR)
            else:
                anom_bin.append(bin_f.mean('time')['var'])

            num_bin.append(len(np.where(~np.isnan(bin_f['dmi']))[0]))

        anom_bin_main.append(anom_bin)
        num_bin_main.append(num_bin)

    return anom_bin_main, num_bin_main, clim

def BinsByCases_noComp(v, v_name, fix_factor, s, mm, c, c_count,
                       bin_limits, bins_by_cases_dmi, bins_by_cases_n34,
                       dates_dir, cases_dir, snr=False, neutro_clim=False,
                       box=False, box_lat=[], box_lon=[], ocean_mask=False):

    def Weights(data):
        weights = np.transpose(
            np.tile(np.cos(np.arange(
                data.lat.min().values,
                data.lat.max().values + 1) * np.pi / 180),
                    (len(data.lon), 1)))
        try:
            data_w = data * weights
        except:
            data_w = data.transpose('time', 'lat', 'lon') * weights

        return data_w

    # 1. se abren los archivos de los índices (completos y se pesan por su SD)
    # tambien los archivos de la variable en cuestion pero para cada "case" = c

    data_dates_dmi_or = xr.open_dataset(dates_dir + 'DMI_' + s +
                                        '_Leads_r_CFSv2.nc')
    data_dates_dmi_or /= data_dates_dmi_or.mean('r').std()

    data_dates_n34_or = xr.open_dataset(dates_dir + 'N34_' + s +
                                        '_Leads_r_CFSv2.nc')

    aux_n34_std = data_dates_n34_or.mean('r').std()
    data_dates_n34_or /= aux_n34_std

    print('1.1 Climatología y case')
    if v != 'hgt':
        end_nc_file = '_detrend_05.nc'
        if v == 'hgt750': # que maravilla...
            end_nc_file = '__detrend_05.nc'
        #end_nc_file = '_no_detrend_05.nc'

    else:
        end_nc_file = '_05.nc'

    if neutro_clim:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_neutros' + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)
    else:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_' + s.lower() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)

    if ocean_mask is True:
        mask = MakeMask(clim, list(clim.data_vars)[0])
        clim = clim * mask

    try:
        case = Weights(
            xr.open_dataset(cases_dir + v + '_' + c + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)

        if ocean_mask is True:
            mask = MakeMask(case, list(case.data_vars)[0])
            case = case * mask

    except:
        print(f"case {c}, no encontrado para {v}")
        aux = clim.mean('time').__mul__(0)
        return aux, aux, aux

    try:
        clim = clim.sel(P=750)
        case = case.sel(P=750)
    except:
        pass

    if v == 'tref' or v == 'prec' or v == 'tsigma':
        lat = np.arange(-60, 20 + 1)
        lon = np.arange(275, 330 + 1)
    else:
        lat = np.linspace(-80, 20, 101)
        lon = np.linspace(0, 359, 360)

    if clim.lat[0] > clim.lat[-1]:
        lat_clim = lat[::-1]
    else:
        lat_clim = lat
    clim = clim.sel(lat=slice(lat_clim[0], lat_clim[-1]),
                    lon=slice(lon[0], lon[-1]))

    if case.lat[0] > case.lat[-1]:
        lat_case = lat[::-1]
    else:
        lat_case = lat
    case = case.sel(lat=slice(lat_case[0], lat_case[-1]),
                    lon=slice(lon[0], lon[-1]))

    print('Anomalía')
    for l in [0, 1, 2, 3]:
        try:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['r', 'time'])
        except:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['time'])

        if l == 0:
            anom = case.sel(time=case.time.dt.month.isin(mm - l)) #- clim_aux
        else:
            anom2 = case.sel(time=case.time.dt.month.isin(mm - l))# - clim_aux
            anom = xr.concat([anom, anom2], dim='time')

    print('1.2')
    anom = anom.sortby(anom.time.dt.month)

    if box is True:
        anom = anom.sel(lat=slice(min(box_lat), max(box_lat)),
                        lon=slice(box_lon[0], box_lon[1]))
        anom = anom.mean(['lon', 'lat'], skipna=True)
        aux_set_ds = ['time']
    else:
        aux_set_ds = ['time', 'lat', 'lon']

    # 2. Vinculo fechas case -> índices DMI y N34 para poder clasificarlos
    # las fechas entre el case variable y el case indices COINCIDEN,
    # DE ESA FORMA SE ELIGIERON LOS CASES VARIABLE
    # pero diferen en orden. Para evitar complicar la selección usando r y L
    # con .sortby(..time.dt.month) en cada caso se simplifica el problema
    # y coinciden todos los eventos en fecha, r y L

    cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates/'

    aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]: 'index'})

    case_sel_dmi = SelectVariables(aux_cases, data_dates_dmi_or)
    case_sel_dmi = case_sel_dmi.sortby(case_sel_dmi.time.dt.month)
    case_sel_dmi_n34 = SelectVariables(aux_cases, data_dates_n34_or)
    case_sel_dmi_n34 = case_sel_dmi_n34.sortby(case_sel_dmi_n34.time.dt.month)

    print('2.1 uniendo var, dmi y n34')
    try:
        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    # No deberia suceder pero con tref hay fechas duplicadas en 2011 y los campos
    # de estas fechas no son iguales. son 4 datos en total.
    except:
        print('error de anios, revisando...')
        times_to_remove = []
        for t in anom.time.values:
            lista = anom.sel(time=t).r.values
            vistos = set()

            try:
                len(lista)
                for valor in lista:
                    if valor in vistos:
                        print(f'error en anio {t}, será removido')
                        times_to_remove.append(t)
                    else:
                        vistos.add(valor)
            except:
                pass

        print(times_to_remove)
        for t in np.unique(times_to_remove):
            anom = anom.sel(time=anom.time != t)
            case_sel_dmi = case_sel_dmi.sel(
                time=case_sel_dmi.time != t)
            case_sel_dmi_n34 = case_sel_dmi_n34.sel(
                time=case_sel_dmi_n34.time != t)

        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    bins_aux_dmi = bins_by_cases_dmi[c_count]
    bins_aux_n34 = bins_by_cases_n34[c_count]
    print("3. Seleccion en cada bin")
    anom_bin_main = list()
    num_bin_main = list()
    # loops en las bins para el dmi segun case
    for ba_dmi in range(0, len(bins_aux_dmi)):
        bins_aux = data_merged.where(
            SelectBins(data_merged.dmi,
                       bin_limits[bins_aux_dmi[ba_dmi]][0],
                       bin_limits[bins_aux_dmi[ba_dmi]][1]))

        anom_bin = list()
        num_bin = list()
        # loop en las correspondientes al n34 segun case
        for ba_n34 in range(0, len(bins_aux_n34)):
            bin_f = bins_aux.where(
                SelectBins(
                    bins_aux.n34,
                    bin_limits[bins_aux_n34[ba_n34]][
                        0] / aux_n34_std.sst.values,
                    bin_limits[bins_aux_n34[ba_n34]][
                        1] / aux_n34_std.sst.values))

            if snr:
                spread = bin_f - bin_f.mean(['time'])
                spread = spread.std('time')
                SNR = bin_f.mean(['time']) / spread
                anom_bin.append(SNR)
            else:
                anom_bin.append(bin_f['var'])

            num_bin.append(len(np.where(~np.isnan(bin_f['dmi']))[0]))

        anom_bin_main.append(anom_bin)
        num_bin_main.append(num_bin)

    return anom_bin_main, num_bin_main, clim


def DetrendClim(data, mm, v_name='prec'):
    # la diferencia es mínima en fitlrar o no tendencia para hacer una climatología,
    # pero para no perder la costumbre de complicar las cosas...

    for l in [0, 1, 2, 3]:
        season_data = data.sel(time=data.time.dt.month.isin(mm - l), L=l)
        aux = season_data.polyfit(dim='time', deg=1)
        if v_name == 'prec':
            aux_trend = xr.polyval(season_data['time'],
                                   aux.prec_polyfit_coefficients[0])  # al rededor de la media
        elif v_name == 'tref':
            aux_trend = xr.polyval(season_data['time'],
                                   aux.tref_polyfit_coefficients[0])  # al rededor de la media
        elif v_name == 'hgt':
            aux_trend = xr.polyval(season_data['time'],
                                   aux.hgt_polyfit_coefficients[0])  # al rededor de la media

        if l == 0:
            season_anom_detrend = season_data - aux_trend
        else:
            aux_detrend = season_data - aux_trend
            season_anom_detrend = xr.concat([season_anom_detrend,
                                             aux_detrend], dim='time')

    return season_anom_detrend.mean(['r', 'time'])

# Plots ################################################################################################################
def PlotComp(comp, comp_var, title='Fig', fase=None, name_fig='Fig',
             save=False, dpi=200, levels=np.linspace(-1.5, 1.5, 13),
             contour=False, cmap='RdBu_r', number_events='', season = '',
             waf=False, px=None, py=None, text=True, SA=False,
             two_variables = False, comp2=None, step = 1,
             levels2=np.linspace(-1.5, 1.5, 13), contour0 = False):

    if SA:
        fig = plt.figure(figsize=(5, 6), dpi=dpi)
    else:
        fig = plt.figure(figsize=(7, 3), dpi=dpi)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    if SA:
        ax.set_extent([270,330, -60,20],crs_latlon)
    else:
        ax.set_extent([0, 359, -90, 10], crs=crs_latlon)


    im = ax.contourf(comp.lon[::step], comp.lat[::step], comp_var[::step,::step],
                     levels=levels,transform=crs_latlon, cmap=cmap, extend='both')
    if contour:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                            transform=crs_latlon, colors='darkgray', linewidths=1)
        ax.clabel(values, inline=1, fontsize=5, fmt='%1.1f')

    if contour0:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=0,
                            transform=crs_latlon, colors='magenta', linewidths=1)
        ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

    if two_variables:
        print('Plot Two Variables')
        comp_var2 = comp2['var'] ######## CORREGIR en caso de generalizar #############
        values2 = ax.contour(comp2.lon, comp2.lat, comp_var2, levels=levels2,
                            transform=crs_latlon, colors='k', linewidths=0.5, alpha=0.6)
        #ax.clabel(values2, inline=1, fontsize=5, fmt='%1.1f')


    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    if SA:
        ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
        ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
    else:
        ax.set_xticks(np.arange(30, 330, 60), crs=crs_latlon)
        ax.set_yticks(np.arange(-90, 10, 10), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    if waf:
        Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 0)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)
        # plot vectors
        lons, lats = np.meshgrid(comp.lon.values, comp.lat.values)
        ax.quiver(lons[::17, ::17], lats[::17, ::17], px_mask[0, ::17, ::17],
                  py_mask[0, ::17, ::17], transform=crs_latlon,pivot='tail',
                  width=0.0014,headwidth=4.1, alpha=0.8, color='k')
                  #, scale=1/10)#, width=1.5e-3, headwidth=3.1,  # headwidht (default3)
                  #headlength=2.2)  # (default5))

    plt.title(title, fontsize=10)
    if text:
        plt.figtext(0.5, 0.01, number_events, ha="center", fontsize=10,
                bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    plt.tight_layout()

    if save:
        plt.savefig(name_fig + str(season) + '_' + str(fase.split(' ', 1)[1]) + '.jpg')
        plt.close()
    else:
        plt.show()

def Plots(data, variable='var', neutral=None, DMI_pos=None, DMI_neg=None,
          N34_pos=None, N34_neg=None, neutral_name='', cmap='RdBu_r',
          dpi=200, mode='', levels=np.linspace(-1.5, 1.5, 13),
          name_fig='', save=False, contour=False, title="", waf=False,
          two_variables=False, data2=None, neutral2=None, levels2=None,
          season=None, text=True, SA=False, contour0=False, step=1,
          px=None, py=None):

    if two_variables == False:
        if data is None:
            if data2 is None:
                print('data None!')
            else:
                data = data2
                print('Data is None!')
                print('Using data2 instead data')
                levels = levels2
                neutral = neutral2

    def Title(DMI_phase, N34_phase, title=title):
        DMI_phase = set(DMI_phase)
        N34_phase = set(N34_phase)
        if mode.split(' ', 1)[0] != 'Simultaneus':
            if mode.split(' ', 1)[1] == 'IODs':
                title = title + mode + ': ' + str(len(DMI_phase)) + '\n' + 'against ' + clim
                number_events = str(DMI_phase)
            else:
                title = title + mode + ': ' + str(len(N34_phase)) + '\n' + 'against ' + clim
                number_events = str(N34_phase)

        elif mode.split(' ', 1)[0] == 'Simultaneus':
            title = title +mode + '\n' + 'IODs: ' + str(len(DMI_phase)) + \
                    ' - ENSOs: ' + str(len(N34_phase)) + '\n' + 'against ' + clim
            number_events = str(N34_phase)
        return title, number_events




    if data[0] != 0:
        comp = data[0] - neutral
        clim = neutral_name
        try:
            comp2 = data2[0] - neutral2
        except:
            comp2 = None
            print('One Variable')

        PlotComp(comp=comp, comp_var=comp[variable],
                 title=Title(DMI_phase=DMI_pos, N34_phase=N34_pos)[0],
                 fase=' - Positive', name_fig=name_fig,
                 save=save, dpi=dpi, levels=levels,
                 contour=contour, cmap=cmap,
                 number_events=Title(DMI_phase=DMI_pos, N34_phase=N34_pos)[1],
                 season=season,
                 waf=waf, px=px, py=py, text=text, SA=SA,
                 two_variables=two_variables,
                 comp2=comp2, step=step,
                 levels2=levels2, contour0=contour0)

    if data[1] != 0:
        comp = data[1] - neutral
        clim = neutral_name
        try:
            comp2 = data2[1] - neutral2
        except:
            comp2 = None
            print('One Variable')

        PlotComp(comp=comp, comp_var=comp[variable],
                 title=Title(DMI_phase=DMI_neg, N34_phase=N34_neg)[0],
                 fase=' - Negative', name_fig=name_fig,
                 save=save, dpi=dpi, levels=levels,
                 contour=contour, cmap=cmap,
                 number_events=Title(DMI_phase=DMI_neg, N34_phase=N34_neg)[1],
                 season=season,
                 waf=waf, px=px, py=py, text=text, SA=SA,
                 two_variables=two_variables,
                 comp2=comp2, step=step,
                 levels2=levels2, contour0=contour0)

def PlotComposite_wWAF(comp, levels, cmap, step1, contour1=True,
                       two_variables=False, comp2=None,
                       levels2=np.linspace(-1, 1, 13), step2=4, mapa='sa',
                       title='title', name_fig='name_fig', dpi=100, save=False,
                       comp_sig=None, color_sig='k', significance=True,
                       linewidht2=.5, color_map='#d9d9d9',
                       out_dir='RUTA', proj='eq', borders=False,
                       third_variable=False, comp3=None,
                       levels_contour3=np.linspace(-1, 1, 13),
                       waf=False, data_waf=None, px=None, py=False,
                       waf_scale=1 / 1000, step_waf=10, hatches='..'):


    if mapa.lower() == 'sa':
        fig_size = (5, 6)
        extent = [270, 330, -60, 20]
        xticks = np.arange(270, 330, 10)
        yticks = np.arange(-60, 40, 20)

    elif mapa.lower() == 'tropical':
        fig_size = (7, 2)
        extent = [40, 280, -20, 20]
        xticks = np.arange(40, 280, 60)
        yticks = np.arange(-20, 20, 20)

    elif mapa.lower() == 'hs':
        fig_size = (9, 3.5)
        extent = [0, 359, -80, 20]
        xticks = np.arange(0, 360, 30)
        yticks = np.arange(-80, 20, 10)
        if proj != 'eq':
            fig_size = (5, 5)

    elif mapa.lower() == 'hs_ex':
        fig_size = (9, 3)
        extent = [0, 359, -60, -20]
        xticks = np.arange(0, 360, 30)
        yticks = np.arange(-60, -20, 10)

    else:
        fig_size = (8, 3)
        extent = [30, 330, -80, 20]
        xticks = np.arange(30, 330, 30)
        yticks = np.arange(-80, 20, 10)
        if proj != 'eq':
            fig_size = (5, 5)

    levels_contour = levels.copy()
    comp_var = comp['var']
    if isinstance(levels, np.ndarray):
        levels_contour = levels[levels != 0]
    else:
        levels_contour.remove(0)

    if two_variables:
        levels_contour2 = levels2.copy()
        comp_var2 = comp2['var']
        if isinstance(levels2, np.ndarray):
            levels_contour2 = levels2[levels2 != 0]
        else:
            levels_contour2.remove(0)

    crs_latlon = ccrs.PlateCarree()
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    if proj == 'eq':
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent(extent, crs=crs_latlon)
    else:
        ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=200))
        ax.set_extent([30, 340, -90, 0],
                      ccrs.PlateCarree(central_longitude=200))

    if two_variables:
        ax.contour(comp2.lon[::step2], comp2.lat[::step2], comp_var2[::step2, ::step2],
                   linewidths=linewidht2, levels=levels_contour2, transform=crs_latlon, colors='k')
    else:
        if contour1:
            ax.contour(comp.lon[::step1], comp.lat[::step1], comp_var[::step1, ::step1],
                       linewidths=.8, levels=levels_contour, transform=crs_latlon, colors='black')

    im = ax.contourf(comp.lon[::step1], comp.lat[::step1], comp_var[::step1, ::step1],
                     levels=levels, transform=crs_latlon, cmap=cmap, extend='both')

    if third_variable:
        comp_var3 = comp3['var']
        tv = ax.contour(comp3.lon[::2], comp3.lat[::2], comp_var3[::2, ::2], levels=levels_contour3,
                        colors=['#D300FF', '#00FF5D'], transform=crs_latlon, linewidths=1.5)
        # tv.monochrome = True
        # for col, ls in zip(tv.collections, tv._process_linestyles()):
        #     col.set_linestyle(ls)

    if significance:
        colors_l = [color_sig, color_sig]
        comp_sig_var = comp_sig['var']
        cs = ax.contourf(comp_sig.lon, comp_sig.lat, comp_sig_var,
                         transform=crs_latlon, colors='none',
                         hatches=[hatches, hatches], extend='lower')
        for i, collection in enumerate(cs.collections):
            collection.set_edgecolor(colors_l[i % len(colors_l)])

        for collection in cs.collections:
            collection.set_linewidth(0.)

    if waf:
        Q60 = np.nanpercentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 60)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)

        Q99 = np.nanpercentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 99)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) > Q99
        # mask array
        px_mask = ma.array(px_mask, mask=M)
        py_mask = ma.array(py_mask, mask=M)

        # plot vectors
        lons, lats = np.meshgrid(data_waf.lon.values, data_waf.lat.values)
        ax.quiver(lons[::step_waf, ::step_waf], lats[::step_waf, ::step_waf],
                  px_mask[0, ::step_waf, ::step_waf], py_mask[0, ::step_waf, ::step_waf],
                  transform=crs_latlon, pivot='tail', width=1.5e-3, headwidth=3, alpha=1,
                  headlength=2.5, color='k', scale=waf_scale)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)
    if borders:
        ax.add_feature(cartopy.feature.BORDERS, facecolor='white',
                       edgecolor=color_map)
    # ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    ax.coastlines(color=color_map, linestyle='-', alpha=1)
    if mapa.lower() == 'sa':
        ax.add_feature(cartopy.feature.BORDERS, alpha=0.7)

    if proj == 'eq':
        ax.gridlines(crs=crs_latlon, linewidth=0, linestyle='-')
        ax.set_xticks(xticks, crs=crs_latlon)
        ax.set_yticks(yticks, crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    else:
        gls = ax.gridlines(draw_labels=True, crs=crs_latlon, lw=0.3, color="gray",
                           y_inline=True, xlocs=range(-180, 180, 30), ylocs=np.arange(-80, 0, 20))

        r_extent = 1.2e7
        ax.set_xlim(-r_extent, r_extent)
        ax.set_ylim(-r_extent, r_extent)
        circle_path = mpath.Path.unit_circle()
        circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                                 circle_path.codes.copy())
        ax.set_boundary(circle_path)
        ax.set_frame_on(False)
        plt.draw()
        for ea in gls._labels:
            pos = ea[2].get_position()
            if (pos[0] == 150):
                ea[2].set_position([0, pos[1]])

    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

################################################################################
def CreateDirectory(out_dir, *args):
    for arg in args:
        if arg is not None:
            if not os.path.exists(os.path.join(out_dir, str(arg))):
                os.mkdir(os.path.join(out_dir, str(arg)))

def DirAndFile(out_dir, dir_results, common_name, names, format='jpg'):
    file_name = f"{'_'.join(names)}_{common_name}.{format}"
    path = os.path.join(out_dir, dir_results, file_name)
    return path

# Esto era de los esquemas pero por alguna razon se borro parte de ese codigo
def moving_average_2d(data, window):
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a numpy array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)

    # The output array has the same dimensions as the input data
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    return convolve2d(data, window, mode='same', boundary='symm')

def RenameDataset(new_name, *args):
    dataset = []
    for arg in args:
        arg2 = arg.rename({list(arg.data_vars)[0]:new_name})

        dataset.append(arg2)

    return tuple(dataset)

################################################################################
# PlotFinal ####################################################################
def SetDataToPlotFinal(*args):
    data_arrays = []
    first = True
    for arg in args:
        if not isinstance(arg, xr.DataArray):
            try:
                arg = xr.DataArray(arg, dims=['lat', 'lon'])
            except:
                arg = arg.to_array()
                if 1 in arg.shape:
                    arg = arg.squeeze()
                arg = xr.DataArray(arg, dims=['lat', 'lon'])
        else:
            if 1 in arg.shape:
                arg = arg.squeeze()

        if first is False:
            if data_arrays[0].lon.values[-1] != arg.lon.values[-1]:
                arg = arg.interp(lon = data_arrays[0].lon.values,
                                 lat = data_arrays[0].lat.values)

        data_arrays.append(arg)
        first = False

    data = xr.concat(data_arrays, dim='plots')
    data = data.assign_coords(plots=range(data.shape[0]))

    return data

def PlotFinal(data, levels, cmap, titles, namefig, map, save, dpi, out_dir,
              data_ctn=None, levels_ctn=None, color_ctn=None,
              data_ctn2=None, levels_ctn2=None, color_ctn2=None,
              data_waf=None, wafx=None, wafy=None, waf_scale=None,
              waf_step=None, waf_label=None, sig_points=None, hatches=None,
              num_cols=None, high=2, width = 7.08661, step=2, cbar_pos = 'H',
              num_cases=False, num_cases_data=None, pdf=False, ocean_mask=False,
              data_ctn_no_ocean_mask=False, data_ctn2_no_ocean_mask=False,
              pcolormesh=False):

    # cantidad de filas necesarias
    if num_cols is None:
        num_cols = 2
    width = width
    plots = data.plots.values
    num_plots = len(plots)
    num_rows = np.ceil(num_plots / num_cols).astype(int)

    crs_latlon = ccrs.PlateCarree()

    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        step_lon = 60
        high = high
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        step_lon = 60
        high = high
    elif map.upper() == 'HS_EX':
        extent = [0, 359, -65, -20]
        step_lon = 60
        high = 2
    elif map.upper() == 'SA':
        extent = [275, 330, -60, 20]
        step_lon = 20
        high = high
    else:
        print(f"Mapa {map} no seteado")
        return

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    i = 0
    for ax, plot in zip(axes.flatten(), plots):
        no_plot = False

        # CONTOUR ------------------------------------------------------------ #
        if data_ctn is not None:
            if levels_ctn is None:
                levels_ctn = levels.copy()
            try:
                if isinstance(levels_ctn, np.ndarray):
                    levels_ctn = levels_ctn[levels_ctn != 0]
                else:
                    levels_ctn.remove(0)
            except:
                pass
            aux_ctn = data_ctn.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values

                ax.contour(data_ctn.lon.values[::step],
                           data_ctn.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.4,
                           levels=levels_ctn, transform=crs_latlon,
                           colors=color_ctn)

        # CONTOUR2 ----------------------------------------------------------- #
        if data_ctn2 is not None:
            if levels_ctn2 is None:
                levels_ctn2 = levels.copy()

            try:
                if isinstance(levels_ctn2, np.ndarray):
                    levels_ctn2 = levels_ctn2[levels_ctn != 0]
                else:
                    levels_ctn2.remove(0)
            except:
                pass
            aux_ctn = data_ctn2.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn2_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values
                ax.contour(data_ctn2.lon.values[::step],
                           data_ctn2.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.5,
                           levels=levels_ctn2, transform=crs_latlon,
                           colors=color_ctn2)

        # CONTOURF OR COLORMESH ---------------------------------------------- #
        aux = data.sel(plots=plot)
        if aux.mean().values != 0:

            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux = aux * mask_ocean.mask

            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            if pcolormesh is True:
                im = ax.pcolormesh(aux.lon.values[::step],
                                   aux.lat.values[::step],
                                   aux_var[::step, ::step],
                                   vmin=np.min(levels), vmax=np.max(levels),
                                   transform=crs_latlon, cmap=cmap)
            else:
                im = ax.contourf(aux.lon.values[::step], aux.lat.values[::step],
                                 aux_var[::step, ::step],
                                 levels=levels,
                                 transform=crs_latlon, cmap=cmap, extend='both')


        else:
            ax.axis('off')
            no_plot=True

        # WAF ---------------------------------------------------------------- #
        if data_waf is not None:
            wafx_aux = wafx.sel(plots=plot)
            wafy_aux = wafy.sel(plots=plot)

            if ocean_mask is True:
                mask_ocean = MakeMask(wafx_aux)
                wafx_aux = wafx_aux * mask_ocean.mask
                wafy_aux = wafy_aux * mask_ocean.mask


            Q60 = np.nanpercentile(np.sqrt(np.add(np.power(wafx_aux, 2),
                                                  np.power(wafy_aux, 2))), 60)
            M = np.sqrt(np.add(np.power(wafx_aux, 2),
                               np.power(wafy_aux, 2))) < Q60
            # mask array
            wafx_mask = ma.array(wafx_aux, mask=M)
            wafy_mask = ma.array(wafy_aux, mask=M)
            Q99 = np.nanpercentile(np.sqrt(np.add(np.power(wafx_aux, 2),
                                                  np.power(wafy_aux, 2))), 99)
            M = np.sqrt(np.add(np.power(wafx_aux, 2),
                               np.power(wafy_aux, 2))) > Q99
            # mask array
            wafx_mask = ma.array(wafx_mask, mask=M)
            wafy_mask = ma.array(wafy_mask, mask=M)

            # plot vectors
            lons, lats = np.meshgrid(data_waf.lon.values, data_waf.lat.values)
            Q = ax.quiver(lons[::waf_step, ::waf_step],
                          lats[::waf_step, ::waf_step],
                          wafx_mask[::waf_step, ::waf_step],
                          wafy_mask[::waf_step, ::waf_step],
                          transform=crs_latlon, pivot='tail',
                          width=1.7e-3, headwidth=4, alpha=1, headlength=2.5,
                          color='k', scale=waf_scale, angles='xy',
                          scale_units='xy')

            ax.quiverkey(Q, 0.85, 0.05, waf_label,
                         f'{waf_label:.1e} $m^2$ $s^{{-2}}$',
                         labelpos='E', coordinates='figure', labelsep=0.05,
                         fontproperties=FontProperties(size=6, weight='light'))

        # SIG ---------------------------------------------------------------- #
        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']
                try:
                    comp_sig_var = aux_sig_points['var']
                except:
                    comp_sig_var = aux_sig_points.values
                cs = ax.contourf(aux_sig_points.lon[::step],
                                 aux_sig_points.lat[::step],
                                 comp_sig_var[::step, ::step],
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower')

                for i2, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i2 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)

        # no plotear --------------------------------------------------------- #
        if no_plot is False:
            if num_cases:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]}), "
                                       f"$N={num_cases_data[plot]}$",
                        transform=ax.transAxes, size=6)
            else:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes, size=6)
            i = i + 1

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2,
                          resolution='110m')
            if map.upper() == 'SA':
                ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                               linestyle='-', linewidth=0.2, color='k')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                              zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            ax.set_xticks(np.arange(0, 360, step_lon), crs=crs_latlon)
            ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            ax.tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=4)
            ax.set_extent(extent, crs=crs_latlon)

            ax.set_aspect('equal')
            ax.set_title(titles[plot], fontsize=6, pad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    #cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.235, 0.03, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=5, pad=1)
        fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        aux_color = cmap.colors[2]
        patch = mpatches.Patch(color=aux_color, label='Ks < 0')

        legend = fig.legend(handles=[patch], loc='lower center', fontsize=8,
                            frameon=True, framealpha=1, fancybox=True)
        legend.set_bbox_to_anchor((0.5, 0.01), transform=fig.transFigure)
        legend.get_frame().set_linewidth(0.5)
        fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{namefig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()

def PlotFinal_Figs12_13(data, levels, cmap, titles, namefig, map, save, dpi,
                        out_dir, data_ctn=None, levels_ctn=None,
                        color_ctn=None, row_titles=None, col_titles=None,
                        clim_plot=None, clim_levels=None,clim_cbar=None,
                        high=2, width = 7.08661, cbar_pos='H', plot_step=3,
                        contourf_clim=False, pdf=True):

    # cantidad de filas necesarias
    num_cols = 5
    num_rows = 5

    plots = data.plots.values
    crs_latlon = ccrs.PlateCarree()

    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        high = high
        xticks = np.arange(0, 360, 60)
        yticks = np.arange(-80, 20, 20)
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        high = high
        xticks = np.arange(0, 360, 60)
        np.arange(-80, 20, 20)
    elif map.upper() == 'HS_EX':
        extent = [0, 359, -65, -20]
        high = high
        xticks = np.arange(0, 360, 60)
    elif map.upper() == 'SA':
        extent = [270, 330, -60, 20]
        high = high
        yticks = np.arange(-60, 15+1, 10)
        xticks = np.arange(275, 330+1, 10)
    else:
        print(f"Mapa {map} no seteado")
        return

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.05, 'hspace': 0.01})

    i2 = 0
    for i, (ax, plot) in enumerate(zip(axes.flatten(), plots)):
        remove_axes = False

        # Seteo filas, titulos ----------------------------------------------- #
        if i == 2 or i == 10:
            ax.set_title(col_titles[i], fontsize=5, pad=2)
            ax.yaxis.set_label_position('left')
            ax.text(-0.05, 0.5, row_titles[i], rotation=90,
                    transform=ax.transAxes, fontsize=5,
                    verticalalignment='center')
        elif i == 3 or i == 4 or i == 11:
            ax.set_title(col_titles[i], fontsize=5, pad=3)
        elif i == 7 or i == 15 or i == 20:
            ax.yaxis.set_label_position('left')
            ax.text(-0.05, 0.5, row_titles[i], rotation=90,
                    transform=ax.transAxes, fontsize=5,
                    verticalalignment='center')

        if i in [4, 9, 14, 17, 22]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_yticks(yticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=3)
            ax.set_extent(extent, crs=crs_latlon)

        if i in [20,21,22, 23, 13, 14]:
            ax.set_xticks(xticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.tick_params(labelsize=3)
            ax.set_extent(extent, crs=crs_latlon)

        ax.tick_params(width=0.5, pad=1, labelsize=4)

        # Plot --------------------------------------------------------------- #
        if plot == 12: # Clima
            if contourf_clim: # Contourf ------------------------------------- #
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) "
                                       f"$N={titles[plot]}$",
                        transform=ax.transAxes, size=4)
                i2 += 1

                cp = ax.contourf(clim_plot.lon.values[::plot_step],
                                 clim_plot.lat.values[::plot_step],
                                 clim_plot['var'][::plot_step, ::plot_step],
                                 levels=clim_levels,
                                 transform=crs_latlon, cmap=clim_cbar,
                                 extend='both')

            else: # Contour -------------------------------------------------- #

                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]})",
                        transform=ax.transAxes, size=4)
                i2 += 1

                cp = ax.contour(clim_plot.lon.values[::plot_step],
                                clim_plot.lat.values[::plot_step],
                                clim_plot['var'][::plot_step, ::plot_step],
                                linewidth=1, levels=clim_levels,
                                transform=crs_latlon, cmap=clim_cbar)

            ax.add_feature(cartopy.feature.LAND, facecolor='white')
            ax.coastlines(color='k', linestyle='-', alpha=1, resolution='110m',
                          linewidth=0.2)
            ax.set_title('Climatology', fontsize=4, pad=1)
            gl = ax.gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
                              zorder=20)

            gl.ylocator = plt.MultipleLocator(20)
            gl.xlocator = plt.MultipleLocator(60)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        else: # Todos los otros ploteos -------------------------------------- #

            # Contour -------------------------------------------------------- #
            if data_ctn is not None:
                if levels_ctn is None:
                    levels_ctn = levels.copy()
                try:
                    if isinstance(levels_ctn, np.ndarray):
                        levels_ctn = levels_ctn[levels_ctn != 0]
                    else:
                        levels_ctn.remove(0)
                except:
                    pass
                aux_ctn = data_ctn.sel(plots=plot)

                if aux_ctn.mean().values != 0:
                    ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) "
                                           f"$N={titles[plot]}$",
                            transform=ax.transAxes, size=4)
                    i2 += 1

                    try:
                        aux_ctn_var = aux_ctn['var'].values
                    except:
                        aux_ctn_var = aux_ctn.values

                    ax.contour(data_ctn.lon.values[::plot_step],
                               data_ctn.lat.values[::plot_step],
                               aux_ctn_var[::plot_step,::plot_step],
                               linewidths=0.4,
                               levels=levels_ctn, transform=crs_latlon,
                               colors=color_ctn)
                else:
                    remove_axes = True

            aux = data.sel(plots=plot)
            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            # Contourf ------------------------------------------------------- #
            if aux.mean().values != 0:
                im = ax.contourf(aux.lon.values[::plot_step],
                                 aux.lat.values[::plot_step],
                                 aux_var[::plot_step,::plot_step],
                                 levels=levels,
                                 transform=crs_latlon, cmap=cmap, extend='both')

                ax.add_feature(cartopy.feature.LAND, facecolor='white',
                               linewidth=0.5)
                #ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.2)
                ax.coastlines(color='k', linestyle='-', alpha=1,
                              linewidth=0.2, resolution='110m')
                gl = ax.gridlines(draw_labels=False, linewidth=0.1,
                                  linestyle='-',
                                  zorder=20)
                gl.ylocator = plt.MultipleLocator(20)
                gl.xlocator = plt.MultipleLocator(60)

            else:
                remove_axes = True

            if remove_axes:
                ax.axis('off')

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.261, 0, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0.05, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        pos = fig.add_axes([0.261, 0, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='vertical')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0.05, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)
    else:
        print(f"cbar_pos {cbar_pos} no valido")

    if save:
        if pdf:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(f"{out_dir}{namefig}.jpg", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def PlotFinal_CompositeByMagnitude(data, levels, cmap, titles, namefig, map,
                                   save, dpi, out_dir, data_ctn=None,
                                   levels_ctn=None, color_ctn=None,
                                   row_titles=None, col_titles=None,
                                   clim_plot=None, clim_levels=None,
                                   clim_cbar=None, high=2, width = 7.08661,
                                   cbar_pos='H', plot_step=3,
                                   clim_plot_ctn=None, clim_levels_ctn=None,
                                   pdf=True, ocean_mask=False,
                                   data_ctn_no_ocean_mask=False,
                                   sig_data=None, hatches=None):

    # cantidad de filas necesarias
    num_cols = 5
    num_rows = 5

    plots = data.plots.values
    crs_latlon = ccrs.PlateCarree()

    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        high = high
        xticks = np.arange(0, 360, 60)
        yticks = np.arange(-80, 20, 20)
        lon_localator = 60
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        high = high
        xticks = np.arange(0, 360, 60)
        np.arange(-80, 20, 20)
        lon_localator = 60
    elif map.upper() == 'HS_EX':
        extent = [0, 359, -65, -20]
        high = high
        xticks = np.arange(0, 360, 60)
        lon_localator = 60
    elif map.upper() == 'SA':
        extent = [270, 330, -60, 20]
        high = high
        yticks = np.arange(-60, 15+1, 20)
        xticks = np.arange(275, 330+1, 20)
        lon_localator = 20
    else:
        print(f"Mapa {map} no seteado")
        return

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.05, 'hspace': 0.01})

    i2 = 0
    for i, (ax, plot) in enumerate(zip(axes.flatten(), plots)):
        remove_axes = False

        # Seteo filas, titulos ----------------------------------------------- #
        if i == 2 or i == 10:
            ax.set_title(f'{col_titles[i]}\n                           ',
                         fontsize=5, pad=2, loc='left')
            ax.yaxis.set_label_position('left')
            ax.text(-0.07, 0.5, row_titles[i], rotation=90,
                    transform=ax.transAxes, fontsize=5,
                    verticalalignment='center')
        elif i == 3 or i == 4 or i == 11:
            ax.set_title(f'{col_titles[i]}\n                           ',
                         fontsize=5, pad=3, loc='left')
        elif i == 7 or i == 15 or i == 20:
            ax.yaxis.set_label_position('left')
            ax.text(-0.07, 0.5, row_titles[i], rotation=90,
                    transform=ax.transAxes, fontsize=5,
                    verticalalignment='center')

        if i in [4, 9, 14, 17, 22]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_yticks(yticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            # ax.tick_params(labelsize=3)
            # ax.set_extent(extent, crs=crs_latlon)

        if i in [20,21,22, 23, 13, 14]:
            ax.set_xticks(xticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.tick_params(labelsize=3)
            #ax.set_extent(extent, crs=crs_latlon)

        ax.tick_params(width=0.5, pad=1, labelsize=4)
        # Plot --------------------------------------------------------------- #
        if plot == 12:  # Clima
            if clim_plot.mean()['var'].values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(clim_plot)
                    clim_plot = clim_plot * mask_ocean.mask

                ax_new = fig.add_axes(
                    [0.365, 0.41, 0.19, 0.18],
                    projection=ccrs.PlateCarree())

                # Contour ---------------------------------------------------- #
                if clim_plot_ctn and clim_plot_ctn.mean()['var'].values != 0:

                    if ocean_mask is True and data_ctn_no_ocean_mask is False:
                        mask_ocean = MakeMask(clim_plot_ctn)
                        clim_plot_ctn = clim_plot_ctn * mask_ocean.mask

                    ax_new.contour(clim_plot_ctn.lon.values[::plot_step],
                                   clim_plot_ctn.lat.values[::plot_step],
                                   clim_plot_ctn['var'][::plot_step, ::plot_step],
                                   linewidths=0.4,
                                   levels=clim_levels_ctn, transform=crs_latlon,
                                   colors=color_ctn)

                # Contourf --------------------------------------------------- #
                cp = ax_new.contourf(clim_plot.lon.values[::plot_step],
                                     clim_plot.lat.values[::plot_step],
                                     clim_plot['var'][::plot_step,
                                     ::plot_step],
                                     levels=clim_levels,
                                     transform=crs_latlon,
                                     cmap=clim_cbar,
                                     extend='both')

                ax_new.set_title('Plot 12', fontsize=5)

                # Barra de colores si es necesario
                cb = plt.colorbar(cp, ax=ax_new, fraction=0.046,
                                  pad=0.02, shrink=0.6,
                                  orientation='horizontal')
                cb.ax.tick_params(labelsize=4, pad=0.1, length=1,
                                  width=0.5)
                for spine in cb.ax.spines.values():
                    spine.set_linewidth(0.5)

                else:
                    pass

                ax_new.add_feature(cartopy.feature.LAND, facecolor='white')
                ax_new.coastlines(color='k', linestyle='-', alpha=1,
                                  resolution='110m', linewidth=0.2)
                ax_new.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) ",
                        transform=ax_new.transAxes, size=4)
                i2 += 1
                ax_new.set_title('Climatology', fontsize=4, pad=1)
                gl = ax_new.gridlines(draw_labels=False, linewidth=0.3,
                                      linestyle='-', zorder=20)

                gl.ylocator = plt.MultipleLocator(20)
                gl.xlocator = plt.MultipleLocator(60)

                for spine in ax_new.spines.values():
                    spine.set_linewidth(0.5)

            ax.axis('off')

        else:
            # Contour -------------------------------------------------------- #
            if data_ctn is not None:
                if levels_ctn is None:
                    levels_ctn = levels.copy()
                try:
                    if isinstance(levels_ctn, np.ndarray):
                        levels_ctn = levels_ctn[levels_ctn != 0]
                    else:
                        levels_ctn.remove(0)
                except:
                    pass

                aux_ctn = data_ctn.sel(plots=plot)
                if ((aux_ctn.mean().values != 0) and
                        (~np.isnan(aux_ctn.mean().values))):

                    if ocean_mask is True and data_ctn_no_ocean_mask is False:
                        mask_ocean = MakeMask(aux_ctn)
                        aux_ctn = aux_ctn * mask_ocean.mask

                    try:
                        aux_ctn_var = aux_ctn['var'].values
                    except:
                        aux_ctn_var = aux_ctn.values

                    ax.contour(data_ctn.lon.values[::plot_step],
                               data_ctn.lat.values[::plot_step],
                               aux_ctn_var[::plot_step,::plot_step],
                               linewidths=0.4,
                               levels=levels_ctn, transform=crs_latlon,
                               colors=color_ctn)
                else:
                    remove_axes = True

            aux = data.sel(plots=plot)
            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values


            if sig_data is not None:
                aux_sig_points = sig_data.sel(plots=plot)
                if aux_sig_points.mean().values != 0:

                    if ocean_mask is True:
                        mask_ocean = MakeMask(aux_sig_points)
                        aux_sig_points = aux_sig_points * mask_ocean.mask

                    # hatches = '....'
                    colors_l = ['k', 'k']
                    try:
                        comp_sig_var = aux_sig_points['var']
                    except:
                        comp_sig_var = aux_sig_points.values
                    cs = ax.contourf(aux_sig_points.lon,
                                     aux_sig_points.lat,
                                     comp_sig_var,
                                     transform=crs_latlon, colors='none',
                                     hatches=[hatches, hatches], extend='lower', zorder=5)

                    for i3, collection in enumerate(cs.collections):
                        collection.set_edgecolor(colors_l[i3 % len(colors_l)])

                    for collection in cs.collections:
                        collection.set_linewidth(0.)


            # Contourf ------------------------------------------------------- #
            if ((aux.mean().values != 0) and
                    (~np.isnan(aux.mean().values))):

                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) "
                                       f"$N={titles[plot]}$",
                        transform=ax.transAxes, size=4)

                i2 += 1

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux)
                    aux_var = aux_var * mask_ocean.mask

                im = ax.contourf(aux.lon.values[::plot_step],
                                 aux.lat.values[::plot_step],
                                 aux_var[::plot_step,::plot_step],
                                 levels=levels,
                                 transform=crs_latlon, cmap=cmap, extend='both', zorder=1)

                ax.add_feature(cartopy.feature.LAND, facecolor='white',
                               linewidth=0.5)
                ax.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
                              resolution='110m')
                gl = ax.gridlines(draw_labels=False, linewidth=0.1,
                                  linestyle='-', zorder=20)
                gl.ylocator = plt.MultipleLocator(20)
                gl.xlocator = plt.MultipleLocator(lon_localator)

            else:
                remove_axes = True

            if remove_axes:
                ax.axis('off')

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.261, 0, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0.05, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        pos = fig.add_axes([0.95, 0.2, 0.02, 0.5])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='vertical')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0, wspace=0.5, hspace=0.25, left=0.02,
                            right=0.9, top=1)
    else:
        print(f"cbar_pos {cbar_pos} no valido")

    if save:
        if pdf:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(f"{out_dir}{namefig}.jpg", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def PlotFinal14(data, levels, cmap, titles, namefig, save, dpi, out_dir,
                sig_points=None, lons=None, levels2=None, cmap2=None,
                high=4, width = 7.08661):

    hatches='///'
    variable = ['Temperature', None, None, 'Precipitation']
    plots = data.plots.values
    num_plots = len(plots)

    crs_latlon = ccrs.PlateCarree()

    fig = plt.figure(figsize=(width, high), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1], wspace=0.05, hspace=.2)

    axs0 = subfigs[0].subplots(4, 3,
                               subplot_kw={'projection': ccrs.PlateCarree(
                                   central_longitude=180)})
    axs0 = axs0.flatten()

    axs1 = subfigs[1].subplots(4, 3,
                               subplot_kw={'projection': ccrs.PlateCarree(
                                   central_longitude=180)})
    axs1 = axs1.flatten()

    axs = [axs0[0], axs0[1], axs0[2],
           axs1[0], axs1[1], axs1[2],
           axs0[3], axs0[4], axs0[5],
           axs1[3], axs1[4], axs1[5],
           axs0[6], axs0[7], axs0[8],
           axs1[6], axs1[7], axs1[8],
           axs0[9], axs0[10], axs0[11],
           axs1[9], axs1[10], axs1[11]]

    i2 = 0
    for i in plots:
        if i >= 0:
            if i in [0, 3]:
                axs[i].text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) "
                                           f"${variable[i]}$",
                            transform=axs[i].transAxes, size=6)
                i2 += 1
            else:
                axs[i].text(-0.005, 1.025, f"({string.ascii_lowercase[i2]})",
                            transform=axs[i].transAxes, size=6)
                i2 += 1

        aux = data.sel(plots=i)
        try:
            aux_var = aux['var'].values
        except:
            aux_var = aux.values

        if i in [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]:
            im0 = axs[i].contourf(aux.lon.values[::2], aux.lat.values[::2],
                                  aux_var[::2,::2],
                                 levels=levels, cmap=cmap, extend='both',
                                 transform=ccrs.PlateCarree())
        else:
            im1 = axs[i].contourf(aux.lon.values[::2], aux.lat.values[::2],
                                  aux_var[::2, ::2],
                                  levels=levels2, cmap=cmap2, extend='both',
                                  transform=ccrs.PlateCarree())

        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=i)
            colors_l = ['k', 'k']
            try:
                comp_sig_var = aux_sig_points['var'].values
            except:
                comp_sig_var = aux_sig_points.values

            mpl.rcParams['hatch.linewidth'] = 0.5
            cs = axs[i].contourf(aux_sig_points.lon[::2],
                                 aux_sig_points.lat[::2],
                                 comp_sig_var[::2,::2],
                                 transform=ccrs.PlateCarree(),
                                 colors='none', hatches=['//////', '//////'],
                                 extend='lower')

            for i3, collection in enumerate(cs.collections):
                collection.set_edgecolor(colors_l[i3 % len(colors_l)])

            for collection in cs.collections:
                collection.set_linewidth(0.)

        axs[i].add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
        #axs[i].add_feature(cartopy.feature.COASTLINE, linewidth=0.2)
        axs[i].coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2, resolution='110m')
        gl = axs[i].gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
                          zorder=20)
        gl.ylocator = plt.MultipleLocator(20)
        gl.xlocator = plt.MultipleLocator(20)

        if i in [1, 4, 7, 10, 13, 16, 19, 22]:
            axs[i].set_title(titles[i], fontsize=6, pad=1)
            axs[i].set_xticks(np.arange(100, 160, 20), crs=crs_latlon)
            axs[i].tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            axs[i].xaxis.set_major_formatter(lon_formatter)
            extent = [lons[i][0], lons[i][1], -60, -20]
            axs[i].tick_params(labelsize=4)
            axs[i].set_extent(extent, crs=crs_latlon)
            axs[i].set_aspect('equal')

        elif i in [2, 5, 8, 11, 14, 17, 20, 23]:
            axs[i].set_xticks(np.arange(280, 340, 20), crs=crs_latlon)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            axs[i].xaxis.set_major_formatter(lon_formatter)
            axs[i].tick_params(width=0.5, pad=1)
            axs[i].tick_params(labelsize=4)
            extent = [lons[i][0], lons[i][1], -60, -20]
            axs[i].set_extent(extent, crs=crs_latlon)
            axs[i].set_aspect('equal')

        else:
            axs[i].set_xticks(np.arange(0, 60, 20), crs=crs_latlon)
            axs[i].set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            axs[i].tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            axs[i].xaxis.set_major_formatter(lon_formatter)
            axs[i].yaxis.set_major_formatter(lat_formatter)
            axs[i].tick_params(labelsize=4)
            extent = [lons[i][0], lons[i][1], -60, -20]
            axs[i].set_extent(extent, crs=crs_latlon)
            axs[i].set_aspect('equal')

        for spine in axs[i].spines.values():
            spine.set_linewidth(0.5)

    pos = subfigs[0].add_axes([0.21, 0.03, 0.5, 0.02])
    cb0 = subfigs[0].colorbar(im0, cax=pos, pad=0.1, orientation='horizontal')
    cb0.ax.tick_params(labelsize=5, pad=1)

    pos = subfigs[1].add_axes([0.21, 0.03, 0.5, 0.02])
    cb1 = subfigs[1].colorbar(im1, cax=pos, pad=0.1, orientation='horizontal')
    cb1.ax.tick_params(labelsize=5, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0.008, hspace=0.35, left=0, top=1)

    if save:
        plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def PlotFinal15_16(data, levels, cmap, titles, namefig, save, dpi, out_dir,
                   sig_points=None, lons=None, high=4.19, width = 7.08661):

    hatches='//////'

    plots = data.plots.values
    num_plots = len(plots)

    crs_latlon = ccrs.PlateCarree()

    fig = plt.figure(figsize=(width, high), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1], wspace=0, hspace=.05)

    axs0 = subfigs[0].subplots(3, 3,
                               subplot_kw={'projection': ccrs.PlateCarree(
                                   central_longitude=180)})
    axs0 = axs0.flatten()
    #subfigs[0].suptitle('Positive Phase', fontsize=20)

    axs1 = subfigs[1].subplots(3, 3,
                               subplot_kw={'projection': ccrs.PlateCarree(
                                   central_longitude=180)})
    axs1 = axs1.flatten()
    #subfigs[1].suptitle('Negative Phase', fontsize=20)

    axs = [axs0[0], axs0[1], axs0[2],
           axs1[0], axs1[1], axs1[2],
           axs0[3], axs0[4], axs0[5],
           axs1[3], axs1[4], axs1[5],
           axs0[6], axs0[7], axs0[8],
           axs1[6], axs1[7], axs1[8]]

    i2 = 0
    for i in plots:
        axs[i].text(-0.005, 1.025, f"({string.ascii_lowercase[i2]})",
                    transform=axs[i].transAxes, size=6)
        i2 += 1
        # if i in [0, 3, 6, 9, 12, 15, 18, 21]:
        #     axs[i].text(-0.005, 1.025, f"({string.ascii_lowercase[i2]})",
        #             transform=axs[i].transAxes, size=6)
        #     i2 += 1

        aux = data.sel(plots=i)
        try:
            aux_var = aux['var'].values
        except:
            aux_var = aux.values

        im = axs[i].contourf(aux.lon.values[::2], aux.lat.values[::2],
                             aux_var[::2,::2],
                             levels=levels, cmap=cmap, extend='both',
                             transform=ccrs.PlateCarree())

        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=i)
            colors_l = ['k', 'k']
            comp_sig_var = aux_sig_points['var']
            mpl.rcParams['hatch.linewidth'] = 0.5
            cs = axs[i].contourf(aux_sig_points.lon[::2],
                                 aux_sig_points.lat[::2],
                                 comp_sig_var[::2,::2],
                                 transform=ccrs.PlateCarree(),
                                 colors='none', hatches=['///////', '///////'],
                                 extend='lower')

            for i3, collection in enumerate(cs.collections):
                collection.set_edgecolor(colors_l[i3 % len(colors_l)])

            for collection in cs.collections:
                collection.set_linewidth(0.)

        axs[i].add_feature(cartopy.feature.LAND, facecolor='white')
        #axs[i].add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        axs[i].coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2, resolution='110m')
        gl = axs[i].gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
                          zorder=20)
        gl.ylocator = plt.MultipleLocator(20)
        gl.xlocator = plt.MultipleLocator(20)

        if i in [1, 4, 7, 10, 13, 16, 19, 22]:
            axs[i].set_title(titles[i], fontsize=6, pad=1)
            axs[i].set_xticks(np.arange(100, 160, 20), crs=crs_latlon)
            axs[i].tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            axs[i].xaxis.set_major_formatter(lon_formatter)
            extent = [lons[i][0], lons[i][1], -60, -20]
            axs[i].tick_params(labelsize=4)
            axs[i].set_extent(extent, crs=crs_latlon)
            axs[i].set_aspect('equal')

        elif i in [2, 5, 8, 11, 14, 17, 20, 23]:
            axs[i].set_xticks(np.arange(280, 340, 20), crs=crs_latlon)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            axs[i].xaxis.set_major_formatter(lon_formatter)
            axs[i].tick_params(width=0.5, pad=1)
            axs[i].tick_params(labelsize=4)
            extent = [lons[i][0], lons[i][1], -60, -20]
            axs[i].set_extent(extent, crs=crs_latlon)
            axs[i].set_aspect('equal')

        else:
            axs[i].set_xticks(np.arange(0, 60, 20), crs=crs_latlon)
            axs[i].set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            axs[i].tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            axs[i].xaxis.set_major_formatter(lon_formatter)
            axs[i].yaxis.set_major_formatter(lat_formatter)
            axs[i].tick_params(labelsize=4)
            extent = [lons[i][0], lons[i][1], -60, -20]
            axs[i].set_extent(extent, crs=crs_latlon)
            axs[i].set_aspect('equal')

        for spine in axs[i].spines.values():
            spine.set_linewidth(0.5)

    pos = subfigs[1].add_axes([-.56, 0.03, 1, 0.02])
    cb = subfigs[1].colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
    cb.ax.tick_params(labelsize=5)

    fig.subplots_adjust(bottom=0.1, wspace=0.02, hspace=0.2, left=0, top=1)

    if save:
        plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def PlotFinalFigS3(data, data_ctn, levels, cmap, title0, namefig, save, dpi,
                   out_dir, data2=None, levels2=None, cmap2=None,
                   data2_ctn=None, titles2=None, high=4, width = 7.08661):

    crs_latlon = ccrs.PlateCarree()

    fig = plt.figure(figsize=(width, high), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[0.95, 1.05],
                             wspace=0.05,
                             hspace=0.1)

    axs0 = subfigs[0].subplots(1, 1,
                               subplot_kw={'projection': ccrs.PlateCarree(
                                   central_longitude=180)})

    axs1 = subfigs[1].subplots(2, 2,
                               subplot_kw={'projection': ccrs.PlateCarree(
                                   central_longitude=180)})
    axs1 = axs1.flatten()
    # esto puede no ser necesario
    axs = [axs1[0], axs1[1], axs1[2], axs1[3]]

    axs0.text(-0.005, 1.025, f"{string.ascii_lowercase[0]}. ",
                transform=axs0.transAxes, size=6)

    aux = data
    try:
        aux_var = aux['var'].values
    except:
        aux_var = aux.values

    im0 = axs0.contourf(aux.lon.values[::2], aux.lat.values[::2],
                        aux_var[::2,::2],
                        levels=levels2, cmap=cmap2, extend='both',
                        transform=ccrs.PlateCarree())

    aux_ctn = data_ctn
    try:
        aux_ctn_var = aux_ctn['var'].values
    except:
        aux_ctn_var = aux_ctn.values

    levels_ctn = levels.copy()
    if isinstance(levels_ctn, np.ndarray):
        levels_ctn = levels_ctn[levels_ctn != 0]
    else:
        levels_ctn.remove(0)

    axs0.contour(data2_ctn.lon.values[::2],
                 data2_ctn.lat.values[::2],
                 aux_ctn_var[::2,::2], linewidths=0.4,
                 levels=levels_ctn, transform=crs_latlon,
                 colors='k')

    axs0.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
    #axs0.add_feature(cartopy.feature.COASTLINE, linewidth=0.2)
    axs0.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
                    resolution='110m')
    gl = axs0.gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
                          zorder=20)
    gl.ylocator = plt.MultipleLocator(20)
    gl.xlocator = plt.MultipleLocator(60)

    axs0.set_xticks(np.arange(0, 360, 60), crs=crs_latlon)
    axs0.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    axs0.xaxis.set_major_formatter(lon_formatter)
    axs0.yaxis.set_major_formatter(lat_formatter)
    axs0.tick_params(labelsize=4)
    axs0.tick_params(width=0.5, pad=1)
    extent = [0, 359, -80, 20]
    axs0.set_extent(extent, crs=crs_latlon)
    axs0.set_aspect('equal')
    axs0.set_title(title0, fontsize=6, pad=1)
    for spine in axs0.spines.values():
        spine.set_linewidth(0.5)

    i2 = 1
    plots = data2.plots.values
    for i in plots:
        axs[i].text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) ",
                            transform=axs[i].transAxes, size=6)
        i2 += 1

        aux = data2.sel(plots=i)
        try:
            aux_var = aux['var'].values
        except:
            aux_var = aux.values

        im1 = axs[i].contourf(aux.lon.values, aux.lat.values, aux_var,
                              levels=levels2, cmap=cmap2, extend='both',
                              transform=ccrs.PlateCarree())

        aux_ctn = data2_ctn.sel(plots=i)
        try:
            aux_ctn_var = aux_ctn['var'].values
        except:
            aux_ctn_var = aux_ctn.values

        levels_ctn = levels2.copy()
        if isinstance(levels_ctn, np.ndarray):
            levels_ctn = levels_ctn[levels_ctn != 0]
        else:
            levels_ctn.remove(0)

        axs[i].add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
        #axs[i].add_feature(cartopy.feature.COASTLINE, linewidth=0.2)
        axs[i].coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2, resolution='110m')
        gl = axs[i].gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
                          zorder=20)
        gl.ylocator = plt.MultipleLocator(20)
        gl.xlocator = plt.MultipleLocator(60)

        axs[i].contour(data2_ctn.lon.values[::2], data2_ctn.lat.values[::2],
                   aux_ctn_var[::2,::2], linewidths=0.4,
                   levels=levels_ctn, transform=crs_latlon,
                   colors='k')

        axs[i].set_xticks(np.arange(0, 360, 60), crs=crs_latlon)
        axs[i].set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        axs[i].xaxis.set_major_formatter(lon_formatter)
        axs[i].yaxis.set_major_formatter(lat_formatter)
        axs[i].tick_params(labelsize=4)
        axs[i].tick_params(width=0.5, pad=1)
        extent = [0, 359, -80, 20]
        axs[i].set_extent(extent, crs=crs_latlon)
        axs[i].set_aspect('equal')

        axs[i].set_title(titles2[i], fontsize=6, pad=1)

        for spine in axs[i].spines.values():
            spine.set_linewidth(0.5)

    # pos = subfigs[0].add_axes([0.96, 0.16, 0.01, 0.75])
    # cb0 = subfigs[0].colorbar(im0, cax=pos, pad=0, orientation='vertical')
    # cb0.ax.tick_params(labelsize=5, pad=1)

    pos = subfigs[1].add_axes([0.25, 0.01, 0.5, 0.02])
    cb1 = subfigs[1].colorbar(im1, cax=pos, pad=1, orientation='horizontal')
    cb1.ax.tick_params(labelsize=5, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                        top=1)

    if save:
        plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def PlotFinalTwoVariables(data, num_cols,
                          levels_r1, cmap_r1,
                          levels_r2, cmap_r2,
                          data_ctn, levels_ctn_r1, levels_ctn_r2, color_ctn,
                          titles, namefig, save, dpi, out_dir, pdf=False,
                          high=2, width = 7.08661, step=1,
                          ocean_mask=False, num_cases=False,
                          num_cases_data=None,
                          sig_points=None, hatches=None,
                          data_ctn_no_ocean_mask=False):

    plots = data.plots.values
    num_plots = len(plots)
    num_rows = np.ceil(num_plots / num_cols).astype(int)

    crs_latlon = ccrs.PlateCarree()

    map == 'SA'
    extent = [275, 330, -60, 20]
    step_lon = 20
    high = high

    change_row = len(data.plots)/2

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    i = 0
    for ax, plot in zip(axes.flatten(), plots):
        no_plot = False

        if plot < change_row:
            levels = levels_r1
            levels_ctn = levels_ctn_r1
            cmap = cmap_r1
            row1=True
        else:
            levels = levels_r2
            levels_ctn = levels_ctn_r2
            cmap = cmap_r2
            row1 = False


        # CONTOUR ------------------------------------------------------------ #
        if data_ctn is not None:
            if levels_ctn is None:
                levels_ctn = levels.copy()
            try:
                if isinstance(levels_ctn, np.ndarray):
                    levels_ctn = levels_ctn[levels_ctn != 0]
                else:
                    levels_ctn.remove(0)
            except:
                pass
            aux_ctn = data_ctn.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values

                ax.contour(data_ctn.lon.values[::step],
                           data_ctn.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.4,
                           levels=levels_ctn, transform=crs_latlon,
                           colors=color_ctn)

        aux = data.sel(plots=plot)
        if aux.mean().values != 0:

            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux = aux * mask_ocean.mask

            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            if row1 is True:
                im_r1= ax.contourf(aux.lon.values[::step],
                                   aux.lat.values[::step],
                                   aux_var[::step, ::step],
                                   levels=levels,
                                   transform=crs_latlon, cmap=cmap,
                                   extend='both')
            else:
                im_r2 = ax.contourf(aux.lon.values[::step],
                                    aux.lat.values[::step],
                                    aux_var[::step, ::step],
                                    levels=levels,
                                    transform=crs_latlon, cmap=cmap,
                                    extend='both')
        else:
            ax.axis('off')
            no_plot=True


        # sig ---------------------------------------------------------------- #
        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']
                try:
                    comp_sig_var = aux_sig_points['var']
                except:
                    comp_sig_var = aux_sig_points.values
                cs = ax.contourf(aux_sig_points.lon[::step],
                                 aux_sig_points.lat[::step],
                                 comp_sig_var[::step, ::step],
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower')

                for i2, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i2 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)


        # no plotear --------------------------------------------------------- #
        if no_plot is False:
            if num_cases:
                ax.text(-0.01, 1.055, f"({string.ascii_lowercase[i]}) "
                                       f"$N={num_cases_data[plot]}$",
                        transform=ax.transAxes, size=6)
            else:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes, size=6)
            i = i + 1

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2,
                          resolution='110m')

            ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                               linestyle='-', linewidth=0.2, color='k')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                              zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            ax.set_xticks(np.arange(0, 360, step_lon), crs=crs_latlon)
            ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            ax.tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=4)
            ax.set_extent(extent, crs=crs_latlon)

            ax.set_aspect('equal')
            ax.set_title(titles[plot], fontsize=6, pad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    pos1 = fig.add_axes([0.87, 0.55, 0.015, 0.35])
    cb1 = fig.colorbar(im_r1, cax=pos1, orientation='vertical')
    cb1.ax.tick_params(labelsize=5, pad=1)

    pos2 = fig.add_axes([0.87, 0.1, 0.015, 0.35])
    cb2 = fig.colorbar(im_r2, cax=pos2, orientation='vertical')
    cb2.ax.tick_params(labelsize=5, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0.05, hspace=0.25, left=0.05,
                        right=0.85, top=0.95)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{namefig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()


    plt.show()

################################################################################
# PDF ##########################################################################
def pdf_fit_normal(data, size, start, end):
    y, x = np.histogram(data)
    #x = (x + np.roll(x, -1))[:-1] / 2.0

    #best_distributions = []
    # for distribution_name in ['norm']:
    #     distribution = getattr(st, distribution_name)

    distribution = st.norm

    # ajuste distribucion a la data
    params = distribution.fit(data)

    # parametros
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)


    # calculo de pdf ajustada y error en el ajuste
    # (por si es necesrio apra otra cosa)
    #pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
    #sse = np.sum(np.power(y - pdf, 2.0))

    # identify if this distribution is better
    # best_distributions.append((distribution, params, sse,
    #                            distribution_name))

    # best_dist = sorted(best_distributions, key=lambda x: x[2])[0]
    # dist = best_dist[0]
    # params =  best_dist[1]
    #
    # arg = params[:-2]
    # loc = params[-2]
    # scale = params[-1]

    return pdf

def PDF_cases(variable, season, box_lons, box_lats, box_name,
              cases, cases_dir):

    if variable == 'prec':
        fix_factor = 30
    elif variable == 'tref':
        fix_factor = 1
    else:
        fix_factor = 9.8

    # climatologia
    neutro = xr.open_dataset(
        f'{cases_dir}{variable}_neutros_{season}_detrend_05.nc')
    neutro = neutro.rename({list(neutro.data_vars)[0]: 'var'})
    neutro = neutro * fix_factor

    mask_ocean = MakeMask(neutro, list(neutro.data_vars)[0])
    neutro = neutro * mask_ocean

    resultados_regiones = {}
    for bl, bt, name in zip(box_lons, box_lats, box_name):
        aux_neutro = neutro.sel(lon=slice(bl[0], bl[1]),
                                lat=slice(bt[0], bt[1]))
        clim = aux_neutro.mean(['lon', 'lat'])

        aux_resultados = {}
        if variable == 'prec':
            if name == 'Patagonia':
                startend = -30
            elif name == 'N-SESA':
                startend = -80
            else:
                startend = -60
        else:
            startend = -2



        aux_clim = clim - clim.mean('time')
        aux_clim = np.nan_to_num(aux_clim['var'].values)

        pdf_clim_full = pdf_fit_normal(aux_clim, 500,
                                       -1 * startend, startend)

        aux_resultados['clim'] = pdf_clim_full

        for c_count, c in enumerate(cases):
            case = xr.open_dataset(
                f'{cases_dir}{variable}_{c}_{season}_detrend_05.nc')
            case = case.rename({list(case.data_vars)[0]: 'var'})
            case = case * fix_factor
            mask_ocean = MakeMask(case, list(case.data_vars)[0])
            case = case * mask_ocean
            case = case.sel(lon=slice(bl[0], bl[1]), lat=slice(bt[0], bt[1]))
            case = case.mean(['lon', 'lat'])

            case_anom = case - clim.mean('time')
            case_anom = np.nan_to_num(case_anom['var'].values)

            pdf_case = pdf_fit_normal(case_anom, 500,
                                      -startend,  startend)

            aux_resultados[c] = pdf_case

        resultados_regiones[name] = aux_resultados

    return resultados_regiones

def PlotPdfs(data, selected_cases, width=5, high=1.2, title='', namefig='fig',
             out_dir='', save=False, dpi=100):

    positive_cases_colors = ['red', '#F47B00', 'forestgreen']
    negative_cases_colors = ['#509DFE', '#00E071', '#FF3AA0']
    colors_cases = [positive_cases_colors, negative_cases_colors]

    fig, axes = plt.subplots(
        1, 2, figsize=(width, high * 2),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.01})

    for ax_count, (ax, cases, color_case) in enumerate(zip(
            axes.flatten(), selected_cases, colors_cases)):
        if ax_count == 1:
            ax.yaxis.tick_right()

        max_y = []
        ax.plot(data['clim'], lw=2.5, color='k', label='Clim.')
        max_y.append(max(data['clim']))
        for c_count, c in enumerate(cases[1::]):
            aux_case = data[c]
            ax.plot(aux_case, lw=2.5, color=color_case[c_count], label=c)
            max_y.append(max(aux_case))

        ax.grid(alpha=0.5)
        ax.legend(loc='best')
        ax.set_ylim(0, max(max_y) + 0.001)

    fig.suptitle(title, fontsize=10)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.tight_layout()
    if save:
        plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def AreaBetween(curva1: pd.Series, curva2: pd.Series) -> float:
    """
    Calcula el área entre dos curvas dadas como pandas.Series,

    Parámetros:
    curva1 : pd.Series -> Primera curva (índices = x, valores = densidad)
    curva2 : pd.Series -> Segunda curva (índices = x, valores = densidad)

    Retorna:
    float -> Área entre las dos curvas
    """
    # Extraer valores x e y de cada curva
    x = curva1.index.values
    y1 = curva1.values
    y2 = curva2.values

    max1 = curva1.idxmax()
    max2 = curva2.idxmax()

    sign = -1*np.sign(max1 - max2)
    if sign == 0:
        sign = 1
    # Calcular el área entre curvas
    area_between = trapz(np.abs(y1 - y2), x)
    if area_between <0: # pasa por el orden de x
        area_between = -1* area_between

    return area_between*sign

def AreaDiferencial(curva1: pd.Series, curva2: pd.Series):
    """
    Calcula el área entre dos curvas, separando el área donde curva2 está por encima
    de curva1 y viceversa.

    Parámetros:
    curva1 : pd.Series -> Primera curva (índices = x, valores = densidad)
    curva2 : pd.Series -> Segunda curva (índices = x, valores = densidad)

    Retorna:
    (area_superior, area_inferior) : tuple[float, float]
        area_superior: área donde curva2 > curva1
        area_inferior: área donde curva2 < curva1
    """
    x = curva1.index.values
    y1 = curva1.values
    y2 = curva2.values

    diff = y2 - y1

    area_superior = trapz(np.where(diff > 0, diff, 0), x[::-1])
    area_inferior = trapz(np.where(diff < 0, -diff, 0), x[::-1])  # usar -diff para que sea positiva

    return area_superior, area_inferior

def AreaBetween0(curva1: pd.Series, curva2: pd.Series) -> float:
    """
    Calcula el área entre dos curvas dadas como pandas.Series,

    Parámetros:
    curva1 : pd.Series -> Primera curva (índices = x, valores = densidad)
    curva2 : pd.Series -> Segunda curva (índices = x, valores = densidad)

    Retorna:
    float -> Área entre las dos curvas
    """
    # Extraer valores x e y de cada curva
    x = curva1.index.values
    y1 = curva1.values
    y2 = curva2.values

    max1 = curva1.idxmax()
    max2 = curva2.idxmax()
    sign = -1*(max1-max2)/abs(max1-max2)

    # Calcular el área entre curvas
    area_between = trapz(np.abs(y1 - y2), x[::-1])
    if area_between <0: # pasa por el orden de x
        area_between = -1* area_between

    return area_between*sign

def hellinger_distance(p: pd.Series, q: pd.Series):
    """
    Calcula la distancia de Hellinger entre dos distribuciones de probabilidad (Series normalizadas).
    """
    return np.sqrt(0.5 * np.sum((np.sqrt(p.values) - np.sqrt(q.values))**2))

from scipy.spatial.distance import jensenshannon

def js_divergence(p: pd.Series, q: pd.Series) -> float:
    p_norm = p / trapz(p.values, p.index.values)
    q_norm = q / trapz(q.values, q.index.values)
    return jensenshannon(p_norm, q_norm, base=2)**2

def PlotPDFTable(df, cmap, levels, title, name_fig='fig',
                 save=False, out_dir='~/', dpi=100, color_thr=0.4):

    fig = plt.figure(dpi=dpi, figsize=(8, 4))
    ax = fig.add_subplot(111)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    im = ax.imshow(df, cmap=cmap, norm=norm, aspect='auto')

    data_array = df.values
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            if np.abs(data_array[i, j]) > color_thr:
                color_num = 'white'
            else:
                color_num = 'k'
            ax.text(j, i, f"{data_array[i, j]:.2f}", ha='center', va='center',
                    color=color_num)

    # Ticks principales en el centro de las celdas (para los labels)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=0, ha='center')

    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)

    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    fig.suptitle(title, size=12)

    ax.margins(0)
    plt.tight_layout()

    if save:
        plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
################################################################################
# Bins #########################################################################
def SelectBins2D(serie, bins_limits):

    bins = []
    bins_limits = [[bins_limits[i], bins_limits[i+1]]
                   for i in range(len(bins_limits)-1)]

    for bl in bins_limits:
        bmin = bl[0]
        bmax = bl[1]
        x = serie.where(serie >= bmin)
        x = x.where(x < bmax)
        bins.append(x)

    bins = xr.concat(bins, dim='bins')

    return bins

def SelectDatesBins(bins, bin_data, min_percentage=0.1):

    bins_name_var = list(bins.data_vars)[0]
    bin_data_name_var = list(bin_data.data_vars)[0]

    bin_data_f = []
    bin_len = []
    for bl in bins.bins.values:
        aux_bins = bins.sel(bins=bl)

        bins_r = []
        bin_len_r = []
        for r in bins.r.values:
            aux = aux_bins.sel(r=r)
            bin_len_r.append(len(aux.where(
                ~np.isnan(aux.sst), drop=True)[bins_name_var]))
            bin_data_sel = bin_data.sel(r=r)
            dates_bins = aux.time[np.where(~np.isnan(aux[bins_name_var]))]
            bin_data_sel = bin_data_sel.sel(
                time=bin_data_sel.time.isin(dates_bins))
            bins_r.append(bin_data_sel)

        bin_len.append(np.sum(bin_len_r))
        bins_r_f = xr.concat(bins_r, dim='r')
        bin_data_f.append(bins_r_f)

    bin_data_f = xr.concat(bin_data_f, dim='bins')

    bin_data_mean = bin_data_f.mean(['r', 'time'], skipna=True)
    bin_data_mean = list(bin_data_mean[bin_data_name_var].values)

    bin_data_std = bin_data_f.std(['r', 'time'], skipna=True)
    bin_data_std = list(bin_data_std[bin_data_name_var].values)

    check = sum(bin_len)*min_percentage

    bin_data_mean, bin_data_std = zip(*[
        (0 if count < check else mean,
         0 if count < check else std)
        for count, mean, std in
        zip(bin_len, bin_data_mean, bin_data_std)
    ])

    return bin_data_mean, bin_data_std, bin_len

def PlotBars(x, bin_n, bin_n_err, bin_n_len,
             bin_d, bin_d_err, bin_d_len,
             title='', name_fig='fig', out_dir=out_dir, save=False,
             ymin=-80, ymax=45, dpi=100, ylabel='Anomaly',
             bar_n_color=None, bar_n_error_color=None, bar_d_color=None,
             bar_d_error_color=None):

    fig = plt.figure(1, figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111)

    plt.hlines(y=0, xmin=-4, xmax=4, color='k')

    ax.bar(x + 0.075, bin_n, color=bar_n_color, alpha=1, width=0.15,
           label='Niño3.4')
    ax.errorbar(x + 0.075, bin_n, yerr=bin_n_err, capsize=4, fmt='o', alpha=1,
                elinewidth=0.9, ecolor=bar_n_error_color, mfc='w',
                mec=bar_n_error_color, markersize=5)

    ax2 = ax.twinx()
    ax2.bar(x + 0.075, bin_n_len, color=bar_n_color, alpha=0.7, width=0.15)

    ax.bar(x - 0.075, np.nan_to_num(bin_d), color=bar_d_color, alpha=1,
           width=0.15, label='DMI')
    ax.errorbar(x - 0.075, bin_d, yerr=bin_d_err, capsize=4, fmt='o', alpha=1,
                elinewidth=0.9, ecolor=bar_d_error_color, mec=bar_d_error_color,
                mfc='w', markersize=5)
    ax2.bar(x - 0.075, bin_d_len, color=bar_d_color, alpha=0.7, width=0.15)

    ax.legend(loc='upper left')
    ax.set_ylim(ymin, ymax)
    ax2.set_ylim(0, 3000)
    ax.set_ylabel(ylabel, fontsize=10)
    ax2.set_ylabel('number of samples', fontsize=10)
    ax.set_xlabel('SST index (of std)', fontsize=15)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.grid(True)
    plt.title(title, fontsize=15)
    plt.xlim(-3.5, 3.5)

    if save:
        plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

################################################################################
# Bins 2D ######################################################################

def PlotBins2D(cases_ordenados, num_ordenados, levels, cmap,
               color_thr, title, save=False, name_fig='fig', out_dir='~/',
               dpi=100, bin_limits=None):
    cases_ordenados = np.flip(cases_ordenados)
    num_ordenados = np.flip(num_ordenados)

    cmap = plt.get_cmap(cmap)  # debe ser objeto colormap no solo str

    fig = plt.figure(dpi=dpi, figsize=(8, 7))
    ax = fig.add_subplot(111)

    norm = BoundaryNorm(levels, cmap.N, clip=True)

    im = ax.imshow(cases_ordenados, cmap=cmap, norm=norm)

    for i in range(0, len(bin_limits)):
        for j in range(0, len(bin_limits)):
            if np.abs(cases_ordenados[i, j]) > color_thr:
                color_num = 'white'
            else:
                color_num = 'k'
            if (~np.isnan(num_ordenados[i, j])) and (num_ordenados[i, j] != 0):
                ax.text(j, i, num_ordenados[i, j].astype(np.int64),
                        ha='center', va='center', color=color_num)

    xylimits = [-.5, -.5 + len(bin_limits)]
    ax.set_xlim(xylimits[::-1])
    ax.set_ylim(xylimits)

    original_ticks = np.arange(-.5, -.5 + len(bin_limits) + 0.5)
    new_tickx = np.unique(bin_limits)
    ax.set_xticks(original_ticks, new_tickx)
    ax.set_yticks(original_ticks, new_tickx)

    ax.set_ylabel('Niño3.4 - SST index (of std)', fontsize=11)
    ax.set_xlabel('DMI - SST index (of std)', fontsize=11)
    fig.suptitle(title, size=12)

    inf_neutro_border = original_ticks[
                            int(np.floor(len(original_ticks) / 2))] - 1
    upp_neutro_border = original_ticks[int(np.ceil(len(original_ticks) / 2))]
    plt.axhline(y=inf_neutro_border, color='k', linestyle='-', linewidth=2)
    plt.axhline(y=upp_neutro_border, color='k', linestyle='-', linewidth=2)
    plt.axvline(x=inf_neutro_border, color='k', linestyle='-', linewidth=2)
    plt.axvline(x=upp_neutro_border, color='k', linestyle='-', linewidth=2)

    ax.margins(0)
    ax.grid(which='major', alpha=0.5, color='k')
    plt.colorbar(im, ticks=levels,
                 fraction=0.046, pad=0.04,
                 boundaries=levels)

    plt.tight_layout()
    if save:
        plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


################################################################################
# SetBinsByCases ###############################################################
def SetBinsByCases(indices, magnitudes, bin_limits, cases):

    mt_dim = (len(magnitudes) * 2 + 1)

    matriz_base = np.full((mt_dim, mt_dim), None, dtype=object)

    columna_fila_centro = {}
    for i, u in zip(indices, ['columna_centro', 'fila_centro']):
        aux_names = []

        for signo in ['n', 'p']:
            if signo == 'p':
                aux_magnitudes = magnitudes[::-1]
            else:
                aux_magnitudes = magnitudes

            for m in aux_magnitudes:
                aux_names.append(f'{m}_{i}{signo}')
            if signo == 'n':
                aux_names.append('clim')

        columna_fila_centro[u] = aux_names

    matriz_base[:, mt_dim // 2] = columna_fila_centro['columna_centro'][::-1]
    matriz_base[mt_dim // 2, :] = columna_fila_centro['fila_centro']

    # combinaciones
    for i in range(mt_dim):
        if i != mt_dim // 2:
            for j in range(mt_dim):
                if j != mt_dim // 2:
                    matriz_base[i, j] = \
                        (f'{matriz_base[:, mt_dim // 2][i]}-'
                         f'{matriz_base[mt_dim // 2, :][j]}')

    cases_magnitude = matriz_base.flatten().tolist()

    bins_limits_pos = []
    bins_limits_neg = []
    bins_limits_neutro = []
    for bl_count, bl in enumerate(bin_limits):
        if sum(bl) > 0:
            bins_limits_pos.append(bl_count)
        elif sum(bl) < 0:
            bins_limits_neg.append(bl_count)
        elif sum(bl) == 0:
            bins_limits_neutro.append(bl_count)

    bins_by_cases_indice1 = []
    bins_by_cases_indice2 = []
    for c in cases:
        check_pos_neg = True

        if (indices[0] in c) and ('puros' in c):
            bins_by_cases_indice2.append(bins_limits_neutro)
        if (indices[1] in c) and ('puros' in c):
            bins_by_cases_indice1.append(bins_limits_neutro)

        if 'neutros' in c:
            bins_by_cases_indice1.append(bins_limits_neutro)
            bins_by_cases_indice2.append(bins_limits_neutro)

        if (f'{indices[0]}_pos' in c) and (f'{indices[1]}_neg' in c):
            bins_by_cases_indice1.append(bins_limits_pos)
            bins_by_cases_indice2.append(bins_limits_neg)
            check_pos_neg = False
        elif (f'{indices[0]}_neg' in c) and (f'{indices[1]}_pos' in c):
            bins_by_cases_indice1.append(bins_limits_neg)
            bins_by_cases_indice2.append(bins_limits_pos)
            check_pos_neg = False

        if (indices[0] in c) and ('pos' in c) and check_pos_neg:
            bins_by_cases_indice1.append(bins_limits_pos)
        elif (indices[0] in c) and ('neg' in c) and check_pos_neg:
            bins_by_cases_indice1.append(bins_limits_neg)

        if (indices[1] in c) and ('pos' in c) and check_pos_neg:
            bins_by_cases_indice2.append(bins_limits_pos)
        elif (indices[1] in c) and ('neg' in c) and check_pos_neg:
            bins_by_cases_indice2.append(bins_limits_neg)

        if ('sim' in c) and ('pos' in c):
            bins_by_cases_indice1.append(bins_limits_pos)
            bins_by_cases_indice2.append(bins_limits_pos)
        elif ('sim' in c) and ('neg' in c):
            bins_by_cases_indice1.append(bins_limits_neg)
            bins_by_cases_indice2.append(bins_limits_neg)

    bin_names = magnitudes + [''] + magnitudes[::-1]

    cases_names = []
    for c_count, c in enumerate(cases):
        aux_h = '-'
        for i1 in bins_by_cases_indice1[c_count]:
            i1_aux = sum(bin_limits[i1])
            i1_aux_mag_name = bin_names[i1]
            i1_aux_h = '_'
            if i1_aux > 0:
                i1_aux_name = indices[0] + 'p'
            elif i1_aux < 0:
                i1_aux_name = indices[0] + 'n'
            elif i1_aux == 0:
                i1_aux_name = ''
                i1_aux_mag_name = ''
                i1_aux_h = ''
                aux_h = ''

            i1_name = f"{i1_aux_mag_name}{i1_aux_h}{i1_aux_name}"

            for i2 in bins_by_cases_indice2[c_count]:
                i2_aux = sum(bin_limits[i2])
                i2_aux_mag_name = bin_names[i2]
                i2_aux_h = '_'

                if i2_aux > 0:
                    i2_aux_name = indices[1] + 'p'
                elif i2_aux < 0:
                    i2_aux_name = indices[1] + 'n'
                elif i2_aux == 0:
                    i2_aux_name = ''
                    i2_aux_mag_name = ''
                    i2_aux_h = ''
                    aux_h = ''

                i2_name = f"{i2_aux_mag_name}{i2_aux_h}{i2_aux_name}"
                case_name = f"{i1_name}{aux_h}{i2_name}"
                cases_names.append(case_name)

    return cases_names, cases_magnitude, \
        bins_by_cases_indice1, bins_by_cases_indice2



def SameDateAs(data, datadate):
    """
    En data selecciona las mismas fechas que datadate
    :param data:
    :param datadate:
    :return:
    """
    return data.sel(time=datadate.time.values)



def PlotBins2DTwoVariables(data_bins, num_bins, bin_limits, num_cols,
                           variable_v1, variable_v2, levels_v1, cmap_v1,
                           levels_v2, cmap_v2, color_thr_v1, color_thr_v2,
                           title, save, name_fig, out_dir, dpi,
                           high=3.5, width=11, pdf=True):

    num_plots = len(data_bins)
    num_rows = np.ceil(num_plots / num_cols).astype(int)
    change_row = len(data_bins)/2

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    for ax, plot in zip(axes.flatten(), range(0, num_plots)):

        if plot < change_row:
            levels = levels_v1
            cmap = cmap_v1
            color_thr = color_thr_v1
            row1=True
            ylabel = variable_v1
        else:
            levels = levels_v2
            cmap = cmap_v2
            color_thr = color_thr_v2
            row1 = False
            ylabel = variable_v2

        data_sel = data_bins[plot]
        if data_sel is None:
            to_plot = False
        else:
            to_plot = True

        num_bins_sel = num_bins[plot]
        cmap = plt.get_cmap(cmap)

        norm = BoundaryNorm(levels, cmap.N, clip=True)

        if to_plot is True:
            if row1 is True:
                im_r1 = ax.imshow(data_sel, cmap=cmap, norm=norm)
            else:
                im_r2 = ax.imshow(data_sel, cmap=cmap, norm=norm)

            for i in range(0, len(bin_limits)):
                for j in range(0, len(bin_limits)):
                    if np.abs(data_sel[i, j]) > color_thr:
                        color_num = 'white'
                    else:
                        color_num = 'k'
                    if (~np.isnan(num_bins_sel[i, j])) and (
                            num_bins_sel[i, j] != 0):
                        ax.text(j, i, num_bins_sel[i, j].astype(np.int64),
                                ha='center', va='center', color=color_num,
                                size=9)

            ax.set_title(title[plot])
            ax.text(-0.01, 1.025, f"({string.ascii_lowercase[plot]})",
                    transform=ax.transAxes, size=9)

            xylimits = [-.5, -.5 + len(bin_limits)]
            ax.set_xlim(xylimits[::-1])
            ax.set_ylim(xylimits)

            original_ticks = np.arange(-.5, -.5 + len(bin_limits) + 0.5)
            new_tickx = np.unique(bin_limits)
            ax.set_xticks(original_ticks, new_tickx)
            ax.set_yticks(original_ticks, new_tickx)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=8)

            inf_neutro_border = original_ticks[
                                    int(np.floor(len(original_ticks) / 2))] - 1
            upp_neutro_border = original_ticks[
                int(np.ceil(len(original_ticks) / 2))]
            ax.axhline(y=inf_neutro_border, color='k', linestyle='-',
                       linewidth=2)
            ax.axhline(y=upp_neutro_border, color='k', linestyle='-',
                       linewidth=2)
            ax.axvline(x=inf_neutro_border, color='k', linestyle='-',
                       linewidth=2)
            ax.axvline(x=upp_neutro_border, color='k', linestyle='-',
                       linewidth=2)

            ax.margins(0)
            ax.grid(which='major', alpha=0.5, color='k')
            if plot == 0 or plot == change_row:
                ax.set_ylabel(ylabel)
                first = False
        else:
            ax.axis('off')


    pos1 = fig.add_axes([0.93, 0.57, 0.015, 0.35])
    cb1 = fig.colorbar(im_r1, cax=pos1, orientation='vertical')
    cb1.ax.tick_params(labelsize=8, pad=1)

    pos2 = fig.add_axes([0.93, 0.1, 0.015, 0.35])
    cb2 = fig.colorbar(im_r2, cax=pos2, orientation='vertical')
    cb2.ax.tick_params(labelsize=8, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0.05, hspace=0.55, left=0.075,
                        right=0.90, top=0.95)

    fig.supylabel('Niño3.4 - SST index (of std)', fontsize=11)
    fig.supxlabel('DMI - SST index (of std)', fontsize=11)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{name_fig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
def spearman_correlation(da_field, da_series):

    def spearman_func(x, y):
        # Si hay valores NaN, se ignoran ambos pares
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3: # se necesitan al menos 3 pares para  Spearman
            return np.nan
        return spearmanr(x[mask], y[mask])[0]

    result = xr.apply_ufunc(
        spearman_func,
        da_field,
        da_series,
        input_core_dims=[["sample"], ["sample"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return result

################################################################################
################################################################################

# def ComputeFieldsByCases(v, v_name, fix_factor, snr,
#                          levels_main, cbar_main, levels_clim, cbar_clim,
#                          title_var, name_fig, dpi,
#                          cases, bin_limits, bins_by_cases_dmi, bins_by_cases_n34,
#                          cases_dir, dates_dir,
#                          figsize=[16, 17], usemask=True, hcolorbar=False, save=True,
#                          proj='eq', obsdates=False, out_dir='~/'):
#     # no, una genialidad... -------------------------------------------------------------------------------------------#
#     sec_plot = [13, 14, 10, 11,
#                 7, 2, 22, 17,
#                 8, 3, 9, 4,
#                 20, 15, 21, 16,
#                 5, 0, 6, 1,
#                 23, 18, 24, 19]
#     row_titles = ['Strong El Niño', None, None, None, None,
#                   'Moderate El Niño', None, None, None, None,
#                   'Neutro ENSO', None, None, None, None,
#                   'Moderate La Niña', None, None, None, None,
#                   'Strong La Niña', None, None, None, None]
#     col_titles = ['Strong IOD - ', 'Moderate IOD - ', 'Neutro IOD', 'Moderate IOD + ', 'Strong IOD + ']
#     num_neutros = [483, 585, 676, 673]
#     porcentaje = 0.1
#     # ------------------------------------------------------------------------------------------------------------------#
#     print('Only SON')
#     print('No climatology')
#     mm = 10
#     for s in ['SON']:
#         n_check = []
#         sec_count = 0
#         # esto no tiene sentido
#         # comp_case_clim = DetrendClim(data, mm, v_name=v_name)
#
#         crs_latlon = ccrs.PlateCarree()
#         if proj == 'eq':
#             fig, axs = plt.subplots(nrows=5, ncols=5,
#                                     subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
#                                     figsize=(figsize[0], figsize[1]))
#         else:
#             fig, axs = plt.subplots(nrows=5, ncols=5,
#                                     subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=200)},
#                                     figsize=(figsize[0], figsize[1]))
#         axs = axs.flatten()
#         # Loop en los cases -{neutro} ---------------------------------------------------------------------------------#
#         for c_count in [0, 1, 2, 3, 4, 5, 6, 7]:  # , 8]:
#             cases_bin, num_bin, aux = BinsByCases(v=v, v_name=v_name, fix_factor=fix_factor,
#                                                   s=s, mm=mm, c=cases[c_count], c_count=c_count,
#                                                   bin_limits=bin_limits, bins_by_cases_dmi=bins_by_cases_dmi,
#                                                   bins_by_cases_n34=bins_by_cases_n34, snr=snr,
#                                                   cases_dir=cases_dir, dates_dir=dates_dir, obsdates=obsdates)
#
#             bins_aux_dmi = bins_by_cases_dmi[c_count]
#             bins_aux_n34 = bins_by_cases_n34[c_count]
#             for b_dmi in range(0, len(bins_aux_dmi)):
#                 for b_n34 in range(0, len(bins_aux_n34)):
#                     n = sec_plot[sec_count]
#                     if proj != 'eq':
#                         axs[n].set_extent([0, 360, -80, 20],
#                                           ccrs.PlateCarree(central_longitude=200))
#                     comp_case = cases_bin[b_dmi][b_n34]
#
#
#                     # if v == 'prec' and s == 'JJA':
#                     #
#                     #     mask2 = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(comp_case_clim)
#                     #     mask2 = xr.where(np.isnan(mask2), mask2, 1)
#                     #     mask2 = mask2.to_dataset(name='prec')
#                     #
#                     #     dry_season_mask = comp_case_clim.where(comp_case_clim.prec>30)
#                     #     dry_season_mask = xr.where(np.isnan(dry_season_mask), dry_season_mask, 1)
#                     #     dry_season_mask *= mask2
#                     #
#                     #     if snr:
#                     #         comp_case['var'] *= dry_season_mask.prec
#                     #     else:
#                     #         comp_case *= dry_season_mask.prec.values
#
#                     if usemask:
#                         mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(aux)
#                         mask = xr.where(np.isnan(mask), mask, 1)
#                         comp_case *= mask
#                     if snr:
#                         comp_case = comp_case['var']
#
#                     if num_bin[b_dmi][b_n34] > num_neutros[mm - 7] * porcentaje:
#                         im = axs[n].contourf(aux.lon, aux.lat, comp_case,
#                                              levels=levels_main, transform=crs_latlon,
#                                              cmap=cbar_main, extend='both')
#
#                         levels_contour = levels_main.copy()
#                         if isinstance(levels_main, np.ndarray):
#                             levels_contour = levels_main[levels_main != 0]
#                         else:
#                             levels_contour.remove(0)
#
#                         axs[n].contour(aux.lon, aux.lat, comp_case,
#                                         levels=levels_contour,
#                                         transform=crs_latlon,
#                                         colors='k', linewidths=1)
#
#                         axs[n].add_feature(cartopy.feature.LAND, facecolor='lightgrey')
#                         axs[n].add_feature(cartopy.feature.COASTLINE)
#                         if proj == 'eq':
#                             axs[n].gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', color='gray')
#                             axs[n].set_xticks([])
#                             axs[n].set_yticks([])
#                             axs[n].set_extent([270,330,-60,20])
#                             # axs[n].set_xticks(x_lon, crs=crs_latlon)
#                             # axs[n].set_yticks(x_lat, crs=crs_latlon)
#                             # lon_formatter = LongitudeFormatter(zero_direction_label=True)
#                             # lat_formatter = LatitudeFormatter()
#                             # axs[n].xaxis.set_major_formatter(lon_formatter)
#                             # axs[n].yaxis.set_major_formatter(lat_formatter)
#                         else:
#                             # polar
#                             gls = axs[n].gridlines(draw_labels=True, crs=crs_latlon, lw=0.3, color="gray",
#                                                    y_inline=True, xlocs=range(-180, 180, 30),
#                                                    ylocs=np.arange(-80, 20, 20))
#                             r_extent = 1.2e7
#                             axs[n].set_xlim(-r_extent, r_extent)
#                             axs[n].set_ylim(-r_extent, r_extent)
#                             circle_path = mpath.Path.unit_circle()
#                             circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
#                                                      circle_path.codes.copy())
#                             axs[n].set_boundary(circle_path)
#                             axs[n].set_frame_on(False)
#                             plt.draw()
#                             for ea in gls._labels:
#                                 pos = ea[2].get_position()
#                                 if (pos[0] == 150):
#                                     ea[2].set_position([0, pos[1]])
#
#                         axs[n].tick_params(labelsize=0)
#
#                         if n == 0 or n == 1 or n == 2 or n == 3 or n == 4:
#                             axs[n].set_title(col_titles[n], fontsize=15)
#
#                         if n == 0 or n == 5 or n == 10 or n == 15 or n == 20:
#                             axs[n].set_ylabel(row_titles[n], fontsize=15)
#
#                         axs[n].xaxis.set_label_position('top')
#                         axs[n].set_xlabel('N=' + str(num_bin[b_dmi][b_n34]), fontsize=12, loc='left', fontweight="bold")
#
#                     else:
#                         n_check.append(n)
#                         axs[n].axis('off')
#
#                     sec_count += 1
#
#         # subtitulos columnas de no ploteados -------------------------------------------------------------------------#
#         for n_aux in [0, 1, 2, 3, 4]:
#             if n_aux in n_check:
#                 if n_aux + 5 in n_check:
#                     axs[n_aux + 10].set_title(col_titles[n_aux], fontsize=15)
#                 else:
#                     axs[n_aux + 5].set_title(col_titles[n_aux], fontsize=15)
#
#         for n_aux in [0, 5, 10, 15, 20]:
#             if n_aux in n_check:
#                 if n_aux + 1 in n_check:
#                     axs[n_aux + 2].set_ylabel(row_titles[n_aux], fontsize=15)
#                 else:
#                     axs[n_aux + 1].set_ylabel(row_titles[n_aux], fontsize=15)
#
#         # Climatologia = NADA en HGT ----------------------------------------------------------------------------------#
#         # en el lugar del neutro -> climatología de la variable (data)
#
#         # if usemask:
#         #     comp_case_clim = comp_case_clim[v_name] * fix_factor * mask
#         # else:
#         #     comp_case_clim = comp_case_clim[v_name] * fix_factor
#
#         # if v_name=='hgt':
#         #     comp_case_clim = 0
#
#         aux0 = aux.sel(r=1, time='1982-10-01').drop(['r', 'L', 'time'])
#         im2 = axs[12].contourf(aux.lon, aux.lat, aux0['var'][0, :, :],
#                                levels=levels_clim, transform=crs_latlon, cmap=cbar_clim, extend='max')
#         axs[12].set_extent([270, 330, -60, 20])
#         # --------------------------------------------------------------------------------------------------------------#
#         axs[12].add_feature(cartopy.feature.LAND, facecolor='grey')
#         axs[12].add_feature(cartopy.feature.COASTLINE)
#
#         if v_name != 'hgt':
#             cb = plt.colorbar(im2, fraction=0.042, pad=0.032, shrink=1, ax=axs[12], aspect=20)
#             cb.ax.tick_params(labelsize=5)
#
#         if proj == 'eq':
#             axs[12].gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', color='gray')
#             axs[12].set_xticks([])
#             axs[12].set_yticks([])
#         else:
#             # polar
#             gls = axs[12].gridlines(draw_labels=True, crs=crs_latlon, lw=0.3, color="gray",
#                                     y_inline=True, xlocs=range(-180, 180, 30), ylocs=np.arange(-80, 0, 20))
#             r_extent = 1.2e7
#             axs[12].set_xlim(-r_extent, r_extent)
#             axs[12].set_ylim(-r_extent, r_extent)
#             circle_path = mpath.Path.unit_circle()
#             circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
#                                      circle_path.codes.copy())
#             axs[12].set_boundary(circle_path)
#             axs[12].set_frame_on(False)
#             axs[12].tick_params(labelsize=0)
#             plt.draw()
#             for ea in gls._labels:
#                 pos = ea[2].get_position()
#                 if (pos[0] == 150):
#                     ea[2].set_position([0, pos[1]])
#
#         if hcolorbar:
#             pos = fig.add_axes([0.2, 0.05, 0.6, 0.01])
#             cbar = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
#         else:
#             pos = fig.add_axes([0.90, 0.2, 0.012, 0.6])
#             cbar = fig.colorbar(im, cax=pos, pad=0.1)
#
#         cbar.ax.tick_params(labelsize=18)
#         if snr:
#             fig.suptitle('Signal-to-Noise ratio:' + title_var + ' - ' + s, fontsize=20)
#         else:
#             fig.suptitle(title_var + ' - ' + s, fontsize=20)
#         # fig.tight_layout() #BUG matplotlib 3.5.0 #Solucionado definitivamnete en 3.6 ?
#         if hcolorbar:
#             fig.subplots_adjust(top=0.93, bottom=0.07)
#         else:
#             fig.subplots_adjust(top=0.93)
#         if save:
#             if snr:
#                 plt.savefig(out_dir + 'SNR_' + name_fig + '_' + s + '.jpg', bbox_inches='tight', dpi=dpi)
#             else:
#                 plt.savefig(out_dir + name_fig + '_' + s + '.jpg', bbox_inches='tight', dpi=dpi)
#
#             plt.close('all')
#         else:
#             plt.show()
#         mm += 1
################################################################################
################################################################################