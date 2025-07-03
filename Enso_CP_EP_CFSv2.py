
import numpy as np
import xarray as xr
from eofs.xarray import Eof


def Compute(data, modeL, save=False):

    weights = np.sqrt(np.abs(np.cos(np.radians(data.lat))))
    try:
        name = list(data.data_vars)[0]
    except:
        name = None

    if modeL is True:
        aux = data*weights
        # r y time a a una misma variable para Eof
        aux_st = aux.rename({'time': 'time2'})
        aux_st = aux_st.stack(time=('r', 'time2'))
        aux_st = aux_st.transpose('time', 'lat', 'lon')

        # eof ------------------------------------#
        try:
            if name is None:
                solver = Eof(aux_st)
            else:
                solver = Eof(aux_st[name])

        except ValueError as ve:
            if str(ve) == 'all input data is missing':
                print('campos faltantes')
                aux_st = aux_st.where(~np.isnan(aux_st), drop=True)

                if name is None:
                    solver = Eof(aux_st)
                else:
                    solver = Eof(aux_st[name])

        eof_L_r = solver.eofsAsCovariance(neofs=2)
        pcs = solver.pcs(pcscaling=1)
        var_per = np.around(solver.varianceFraction(neigs=2).values * 100, 1)

        #pcs = pcs.rename({'time2': 'time'})
        pc1_f_r = pcs[:,0]
        pc2_f_r = pcs[:,1]
        eof_f_r = eof_L_r

        pc1_f_em = None
        pc2_f_em = None
        eof_f_em = None

    else:

        for l in [0, 1, 2, 3]:
            print('L:' + str(l))
            # Todos las runs ------------------------------------------------- #

            aux = data.sel(time=data['L'] == l) * weights  # .mean('r')

            # r y time a a una misma variable para Eof
            aux_st = aux.rename({'time': 'time2'})
            aux_st = aux_st.stack(time=('r', 'time2'))
            aux_st = aux_st.transpose('time', 'lat', 'lon')

            # eof ------------------------------------#
            try:
                if name is None:
                    solver = Eof(aux_st)
                else:
                    solver = Eof(xr.DataArray(aux_st[name]))

            except ValueError as ve:
                if str(ve) == 'all input data is missing':
                    print('Lead ' + str(l) + ' con campos faltantes')
                    aux_st = aux_st.where(~np.isnan(aux_st), drop=True)

                    if name is None:
                        solver = Eof(aux_st)
                    else:
                        solver = Eof(aux_st[name])

            eof_l_r = solver.eofsAsCovariance(neofs=2)
            pcs = solver.pcs(pcscaling=1).unstack()

            pc1_l_r = pcs.sel(mode=0)
            pc2_l_r = pcs.sel(mode=1)

            print('Done EOF r')
            del aux_st
            del solver

            # EM ---
            aux = aux.mean('r')

            # eof ------------------------------------#
            try:
                if name is None:
                    solver = Eof(xr.DataArray(aux))
                else:
                    solver = Eof(xr.DataArray(aux[name]))

            except ValueError as ve:
                if str(ve) == 'all input data is missing':
                    print('Lead ' + str(l) + ' con campos faltantes')
                    aux = aux.where(~np.isnan(aux), drop=True)
                    if name is None:
                        solver = Eof(xr.DataArray(aux))
                    else:
                        solver = Eof(xr.DataArray(aux[name]))

            eof_l_em = solver.eofsAsCovariance(neofs=2)
            pcs = solver.pcs(pcscaling=1).unstack()

            pc1_l_em = pcs.sel(mode=0)
            pc2_l_em = pcs.sel(mode=1)

            # para guardar...
            if l == 0:
                pc1_f_r = pc1_l_r.drop('L')
                pc2_f_r = pc2_l_r.drop('L')
                eof_f_r = eof_l_r

                pc1_f_em = pc1_l_em.drop('L')
                pc2_f_em = pc2_l_em.drop('L')
                eof_f_em = eof_l_em

            else:
                pc1_f_r = xr.concat([pc1_f_r, pc1_l_r.drop('L')], dim='time2')
                pc2_f_r = xr.concat([pc2_f_r, pc2_l_r.drop('L')], dim='time2')
                eof_f_r = xr.concat([eof_f_r, eof_l_r], dim='L')

                pc1_f_em = xr.concat([pc1_f_em, pc1_l_em.drop('L')], dim='time')
                pc2_f_em = xr.concat([pc2_f_em, pc2_l_em.drop('L')], dim='time')
                eof_f_em = xr.concat([eof_f_em, eof_l_em], dim='L')

            print('Done concat')

    pc1_f_r.rename({'time2':'time'})
    pc2_f_r.rename({'time2':'time'})

    return pc1_f_r, pc2_f_r, eof_f_r, pc1_f_em, pc2_f_em, eof_f_em

sst = xr.open_dataset('/pikachu/datos/luciano.andrian/cases_fields/sst_son.nc')
sst = sst.sel(lon=slice(110,290), lat=slice(-10,10))

pc1_f_r, pc2_f_r, eof_f_r, pc1_f_em, pc2_f_em, eof_f_em = Compute(sst, False)


