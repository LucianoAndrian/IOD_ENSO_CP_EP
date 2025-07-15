"""
Calculo de los indices EP y CP en CFSv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
save = True

# ---------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
from eofs.xarray import Eof
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
def Compute(data, modeL):

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
        #var_per = np.around(solver.varianceFraction(neigs=2).values * 100, 1)

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
                pc1_f_r = pc1_l_r#.drop('L')
                pc2_f_r = pc2_l_r#.drop('L')
                eof_f_r = eof_l_r

                pc1_f_em = pc1_l_em#.drop('L')
                pc2_f_em = pc2_l_em#.drop('L')
                eof_f_em = eof_l_em

            else:
                # pc1_f_r = xr.concat([pc1_f_r, pc1_l_r.drop('L')], dim='time2')
                # pc2_f_r = xr.concat([pc2_f_r, pc2_l_r.drop('L')], dim='time2')
                # eof_f_r = xr.concat([eof_f_r, eof_l_r], dim='L')
                #
                # pc1_f_em = xr.concat([pc1_f_em, pc1_l_em.drop('L')], dim='time')
                # pc2_f_em = xr.concat([pc2_f_em, pc2_l_em.drop('L')], dim='time')
                # eof_f_em = xr.concat([eof_f_em, eof_l_em], dim='L')

                pc1_f_r = xr.concat([pc1_f_r, pc1_l_r], dim='time2')
                pc2_f_r = xr.concat([pc2_f_r, pc2_l_r], dim='time2')
                eof_f_r = xr.concat([eof_f_r, eof_l_r], dim='L')

                pc1_f_em = xr.concat([pc1_f_em, pc1_l_em], dim='time')
                pc2_f_em = xr.concat([pc2_f_em, pc2_l_em], dim='time')
                eof_f_em = xr.concat([eof_f_em, eof_l_em], dim='L')


            print('Done concat')

    pc1_f_r = pc1_f_r.rename({'time2':'time'})
    pc2_f_r = pc2_f_r.rename({'time2':'time'})

    return pc1_f_r, pc2_f_r, eof_f_r, pc1_f_em, pc2_f_em, eof_f_em

def CheckSign(eof, pc1, pc2):

    pc1_f = []
    pc2_f = []
    for l in eof.L.values:
        # try:
        #     print('test0')
        sign_pc1 = np.sign(eof.sel(mode=0, L=l, lon=slice(210, 250)).mean())
        sign_pc2 = np.sign(eof.sel(mode=1, L=l, lon=slice(150, 180)).mean())
        # except:
        #     print('test1')
        #     sign_pc1 = np.sign(eof.sel(mode=0, lon=slice(210, 250))[l,:,:].mean())
        #     sign_pc2 = np.sign(eof.sel(mode=1, lon=slice(150, 180))[l,:,:].mean())

        if sign_pc1 < 0:
            pc1_L = pc1.sel(time=pc1.time.dt.month.isin(10 - l))*sign_pc1
        else:
            pc1_L = pc1.sel(time=pc1.time.dt.month.isin(10 - l))

        if sign_pc2 < 0:
            pc2_L = pc2.sel(time=pc2.time.dt.month.isin(10 - l))*sign_pc2
        else:
            pc2_L = pc2.sel(time=pc2.time.dt.month.isin(10 - l))


        pc1_f.append(pc1_L)
        pc2_f.append(pc2_L)

    pc1_f = xr.concat(pc1_f, 'time')
    pc2_f = xr.concat(pc2_f, 'time')

    return pc1_f, pc2_f

def CheckEvent(sst, cp, ep, year, lead, r):
    month = 10-lead
    if month < 10:
        month = '0'+str(month)

    cp_val = np.round(cp.sel(r=r, time=f'{year}-{month}-01').values, 3)
    ep_val = np.round(ep.sel(r=r, time=f'{year}-{month}-01').values, 3)

    plt.imshow(sst.sel(r=r, time=f'{year}-{month}-01')['sst'][0, :, :])
    plt.title(f'CP:{cp_val}, EP:{ep_val}')
    plt.show()
# ---------------------------------------------------------------------------- #

sst = xr.open_dataset('/pikachu/datos/luciano.andrian/cases_fields/sst_son.nc')

# Takahashi et al. 2011 ------------------------------------------------------ #
sst_sel = sst.sel(lon=slice(110,290), lat=slice(-10,10))
pc1_f_r, pc2_f_r, eof_f_r, _, _, _ = Compute(sst_sel, False)

# En algunos leads el oef puede tener signo cambiado
# Acomodo los signos de cada pc en funcion del EOF
pc1_ch, pc2_ch = CheckSign(eof_f_r, pc1_f_r, pc2_f_r)

cp = (pc1_ch + pc2_ch)/np.sqrt(2)
cp = cp.to_dataset(name='sst')
ep = (pc1_ch - pc2_ch)/np.sqrt(2)
ep = ep.to_dataset(name='sst')

# Tedeschi et al. 2014 ------------------------------------------------------- #
cp_td = sst.sel(lon=slice(160, 210), lat=slice(-5,5)).mean(['lon', 'lat'])
ep_td = sst.sel(lon=slice(220, 270), lat=slice(-5,5)).mean(['lon', 'lat'])

# Sulivan et al. 2016 -------------------------------------------------------- #
n3 = sst.sel(lon=slice(210, 270), lat=slice(-5,5)).mean(['lon', 'lat'])
n4 = sst.sel(lon=slice(200, 210), lat=slice(-5,5)).mean(['lon', 'lat'])

n3 = (n3 - n3.mean('time'))/n3.std('time')
n4 = (n4 - n4.mean('time'))/n4.std('time')

ep_n = n3 - 0.5*n4
cp_n = n4 - 0.5*n3

# ---------------------------------------------------------------------------- #
if save:
    print('Saving...')
    cp.to_netcdf(f'{out_dir}CP_SON_Leads_r_CFSv2.nc')
    ep.to_netcdf(f'{out_dir}EP_SON_Leads_r_CFSv2.nc')

    cp_td.to_netcdf(f'{out_dir}/aux_ep_cp_t/CP_Td_SON_Leads_r_CFSv2.nc')
    ep_td.to_netcdf(f'{out_dir}/aux_ep_cp_t/EP_Td_SON_Leads_r_CFSv2.nc')

    cp_n.to_netcdf(f'{out_dir}/aux_ep_cp_n/CP_n_SON_Leads_r_CFSv2.nc')
    ep_n.to_netcdf(f'{out_dir}/aux_ep_cp_n/EP_n_SON_Leads_r_CFSv2.nc')

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')