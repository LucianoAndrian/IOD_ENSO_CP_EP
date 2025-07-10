
import xarray as xr
import numpy as np

index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'


sd_dmi_s = xr.open_dataset(index_dir + 'DMI_SON_Leads_r_CFSv2.nc').std()
sd_ep_s = xr.open_dataset(index_dir + 'EP_SON_Leads_r_CFSv2.nc').std()
sd_cp_s = xr.open_dataset(index_dir + 'CP_SON_Leads_r_CFSv2.nc').std()

def SelectParIndex(case, idx1_name, idx2_name,
                   idx1_sd, idx2_sd, by_r=False,
                   open_idx1=False, open_idx2=False):

    case = case.replace("CFSv2_", "")
    cases_dir = '/pikachu/datos/luciano.andrian/cases_fields_EP_CP/'
    index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

    try:
        aux = xr.open_dataset(f'{cases_dir}{idx1_name}_{case}').__mul__(1/idx1_sd)
        aux_var_name = list(aux.data_vars)[0]
        aux2 = xr.open_dataset(f'{cases_dir}{idx2_name}_{case}').__mul__(1/idx2_sd)
        aux2_var_name = list(aux.data_vars)[0]
        check = True
    except:
        check = False

    if check is True:
        if by_r:
            if open_idx1:
                idx1 = (xr.open_dataset(
                    f'{index_dir}{idx1_name}_SON_Leads_r_CFSv2.nc')
                        .__mul__(1 / idx1_sd))

                aux3 = idx1.sel(r=aux.r, time=aux.time)
                aux3_var_name = list(aux.data_vars)[0]

                if len(np.where(aux3.L.values == aux.L.values)[0]):
                    return aux[aux_var_name].values.round(2), \
                        aux3[aux3_var_name].values.round(2)
                else:
                    print('Error: CASES')
                    return [], []

            if open_idx2:
                idx2 = (xr.open_dataset(
                    f'{index_dir}{idx2_name}_SON_Leads_r_CFSv2.nc')
                        .__mul__(1 / idx2_sd))

                aux3 = idx2.sel(r=aux.r, time=aux.time)
                aux3_var_name = list(aux.data_vars)[0]

                if len(np.where(aux3.L.values == aux.L.values)[0]):
                    return aux[aux_var_name].values.round(2), \
                        aux3[aux3_var_name].values.round(2)
                else:
                    print('Error: CASES')
                    return [], []
        else:
            aux2 = aux2.sel(time=aux2.time.isin([aux.time.values]))

            if len(aux2.time) == len(aux.time):
                return aux[aux_var_name].values.round(2), \
                    aux2[aux2_var_name].values.round(2)
            else:
                print('Error: CASES')
                return [], []

    else:
        return [], []

def PlotScatter(idx1_name, idx2_name, idx1_sd, idx2_sd, save=False, out_dir=''):
    idx1_name = idx1_name.upper()
    idx1 = idx1_name.lower()
    idx2_name = idx2_name.upper()
    idx2 = idx2_name.lower()

    case = f'CFSv2_neutros_SON.nc'
    idx1_neutros, idx2_neutros = SelectParIndex(case=case,
                                          idx1_name=idx1_name,
                                          idx2_name=idx2_name,
                                          idx1_sd=idx1_sd, idx2_sd=idx2_sd,
                                          by_r=False,
                                          open_idx1=False, open_idx2=False)

    case = f'CFSv2_simultaneos_dobles_{idx1}_{idx2}_pos_SON.nc'
    idx1_sim_pos, idx2_sim_pos = SelectParIndex(case=case,
                                             idx1_name=idx1_name,
                                             idx2_name=idx2_name,
                                             idx1_sd=idx1_sd, idx2_sd=idx2_sd,
                                             by_r=False,
                                             open_idx1=False, open_idx2=False)

    case = f'CFSv2_simultaneos_dobles_{idx1}_{idx2}_neg_SON.nc'
    idx1_sim_neg, idx2_sim_neg = SelectParIndex(case=case,
                                             idx1_name=idx1_name,
                                             idx2_name=idx2_name,
                                             idx1_sd=idx1_sd, idx2_sd=idx2_sd,
                                             by_r=False,
                                             open_idx1=False, open_idx2=False)

    case = f'CFSv2_simultaneos_dobles_op_{idx1}_pos_{idx2}_neg_SON.nc'
    idx1_pos_idx2_neg, idx2_in_idx1_pos_idx2_neg = SelectParIndex(case=case,
                                             idx1_name=idx1_name,
                                             idx2_name=idx2_name,
                                             idx1_sd=idx1_sd, idx2_sd=idx2_sd,
                                             by_r=False,
                                             open_idx1=False, open_idx2=False)

    case = f'CFSv2_simultaneos_dobles_op_{idx1}_neg_{idx2}_pos_SON.nc'
    idx1_neg_idx2_pos, idx2_in_idx1_neg_idx2_pos = SelectParIndex(case=case,
                                             idx1_name=idx1_name,
                                             idx2_name=idx2_name,
                                             idx1_sd=idx1_sd, idx2_sd=idx2_sd,
                                             by_r=False,
                                             open_idx1=False, open_idx2=False)

    case = f'CFSv2_puros_{idx1}_neg_SON.nc'
    idx1_puros_neg, idx2_in_idx1_puros_neg = SelectParIndex(case=case,
                                                        idx1_name=idx1_name,
                                                        idx2_name=idx2_name,
                                                        idx1_sd=idx1_sd,
                                                        idx2_sd=idx2_sd,
                                                        by_r=False,
                                                        open_idx1=False,
                                                        open_idx2=False)

    case = f'CFSv2_puros_{idx1}_pos_SON.nc'
    idx1_puros_pos, idx2_in_idx1_puros_pos = SelectParIndex(case=case,
                                                        idx1_name=idx1_name,
                                                        idx2_name=idx2_name,
                                                        idx1_sd=idx1_sd,
                                                        idx2_sd=idx2_sd,
                                                        by_r=False,
                                                        open_idx1=False,
                                                        open_idx2=False)

    case = f'CFSv2_puros_{idx2}_pos_SON.nc'
    idx2_puros_pos, idx1_in_idx2_puros_pos = SelectParIndex(case=case,
                                                       idx1_name=idx2_name,
                                                       idx2_name=idx1_name,
                                                       idx1_sd=idx2_sd,
                                                       idx2_sd=idx1_sd,
                                                       by_r=False,
                                                       open_idx1=False,
                                                       open_idx2=False)

    case = f'CFSv2_puros_{idx2}_neg_SON.nc'
    idx2_puros_neg, idx1_in_idx2_puros_neg = SelectParIndex(case=case,
                                                       idx1_name=idx2_name,
                                                       idx2_name=idx1_name,
                                                       idx1_sd=idx2_sd,
                                                       idx2_sd=idx1_sd,
                                                       by_r=False,
                                                       open_idx1=False,
                                                       open_idx2=False)

    case = f'CFSv2_puros_{idx2}_neg_SON.nc'
    idx2_puros_neg, idx1_in_idx2_puros_neg = SelectParIndex(case=case,
                                                       idx1_name=idx2_name,
                                                       idx2_name=idx1_name,
                                                       idx1_sd=idx2_sd,
                                                       idx2_sd=idx1_sd,
                                                       by_r=False,
                                                       open_idx1=False,
                                                       open_idx2=False)


    # case = f'CFSv2_todo_{idx1}_pos_SON.nc'
    # idx1_todo_pos, idx2_in_idx1_todo_pos = SelectParIndex(case=case,
    #                                      idx1_name=idx1_name,
    #                                      idx2_name=idx2_name,
    #                                      idx1_sd=idx2_sd, idx2_sd=idx1_sd,
    #                                      by_r=False,
    #                                      open_idx1=False,
    #                                      open_idx2=False)
    #
    # case = f'CFSv2_todo_{idx1}_neg_SON.nc'
    # idx1_todo_neg, idx2_in_idx1_todo_neg = SelectParIndex(case=case,
    #                                      idx1_name=idx1_name,
    #                                      idx2_name=idx2_name,
    #                                      idx1_sd=idx2_sd, idx2_sd=idx1_sd,
    #                                      by_r=False,
    #                                      open_idx1=False,
    #                                      open_idx2=False)


    import matplotlib.pyplot as plt
    dpi = 100
    in_label_size = 13
    label_legend_size = 12
    tick_label_size = 11
    scatter_size_fix = 3
    fig, ax = plt.subplots(dpi=dpi, figsize=(7.08661, 7.08661))

    # todos
    ax.scatter(x=idx1_neutros, y=idx2_neutros, marker='o',
               label=f'{idx1_name} vs. {idx2_name}',
               s=30 * scatter_size_fix, edgecolor='k', color='dimgray', alpha=1)
    # dmi puros
    ax.scatter(x=idx1_puros_pos, y=idx2_in_idx1_puros_pos, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#8B1E1E',
               alpha=1, label=f'{idx1_name}+')
    ax.scatter(x=idx1_puros_neg, y=idx2_in_idx1_puros_neg, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#7CCD73',
               alpha=1, label=f'{idx1_name}-')

    # ep puros
    ax.scatter(x=idx1_in_idx2_puros_pos, y=idx2_puros_pos, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='navy', alpha=1,
               label=f'{idx2_name}+')
    ax.scatter(x=idx1_in_idx2_puros_neg, y=idx2_puros_neg, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#DE00FF',
               alpha=1, label=f'{idx2_name}-')

    # sim
    ax.scatter(x=idx1_sim_pos, y=idx2_sim_pos, marker='o',
               s=30 * scatter_size_fix,
               edgecolor='k', color='#FF5B12', alpha=1,
               label=f'{idx1_name}+ & {idx2_name}+')
    ax.scatter(x=idx1_sim_neg, y=idx2_sim_neg, marker='o',
               s=30 * scatter_size_fix,
               edgecolor='k', color='#63A6FF', alpha=1,
               label=f'{idx1_name}- & {idx2_name}-')

    # sim opp. sing
    ax.scatter(x=idx1_pos_idx2_neg, y=idx2_in_idx1_pos_idx2_neg, marker='o',
               s=30*scatter_size_fix, edgecolor='k', color='#FF9232', alpha=1,
               label=f'{idx1_name}+ & {idx2_name}-')
    ax.scatter(x=idx1_neg_idx2_pos, y=idx2_in_idx1_neg_idx2_pos, marker='o',
                s=30*scatter_size_fix, edgecolor='k', color='gold', alpha=1,
               label=f'{idx1_name}- & {idx2_name}+')

    # # tdo
    # ax.scatter(x=idx1_todo_pos, y=idx2_in_idx1_todo_pos, marker='x',
    #            label=f'{idx1_name} vs. {idx2_name} todo',
    #            s=10 * scatter_size_fix, edgecolor='k', color='k', alpha=1)
    #
    # ax.scatter(x=idx1_todo_neg, y=idx2_in_idx1_todo_neg, marker='x',
    #            label=f'{idx1_name} vs. {idx2_name} todo',
    #            s=10 * scatter_size_fix, edgecolor='k', color='k', alpha=1)

    ax.legend(loc=(.01, .57), fontsize=label_legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, pad=1)
    ax.set_ylim((-5, 5))
    ax.set_xlim((-5, 5))
    ax.axhspan(-.5 , .5,#/ idx1_sd.sst.values, .5 / idx1_sd.sst.values,
               alpha=0.2, color='black', zorder=0)
    ax.axvspan(-.5 , .5,#/ idx2_sd.sst.values, .5 / idx2_sd.sst.values,
               alpha=0.2, color='black', zorder=0)
    ax.set_xlabel(f'{idx1_name}', size=in_label_size)
    ax.set_ylabel(f'{idx2_name}', size=in_label_size)
    ax.text(-4.9, 4.6, f'{idx2_name}+/{idx1_name}-', dict(size=in_label_size))
    ax.text(-.2, 4.6,  f'{idx2_name}+', dict(size=in_label_size))
    ax.text(+3.7, 4.6, f'{idx2_name}+/{idx1_name}+', dict(size=in_label_size))
    ax.text(+4.2, -.1,  f'{idx1_name}+', dict(size=in_label_size))
    ax.text(+3.7, -4.9, f'{idx2_name}-/{idx1_name}+', dict(size=in_label_size))
    ax.text(-.2, -4.9, f'{idx2_name}-', dict(size=in_label_size))
    ax.text(-4.9, -4.9, f'{idx2_name}-/{idx1_name}-', dict(size=in_label_size))
    ax.text(-4.9, -.1, f'{idx1_name}-', dict(size=in_label_size))
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig(out_dir + 'f08.pdf', dpi=dpi, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


PlotScatter(idx1_name='DMI', idx2_name='EP',
            idx1_sd=sd_dmi_s, idx2_sd= sd_ep_s, save=False, out_dir='')

PlotScatter(idx1_name='DMI', idx2_name='CP',
            idx1_sd=sd_dmi_s, idx2_sd= sd_cp_s, save=False, out_dir='')

PlotScatter(idx1_name='EP', idx2_name='CP',
            idx1_sd=sd_ep_s, idx2_sd= sd_cp_s, save=False, out_dir='')