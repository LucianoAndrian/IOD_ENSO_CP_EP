"""
Comparacion de indices
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/'
index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# ---------------------------------------------------------------------------- #
import xarray as xr
import matplotlib.pyplot as plt

if save:
    dpi=300
else:
    dpi=300

# ---------------------------------------------------------------------------- #
def PlotScatter(idx1, idx2, idx1_name, idx2_name, save=save,
                out_dir=out_dir,  name_fig='fig', dpi=dpi):

    in_label_size = 13
    label_legend_size = 12
    tick_label_size = 11
    scatter_size_fix = 3
    fig, ax = plt.subplots(dpi=dpi, figsize=(7.08661, 7.08661))

    # todos
    ax.scatter(x=idx1, y=idx2, marker='x',
               s=5 * scatter_size_fix, edgecolor='k', color='k', alpha=1)

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
    plt.title(f'{idx1_name}-{idx2_name}')

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', dpi=dpi, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()

def OpenAndSet(index_dir, name_file):
    index = xr.open_dataset(index_dir + name_file).sst
    index_stk = index.stack(index=('r', 'time'))
    index_stk = index_stk/index_stk.std('index')

    return index_stk
# ---------------------------------------------------------------------------- #
# CP:
cp = OpenAndSet(index_dir, 'CP_SON_Leads_r_CFSv2.nc')
cp_td = OpenAndSet(index_dir, '/aux_ep_cp_t/CP_Td_SON_Leads_r_CFSv2.nc')
cp_n = OpenAndSet(index_dir, '/aux_ep_cp_n/CP_n_SON_Leads_r_CFSv2.nc')

PlotScatter(idx1=cp, idx2=cp_td, idx1_name='CP', idx2_name='CP_Td',
            save=save, out_dir=out_dir,  name_fig='CP_vs_CP_td',
            dpi=dpi)

PlotScatter(idx1=cp, idx2=cp_n, idx1_name='CP', idx2_name='CP_n',
            save=save, out_dir=out_dir,  name_fig='CP_vs_CP_n',
            dpi=dpi)

PlotScatter(idx1=cp_td, idx2=cp_n, idx1_name='CP_td', idx2_name='CP_n',
            save=save, out_dir=out_dir,  name_fig='CP_n_vs_CP_td',
            dpi=dpi)

# ---------------------------------------------------------------------------- #
# EP
ep = OpenAndSet(index_dir, 'EP_SON_Leads_r_CFSv2.nc')
ep_td = OpenAndSet(index_dir, '/aux_ep_cp_t/EP_Td_SON_Leads_r_CFSv2.nc')
ep_n = OpenAndSet(index_dir, '/aux_ep_cp_n/EP_n_SON_Leads_r_CFSv2.nc')

PlotScatter(idx1=ep, idx2=ep_td, idx1_name='EP', idx2_name='EP_Td',
            save=save, out_dir=out_dir,  name_fig='EP_vs_EP_td',
            dpi=dpi)

PlotScatter(idx1=ep, idx2=ep_n, idx1_name='EP', idx2_name='EP_n',
            save=save, out_dir=out_dir,  name_fig='EP_vs_EP_n',
            dpi=dpi)

PlotScatter(idx1=ep_td, idx2=ep_n, idx1_name='EP_td', idx2_name='EP_n',
            save=save, out_dir=out_dir,  name_fig='EP_td_vs_EP_n',
            dpi=dpi)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #