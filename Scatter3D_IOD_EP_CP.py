"""
Scatter plot DMI, EP, CP para exploracion
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/IOD_ENSO_CP_EP/salidas/frames3d/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
import matplotlib.cm as cm
from itertools import product
import numpy as np

# ---------------------------------------------------------------------------- #
dates_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# ---------------------------------------------------------------------------- #

def PlotScatter3D(x, y, z, elev, azim, name_fig, save, out_dir, alpha=0.8,
                  thr=0.5):
    dpi = 150

    # clasificacion por thr
    def encode(v):
        return np.where(v > thr, 1, np.where(v < -thr, -1, 0))

    dx = encode(x)
    dy = encode(y)
    dz = encode(z)

    markers = ['o', '^', 's', 'P', '*', 'X', 'D', 'v', 'h', '<']

    # combinaciones y colores, label, marker
    combos = []
    color_map = {}
    label_map = {}
    marker_map = {}

    base_cmap = cm.get_cmap('hsv', 27)
    combo_colors = [to_hex(base_cmap(i)) for i in range(27)]

    i = 0
    for dx_i, dy_i, dz_i in product([-1, 0, 1], repeat=3):
        label_parts = []
        for val, name in zip([dx_i, dy_i, dz_i], ['DMI', 'EP', 'CP']):
            if val == 1:
                label_parts.append(f'{name}+')
            elif val == -1:
                label_parts.append(f'{name}-')
            else:
                label_parts.append(f'')

        label = ' '.join([p for p in label_parts if p])
        key = (dx_i, dy_i, dz_i)
        combos.append(key)
        color_map[key] = combo_colors[i]
        label_map[key] = label if label else 'Neutral'
        marker_map[key] = markers[i % len(markers)]
        i += 1

    # ------------------------------------------------------------------------ #
    fig = plt.figure(dpi=dpi, figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    ax.plot([-4.5, 4.5], [0, 0], [0, 0], color='gray', linestyle='--', alpha=1)
    ax.plot([0, 0], [-4.5, 4.5], [0, 0], color='gray', linestyle='--', alpha=1)
    ax.plot([0, 0], [0, 0], [-4.5, 4.5], color='gray', linestyle='--', alpha=1)

    ax.set_xlabel('DMI')
    ax.set_ylabel('EP')
    ax.set_zlabel('CP')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_zlim([-4.5, 4.5])

    for k in combos:
        mask = (dx == k[0]) & (dy == k[1]) & (dz == k[2])
        if not np.any(mask):
            continue
        ax.scatter(x[mask], y[mask], z[mask],
                   c=color_map[k],
                   marker=marker_map[k],
                   alpha=alpha,
                   label=label_map[k],
                   edgecolors='k', linewidths=0.3,
                   s=40,  # tamaño puntos, ajustá si querés
                   depthshade=True,
                   zorder=1)

    ncol = 4
    fig.subplots_adjust(bottom=0.25)

    legend_elements = [
        Line2D([0], [0], marker=marker_map[k], color='w',
               label=label_map[k],
               markerfacecolor=color_map[k],
               markeredgecolor='k', markersize=8,
               linestyle='None')
        for k in combos if np.any((dx == k[0]) & (dy == k[1]) & (dz == k[2]))
    ]

    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.1),
               ncol=ncol,
               title='Categorías',
               fontsize='small',
               frameon=False)

    plt.tight_layout()

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', dpi=dpi, bbox_inches='tight',
                    pad_inches=0.05)
        plt.close()
    else:
        plt.show()

def generate_rotation_sequence(x, y, z, out_dir,
                               steps_per_transition=30, alpha=0.8, thr=0.5,
                               save=False):
    frame_num = 0
    positions = [(0, 0), (90, -90), (0, -90), (0, 90), (-90, 180)]

    # auxiliar para interpolar ángulos (lineal)
    def interp_angles(start, end, num):
        return np.linspace(start, end, num)

    for i in range(len(positions)):
        start = positions[i]
        end = positions[(i + 1) % len(positions)]  # ciclo cerrado

        elev_seq = interp_angles(start[0], end[0], steps_per_transition)
        azim_seq = interp_angles(start[1], end[1], steps_per_transition)

        for elev, azim in zip(elev_seq, azim_seq):
            filename = f'frame_{frame_num:03d}'
            print(
                f'Generating {filename} with elev={elev:.1f}, azim={azim:.1f}')
            PlotScatter3D(x, y, z,
                          elev=elev, azim=azim,
                          name_fig=filename, save=save, out_dir=out_dir,
                          alpha=alpha, thr=thr)
            frame_num += 1

# ---------------------------------------------------------------------------- #
dmi = xr.open_dataset(dates_dir + 'DMI_SON_Leads_r_CFSv2.nc')
ep = xr.open_dataset(dates_dir + 'EP_SON_Leads_r_CFSv2.nc')
cp = xr.open_dataset(dates_dir + 'CP_SON_Leads_r_CFSv2.nc')

x = dmi.stack(time2=('time', 'r')).sst.values/dmi.std(['r', 'time']).sst.values
y = ep.stack(time2=('time', 'r')).sst.values/ep.std(['r', 'time']).sst.values
z = cp.stack(time2=('time', 'r')).sst.values/cp.std(['r', 'time']).sst.values

# ---------------------------------------------------------------------------- #
generate_rotation_sequence(x, y, z, out_dir=out_dir, steps_per_transition=45,
                           save=save)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #