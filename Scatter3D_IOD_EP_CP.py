
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

dates_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

dmi = xr.open_dataset(dates_dir + 'DMI_SON_Leads_r_CFSv2.nc')
ep = xr.open_dataset(dates_dir + 'EP_SON_Leads_r_CFSv2.nc')
cp = xr.open_dataset(dates_dir + 'CP_SON_Leads_r_CFSv2.nc')

x = dmi.stack(time2=('time', 'r')).sst.values/dmi.std(['r', 'time']).sst.values
y = ep.stack(time2=('time', 'r')).sst.values/ep.std(['r', 'time']).sst.values
z = cp.stack(time2=('time', 'r')).sst.values/cp.std(['r', 'time']).sst.values


def plot_quadrants(ax, array, fixed_coord, cmap, alpha):
    nx, ny, nz = array.shape

    # Rango espacial deseado
    coord_min, coord_max = -4.5, 4.5

    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape

    # Crear vectores de coordenadas para cada dimensión en rango [-4.5, 4.5]
    # Para cada mitad de la dimensión, usamos la mitad del rango total
    half_len = (coord_max - coord_min) / 2  # 9/2 = 4.5

    # En cada cuadrante se representan n0/2 y n1/2 puntos, así que usamos linspace para esos tamaños
    coord0 = np.linspace(coord_min, coord_min + half_len, n0 // 2)
    coord1 = np.linspace(coord_min, coord_min + half_len, n1 // 2)

    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    min_val = array.min()
    max_val = array.max()

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))

        if fixed_coord == 'x':
            # Para Y y Z: se crean las mallas para el tamaño del cuadrante
            Y, Z = np.meshgrid(coord0, coord1, indexing='ij')
            X = (coord_min + half_len) * np.ones_like(
                Y)  # plano fijo en x = 0 para el centro, se ajusta abajo

            # El offset depende del cuadrante: sumamos half_len para mover a la derecha o arriba
            Y_offset = (i // 2) * half_len
            Z_offset = (i % 2) * half_len

            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=True, alpha=alpha)

        elif fixed_coord == 'y':
            X, Z = np.meshgrid(coord0, coord1, indexing='ij')
            Y = (coord_min + half_len) * np.ones_like(X)

            X_offset = (i // 2) * half_len
            Z_offset = (i % 2) * half_len

            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=True, alpha=alpha)

        elif fixed_coord == 'z':
            X, Y = np.meshgrid(coord0, coord1, indexing='ij')
            Z = (coord_min + half_len) * np.ones_like(X)

            X_offset = (i // 2) * half_len
            Y_offset = (i % 2) * half_len

            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1,
                            facecolors=facecolors, shade=True, alpha=alpha)

def PlotScatter3D(x, y, z, elev, azim, name_fig, save, out_dir, alpha_walls=0.1):

    dpi = 100
    if save:
        dpi=300

    mask_neutral = (np.abs(x) < 0.5) & (np.abs(y) < 0.5) & (np.abs(z) < 0.5)
    point_colors = np.empty_like(x, dtype='<U10')
    point_colors[mask_neutral] = 'k'

    quadrant_index = (x > 0).astype(int) * 4 + (y > 0).astype(int) * 2 + (
                z > 0).astype(int)

    colors = np.array([
        'navy',  # 0: - - -
        'blue',  # 1: - - +
        'green',  # 2: - + -
        'cyan',  # 3: - + +
        'magenta',  # 4: + - -
        'red',  # 5: + - +
        '#FF5B12',  # 6: + + -
        '#8B1E1E'  # 7: + + +
    ])
    labels = [
        'DMI-, EP-, CP-',
        'DMI-, EP-, CP+',
        'DMI-, EP+, CP-',
        'DMI-, EP+, CP+',
        'DMI+, EP-, CP-',
        'DMI+, EP-, CP+',
        'DMI+, EP+, CP-',
        'DMI+, EP+, CP+'
    ]

    for i in range(8):
        point_colors[(quadrant_index == i) & ~mask_neutral] = colors[i]


    fig = plt.figure(dpi=dpi, figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')

    nx, ny, nz = 70, 100, 50
    r_square = (np.mgrid[-1:1:1j * nx, -1:1:1j * ny, -1:1:1j * nz] ** 2).sum(0)
    ax.set_box_aspect(r_square.shape)
    cmap = 'binary'
    plot_quadrants(ax, r_square, 'x', cmap=cmap, alpha=alpha_walls)
    plot_quadrants(ax, r_square, 'y', cmap=cmap, alpha=alpha_walls)
    plot_quadrants(ax, r_square, 'z', cmap=cmap, alpha=alpha_walls)

    sc = ax.scatter(x, y, z, c=point_colors, marker='.', alpha=0.8,
                    depthshade=True, zorder=1)

    # Planos guía
    ax.plot([-4.5, 4.5], [0, 0], [0, 0], color='gray', linestyle='--',
            alpha=1)  # Eje X
    ax.plot([0, 0], [-4.5, 4.5], [0, 0], color='gray', linestyle='--',
            alpha=1)  # Eje Y
    ax.plot([0, 0], [0, 0], [-4.5, 4.5], color='gray', linestyle='--',
            alpha=1)  # Eje Z

    # range_ = np.linspace(-4.5, 4.5, 10)

    # # Plano Y=0 (X-Z)
    # X, Z = np.meshgrid(range_, range_)
    # Y = np.zeros_like(X)
    # ax.plot_surface(X, Y, Z, color='white', alpha=alpha_walls)
    #
    # # Plano X=0 (Y-Z)
    # Y, Z = np.meshgrid(range_, range_)
    # X = np.zeros_like(Y)
    # ax.plot_surface(X, Y, Z, color='white', alpha=alpha_walls)
    #
    # # Plano Z=0 (X-Y)
    # X, Y = np.meshgrid(range_, range_)
    # Z = np.zeros_like(X)
    # ax.plot_surface(X, Y, Z, color='white', alpha=alpha_walls)
    #
    ax.set_xlabel('DMI')
    ax.set_ylabel('EP')
    ax.set_zlabel('CP')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_zlim([-4.5, 4.5])

    legend_elements = [
        Line2D([0], [0], marker='.', color='w', label=label,
               markerfacecolor=color, markeredgecolor=color, markersize=8,
               linestyle='None')
        for label, color in zip(labels, colors)
    ]

    legend_elements.insert(0, Line2D([0], [0], marker='.', color='w',
                                     label='|DMI|,|EP|,|CP| < 0.5',
                                     markerfacecolor='k',
                                     markeredgecolor='k', markersize=8,
                                     linestyle='None'))

    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.05, 1), title='Categorías')

    plt.tight_layout()

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', dpi=dpi)
        plt.close()
    else:
        plt.show()

# EP vs CP
PlotScatter3D(x, y, z, elev=0, azim=0,
              name_fig='', save=False, out_dir='')

# DMI vs EP
PlotScatter3D(x, y, z, elev=90, azim=-90,
              name_fig='', save=False, out_dir='')

# DMI  vs CP
PlotScatter3D(x, y, z, elev=0, azim=-90,
              name_fig='', save=False, out_dir='')

PlotScatter3D(x, y, z, elev=25, azim=45,
              name_fig='', save=False, out_dir='')

PlotScatter3D(x, y, z, elev=25, azim=45,
              name_fig='', save=False, out_dir='', alpha_walls=0.5)
