"""
Barras de colores y escalas varias
"""
# ---------------------------------------------------------------------------- #
from matplotlib import colors
import numpy as np

# Colorbars ------------------------------------------------------------------ #
def get_cbars(VarName, show_cbars=False):
    # ---------------------------------------------------------------------------- #
    cbar_sst = colors.ListedColormap(['#B98200', '#CD9E46', '#E2B361', '#E2BD5A',
                                      '#FFF1C6', 'white', '#B1FFD0', '#7CEB9F',
                                      '#52D770', '#32C355', '#1EAF3D'][::-1])
    cbar_sst.set_over('#9B6500')
    cbar_sst.set_under('#009B2E')
    cbar_sst.set_bad(color='white')


    cbar = colors.ListedColormap([
                                     '#641B00', '#892300', '#9B1C00', '#B9391B',
                                     '#CD4838', '#E25E55',
                                     '#F28C89', '#FFCECC', '#FFE6E6', 'white',
                                     '#E6F2FF', '#B3DBFF',
                                     '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF',
                                     '#014A9B', '#013A75',
                                     '#012A52'
                                 ][::-1])

    cbar.set_over('#4A1500')
    cbar.set_under('#001F3F')
    cbar.set_bad(color='white')

    cbar_snr = colors.ListedColormap(['#070B4F', '#2E07AC', '#387AE4', '#6FFE9B',
                                      '#FFFFFF',
                                      '#FFFFFF', '#FFFFFF',
                                      '#FEB77E', '#CA3E72', '#782281', '#251255'])
    cbar_snr.set_over('#251255')
    cbar_snr.set_under('#070B4F')
    cbar_snr.set_bad(color='white')

    cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                     '#B4E2DB',
                                     'white',
                                     '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07',
                                     '#543005', ][::-1])
    cbar_pp.set_under('#3F2404')
    cbar_pp.set_over('#00221A')
    cbar_pp.set_bad(color='white')

    cbar_pp_11 = colors.ListedColormap(['#004C42', '#0C7169', '#79C8BC',
                                        '#B4E2DB',
                                        'white',
                                        '#F1DFB3', '#DCBC75', '#995D13',
                                        '#6A3D07'][::-1])
    cbar_pp_11.set_under('#6A3D07')
    cbar_pp_11.set_over('#004C42')
    cbar_pp_11.set_bad(color='white')

    cbar_pp_19 = colors.ListedColormap(['#001912', '#003C30', '#004C42',
                                        '#0C7169', '#3FA293', '#79C8BC',
                                        '#A1D7CD', '#CFE9E5', '#E8F4F2',
                                        'white',
                                        '#F6E7C8', '#F1DFB3', '#DCBC75',
                                        '#C89D4F', '#B17B2C', '#995D13',
                                        '#7F470E', '#472705', '#2F1803'][::-1])

    cbar_pp_19.set_under('#230F02')
    cbar_pp_19.set_over('#0D2D25')
    cbar_pp_19.set_bad('white')

    cbar_ks = colors.ListedColormap(['#C2FAB6', '#FAC78F'])

    cbar_ks.set_under('#5DCCBF')
    cbar_ks.set_over('#FA987C')
    cbar_ks.set_bad(color='white')

    cbar_snr_t = colors.ListedColormap(['#00876c', '#439981', '#6aaa96',
                                        '#8cbcac','#aecdc2', '#cfdfd9',
                                        '#FFFFFF',
                                        '#f1d4d4', '#f0b8b8', '#ec9c9d',
                                        '#e67f83', '#de6069', '#d43d51'])

    cbar_snr_t.set_under('#006D53')
    cbar_snr_t.set_over('#AB183F')
    cbar_snr_t.set_bad(color='white')

    cbar_snr_pp = colors.ListedColormap(['#a97300', '#bb8938', '#cc9f5f',
                                         '#dbb686', '#e9cead','#f5e6d6',
                                         '#ffffff',
                                         '#dce9eb', '#b8d4d8', '#95bfc5',
                                         '#70aab2', '#48959f', '#00818d'])

    cbar_snr_pp.set_under('#6A3D07')
    cbar_snr_pp.set_over('#1E6D5A')
    cbar_snr_pp.set_bad(color='white')

    cbar_bins2d = colors.ListedColormap(['#9B1C00', '#CD4838', '#E25E55',
                                         '#F28C89', '#FFCECC',
                                         'white', 'white',
                                         '#B3DBFF', '#83B9EB', '#5E9AD7',
                                         '#3C7DC3','#014A9B'][::-1])
    cbar_bins2d.set_over('#641B00')
    cbar_bins2d.set_under('#012A52')
    cbar_bins2d.set_bad(color='white')

    cbar_pp_bins2d = colors.ListedColormap(['#003C30', '#004C42', '#0C7169',
                                            '#79C8BC', '#B4E2DB',
                                            'white', 'white',
                                            '#F1DFB3', '#DCBC75', '#995D13',
                                            '#6A3D07', '#543005', ][::-1])
    cbar_pp_bins2d.set_under('#3F2404')
    cbar_pp_bins2d.set_over('#00221A')
    cbar_pp_bins2d.set_bad(color='white')

    cbars = {
        "sst": cbar_sst,
        "cbar_rdbu": cbar,
        "snr": cbar_snr,
        "pp": cbar_pp,
        "pp_11": cbar_pp_11,
        "pp_19": cbar_pp_19,
        "ks": cbar_ks,
        "snr_t": cbar_snr_t,
        "snr_pp": cbar_snr_pp,
        "bins2d": cbar_bins2d,
        "pp_bins2d": cbar_pp_bins2d,
    }

    if show_cbars:
        print(list(cbars.keys()))
    else:
        return cbars[VarName]


# scales --------------------------------------------------------------------- #
def get_scales(VarName, show_scales=False):
    scales = {
        "hgt_regre": [-300, -200, -100, -50, -25, 0, 25, 50, 100, 200, 300],
        "hgt_regre2": [-300, -270, -240, -210, -180, -150, -120, -90, -60, -30,
                       0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
        "hgt750_regre": [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0,
                         10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "pp_regre": np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45]),
        "t_regre": [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1],
        "t_val": [-3, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 3],
        "pp_val": [-60, -30, -15, -5, 0, 5, 15, 30, 60],
        "hgt_comp": [-500, -300, -200, -100, -50, 0, 50, 100, 200, 300, 500],
        "t_comp": [-1.5, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 1.5],
        "pp_comp": [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40],
        "t_comp_cfsv2": [-1, -0.8, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.8, 1],
        "pp_comp_cfsv2": np.array([-45, -30, -20, -10, -2.5, 0, 2.5, 10, 20, 30, 45]),
        "hgt_comp_cfsv2": [-375, -275, -175, -75, -25, 0, 25, 75, 175, 275, 375],
        "snr": [-1, -0.8, -0.6, -0.5, -0.1, 0, 0.1, 0.5, 0.6, 0.8, 1],
        "hgt_ind": [-575, -475, -375, -275, -175, -75, 0,
                          75, 175, 275, 375, 475, 575],
        "vpot200_cfsv2" : np.array([-2, -1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2])*1e7
    }

    if show_scales:
        print(list(scales.keys()))
    else:
        return scales[VarName]

################################################################################