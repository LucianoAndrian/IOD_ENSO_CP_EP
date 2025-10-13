"""
Funcion de seleccion de archivos de NMME
"""
# ---------------------------------------------------------------------------- #
import glob

# ---------------------------------------------------------------------------- #
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
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #