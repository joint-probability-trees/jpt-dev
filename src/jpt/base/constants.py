import numpy as np
import matplotlib

sepcomma = ',\n'
sepsemi = ';\n'
sepn = '\n'

#################
# TIME AND DATE #
#################
TMPFILESTRFMT = 'TMP_%Y%m%d_%H-%M-%S'
FILESTRFMT = "%Y-%m-%d_%H:%M"

##########
# COLORS #
##########
# LIGHT
lightblue = '#b3ffff80'
darkblue = '#3f9fff80'
green = '#b3ff4c80'
yellow = '#ffff4c80'
orange = '#ffbe4980'
red = '#FF4A4980'

# FULL
dlightblue = '#b3ffffff'
ddarkblue = '#3f9fffff'
dgreen = '#b3ff4cff'
dyellow = '#ffff4cff'
dorange = '#ffbe49ff'
dred = '#FF4A49ff'

plotcolormap = 'cividis'  # or viridis


# ----------------------------------------------------------------------------------------------------------------------
# Fix different style names in matplotlib versions
matplotlib_version = matplotlib.__version_info__.major * 10 + matplotlib.__version_info__.minor

if matplotlib_version > 37:
    plotstyle = 'seaborn-v0_8-deep'
else:
    plotstyle = 'seaborn-deep'


# ----------------------------------------------------------------------------------------------------------------------

avalailable_colormaps = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
    'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
    'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired',
    'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
    'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
    'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
    'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
    'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
    'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
    'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
    'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2',
    'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno',
    'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
    'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
    'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain',
    'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
    'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
]


class SYMBOL:
    LAND = '\u2227'
    IN = '\u2208'
    LT = '<'
    GT = '>'
    LTE = '\u2264'
    GTE = '\u2265'
    THETA = '\u03D1'
    ARROW_BAR_LEFT = '\u21E4'
    ARROW_BAR_RIGHT = '\u21E5'


# ----------------------------------------------------------------------------------------------------------------------
# Numeric constants

class Epsilon:
    '''
    Class representing the smallest machine-representable increment or decrement
    of a 64bit float based on numpy data types.
    '''

    def __radd__(self, x):
        return np.nextafter(x, x + 1)

    def __rsub__(self, x):
        return np.nextafter(x, x - 1)


eps = Epsilon()
