import matplotlib as mpl
import matplotlib.pyplot as plt

# Edit the formatting of matplotlib ----------------------------------------------

mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['font.weight'] = 900
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.edgecolor'] = 3 * [0.]
mpl.rcParams['xtick.color'] = 3 * [0.]
mpl.rcParams['ytick.color'] = 3 * [0.]
mpl.rcParams['xtick.major.pad'] = 5
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
# mpl.rcParams['axes.titlepad'] = 20

mpl.rcParams['font.size'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

# plt.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}\bfseries\boldmath"]
# plt.rc('font', weight=400)
# plt.ion()

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Helvetica Neue"
# mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"

mpl.rcParams["figure.autolayout"] = True


def hide_frame(*axes): # function to hide frame from figures
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


# import matplotlib.colors as colors
# import matplotlib.cm as cm
# plt.cm.Reds(np.linspace(0,1,n))