###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################
from matplotlib import rcParams
from pylab import rc

# Fonts:
#rc('font', **{'family': 'serif', 'serif': ['PT Serif']})
#rc('text', usetex=True)
#rc('text', usetex=True, preamble=r'\usepackage[cm]{sfmath}')
rc('font', **{'family': 'serif'})


rcParams['figure.figsize'] = [16/1.5, 9/1.5]

rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelsize'] = 16
rcParams['axes.titlepad'] = 8
rcParams['axes.titlesize'] = 18

rcParams['lines.linewidth'] = 2.

rcParams['image.cmap'] = 'plasma'

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.labelsize']= 14
rcParams['ytick.labelsize']= 14

rcParams['font.size'] = 14.

rcParams['xtick.major.width'] = 1.2
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.pad']='7'

rcParams['xtick.minor.visible']=True
rcParams['xtick.minor.width'] = 1.
rcParams['xtick.minor.size'] = 3

rcParams['ytick.major.width'] = 1.2
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.pad']='7'

rcParams['ytick.minor.visible']=True
rcParams['ytick.minor.width'] = 1.
rcParams['ytick.minor.size'] = 3


rcParams['legend.frameon']= False
rcParams['legend.fontsize'] = 14

rcParams['mathtext.default'] = 'regular' 

###############################################################