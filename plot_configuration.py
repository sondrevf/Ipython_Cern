import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
# Use latex (must be before use_mathtext:True)
# Use latex (must be before use_mathtext:True)
flag_latex=0
if flag_latex:
    size0 = 14
    size1 = 10
    # To be able to use SI units as micro (automatically upright)
    params = {'text.usetex':True,# 'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{cmbright}']}#,r'\usepackage{mathastext}']}
        'text.latex.preamble': [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]
 }
    mpl.rcParams.update(params)
else:
    size0 = 10
    size1 = 9
    mpl.rcParams.update({'axes.formatter.use_mathtext':True}) # Change from 1e4 to 10^4  (NOT IF USING TEX)

# Numpy printing
np.set_printoptions(precision=3,suppress=False)


# Set style of plots
i_layout = 1  #0=paper, 1=thesis
#size1 = [8,9][i_layout]
colwidth= [ 4  , 146.8/25.4  *.48  *1.2][i_layout]
figwidth = 6
figwidthM= figwidth*[1,.75/.48][i_layout]
figheight=[4,5][i_layout] #4 for papers, 5 for thesis
figheightpres=5

ticksize  = size1* figwidth/colwidth
labelsize = size0* figwidth/colwidth
titlesize = size0* figwidth/colwidth


# mpl.style.use('classic')
mpl.rcParams.update({'font.size':ticksize })
mpl.rcParams.update({'legend.fontsize':ticksize })# legend
mpl.rcParams.update({'xtick.labelsize':ticksize , 'ytick.labelsize':ticksize,
                     'xtick.direction':'in',      'ytick.direction':'in',
                     'xtick.major.size':6,        'ytick.major.size':6})
mpl.rcParams.update({'axes.titlesize':titlesize})       # Title
mpl.rcParams.update({'axes.labelsize':labelsize})    # x,y,cbar labels
mpl.rcParams.update({'figure.titlesize' :titlesize})
mpl.rcParams.update({'savefig.bbox':'tight'})
mpl.rcParams.update({'axes.formatter.limits':[-3,3]})
mpl.rcParams.update({'figure.figsize':[figwidth,figheight]})
mpl.rcParams.update({'image.cmap':'rainbow'})
mpl.rcParams.update({'image.cmap':'jet'})
mpl.rcParams.update({'axes.formatter.useoffset':False})   # no offset of axis
mpl.rcParams.update({'xtick.top':True,  'ytick.right':True})

mpl.rcParams.update({'lines.markeredgewidth':2,
                     'lines.markersize':6})

#Legend
mpl.rcParams.update({'legend.handletextpad':0.3,'legend.borderaxespad':0.1,
                    'legend.handlelength':1,'legend.labelspacing':0.2,'legend.borderpad':0.3})


# To use unicode
# mpl.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'})
# To get sans-serif
# params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# plt.rcParams.update(params)

# To get sans-serif
mpl.rcParams.update({'font.style':'normal'})
mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'mathtext.fontset':'dejavuserif'})
#mpl.rcParams.update({'font.serif':'cm'})
#mpl.rcParams.update({'font.sans-serif':['Computer Modern Sans Serif','Geneva']})

#rc = {'font.style':'normal'}
#rc = {  'font.family' : 'sans-serif', 
#        'font.sans-serif':['Geneva',],   # Computer Modern Sans serif
###        "mathtext.fontset" : "custom",
###        'mathtext.it': 'cm',
###        'mathtext.rm': 'Helvetica',
#       }
#mpl.rcParams.update(rc)
#mpl.rcParams['mathtext.fontset'] = 'custom'
##bs = 'Bitstream Vera Sans'
#mpl.rcParams['mathtext.rm'] = 'Computer Modern Sans Serif'
#mpl.rcParams['mathtext.it'] = 'cm:italic'
#mpl.rcParams['mathtext.bf'] = 'cm:bold'
#mpl.rc('font', **{'sans-serif' : 'dejavusans','family' : 'sans-serif'})

# # To use unicode
# # mpl.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'})
# # To get sans-serif
# # params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# # plt.rcParams.update(params)
# 
# # To get sans-serif
# mpl.rcParams.update({'font.style':'normal'})
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# # mpl.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'})
# 

markers = ['*','x','+','1','2','3','4','>','<','o','^','v']
cmap = mpl.pyplot.get_cmap("tab10")
colors = [cmap(i) for i in range(20)]



def short_exp(number):
    if number==0:
        return '0'
    else:
        exp = int(np.log10(number))
        return str(number/10**exp)[0]+'e%d'%exp
def sci(num,dec=0):
    if abs(num)>0:
        exp = int(np.log10(abs(num))+100)-100
        num = num/10**exp
        sci = '%.*fe%d'%(dec,num,exp)
    else:
        sci = '0'
    return sci

def short_float(number):
    if number<1:
        return str(number)
    else:
        return "%d"%int(number+0.5)

def sci_not(num,dec):
    if num>0:
        exp = int(np.log10(num)+100)-100
        sci = (r"$%.10f"%(num/10**exp))[:dec+2+1*(dec>0)]+r"\cdot10^{%d}$"%exp
    else:
        sci = '$0$'
    return sci

def mean_nabo(f,n=3):
    g = np.zeros_like(f)
    for i in range(n):
        g[i] = np.mean(f[:i])
        g[-i-1] = np.mean(f[-(i+1):])

        g[n:-n] += np.roll(f,i+1)[n:-n]
        g[n:-n] += np.roll(f,-(i+1))[n:-n]
    g[n:-n] += f[n:-n]
    g[n:-n] = g[n:-n]/(2*n+1)
    return g

print('Finished configuring the plots.')

