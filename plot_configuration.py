import matplotlib as mpl
import numpy as np
# Use latex (must be before use_mathtext:True)
flag_latex=0
if flag_latex:
    # To be able to use SI units as micro (automatically upright)
    params = {'text.usetex':True, 'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{cmbright}',r'\usepackage{mathastext}']}
    mpl.rcParams.update(params)
else:
    mpl.rcParams.update({'axes.formatter.use_mathtext':True}) # Change from 1e4 to 10^4  (NOT IF USING TEX)

# Numpy printing
np.set_printoptions(precision=3,suppress=False)


# Set style of plots
colwidth=4  #3.5
figwidth=6
figheight=4 #4 for papers
figheightpres=5
ticksize= 9 *figwidth/colwidth
labelsize=10 *figwidth/colwidth
titlesize = 10 *figwidth/colwidth

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
mpl.rcParams.update({'mathtext.fontset':'cm'})
mpl.rcParams.update({'font.style':'normal'})
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
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# mpl.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'})


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
    if abs(number)<1 and number!=0:
        return str(number)
    else:
        return "%d"%int(number+0.5)

def sci_not(num,dec):
    if num>0:
        exp = int(np.log10(num)+100)-100
        sci = (r"$%.*f"%(dec,num/10**exp))+r"\cdot10^{%d}$"%exp #[:dec+2+1*(dec>0)]
    else:
        sci = '$0$'
    return sci
    
print('Finished configuring the plots.')
