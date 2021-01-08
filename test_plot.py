# %reset -f
import numpy as np

import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import scipy.special as spec
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import pickle as pkl

#Interpolation of points
from scipy.interpolate import griddata


# Plotting preparation
import sys
sys.path.append('../')
from plot_configuration import *

import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

plt.figure()
plt.title(r'a$a\mathrm{a}\mathrm{\mu}$')
plt.plot(np.logspace(-4,4,9))
plt.show()
