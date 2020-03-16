import numpy as np
import matplotlib.pyplot as plt

from PySSD.Distribution import Distribution,Gaussian
from PySSD.Detuning import LinearDetuning
from PySSD.Dispersion import Dispersion
from PySSD.Integrator import FixedTrapezoidalIntegrator, SimpsonIntegrator, TrapzIntegrator, DblquadIntegrator

#setupQx = 0.31
ax = 3e-5
bx = 0
f_rev = 11.2455e3
Qs = 0.002
Qx=0.31

# Noise 
sigma_ibs=0
D_ibs = f_rev*sigma_ibs**2/2
sigma_k = 1
Pnoise = f_rev*sigma_k**2
D_k = f_rev*sigma_k**2/2

# Modewmode__DQ = [3e-5+1j*2.7e-5]
wmode__DQ = [5e-5+1j*2.86e-5]
# wmode__DQ = [14e-5+1j*0.3e-5]
wmodeQ0 = [Qx]

# PySSD
distribution = Gaussian()
detuning = LinearDetuning(Qx,ax,bx)
integrator = TrapzIntegrator(distribution, detuning, maxJ=18,n_steps=1000)

# Stability diagram parameters
nQ = 60
integrator_epsilon=1e-6 *ax/1e-4 *4
J = np.linspace(3.8,3.9,nQ)
incoQ = Qx*0 + detuning(J,0)

# lebedev
freeQs = J*ax + Qx
cohDQs = np.zeros((len(freeQs)), dtype='complex')
for k, q in enumerate(freeQs):
    cohDQs[k] = (2*integrator.integrate(q,epsilon=integrator_epsilon*1) 
                  -integrator.integrate(q,epsilon=integrator_epsilon*2) )

R = -1/cohDQs
dQ = wmode__DQ

eps =  1+ dQ*R
Dl = D_k *1/np.abs(eps)**2 



# Plotting
plt.figure()
plt.title("Stability diagram and mode")
plt.plot(cohDQs.real,cohDQs.imag)
plt.plot(wmode__DQ[0].real,wmode__DQ[0].imag,'rx')
plt.tight_layout()

plt.figure()
plt.title("Diffusion coefficient")
plt.plot(J,Dl,'r')
plt.xlabel('J')
plt.tight_layout()
