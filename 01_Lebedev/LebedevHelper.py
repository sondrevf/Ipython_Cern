# Lebedev functions
import numpy as np
import scipy as sp
pi2 = np.pi*2
sigdpp= 1.129E-4

def func_g2(g):
    return g/2
#     return 1-np.sqrt(1-g+g**2*0.5)


def lebfac(g,dq):
    g2 = func_g2(g)
    return pi2**2*(1-g2)**2*(dq)**2 / (pi2**2*(1-g2)*(dq)**2  + g2**2)
#     return pi2**2*(1-g2)**2 / (pi2**2*(1-g2) + g2**2/(dq)**2)    

def anal_DJx(J,g2,Qp,sigdpp= 1.129E-4):
    dq = Qp*sigdpp*np.sqrt(2*J)
    a = pi2**2*(1-g2)**2*dq**2
    b = pi2**2*(1-g2)*dq**2
    c = g2**2
    return a/b*(1-1/np.sqrt(b/c+1))

def anal_DJx2(J,g2,Qp,sigdpp= 1.129E-4):
    return (1-g2)*(1-1/np.sqrt(1+2*pi2**2*(1-g2)/g2**2*Qp**2*sigdpp**2*J))

def dist(d,sigdpp=1.129e-4):
    return  1/(np.sqrt(2*np.pi)*sigdpp) * np.exp(-d**2/sigdpp**2*0.5)

def intQx(Jy,Jx,Qx0,a,b,exp):
    return Qx(Qx0,a,b,Jx,Jy)**(exp)*np.exp(-Jx-Jy)

# Tune shfit
def dQxOctu(Jx,Jy,a,b):
    return a*(Jx-1) + b*(Jy-1)

def dQxChroma(delta,Qpx):
    return Qpx*delta

# Lebedev expression
def octu_leb(Jx,Jy,a,b,g):
    dq = dQxOctu(Jx,Jy,a,b)
    return lebfac(g,dq)

def octuQp_leb(Jx,Jy,delta,a,b,g,Qpx):
    dq = dQxOctu(Jx,Jy,a,b) + dQxChroma(delta,Qpx)
    return lebfac(g,dq)

#Integrand
def int_octu_leb(Jx,Jy,a,b,g):
    return octu_leb(Jx,Jy,a,b,g)*np.exp(-Jx-Jy)

def intJy_octu_leb(Jy,Jx,a,b,g):
    return octu_leb(Jx,Jy,a,b,g)*np.exp(-Jy)

def int_octuQp_leb(Jx,Jy,delta,a,b,g,Qp,sigdpp):
    return octuQp_leb(Jx,Jy,delta,a,b,g,Qp)*np.exp(-Jx-Jy)*dist(delta,sigdpp)

def chi2_dist(J):
    return sp.stats.chi2.pdf(J*2,2)*2


def abs_action(a,b):
    return (a**2+b**2)*0.5

def rel_action(a,b):
    return 0.5*((a-np.average(a,weights=w))**2+(b-np.average(b,weights=w))**2)



def center_edges(x):
    return x[1:] - (x[1]-x[0])*0.5

def edges_to_center(edges):
    return (edges[1:]+edges[:-1])*.5



# Functions from Lebedev idea
def func_dmu(a,J,Javg=1):
    return pi2 * a * (J-Javg)

def func_dDmudJ(a,J):
    return pi2 * a

def JK(J0,k,theta):
    return J0 + k**2/2 + np.sqrt(2*J0)*k*np.sin(theta) #p \propto sin(phi)

def LL(g2,dmu):
    return (1-g2)**2*(dmu)**2       / (g2**2 + (1-g2)*(dmu)**2)

def MM(g2,dmu):
    return (1-g2)*(g2)*(dmu)        / (g2**2 + (1-g2)*(dmu)**2)

def NN(g2,dmu):
    return (1-g2)*(1-g2/2)*(dmu)**2 / (g2**2 + (1-g2)*(dmu)**2)

def dLLdm(g2,dmu):
    return (1-g2)**2*g2**2*2*(dmu)            / (g2**2 + (1-g2)*(dmu)**2)**2

def dMMdm(g2,dmu):
    return (1-g2)*(g2)*(g2**2-(1-g2)*(dmu)**2)/ (g2**2 + (1-g2)*(dmu)**2)**2

def dNNdm(g2,dmu):
    return (1-g2)*(1-g2/2)*g2**2*2*(dmu)      / (g2**2 + (1-g2)*(dmu)**2)**2

def deltaJ(J0,phi0,k,g2,dmu):
    A = np.sqrt(2*J0)
    cphi = np.cos(phi0)
    sphi = np.sin(phi0)
    L = LL(g2,dmu) ; M = MM(g2,dmu) ; N = NN(g2,dmu)
    dL = dLLdm(g2,dmu)*dDmudJ(J0)
    dM = dMMdm(g2,dmu)*dDmudJ(J0)
    dN = dNNdm(g2,dmu)*dDmudJ(J0)
    return  k*A*(M*cphi + N*sphi) +\
            k**2*(L/2 + J0*sphi * (dM *cphi +dN*sphi)) +\
            k**3*A/4*(dL*sphi + dM *cphi +dN*sphi) +\
            k**4/8*dL