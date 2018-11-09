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
#     return pi2**2*(1-g2)**2 / (pi2**2*(1-g2) + g2**2/(dq)**2)
    return pi2**2*(1-g2)**2*(dq)**2 / (pi2**2*(1-g2)*(dq)**2  + g2**2)
    

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


def dQxOctu(Jx,Jy,a,b):
    return a*(Jx-1) + b*(Jy-1)

def octu_leb(Jx,Jy,a,b,g):
    dq = dQxOctu(Jx,Jy,a,b)
    return lebfac(g,dq)

def int_octu_leb(Jx,Jy,a,b,g):
    return octu_leb(Jx,Jy,a,b,g)*np.exp(-Jx-Jy)

def intJy_octu_leb(Jy,Jx,a,b,g):
    return octu_leb(Jx,Jy,a,b,g)*np.exp(-Jy)

def chi2_dist(J):
    return sp.stats.chi2.pdf(J*2,2)*2



def center_edges(x):
    return x[1:] - (x[1]-x[0])*0.5

def edges_to_center(edges):
    return (edges[1:]+edges[:-1])*.5
