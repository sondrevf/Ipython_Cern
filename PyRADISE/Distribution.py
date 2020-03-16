import numpy as np
from scipy import interpolate

from PySSD.Distribution import Distribution

class MyDistribution1D(Distribution):
    def __init__(self,Jb=0,Jc=0,Psic=0,dPsi=0,bool_radial=0,interp_order=1):
        self._Jb=Jb.tolist() 
        if bool_radial and 0:
            Jbsq = np.sqrt(Jb)
            self._Jc = Jbsq[1:]*Jbsq[:-1]
        else:
            self._Jc=Jc.tolist()
        self._Psic=Psic.tolist() #+[0]
        self._dPsi=dPsi.tolist() #+[0]
#         self.interp_psi     = interpolate.interp1d(self._Jb,self._Psic,kind='previous')
#         self.interp_dpsidJx = interpolate.interp1d(self._Jc,self._dPsi,kind='previous',bounds_error=False,fill_value="extrapolate")
        kind = ["nearest",'slinear','quadratic','cubic'][interp_order]
        self.interp_psi     = interpolate.interp1d(self._Jc,self._Psic,kind=kind,bounds_error=False,fill_value="extrapolate")
        self.interp_dPsidJx = interpolate.interp1d(self._Jb[1:-1],self._dPsi,kind=kind,bounds_error=False,fill_value="extrapolate")

    def getValue(self, jx, jy):
        return (self.interp_psi(jx)*np.exp(-jy))[0]

    def getDJx(self, jx, jy):
        return (self.interp_dPsidJx(jx)*np.exp(-jy))[0]

    def getDJy(self, jx, jy):
        return (-self.getValue(jx, jy))[0]
    

    
class MyDistribution1Dy(MyDistribution1D):
    def __init__(self,Jb=0,Jc=0,Psic=0,dPsi=0,bool_radial=0,interp_order=1):
        MyDistribution1D.__init__(self,Jb,Jc,Psic,dPsi,bool_radial,interp_order)
#     def getValue(self, jx, jy):
#         return super().getValue(jy,jx)
    def getDJx(self,jy,jx):
        return super().getDJy(jx,jy)
    def getDJy(self,jy,jx):
        return super().getDJx(jx,jy)
    
    
class MyDistribution2D(Distribution):
    def __init__(self,Jxbx=0,Jxc=0,Jyby=0,Jyc=0,Psic=0,dPsidJx=0,dPsidJy=0,bool_radial=0,interp_order=0):
        self._Jxbx= Jxbx.copy()
        self._Jyby= Jyby.copy()
        if bool_radial :
            Jxbxsq = np.sqrt(Jxbx)
            self._Jxc = Jxbxsq[1:]*Jxbxsq[:-1]
            Jybysq = np.sqrt(Jyby)
            self._Jyc = Jybysq[1:]*Jybysq[:-1]
        else:
            self._Jxc=Jxc.copy()
            self._Jyc=Jyc.copy()
                
        self._Psic=Psic.copy() 
        self._dPsidJx=dPsidJx.copy()
        self._dPsidJy=dPsidJy.copy()
        self._interp_order= interp_order
        if interp_order==0:
            temp_interp_psi = interpolate.RegularGridInterpolator((self._Jxc,self._Jyc),self._Psic.T,
                                   method='nearest',bounds_error=False,fill_value=None)
            temp_interp_dPsidJx = interpolate.RegularGridInterpolator((self._Jxbx[1:-1],self._Jyc),self._dPsidJx.T,
                                   method='nearest',bounds_error=False,fill_value=None)
            temp_interp_dPsidJy = interpolate.RegularGridInterpolator((self._Jxc,self._Jyby[1:-1]),self._dPsidJy.T,
                                   method='nearest',bounds_error=False,fill_value=None)
            self.interp_psi     = temp_interp_psi
            self.interp_dPsidJx = temp_interp_dPsidJx
            self.interp_dPsidJy = temp_interp_dPsidJy
#             self.interp_psi     = lambda a,b: temp_interp_psi((a,b))
#             self.interp_dPsidJx = lambda a,b: temp_interp_dPsidJx((a,b))
#             self.interp_dPsidJy = lambda a,b: temp_interp_dPsidJy((a,b))
        else :
            J0 = [None,0][bool_radial]
            J0 = None
            temp_interp_psi     = interpolate.RectBivariateSpline(self._Jyc,self._Jxc,self._Psic,
                                                                 kx=interp_order,ky=interp_order,bbox=[0,Jyby[-1],0,Jxbx[-1]])
            temp_interp_dPsidJx = interpolate.RectBivariateSpline(self._Jyc,self._Jxbx[1:-1],self._dPsidJx,
                                                                 kx=interp_order,ky=interp_order,bbox=[J0,Jyby[-1],J0,Jxbx[-1]])
            temp_interp_dPsidJy = interpolate.RectBivariateSpline(self._Jyby[1:-1],self._Jxc,self._dPsidJy,
                                                                 kx=interp_order,ky=interp_order,bbox=[J0,Jyby[-1],J0,Jxbx[-1]])
            #Notice shift of input a,b -> b,a
            self.interp_psi     = temp_interp_psi
            self.interp_dPsidJx = temp_interp_dPsidJx
            self.interp_dPsidJy = temp_interp_dPsidJy
#             self.interp_psi = lambda a,b:     temp_interp_psi(b,a,grid=False)
#             self.interp_dPsidJx = lambda a,b: temp_interp_dPsidJx(b,a,grid=False)
#             self.interp_dPsidJy = lambda a,b: temp_interp_dPsidJy(b,a,grid=False)
                


    def getValue(self, jx, jy=0):
        if self._interp_order==0:
            return self.interp_psi((jx,jy))
        else:
            return self.interp_psi(jy,jx,grid=False)

    def getDJx(self, jx, jy=0):
        if self._interp_order==0:
            return self.interp_dPsidJx((jx,jy))
        else:
            return self.interp_dPsidJx(jy,jx,grid=False)
#         return self.interp_dPsidJx(jx,jy)

    def getDJy(self, jx, jy=0):
        if self._interp_order==0:
            return self.interp_dPsidJx((jx,jy))
        else:
            return self.interp_dPsidJx(jy,jx,grid=False)
#         return self.interp_dPsidJy(jx,jy)

class MyDistribution2Dy(MyDistribution2D):
    def __init__(self,Jxbx=0,Jxc=0,Jyby=0,Jyc=0,Psic=0,dPsidJx=0,dPsidJy=0,bool_radial=0,interp_order=0):
        MyDistribution2D.__init__(self,Jyby,Jyc,Jxbx,Jxc,Psic.T,dPsidJy.T,dPsidJx.T,bool_radial,interp_order)
        return
