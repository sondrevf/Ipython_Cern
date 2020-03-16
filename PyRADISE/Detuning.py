import numpy as np


class DetuningClass(object):
#     @abstractmethod
    def __init__(self,*args):
        pass
    def dQx(self,Jx,Jy):
        pass
    def dQy(self,Jx,Jy):
        pass
    def dQxdJx(self,Jx,Jy):
        pass
    def dQxdJy(self,Jx,Jy):
        pass
    def dQydJx(self,Jx,Jy):
        pass
    def dQydJy(self,Jx,Jy):
        pass
        
class LinearDetuningClass(DetuningClass):
    def __init__(self,Q0x,Q0y,ax,bx,ay,by,call_plane=0):
        self.Q0x = Q0x
        self.Q0y = Q0y
        self.ax0 = ax
        self.bx0 = bx
        self.ay0 = ay
        self.by0 = by
        self.ax = ax
        self.bx = bx
        self.ay = ay
        self.by = by
        self.call_plane = call_plane
    def scale(self,scale):
        self.ax = self.ax0*scale
        self.bx = self.bx0*scale
        self.ay = self.ay0*scale
        self.by = self.by0*scale
    def dQx(self,Jx,Jy):
        return self.ax*Jx + self.bx*Jy
    def dQy(self,Jx,Jy):
        return self.by*Jx + self.ay*Jy
    def dQxdJx(self,Jx,Jy):
        return self.ax
    def dQxdJy(self,Jx,Jy):
        return self.bx
    def dQydJx(self,Jx,Jy):
        return self.by
    def dQydJy(self,Jx,Jy):
        return self.ay
    def __call__(self,Jx,Jy):
        if self.call_plane==0:
            return self.Q0x + self.dQx(Jx,Jy)
        else:
            return self.Q0y + self.dQy(Jy,Jx) # opposite order because of PySSD
    def callVertical(self):
        return LinearDetuningClassY(self)
    def callScaledStrength(self,scale):
        return LinearDetuningClassScale(self,scale)
