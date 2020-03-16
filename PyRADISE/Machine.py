import copy 
import numpy as np
from PyRADISE.Coefficients import find_DiffusionSidebandWeight

class MachineClass(object):
    def __init__(self,detuning,noise,gx=0,gy=0,f_rev = 11245,
                 wmodeQ0x=[],wmode__DQx=[],wmodeDipmx=[],wmodeQ0y=[],wmode__DQy=[],wmodeDipmy=[],
                 Qpx=0,Qpy=0,Qs=0.002,sigma_dpp=1e-4):
        self.Q = detuning
        self.Qy= copy.deepcopy(detuning)
        self.Q.call_plane=0 ; self.Qy.call_plane=1
        self.N = noise
        
        self.gx = gx
        self.gy = gy
        self.f_rev = f_rev
        
        #Longitudinal dynamics
        self.Qpx= Qpx
        self.Qpy= Qpy 
        self.Qs = Qs 
        self.sigma_dpp = sigma_dpp
        self.factorsSBx,self.ordersSBx=find_DiffusionSidebandWeight(np.abs(self.Qpx*self.sigma_dpp/self.Qs),debug=1)
        self.factorsSBy,self.ordersSBy=find_DiffusionSidebandWeight(np.abs(self.Qpy*self.sigma_dpp/self.Qs),debug=1)
        
        # Wake in np.arrays
        self.flag_wmode= np.size(wmode__DQx)+np.size(wmode__DQy)>0
        self.nWmodex = nWmodex = np.min([np.size(wmode__DQx),np.size(wmodeQ0x),np.size(wmodeDipmx)])
        self.wmodeQ0x   = np.array([wmodeQ0x  ]) if np.shape(wmodeQ0x  )==() else np.array(wmodeQ0x[:nWmodex])
        self.wmode__DQx = np.array([wmode__DQx]) if np.shape(wmode__DQx)==() else np.array(wmode__DQx[:nWmodex])
        self.wmodeLdDQx = np.zeros_like(self.wmode__DQx)
#        self.wmodeMgDQx = np.zeros_like(self.wmode__DQx)
        self.wmodeDipmx = np.array([wmodeDipmx]) if np.shape(wmodeDipmx)==() else np.array(wmodeDipmx[:nWmodex])
        
        self.nWmodey = nWmodey = np.min([np.size(wmode__DQy),np.size(wmodeQ0y),np.size(wmodeDipmy)])
        self.wmodeQ0y   = np.array([wmodeQ0y  ]) if np.shape(wmodeQ0y  )==() else np.array(wmodeQ0y[:nWmodey])
        self.wmode__DQy = np.array([wmode__DQy]) if np.shape(wmode__DQy)==() else np.array(wmode__DQy[:nWmodey])
        self.wmodeLdDQy = np.zeros_like(self.wmode__DQy)
#        self.wmodeMgDQy = np.zeros_like(self.wmode__DQy)
        self.wmodeDipmy = np.array([wmodeDipmy]) if np.shape(wmodeDipmy)==() else np.array(wmodeDipmy[:nWmodey])
        
