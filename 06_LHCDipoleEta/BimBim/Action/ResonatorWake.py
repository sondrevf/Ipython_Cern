import numpy as np

class ResonatorWake:

    def __init__(self,shuntImpedance,resonanceFrequency=1E9,qualityFactor=1.0,quadWakeRatio=0.0):
        self._omegar = 2.0*np.pi*resonanceFrequency
        self._R = shuntImpedance
        self._Qr = qualityFactor
        self._QPrime = np.sqrt(qualityFactor**2-0.25)
        self._omegaPrime = self._QPrime*self._omegar/self._Qr
        #self._omega1 = self._omegar*(0.5j+self._QPrime)/qualityFactor
        #self._omega2 = self._omegar*(0.5j-self._QPrime)/qualityFactor # for omegar < 0.5

        self._quadWakeRatio = quadWakeRatio

    # distance in ns
    def getWake(self,distance):
        #retVal = np.real(-0.5j*self._omegar*self._R*(np.exp(1E-9j*self._omega1*distance)-np.exp(1E-9j*self._omega2*distance))/self._Qprime) # for omegar < 0.5
        retVal = self._omegar*self._R*np.exp(-1E-9*self._omegar*distance/(2.0*self._Qr))*np.sin(1E-9*self._omegaPrime*distance)/self._QPrime
        return retVal,retVal,self._quadWakeRatio*retVal,self._quadWakeRatio*retVal
