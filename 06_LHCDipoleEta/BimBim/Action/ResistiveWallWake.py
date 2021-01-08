import numpy as np

class ResistiveWallWake:

    def __init__(self,resistivity,quadWakeRatio=0.0):
        self._resistivity = resistivity

        self._quadWakeRatio = quadWakeRatio

    # distance in ns
    def getWake(self,distance):
        retVal = self._resistivity/np.sqrt(1E-9*distance) #TODO
        return retVal,retVal,self._quadWakeRatio*retVal,self._quadWakeRatio*retVal
