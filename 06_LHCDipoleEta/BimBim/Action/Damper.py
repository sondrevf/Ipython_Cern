import scipy.sparse as spm
import numpy as np
from BimBim.Action import Action
from ..Error import BimBimError
from ..Matrix import printMatrix

class Damper(Action.Action):
    _gainX=0
    _gainY=0
    _cosphaseX=1.0
    _sinphaseX=0.0
    _cosphaseY=1.0
    _sinphaseY=0.0
    
    def __init__(self,gainX,gainY=0.0,phaseX=0.0,phaseY=0.0):
        self._gainX = gainX
        self._gainY = gainY
        self._cosphaseX = np.cos(phaseX)
        self._sinphaseX = np.sin(phaseX)
        self._cosphaseY = np.cos(phaseY)
        self._sinphaseY = np.sin(phaseY)
        #print(self._cosphaseX,self._sinphaseX)

    #Build perfect damper matrix
    def getDamperMatrix(self,basis):
        retVal = spm.identity(basis.getBunchBlockSize(),format='dok')
        if basis.getNDim() == 2:
            for iSlice in range(basis.getNSlice()):
                for iRing in range(basis.getNRing()):
                    j = basis.getIndexForSlice(iSlice,iRing)
                    for i in range(0,basis.getBunchBlockSize(),2):
                        retVal[i+1,j+1] -= 2.0*self._gainX*self._cosphaseX*basis.getWeight(iSlice,iRing)
                        retVal[i+1,j] -= 2.0*self._gainX*self._sinphaseX*basis.getWeight(iSlice,iRing)
        elif basis.getNDim() == 4:
            for iSlice in range(basis.getNSlice()):
                for iRing in range(basis.getNRing()):
                    j = basis.getIndexForSlice(iSlice,iRing)
                    for i in range(0,basis.getBunchBlockSize(),2):
                        retVal[i+1,j+1] -= 2.0*self._gainX*self._cosphaseX*basis.getWeight(iSlice,iRing)
                        retVal[i+1,j] -= 2.0*self._gainX*self._sinphaseX*basis.getWeight(iSlice,iRing)
                        retVal[i+3,j+3] -= 2.0*self._gainY*self._cosphaseY*basis.getWeight(iSlice,iRing)
                        retVal[i+3,j+2] -= 2.0*self._gainY*self._sinphaseY*basis.getWeight(iSlice,iRing)
        else:
            raise BimBimError('Damper is not implemented in '+str(basis.getNDim())+' dimensions')
        return retVal
      
    def getMatrix(self,beams,pos,basis):
        retVal = spm.identity(basis.getSize(),format='dok')
        if beams.getBunchB1(pos) != None:
            dMatrix = self.getDamperMatrix(basis)
            dMatrix = basis.getBunchProjection(1,beams.getBunchB1(pos).getNumber(),dMatrix)
            retVal = retVal.dot(dMatrix)
        if beams.getBunchB2(pos) != None:
            dMatrix = self.getDamperMatrix(basis)
            dMatrix = basis.getBunchProjection(2,beams.getBunchB2(pos).getNumber(),dMatrix)
            retVal = retVal.dot(dMatrix)
        myFile = open('debug_damper.mat','w')
        printMatrix(myFile,retVal)
        myFile.close()
        return retVal

    def setGain(self,gainX,gainY=None):
        self._gainX = gainX
        if gainY != None:
            self._gainY = gainY
            
            
class SingleTurnADT(Action.Action):
    _gainX=0
    _gainY=0
    _omega=0
    _soffset = 0
    
    # omega needs to be multiplied by sigs : omega = 2*pi*frec*sigs/c (TODO give bunch as an argument for getDamperMatrix instead)
    # soffset is an offset with in units of the bunch length (same issue as above)
    def __init__(self,gainX,phaseX=0.0,gainHTX=0.0,phaseHTX=0.0,gainY=0.0,phaseY=0.0,gainHTY=0.0,phaseHTY=0.0,omega=0,soffset=0):
        self._gainX = gainX
        self._cosphaseX = np.cos(phaseX)
        self._sinphaseX = np.sin(phaseX)
        self._gainHTX = gainHTX
        self._cosphaseHTX = np.cos(phaseHTX)
        self._sinphaseHTX = np.sin(phaseHTX)
        self._gainY = gainY
        self._cosphaseY = np.cos(phaseY)
        self._sinphaseY = np.sin(phaseY)
        self._gainHTY = gainHTY
        self._cosphaseHTY = np.cos(phaseHTY)
        self._sinphaseHTY = np.sin(phaseHTY)
        self._omega = omega
        self._soffset = soffset

    #Build ADT matrix
    def getDamperMatrix(self,basis):
        retVal = spm.identity(basis.getBunchBlockSize(),format='dok')
        if basis.getNDim() == 2:
            Isum = 0
            Qsum = 0
            for iSlice in range(basis.getNSlice()):
                for iRing in range(basis.getNRing()):
                    print(iSlice,iRing,self._omega*(basis.getSPosition(iSlice,iRing,1.0)+self._soffset)/(2.0*np.pi))
                    Isum += basis.getWeight(iSlice,iRing)*np.cos(self._omega*(basis.getSPosition(iSlice,iRing,1.0)+self._soffset))
                    Qsum += basis.getWeight(iSlice,iRing)*np.sin(self._omega*(basis.getSPosition(iSlice,iRing,1.0)+self._soffset))
            norm = Isum*Isum+Qsum*Qsum
            #print norm,Isum,Qsum
            for iSlice in range(basis.getNSlice()):
                for iRing in range(basis.getNRing()):
                    j = basis.getIndexForSlice(iSlice,iRing)
                    
                    Idiff = basis.getWeight(iSlice,iRing)*np.cos(self._omega*(basis.getSPosition(iSlice,iRing,1.0)+self._soffset))
                    Qdiff = basis.getWeight(iSlice,iRing)*np.sin(self._omega*(basis.getSPosition(iSlice,iRing,1.0)+self._soffset))
                    
                    kickRes = -2.0*self._gainX*self._cosphaseX*(Idiff*Isum+Qdiff*Qsum)/norm
                    kickRea = -2.0*self._gainX*self._sinphaseX*(Idiff*Isum+Qdiff*Qsum)/norm
                    kickHTRes = -2.0*self._gainHTX*self._cosphaseHTX*(Qdiff*Isum-Idiff*Qsum)/norm
                    kickHTRea = -2.0*self._gainHTX*self._sinphaseHTX*(Qdiff*Isum-Idiff*Qsum)/norm
                    for i in range(0,basis.getBunchBlockSize(),2):
                        retVal[i+1,j+1] = kickRes+kickHTRes
                        retVal[i+1,j] = kickRea+kickHTRea
                        if i==j:
                            retVal[i+1,j+1] += 1.0
        else:
            raise BimBimError('Damper is not implemented in '+str(basis.getNDim())+' dimensions')
        return retVal
      
    def getMatrix(self,beams,pos,basis):
        retVal = spm.identity(basis.getSize(),format='dok')
        if beams.getBunchB1(pos) != None:
            dMatrix = self.getDamperMatrix(basis)
            dMatrix = basis.getBunchProjection(1,beams.getBunchB1(pos).getNumber(),dMatrix)
            retVal = retVal.dot(dMatrix)
        if beams.getBunchB2(pos) != None:
            dMatrix = self.getDamperMatrix(basis)
            dMatrix = basis.getBunchProjection(2,beams.getBunchB2(pos).getNumber(),dMatrix)
            retVal = retVal.dot(dMatrix)
        #myFile = open('debug_damper.mat','w')
        #printMatrix(myFile,retVal)
        #myFile.close()
        return retVal

    def setGain(self,gainX,gainY=None):
        self._gainX = gainX
        if gainY != None:
            self._gainY = gainY
