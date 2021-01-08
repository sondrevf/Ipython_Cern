from BimBim.Action import Action
import numpy as np
from time import time
import scipy.sparse as spm
from ..Matrix import printMatrix,sparse_insert
from ..Error import BimBimError

class Transport(Action.Action):
    _phaseX = 0;
    _phaseY = 0;
    _betaX = 0;
    _betaY = 0;
    _qs = 0;
    _chromaX = 0;
    _chromaY = 0;
    _2ndOrderChromaX = 0;
    _2ndOrderChromaY = 0;
    _transition = -1;
    _clight = 2.99792458e8;
    
    _keepInMemory = True;
    _syncBetMaps = None;

    #phaseX,betaX,phaseY,betaY,chromaX,chromaY can be different for each beams -> give an array [value for B1, value for B2]
    def __init__(self,phaseX,betaX,qs,phaseY=0,betaY=0,chromaX=0,chromaY=0,keepInMemory=True,transition = -1,secondOrderChromaX=0.0,secondOrderChromaY=0.0):
        self._syncBetMaps = {};
        
        self.setPhaseX(phaseX);
        self.setPhaseY(phaseY);
        self.setBetaX(betaX);
        self.setBetaY(betaY);
        self.setQs(qs);
        self.setChromaX(chromaX);
        self.setChromaY(chromaY);
        self.set2ndOrderChromaX(secondOrderChromaX);
        self.set2ndOrderChromaY(secondOrderChromaY);
        self._keepInMemory = keepInMemory;
        self._transition = transition;
    
    def setPhaseX(self,phaseX):
        self._phaseX = [];
        try:
            for i in range(len(phaseX)):
                self._phaseX.append(phaseX[i]);
        except TypeError:
            self._phaseX.append(phaseX);
            self._phaseX.append(phaseX);
        self._syncBetMaps.clear();

    def setPhaseY(self,phaseY):
        self._phaseY = [];
        try:
            for i in range(len(phaseY)):
                self._phaseY.append(phaseY[i]);
        except TypeError:
            self._phaseY.append(phaseY);
            self._phaseY.append(phaseY);
        self._syncBetMaps.clear();

    def setBetaX(self,betaX):
        self._betaX = [];
        try:
            for i in range(len(betaX)):
                self._betaX.append(betaX[i]);
        except TypeError:
            self._betaX.append(betaX);
            self._betaX.append(betaX);
        self._syncBetMaps.clear();

    def setBetaY(self,betaY):
        self._betaY = [];
        try:
            for i in range(len(betaY)):
                self._betaY.append(betaY[i]);
        except TypeError:
            self._betaY.append(betaY);
            self._betaY.append(betaY);
        self._syncBetMaps.clear();

    def setChromaX(self,chromaX):
        self._chromaX = [];
        try:
            for i in range(len(chromaX)):
                self._chromaX.append(chromaX[i]);
        except TypeError:
            self._chromaX.append(chromaX);
            self._chromaX.append(chromaX);
        self._syncBetMaps.clear();

    def setChromaY(self,chromaY):
        self._chromaY = [];
        try:
            for i in range(len(chromaY)):
                self._chromaY.append(chromaY[i]);
        except TypeError:
            self._chromaY.append(chromaY);
            self._chromaY.append(chromaY);
        self._syncBetMaps.clear();();

    def set2ndOrderChromaX(self,secondOrderChromaX):
        self._2ndOrderChromaX = [];
        try:
            for i in range(len(secondOrderChromaX)):
                self._2ndOrderChromaX.append(secondOrderChromaX[i]);
        except TypeError:
            self._2ndOrderChromaX.append(secondOrderChromaX);
            self._2ndOrderChromaX.append(secondOrderChromaX);
        self._syncBetMaps.clear();

    def set2ndOrderChromaY(self,secondOrderChromaY):
        self._2ndOrderChromaY = [];
        try:
            for i in range(len(secondOrderChromaY)):
                self._2ndOrderChromaY.append(secondOrderChromaY[i]);
        except TypeError:
            self._2ndOrderChromaY.append(secondOrderChromaY);
            self._2ndOrderChromaY.append(secondOrderChromaY);
        self._syncBetMaps.clear();

    def setQs(self,qs):
        self._qs = [];
        try:
            for i in range(len(qs)):
                self._qs.append(qs[i]);
        except TypeError:
            self._qs.append(qs);
            self._qs.append(qs);
        self._syncBetMaps.clear();();
        
    def getPhaseX(self,beamNumber=1):
        return self._phaseX[beamNumber-1];
        
    def getPhaseY(self,beamNumber=1):
        return self._phaseY[beamNumber-1];

    def getBetaX(self,beamNumber=1):
        return self._betaX[beamNumber-1];
    
    def getBetaY(self,beamNumber=1):
        return self._betaY[beamNumber-1];

    def getChromaX(self,beamNumber=1):
        return self._chromaX[beamNumber-1];
    
    def getChromaY(self,beamNumber=1):
        return self._chromaY[beamNumber-1];

    def get2ndOrderChromaX(self,beamNumber=1):
        return self._2ndOrderChromaX[beamNumber-1];

    def get2ndOrderChromaY(self,beamNumber=1):
        return self._2ndOrderChromaY[beamNumber-1];
        
    def getQs(self,beamNumber=1):
        return self._qs[beamNumber-1];

    def C_ij(self,i,j,nSlice,qs):
        phi = np.pi*(qs-1.0+(float(j)-float(i))/float(nSlice))
        return np.sin(float(nSlice)*phi)/(float(nSlice)*np.sin(phi))

    def getCirc(self,nSlice,qs):
        fullMat = np.array([[1.0 if j==i+1 or (i == nSlice-1 and j == 0) else 0.0 for j in range(nSlice)]for i in range(nSlice)]);
        val,vect = np.linalg.eig(fullMat);
        retVal = np.dot(vect,np.diag((val+0j)**(qs*nSlice)));
        retVal = np.dot(retVal,np.linalg.inv(vect));
        return np.real(retVal);

    #Define circulant matrix
    def getCircMap(self,basis,beamNumber):
        #retVal = spm.dok_matrix([[self.C_ij(j,i,basis.getNSlice(),self.getQs(beamNumber)) for j in range(basis.getNSlice())] for i in range(basis.getNSlice())])
        retVal = self.getCirc(basis.getNSlice(),self.getQs(beamNumber))
        retVal = spm.kron(spm.identity(basis.getNRing(),format='dok'),retVal,format='dok')
        #myFile = open('debug_circmap.mat','w');
        #printMatrix(myFile,retVal);
        #myFile.close();
        return retVal

    def getBetMap(self,basis,beamNumber,phX,phY):
        cosX = np.cos(2.0*np.pi*phX);
        sinX = np.sin(2.0*np.pi*phX);
        if basis.getNDim() == 2:
            retVal = spm.dok_matrix([[cosX,self.getBetaX(beamNumber)*sinX],[-1.0/self.getBetaX(beamNumber)*sinX,cosX]]);
        elif basis.getNDim() == 4:
            cosY = np.cos(2.0*np.pi*phY);
            sinY = np.sin(2.0*np.pi*phY);
            tmpArray = np.zeros((4,4));
            tmpArray[0][0]=cosX;
            tmpArray[0][1]=self.getBetaX(beamNumber)*sinX;
            tmpArray[1][1]=cosX;
            tmpArray[1][0]=-1.0/self.getBetaX(beamNumber)*sinX;
            tmpArray[2][2]=cosY;
            tmpArray[2][3]=self.getBetaY(beamNumber)*sinY;
            tmpArray[3][3]=cosY;
            tmpArray[3][2]=-1.0/self.getBetaY(beamNumber)*sinY;
            retVal = spm.dok_matrix(tmpArray);
        else:
            raise BimBimError('Transport is not implemented in '+str(self.getNDim())+' dimensions');
        return retVal;

    #Build incoming chromaticity phase shift matrix
    def getChromaMap(self,basis,beamNumber,sigp,direction=1.0):
        chrmat = spm.dok_matrix((basis.getBunchBlockSize(),basis.getBunchBlockSize()))
        for iSlice in range(basis.getNSlice()):
            for iRing in range(basis.getNRing()):
                phaseX = direction*(self.getChromaX(beamNumber)*basis.getDPoverP(iSlice,iRing,sigp,self._transition)+0.5*self.get2ndOrderChromaX(beamNumber)*basis.getDPoverP(iSlice,iRing,sigp,self._transition)**2);
                phaseY = direction*(self.getChromaY(beamNumber)*basis.getDPoverP(iSlice,iRing,sigp,self._transition)+0.5*self.get2ndOrderChromaY(beamNumber)*basis.getDPoverP(iSlice,iRing,sigp,self._transition)**2);
                betMap = self.getBetMap(basis,beamNumber,phaseX,phaseY);
                index = basis.getNDim()*(iSlice+iRing*basis.getNSlice());
                sparse_insert(chrmat,betMap,index);
        #myFile = open('debug_chroma.mat','w');
        #printMatrix(myFile,chrmat);
        #myFile.close();
        return chrmat;

    #Build incoming chromaticity phase shift matrix
    def getChromaMap2(self,basis,beamNumber,sigp,direction):
        frev = 11245.5
        eta = 3.225e-4
        phfact = 2*np.pi*frev/(eta*self._clight)
        chrmat = spm.dok_matrix((basis.getBunchBlockSize(),basis.getBunchBlockSize()))
        for iSlice in range(basis.getNSlice()):
            for iRing in range(basis.getNRing()):
                phaseX = direction*self.getChromaX(beamNumber)*phfact*basis.getSPosition(iSlice,iRing,0.0755)/(2.0*np.pi);
                phaseY = direction*self.getChromaY(beamNumber)*phfact*basis.getSPosition(iSlice,iRing,0.0755)/(2.0*np.pi);
                betMap = self.getBetMap(basis,beamNumber,phaseX,phaseY);
                index = basis.getNDim()*(iSlice+iRing*basis.getNSlice());
                sparse_insert(chrmat,betMap,index);
        # myFile = open('debug_chroma.mat','w');
        # printMatrix(myFile,chrmat);
        # myFile.close();
        return chrmat;

    def getSyncBetMap(self,basis,beamNumber,bunch):
        key = (basis,beamNumber,bunch.getSigP());
        if key in self._syncBetMaps.keys():
            return self._syncBetMaps[key];
        else:
            betMap = self.getBetMap(basis,beamNumber,self.getPhaseX(beamNumber),self.getPhaseY(beamNumber));
            circMap = self.getCircMap(basis,beamNumber);
            chromaIn = self.getChromaMap(basis,beamNumber,bunch.getSigP(),direction=1.0);
            #chromaIn = self.getChromaMap2(basis,beamNumber,bunch.getSigP(),direction=1.0);
            #chromaOut = self.getChromaMap2(basis,beamNumber,bunch.getSigP(),direction=-1.0);
            retVal = spm.kron(circMap,betMap,format='dok');
            #retVal = chromaOut.dot(retVal);
            retVal = retVal.dot(chromaIn);
            self._syncBetMaps[key] = retVal;
            return retVal;

    def getMatrix(self,beams,pos,basis):
        bunchB1 = beams.getBunchB1(pos);
        bunchB2 = beams.getBunchB2(pos);
        if bunchB1 != None and bunchB2 == None:
            #time0 = time();
            beamNumber = 1;
            syncBet = self.getSyncBetMap(basis,beamNumber,bunchB1);
            retVal = basis.getBunchProjection(beamNumber,bunchB1.getNumber(),syncBet);
            #myFile = open('debug_transportB1.mat','w');
            #printMatrix(myFile,retVal);
            #myFile.close();
            #time1 = time();
            #print 'Time to build transport matrix for B1b'+str(bunchB1.getNumber()),time1-time0
            return retVal
        elif bunchB1 == None and bunchB2 != None:
            #time0 = time();
            beamNumber = 2;
            syncBet = self.getSyncBetMap(basis,beamNumber,bunchB2);
            retVal = basis.getBunchProjection(beamNumber,bunchB2.getNumber(),syncBet);
            #myFile = open('debug_transportB2.mat','w');
            #printMatrix(myFile,retVal);
            #myFile.close();
            #time1 = time();
            #print 'Time to build transport matrix for B2b'+str(bunchB2.getNumber()),time1-time0
            return retVal;
        else:
            #time0 = time();
            beamNumber = 1;
            syncBet = self.getSyncBetMap(basis,beamNumber,bunchB1);
            projB1 = basis.getBunchProjection(beamNumber,bunchB1.getNumber(),syncBet);
            beamNumber = 2;
            syncBet = self.getSyncBetMap(basis,beamNumber,bunchB2);
            projB2 = basis.getBunchProjection(beamNumber,bunchB2.getNumber(),syncBet);
            retVal = projB1.dot(projB2);
            #myFile = open('debug_transport.mat','w');
            #printMatrix(myFile,retVal);
            #myFile.close();
            #time1 = time();
            #print 'Time to build transport matrix for B1b'+str(bunchB1.getNumber())+' and B2b'+str(bunchB2.getNumber()),time1-time0
            return retVal;
