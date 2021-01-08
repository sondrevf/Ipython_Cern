import scipy.sparse as spm
import numpy as np
from BimBim.Error import BimBimError
from BimBim.Matrix import printMatrix,sparse_insert
from scipy.special import erf

# Description : beam X bunch X ring X slice X transverse plane

class Basis:
    _nBeam = 1;
    _nBunchB1 = 0;
    _nBunchB2 = 0;
    _nSlice = 0;
    _nRing = 0;
    _nDim = 4;
    _basisSize = 0;
    _bunchBlockSize = 0;
    _beamBlockSizeB1 = 0;
    _beamBlockSizeB2 = 0;
    
    _minAmpl = 0.0;
    _maxAmpl = 3.0; # sigma
    _ringDistribution = None;
    
    _equiDistantRing = False;
    
    def __init__(self,nBunchB1,nBunchB2,nSlice,nRing,nDim,equiDistantRing=False):
        #if nSlice%2 == 0:
        #    raise BimBimError('The number of slice has to be odd ('+str(nSlice)+')');
        if nBunchB2 > 0:
            self._nBeam = 2;
        self._nBunchB1 = nBunchB1;
        self._nBunchB2 = nBunchB2;
        self._nSlice = nSlice;
        self._nRing = nRing;
        self._nDim = nDim;
        self._basisSize = (self._nBunchB1+self._nBunchB2)*self._nSlice*self._nRing*self._nDim;
        self._bunchBlockSize = self._nSlice*self._nRing*self._nDim;
        self._beamBlockSizeB1 = self._nBunchB1*self._bunchBlockSize;
        self._beamBlockSizeB2 = self._nBunchB2*self._bunchBlockSize;
        self._equiDistantRing = equiDistantRing;
        self._generateRingDistribution();

    def _generateRingDistribution(self):
        boundaries = [];
        if self._nRing > 1:
            if self._equiDistantRing:
                dx = (self._maxAmpl - self._minAmpl)/self._nRing;
                boundaries = dx*np.arange(self._nRing+1)
                #for i in range(self._nRing+1):
                #    boundaries.append(i*dx);
            else:
                boundaries = np.zeros(self._nRing+1,dtype=float)
                for i in range(1,self._nRing):
                    lbd = boundaries[i-1]
                    boundaries[i]= np.sqrt(-2.0*(np.log(np.exp(-lbd**2/2)-1.0/self._nRing)))
                boundaries[self._nRing] = self._maxAmpl
        else:
            boundaries.append(0.0);
            boundaries.append(self._maxAmpl);
        
        radius = [];
        weight = [];
        for i in range(self._nRing):
            lbd = boundaries[i];
            ubd = boundaries[i+1];
            weight.append(np.exp(-lbd**2/2)-np.exp(-ubd**2/2))
            radius.append((np.sqrt(np.pi/2)*(erf(ubd/np.sqrt(2))-erf(lbd/np.sqrt(2)))-(ubd*np.exp(-ubd**2/2)-lbd*np.exp(-lbd**2/2)))/weight[-1])

            #weight.append(erf(ubd/np.sqrt(2))-erf(lbd/np.sqrt(2))) # From EG
            #radius.append(np.sqrt(2/np.pi)*(np.exp(-lbd**2/2)-np.exp(-ubd**2/2))/(erf(ubd/np.sqrt(2))-erf(lbd/np.sqrt(2)))) # from EG
         
        self._ringDistribution = [radius,weight,boundaries]

    def getNBeam(self):
        return self._nBeam;
        
    def getSize(self):
        return self._basisSize;
        
    def getNSlice(self):
        return self._nSlice;
        
    def getNRing(self):
        return self._nRing;
    
    def getNDim(self):
        return self._nDim;
    
    def getIndexForBeam(self,beam):
        if beam == 1:
            return 0;
        else:
            return self._nBunchB1*self._bunchBlockSize;
    
    def getIndexForBunch(self,bunch,beam=1):
        if beam == 1:
            return bunch*self._bunchBlockSize;
        else:
            return (self._nBunchB1 + bunch)*self._bunchBlockSize;
    
    def getIndexForSlice(self,iSlice,iRing,beam=1,bunch=0):
        retVal = bunch*self._bunchBlockSize + self._nDim*(iRing*self._nSlice + iSlice);
        if beam == 1:
            return retVal;
        else:
            retVal += self._nBunchB1*self._bunchBlockSize;
            return retVal;

    def getBunchBlockSize(self):
        return self._bunchBlockSize;
        
    def getBeamBlockSize(self,beam):
        if beam==1:
            return self._beamBlockSizeB1;
        else:
            return self._beamBlockSizeB2;

    #Project the matrix for a single bunch into the basis (bunch block only)
    def getBunchProjection(self,beam,bunch,inputMatrix):
        if inputMatrix.get_shape() != (self._bunchBlockSize,self._bunchBlockSize):
            raise BimBimError('Matrix could not be projected : wrong shape '+str(inputMatrix.get_shape())+' instead of'+str((self._bunchBlockSize,self._bunchBlockSize)));
        matrix = inputMatrix.todok();
        proj = spm.identity(self._basisSize,format='dok');
        index = self.getIndexForBunch(bunch,beam);
        sparse_insert(proj,matrix,index);
        #myFile = open('debug_proj.mat','w');
        #printMatrix(myFile,proj);
        #myFile.close();
        return proj;
        
    #Project the matrix for a single bunch into the basis (all line i.e. all bunch depdencies)
    def getFullBunchProjection(self,beam,bunch,inputMatrix):
        if inputMatrix.get_shape() != (self._bunchBlockSize,self.getBeamBlockSize(beam)):
            raise BimBimError('Matrix could not be projected : wrong shape '+str(inputMatrix.get_shape())+' instead of'+str((self._bunchBlockSize,self.getBeamBlockSize(beam))));
        matrix = inputMatrix.todok();
        proj = spm.identity(self._basisSize,format='dok');
        sparse_insert(proj,matrix,self.getIndexForBunch(bunch,beam),self.getIndexForBeam(beam));
        #myFile = open('debug_proj.mat','w');
        #printMatrix(myFile,proj);
        #myFile.close();
        return proj;        
        
    #project the matrix of two bunches into the basis
    def getTwoBunchProjection(self,bunchB1,bunchB2,inputMatrix):
        if inputMatrix.get_shape() != (2*self._bunchBlockSize,2*self._bunchBlockSize):
            raise BimBimError('Matrix could not be projected : wrong shape '+str(inputMatrix.get_shape())+' instead of'+str((2*self._bunchBlockSize,2*self._bunchBlockSize)));
        retVal = spm.identity(self._basisSize,format='dok');
        matrix = inputMatrix.todok();
        for i in range(self._bunchBlockSize):
            for j in range(self._bunchBlockSize):
                retVal[i+self.getIndexForBunch(bunchB1,1),j+self.getIndexForBunch(bunchB1,1)] = matrix[i,j]; #copy diagonal term of B1 bunch
                retVal[i+self.getIndexForBunch(bunchB2,2),j+self.getIndexForBunch(bunchB2,2)] = matrix[i+self._bunchBlockSize,j+self._bunchBlockSize]; #copy diagonal term of B2 bunch
                retVal[i+self.getIndexForBunch(bunchB1,1),j+self.getIndexForBunch(bunchB2,2)] = matrix[i,j+self._bunchBlockSize]; # copy upper left block
                retVal[i+self.getIndexForBunch(bunchB2,2),j+self.getIndexForBunch(bunchB1,1)] = matrix[i+self._bunchBlockSize,j]; # copy lower right block
        return retVal;
        
    # project the matrix of interaction of two element into the basis
    def getTwoElementProjection(self,bunchB1,iSliceB1,iRingB1,bunchB2,iSliceB2,iRingB2,inputMatrix):
        if inputMatrix.get_shape() != (2*self._nDim,2*self._nDim):
            raise BimBimError('Matrix could not be projected : wrong shape '+str(inputMatrix.get_shape())+' instead of'+str((2*self._bunchBlockSize,2*self._bunchBlockSize)));       
        retVal = spm.identity(self._basisSize,format='dok');
        for i in range(self._nDim):
            for j in range(self._nDim):
                retVal[i+self.getIndexForSlice(iSliceB1,iRingB1,1,bunchB1),j+self.getIndexForSlice(iSliceB1,iRingB1,1,bunchB1)] = inputMatrix[i,j];
                retVal[i+self.getIndexForSlice(iSliceB2,iRingB2,2,bunchB2),j+self.getIndexForSlice(iSliceB2,iRingB2,2,bunchB2)] = inputMatrix[i+self._nDim,j+self._nDim];
                retVal[i+self.getIndexForSlice(iSliceB1,iRingB1,1,bunchB1),j+self.getIndexForSlice(iSliceB2,iRingB2,2,bunchB2)] = inputMatrix[i,j+self._nDim];
                retVal[i+self.getIndexForSlice(iSliceB2,iRingB2,2,bunchB2),j+self.getIndexForSlice(iSliceB1,iRingB1,1,bunchB1)] = inputMatrix[i+self._nDim,j];
        return retVal;
        
    def getRingRadius(self,iRing):
        return self._ringDistribution[0][iRing];

    def getRingWeight(self,iRing):
        return self._ringDistribution[1][iRing];

    def getWeight(self,iSlice,iRing):
        return self.getRingWeight(iRing)/self._nSlice;

    def getWeightMatrix(self):
        retVal = np.zeros(shape=(self._nSlice*self._nRing,self._nSlice*self._nRing));
        for line in range(self._nSlice*self._nRing):
            for iSlice in range(self._nSlice):
                for iRing in range(self._nRing):
                    retVal[line,iSlice + iRing*self._nSlice] = self.getWeight(iSlice,iRing);
        return retVal;

    def getSliceAngle(self,iSlice):
        return (iSlice*2+1)*np.pi/float(self._nSlice);

    def getSPosition(self,iSlice,iRing,sigs):
        return self.getRingRadius(iRing)*sigs*np.cos(self.getSliceAngle(iSlice));

    def getDPoverP(self,iSlice,iRing,sigp,transition):
        return transition*self.getRingRadius(iRing)*sigp*np.sin(self.getSliceAngle(iSlice));

    def getSDiff(self,iSliceTest,iRingTest,iSliceSource,iRingSource,sigs,threshold=1E-12):
        retVal = self.getSPosition(iSliceSource,iRingSource,sigs)-self.getSPosition(iSliceTest,iRingTest,sigs);
        if abs(retVal) > threshold:
            return retVal;
        else:
            return 0.0;

    def getDiscretisationIndicesFromFullVectorIndex(self,index):
        intraBunchIndex = index%self._bunchBlockSize
        bunch = int((index-intraBunchIndex)/self._bunchBlockSize)
        if bunch < self._nBunchB1:
            beam = 1
        else:
            beam = 2
            bunch -= self._nBunchB1
        dim = intraBunchIndex%self._nDim
        intraBunchIndex = int((intraBunchIndex-dim)/self._nDim)
        iRing = intraBunchIndex%self._nRing
        iSlice = int((intraBunchIndex-iRing)/self._nRing)
        return iSlice,iRing,dim,beam,bunch

    def getLongitudinalPhaseSpacePositionFromFullVectorIndex(self,index,sigs,sigp,transition=-1):
        iSlice,iRing,dim,beam,bunch = self.getDiscretisationIndicesFromFullVectorIndex(index)
        return self.getSPosition(iSlice,iRing,sigs),self.getDPoverP(iSlice,iRing,sigp,transition)

    #dof = 0 for horizontal position, 1 for horizontal divergence, 2 for vertical position, 2 for vertical divergence
    def getModeFirstOrderMoment(self,eigvec,dof=0):
        if dof > self._nDim:
            raise BimBimError('Cannot compute first order moment for request degree of freedom '+str(dof)+' > '+str(self._nDim)+'D');

        retValB1 = [];
        retValB2 = [];
        for bunch in range(self._nBunchB1):
            retValB1.append(0.0);
            norm = 0.0
            for iSlice in range(self._nSlice):
                for iRing in range(self._nRing):
                    norm += np.abs(eigvec[self.getIndexForSlice(iSlice,iRing,1,bunch)+dof])**2 # Not clear why this norm is not always 0.5
                    retValB1[bunch] += eigvec[self.getIndexForSlice(iSlice,iRing,1,bunch)+dof]*self.getWeight(iSlice,iRing)
            retValB1[bunch] /= np.sqrt(norm/self._nSlice/self._nRing)
        for bunch in range(self._nBunchB2):
            retValB2.append(0.0);
            tmp1 = 0.0
            tmp2 = 0.0
            for iSlice in range(self._nSlice):
                for iRing in range(self._nRing):
                    norm += np.abs(eigvec[self.getIndexForSlice(iSlice,iRing,2,bunch)+dof])**2
                    retValB2[bunch] += eigvec[self.getIndexForSlice(iSlice,iRing,2,bunch)+dof]*self.getWeight(iSlice,iRing);
            retValB2[bunch] /= np.sqrt(norm/self._nSlice/self._nRing)
        return retValB1,retValB2;

    #dof = 0 for horizontal position 2 for vertical position
    def getSingleBunchModeComplexMatrixRepresentation(self,eigvec,dof=0,beam=1,bunch=0):
        if dof > self._nDim:
            raise BimBimError('Cannot compute first order moment for request degree of freedom '+str(dof)+' > '+str(self._nDim)+'D');
        if dof == 1 or dof == 3:
            raise BimBimError('Cannot compute first order moment for request degree of freedom '+str(dof)+' (only 0 for horizontal and 2 for vertical)');
        retMat = np.zeros((self._nSlice,self._nRing),dtype=complex)
        for iSlice in range(self._nSlice):
            for iRing in range(self._nRing):
                retMat[iSlice,iRing] = eigvec[self.getIndexForSlice(iSlice,iRing,beam,bunch)+dof]+1j*eigvec[self.getIndexForSlice(iSlice,iRing,beam,bunch)+dof+1]
        return retMat

    def getLongitudinalProfileDiscretePoints(self,nZ=100):
        return np.arange(-self._maxAmpl,self._maxAmpl+self._maxAmpl/nZ,2.0*self._maxAmpl/nZ)

    def getElementLongitudinalProfileFromWeight(self,iSlice,iRing=0,nTheta=1000,nR=100,nZ=100):
        longitudinalProfile = np.zeros_like(self.getLongitudinalProfileDiscretePoints(nZ))
        dTheta = 2.0*np.pi/self._nSlice/nTheta
        for theta in np.arange(self.getSliceAngle(iSlice)-np.pi/self._nSlice,self.getSliceAngle(iSlice)+np.pi/self._nSlice,dTheta):
            dR = (self._ringDistribution[2][iRing+1]-self._ringDistribution[2][iRing])/nR
            for R in np.arange(self._ringDistribution[2][iRing],self._ringDistribution[2][iRing+1],dR):
                zIndex = int(np.floor(nZ*R*np.cos(theta)/(2.0*self._maxAmpl)+nZ/2+0.5))
                longitudinalProfile[zIndex] += self.getWeight(iSlice,iRing)/nTheta/nR/(2.0*np.pi) # Why 2pi?
        return longitudinalProfile*nZ

    def getLongitudinalProfileFromWeight(self,nTheta=100,nR=100,nZ=100):
        longitudinalProfile = np.zeros_like(self.getLongitudinalProfileDiscretePoints(nZ))
        for iSlice in range(self._nSlice):
            for iRing in range(self._nRing):
                longitudinalProfile += self.getElementLongitudinalProfileFromWeight(iSlice,iRing,nTheta,nR,nZ)

        return longitudinalProfile

    def getElementLongitudinalProfile(self,iSlice,iRing=0,nTheta=1000,nR=100,nZ=100):
        longitudinalProfile = np.zeros_like(self.getLongitudinalProfileDiscretePoints(nZ))
        dTheta = 2.0*np.pi/self._nSlice/nTheta
        for theta in np.arange(self.getSliceAngle(iSlice)-np.pi/self._nSlice,self.getSliceAngle(iSlice)+np.pi/self._nSlice,dTheta):
            dR = (self._ringDistribution[2][iRing+1]-self._ringDistribution[2][iRing])/nR
            for R in np.arange(self._ringDistribution[2][iRing],self._ringDistribution[2][iRing+1],dR):
                zIndex = int(np.floor(nZ*R*np.cos(theta)/(2.0*self._maxAmpl)+nZ/2+0.5))
                longitudinalProfile[zIndex] += R*np.exp(-R**2/2)*dR*dTheta/(2.0*np.pi)**2  # Why 2pi?
        return longitudinalProfile*nZ


    def getLongitudinalProfile(self,nTheta=100,nR=100,nZ=100):
        longitudinalProfile = np.zeros_like(self.getLongitudinalProfileDiscretePoints(nZ))
        for iSlice in range(self._nSlice):
            for iRing in range(self._nRing):
                longitudinalProfile += self.getElementLongitudinalProfile(iSlice,iRing,nTheta,nR,nZ)

        return longitudinalProfile


