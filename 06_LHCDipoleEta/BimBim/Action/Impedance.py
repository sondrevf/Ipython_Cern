import scipy.sparse as spm
from time import time
import numpy as np
import os
import pickle

from BimBim.Action import Action
from ..Error import BimBimError
from ..Matrix import printMatrix,sparse_insert,sparse_add


class Impedance(Action.Action):

    # deactivate keepInMemory if all bunches are different
    # betas correspond to the ratio with respect to the tracking beta
    def __init__(self,wakeDefinition,betaX,betaY,quadWake=False,keepInMemory = True,pickleFileName=None,intensityScaling = 1.0):
        self._clight = 2.99792458e8;
        self._GeVToKg = 1.78266113385e-27;
        self._qe = 1.60217733e-19;

        self._betaX = betaX;
        self._betaY = betaY;
        self._quadWake = quadWake;
        self._wakeDefinition = wakeDefinition
        self._keepInMemory = keepInMemory;
        self._intensityScaling = intensityScaling;
        if pickleFileName == None:
            self._singleBunchMatrices = {};
        elif not os.path.exists(pickleFileName):
            print('Pickle does not exist',pickleFileName);
            self._singleBunchMatrices = {};
        else:
            print('Unpickling impedance matrix')
            pickleFile = open(pickleFileName,'rb');
            denseSingleBunchMatrices = pickle.load(pickleFile);
            pickleFile.close();
            for key in denseSingleBunchMatrices.keys():
                self._singleBunchMatrices[key] = spm.dok_matrix(denseSingleBunchMatrices[key])
        
    def pickleMatrices(self,fileName):
        print('Pickling impedance matrix')
        pickleFile = open(fileName,'wb');
        denseSingleMatrices = {}
        for key in self._singleBunchMatrices.keys():
            denseSingleBunchMatrices[key] = self._singleBunchMatrices[key].todense()
        pickle.dump(denseSingleBunchMatrices,pickleFile);
        pickleFile.close();
    
    def getSingleBunchImpedanceMatrix(self,basis,bunch):
        cst = self._qe**2/(bunch.getMass()*self._GeVToKg*bunch.getGamma()*bunch.getBeta()**2*self._clight**2)*bunch.getIntensity()*self._intensityScaling;
        sfact = self._clight/(bunch.getBeta()*1.0e9);
        key = (basis.getNSlice(),basis.getNRing(),basis.getNDim(),cst,sfact,bunch.getSigS());
        if key in self._singleBunchMatrices.keys():
            return self._singleBunchMatrices[key];
        else:
            zMatrix = spm.identity(basis.getBunchBlockSize(),format='dok');
            for iSliceSource in range(basis.getNSlice()):
                for iSliceTest in range(basis.getNSlice()):
                    for iRingSource in range(basis.getNRing()):
                        for iRingTest in range(basis.getNRing()):
                            distance = basis.getSDiff(iSliceTest,iRingTest,iSliceSource,iRingSource,bunch.getSigS())/sfact;
                            if distance > 0.0:
                                dipx,dipy,quadx,quady = self._wakeDefinition.getWake(distance);
                                weight = basis.getWeight(iSliceSource,iRingSource);
                                if basis.getNDim() == 2:
                                    basisIndexForTest = basis.getIndexForSlice(iSliceTest,iRingTest);
                                    zMatrix[basisIndexForTest+1,basis.getIndexForSlice(iSliceSource,iRingSource)] = dipx*cst*self._betaX*weight;
                                    if self._quadWake:
                                        zMatrix[basisIndexForTest+1,basisIndexForTest] += quadx*cst*self._betaX*weight;
                                elif basis.getNDim() == 4:
                                    basisIndexForTest = basis.getIndexForSlice(iSliceTest,iRingTest);
                                    zMatrix[basisIndexForTest+1,basis.getIndexForSlice(iSliceSource,iRingSource)] = dipx*cst*self._betaX*weight;
                                    zMatrix[basisIndexForTest+3,basis.getIndexForSlice(iSliceSource,iRingSource)+2] = dipy*cst*self._betaY*weight;
                                    if self._quadWake:
                                        zMatrix[basisIndexForTest+1,basisIndexForTest] = quadx*cst*self._betaX*weight;
                                        zMatrix[basisIndexForTest+3,basisIndexForTest+2] = quady*cst*self._betaY*weight;
                                else:
                                    raise BimBimError('Impedance is not implemented in '+str(basis.getNDim())+' dimensions');
            self._singleBunchMatrices[key] = zMatrix;
            return zMatrix;
    
    def getCouplingMatrix(self,basis,beam,sourceBunch,testBunch):
        retVal = spm.dok_matrix((basis.getBunchBlockSize(),basis.getBeamBlockSize(beam)));
        sourceBunchIndex = basis.getIndexForBunch(sourceBunch.getNumber());
        testBunchIndex = basis.getIndexForBunch(testBunch.getNumber());
        distance = testBunch.getSPosition()-sourceBunch.getSPosition();
        dipx,dipy,quadx,quady = self._wakeDefinition.getWake(distance);
        cst = self._qe**2/(sourceBunch.getMass()*self._GeVToKg*sourceBunch.getGamma()*sourceBunch.getBeta()**2*self._clight**2)*sourceBunch.getIntensity()*self._intensityScaling;
        for iSliceSource in range(basis.getNSlice()):
            for iSliceTest in range(basis.getNSlice()):
                for iRingSource in range(basis.getNRing()):
                    for iRingTest in range(basis.getNRing()):
                        weight = basis.getWeight(iSliceSource,iRingSource);
                        if basis.getNDim() == 2:
                            basisIndexForTest = basis.getIndexForSlice(iSliceTest,iRingTest);
                            retVal[basisIndexForTest+1,sourceBunchIndex+basis.getIndexForSlice(iSliceSource,iRingSource)] = dipx*cst*self._betaX*weight;
                            if self._quadWake:
                                retVal[basisIndexForTest+1,testBunchIndex + basisIndexForTest] = quadx*cst*self._betaX*weight;
                        elif basis.getNDim() == 4:
                            basisIndexForTest = basis.getIndexForSlice(iSliceTest,iRingTest);
                            retVal[basisIndexForTest+1,sourceBunchIndex + basis.getIndexForSlice(iSliceSource,iRingSource)] = dipx*cst*self._betaX*weight;
                            retVal[basisIndexForTest+3,sourceBunchIndex + basis.getIndexForSlice(iSliceSource,iRingSource)+2] = dipy*cst*self._betaY*weight;
                            if self._quadWake:
                                retVal[basisIndexForTest+1,testBunchIndex + basis.getIndexForSlice(iSliceSource,iRingSource)] = quadx*cst*self._betaX*weight;
                                retVal[basisIndexForTest+3,testBunchIndex + basis.getIndexForSlice(iSliceSource,iRingSource)+2] = quady*cst*self._betaY*weight;
                        else:
                            raise BimBimError('Multibunch impedance is not implemented in '+str(basis.getNDim())+' dimensions');
        return retVal;

    #Build impedance matrix from wake table
    def getImpedanceMatrix(self,basis,beams,beam,bunch):
        singleMatrix = self.getSingleBunchImpedanceMatrix(basis,bunch);
        #singleMatrix = spm.identity(basis.getBunchBlockSize(),format='dok');
        zMatrix = spm.dok_matrix((basis.getBunchBlockSize(),basis.getBeamBlockSize(beam)));
        sparse_insert(zMatrix,singleMatrix,0,basis.getIndexForBunch(bunch.getNumber()));
        for sourceBunch in beams.getBunchConfig(beam):
            if sourceBunch != None:
                if bunch.getSPosition() > sourceBunch.getSPosition():
                    cMatrix = self.getCouplingMatrix(basis,beam,sourceBunch,bunch);
                    sparse_add(zMatrix,cMatrix); # element wise addition
        return zMatrix;

    def getMatrix(self,beams,pos,basis):
        time0 = time();
        retVal = spm.identity(basis.getSize(),format='dok')
        bunchB1 = beams.getBunchB1(pos);
        if bunchB1 != None :
            beam = 1;
            bunchNb = bunchB1.getNumber()
            #print('Building impedance matrix for B1b'+str(bunchNb));
            zMatrix = self.getImpedanceMatrix(basis,beams,beam,bunchB1);
            zMatrix = basis.getFullBunchProjection(beam,bunchNb,zMatrix);
            #myFile = open('debug_impedance_B'+str(beam)+'b'+str(bunchNb)+'.mat','w');
            #printMatrix(myFile,zMatrix);
            #myFile.close();
            retVal = retVal.dot(zMatrix);
        bunchB2 = beams.getBunchB2(pos);
        if bunchB2 != None :
            beam = 2;
            bunchNb = bunchB2.getNumber()
            #print('Building impedance matrix for B2b'+str(bunchNb));
            zMatrix = self.getImpedanceMatrix(basis,beams,beam,bunchB2);
            zMatrix = basis.getFullBunchProjection(beam,bunchNb,zMatrix);
            #myFile = open('debug_impedance_B'+str(beam)+'b'+str(bunchNb)+'.mat','w');
            #printMatrix(myFile,zMatrix);
            #myFile.close();
            retVal = retVal.dot(zMatrix);
        time1 = time();
        #print('Time to build impedance matrix',time1-time0);
        #myFile = open('debug_impedance.mat','w');
        #printMatrix(myFile,retVal);
        #myFile.close();
        return retVal;
