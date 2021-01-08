import scipy.sparse as spm
import numpy as np
from BimBim.Action import Action
from ..Error import BimBimError
from ..Matrix import printMatrix


#Note : slice and rings are equipopulated
class FlatBeamBeam(Action.Action):
    _beta = 0;
    _sepX = 0;
    _sepY = 0;
    _XYCoupling = True;
 
    def __init__(self,beta=1.0,sepX=0.0,sepY=0.0,XYCoupling=True):
        self._beta = beta;
        self._sepX = sepX;
        self._sepY = sepY;
        self._XYCoupling = XYCoupling;

    def bb_force_lr(self,bunch):
        if abs(self._sepX) < 1E-10 and abs(self._sepY) < 1E-10:
            kbbxx = -0.5*bunch.getIntensity()*bunch.getR0()/(bunch.getEmittance()*self._beta);
            kbbyy = -0.5*bunch.getIntensity()*bunch.getR0()/(bunch.getEmittance()*self._beta);
            kbbxy = 0.0;
        else:
            sigma = np.sqrt(self._beta*bunch.getEmittance()/bunch.getGamma());
            x = self._sepX*sigma;
            y = self._sepY*sigma;
            sigma = np.sqrt(2)*sigma; # coherent kick
            r = np.sqrt(x**2+y**2);
            E = np.exp(-r**2/(2.0*sigma**2));
            A1 = 2.0*bunch.getIntensity()*bunch.getR0()/(bunch.getGamma());
            B1 = (1.0/r**2-2.0*x**2/r**4)*(1.0-E);
            B2 = x**2*E/(r**2*sigma**2);
            kbbxx = -A1*(B1+B2);
            B1 = (1.0/r**2-2.0*y**2/r**4)*(1.0-E);
            B2 = y**2*E/(r**2*sigma**2);
            kbbyy = -A1*(B1+B2);
            if self._XYCoupling:
                B1 = -2/r**4*(1.0-E);
                B2 = E/(2.0*r**2*sigma**2);
                kbbxy = -A1*x*y*(B1+B2);
            else:
                kbbxy = 0.0;
        return kbbxx,kbbyy,kbbxy;

    def lrbb(self,bunchB1,bunchB2,basis):
        nChunk = basis.getNSlice()*basis.getNRing();
        kxB2,kyB2,kxyB2 = self.bb_force_lr(bunchB1);
        kxB1,kyB1,kxyB1 = self.bb_force_lr(bunchB2);
        if basis.getNDim() == 2:
            ondiagB1 = spm.kron(spm.identity(nChunk,format='dok'),spm.dok_matrix([[1.0,0.0],[-kxB1,1.0]]),format='dok');
            offdiagB1 = spm.kron(basis.getWeightMatrix(),spm.dok_matrix([[0.0,0.0],[kxB1,0.0]]),format='dok');
            ondiagB2 = spm.kron(spm.identity(nChunk,format='dok'),spm.dok_matrix([[1.0,0.0],[-kxB2,1.0]]),format='dok');
            offdiagB2 = spm.kron(basis.getWeightMatrix(),spm.dok_matrix([[0.0,0.0],[kxB2,0.0]]),format='dok');
            tmp1 = spm.vstack([ondiagB1,offdiagB2])
            tmp2 = spm.vstack([offdiagB1,ondiagB2])
            bbmat = spm.hstack([tmp1,tmp2])
            #myFile = open('debug_fBBmat.mat','w');
            #printMatrix(myFile,bbmat);
            #myFile.close();
            return bbmat
        elif basis.getNDim() == 4:
            ondiagB1 = spm.kron(spm.identity(nChunk,format='dok'),spm.dok_matrix([[1.0,0.0,0.0,0.0],[-kxB1,1.0,-kxyB1,0.0],[0.0,0.0,1.0,0.0],[-kxyB1,0.0,-kyB1,1.0]]),format='dok');
            offdiagB1 = spm.kron(basis.getWeightMatrix(),spm.dok_matrix([[0.0,0.0,0.0,0.0],[kxB1,0.0,kxyB1,0.0],[0.0,0.0,0.0,0.0],[kxyB1,0.0,kyB1,0.0]]),format='dok');
            ondiagB2 = spm.kron(spm.identity(nChunk,format='dok'),spm.dok_matrix([[1.0,0.0,0.0,0.0],[-kxB2,1.0,-kxyB2,0.0],[0.0,0.0,1.0,0.0],[-kxyB2,0.0,-kyB2,1.0]]),format='dok');
            offdiagB2 = spm.kron(basis.getWeightMatrix(),spm.dok_matrix([[0.0,0.0,0.0,0.0],[kxB2,0.0,kxyB2,0.0],[0.0,0.0,0.0,0.0],[kxyB2,0.0,kyB2,0.0]]),format='dok');
            tmp1 = spm.vstack([ondiagB1,offdiagB2])
            tmp2 = spm.vstack([offdiagB1,ondiagB2])
            bbmat = spm.hstack([tmp1,tmp2])
            #myFile = open('debug_fBBmat.mat','w');
            #printMatrix(myFile,bbmat);
            #myFile.close();
            return bbmat
        else:
            raise BimBimError("Flat beam-beam is not defined in "+str(basis.getNDim())+" dimensions");

    def getMatrix(self,beams,pos,basis):
        bunchB1 = beams.getBunchB1(pos);
        bunchB2 = beams.getBunchB2(pos);
        if bunchB1 != None and bunchB2 != None:
            #print('Flat BB between B1b'+str(bunchB1.getNumber())+' and B2b'+str(bunchB2.getNumber()));
            bbmat = self.lrbb(bunchB1,bunchB2,basis);
            retVal = basis.getTwoBunchProjection(bunchB1.getNumber(),bunchB2.getNumber(),bbmat)
            #myFile = open('debug_bb.mat','w');
            #printMatrix(myFile,retVal);
            #myFile.close();
            return retVal;
        return spm.identity(basis.getSize(),format='dok');
