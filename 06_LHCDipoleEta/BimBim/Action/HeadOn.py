import scipy.sparse as spm
import numpy as np
from BimBim.Action import Action
from ..Error import BimBimError
from ..Matrix import dotProd

# Note : The virtual drift is done neglecting the Xing angle and the linearized coherent kick takes into account
#        the beam position and size at the location of the interaction (including hour glass and Xing angle)

class HeadOn(Action.Action):
    _betaStarX = 0.0; # round beam (for now)
    _sepStarX = 0; # full sep
    _sepStarY = 0;
    _XingX = 0.0; # full Xing
    _XingY = 0.0;
    _phaseAdvance = False;
    _hourglass = True;
    _XYCoupling = True;
    
    def __init__(self,betaStarX,sepStarX=0.0,sepStarY=0.0,XingX=0.0,XingY=0.0,hourglass=True,phaseAdvance = False,XYCoupling=True):
        self._betaStarX = betaStarX;
        self._sepStarX = sepStarX;
        self._sepStarY = sepStarY;
        self._XingX = XingX;
        self._XingY = XingY;
        self._phaseAdvance = phaseAdvance;
        self._hourglass = hourglass;
        self._XYCoupling = XYCoupling;
        
    def generateCollisionSchedule(self,bunchB1,bunchB2,basis):
        scheds = [];
        taus = [];
        for iSliceB1 in range(basis.getNSlice()):
            for iRingB1 in range(basis.getNRing()):
                for iSliceB2 in range(basis.getNSlice()):
                    for iRingB2 in range(basis.getNRing()):
                        tau = basis.getSPosition(iSliceB2,iRingB2,bunchB2.getSigS())+basis.getSPosition(iSliceB1,iRingB1,bunchB1.getSigS());
                        sched = [iSliceB1,iRingB1,iSliceB2,iRingB2];
                        if len(taus) == 0 or tau < taus[-1]:
                            taus.append(tau);
                            scheds.append(sched);
                        elif tau >= taus[0]:
                            taus.insert(0,tau);
                            scheds.insert(0,sched);
                        else:
                            for i in range(len(taus)-1):
                                if taus[i] >= tau and tau >= taus[i+1]:
                                    taus.insert(i,tau);
                                    scheds.insert(i,sched);
                                    break;
        return scheds;
    
    def getVirtualDrift(self,sCP,basis):
        if basis.getNDim() == 2:
            drift = spm.identity(4,format='dok');
            drift[0,1] = sCP;
            drift[2,3] = -sCP;
        elif basis.getNDim() == 4:
            drift = spm.identity(8,format='dok');
            drift[0,1] = sCP;
            drift[2,3] = sCP;
            drift[4,5] = -sCP;
            drift[6,7] = -sCP;
        return drift;

    def getBeta(self,s):
        if self._hourglass:
            return self._betaStarX*(1.0+s**2/self._betaStarX**2);
        else:
            return self._betaStarX;

    def bb_force(self,bunch,sepX,sepY,beta):
        if abs(sepX) < 1E-10 and abs(sepY) < 1E-10:
            kbbxx = -0.5*bunch.getIntensity()*bunch.getR0()/(bunch.getEmittance()*beta);
            kbbyy = -0.5*bunch.getIntensity()*bunch.getR0()/(bunch.getEmittance()*beta);
            kbbxy = 0.0;
        else:
            sigma = np.sqrt(beta*bunch.getEmittance()/bunch.getGamma());
            x = sepX;
            y = sepY;
            r = np.sqrt(x**2+y**2);
            E = np.exp(-r**2/(4.0*sigma**2));
            A1 = 2.0*bunch.getIntensity()*bunch.getR0()/(bunch.getGamma());
            B1 = (1.0/r**2-2.0*x**2/r**4)*(1.0-E);
            B2 = x**2*E/(2.0*r**2*sigma**2);
            kbbxx = -A1*(B1+B2);
            B1 = (1.0/r**2-2.0*y**2/r**4)*(1.0-E);
            B2 = y**2*E/(2.0*r**2*sigma**2);
            kbbyy = -A1*(B1+B2);
            if self._XYCoupling:
                B1 = -2/r**4*(1.0-E);
                B2 = E/(2.0*r**2*sigma**2);
                kbbxy = -A1*x*y*(B1+B2);
            else:
                kbbxy = 0;
        #return 0,0,0
        return kbbxx,kbbyy,kbbxy;

    def getBeamBeamKick(self,bunchB1,weightB1,bunchB2,weightB2,sCP,basis):
        sigma = np.sqrt(self._betaStarX*bunchB1.getEmittance()/bunchB1.getGamma());
        sepX = 2.0*sCP*np.tan(self._XingX/2.0);
        sepY = 2.0*sCP*np.tan(self._XingY/2.0);
        beta = self.getBeta(sCP);
        kxB2,kyB2,kxyB2 = self.bb_force(bunchB1,self._sepStarX*sigma+sepX,self._sepStarY*sigma+sepY,beta);
        sigma = np.sqrt(self._betaStarX*bunchB2.getEmittance()/bunchB2.getGamma());
        kxB1,kyB1,kxyB1 = self.bb_force(bunchB2,self._sepStarX*sigma+sepX,self._sepStarY*sigma+sepY,beta);
        if basis.getNDim() == 2:
            bbmat = spm.identity(4,format='dok');
            bbmat[1,0] = -kxB1*weightB2;
            bbmat[1,2] = kxB1*weightB2;
            
            bbmat[3,0] = kxB2*weightB1;
            bbmat[3,2] = -kxB2*weightB1;
        elif basis.getNDim() == 4:
            bbmat = spm.identity(8,format='dok');
            bbmat[1,0] = -kxB1*weightB2;
            bbmat[1,2] = -kxyB1*weightB2;
            bbmat[1,4] = kxB1*weightB2;
            bbmat[1,6] = kxyB1*weightB2;
            
            bbmat[3,0] = -kxyB1*weightB2;
            bbmat[3,2] = -kyB1*weightB2;
            bbmat[3,4] = kxyB1*weightB2;
            bbmat[3,6] = kyB1*weightB2;
            
            bbmat[5,0] = kxB2*weightB1;
            bbmat[5,2] = kxyB2*weightB1;
            bbmat[5,4] = -kxB2*weightB1;
            bbmat[5,6] = -kxyB2*weightB1;
            
            bbmat[7,0] = kxyB2*weightB1;
            bbmat[7,2] = kyB2*weightB1;
            bbmat[7,4] = -kxyB2*weightB1;
            bbmat[7,6] = -kyB2*weightB1;
        return bbmat;
        
    def headOn(self,bunchB1,bunchB2,basis):
        fullBBMat = spm.identity(basis.getSize(),format='dok');
        collSched = self.generateCollisionSchedule(bunchB1,bunchB2,basis);
        for collision in collSched:
            iSliceB1 = collision[0];
            iRingB1 = collision[1];
            iSliceB2 = collision[2];
            iRingB2 = collision[3];
            sB1 = basis.getSPosition(iSliceB1,iRingB1,bunchB1.getSigS());
            sB2 = basis.getSPosition(iSliceB2,iRingB2,bunchB2.getSigS());
            sCP = (sB1-sB2)/2.0;
            virtDrift = self.getVirtualDrift(sCP,basis);
            invVirtDrift = self.getVirtualDrift(-sCP,basis);
            bbKick = self.getBeamBeamKick(bunchB1,basis.getWeight(iSliceB1,iRingB1),bunchB2,basis.getWeight(iSliceB2,iRingB2),sCP,basis);
            tmpBBMat = dotProd([invVirtDrift,bbKick,virtDrift]);
            fullBBMat = basis.getTwoElementProjection(bunchB1.getNumber(),iSliceB1,iRingB1,bunchB2.getNumber(),iSliceB2,iRingB2,tmpBBMat).dot(fullBBMat);
        return fullBBMat;
    
    def getMatrix(self,beams,pos,basis):
        bunchB1 = beams.getBunchB1(pos);
        bunchB2 = beams.getBunchB2(pos);
        if self._phaseAdvance:
            raise BimBimError('Phase advane is not yet implemented in HeadOn.');
        if bunchB1 != None and bunchB2 != None:
            retVal = self.headOn(bunchB1,bunchB2,basis);
            return retVal
        return spm.identity(basis.getSize());
