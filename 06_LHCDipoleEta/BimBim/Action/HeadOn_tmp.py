import scipy.sparse as spm
import numpy as np
from BimBim.Action import Action
from ..Error import BimBimError

class HeadOn(Action.Action):
    _beta=0;
    _phaseAdvance = True;
    
    def __init__(self,beta,phaseAdvance = True):
        self._beta = beta;
        self._phaseAdvance = phaseAdvance;

    def bets(betx,s):
        return betx*(1+s**2/betx**2)
  
    def bb_force(xi,betx,nslice,nrings,s):
        kbb = -2.0*np.pi*xi/(nslice*nrings*bets(betx,s))
        return kbb

    #Define initial drift -> moves all slices except 1st and last out of collision point
    def indrift(nslice,nrings,nbeams,distmat):
        dist = -transpose(distmat)[0]
        drift = spm.lil_matrix((nslice*nrings,nslice*nrings))
        for i in range(nslice*nrings):
            drift[i,i]=dist[i]
        drift = directProd(idMatrix(nbeams),idMatrix(2*nslice*nrings)+directProd(drift.tocsc(),spm.dok_matrix([[0.0,1.0],[0.0,0.0]])))
        return drift
 
#Define final drift -> moves all the slice back to collision point
    def outdrift(nslice,nrings,nbeams,distmat):
        dist = -np.array(distmat)[(nslice-1)/2]
        drift = spm.lil_matrix((nslice*nrings,nslice*nrings))
        for i in range(nslice*nrings):
            drift[i,i]=dist[i]
        drift = directProd(idMatrix(nbeams),idMatrix(2*nslice*nrings)+directProd(drift.tocsc(),spm.dok_matrix([[0.0,1.0],[0.0,0.0]])))
        return drift

    #Build beam-beam kick matrix from distance between slices
    def bbkick(nslice,bbslice,nring,nbeams,distmat,kbb):
        offdiag = [[0.0 for j in range(bbslice*nrings)] for i in range(bbslice*nrings)]
        ondiag = []
        for i in range(distmat.shape[0]):
            tot = 0.0
            for j in range(distmat.shape[1]):
                if distmat[i][j]==0.0:
                    offdiag[i][j]=1.0
                    tot = tot + 1.0
                    ondiag.append(-tot*float(nslice/bbslice))
        ondiag = idMatrix(nslice*nrings*2)+kbb*directProd(cp_linalg.block_diag(*np.array(ondiag).repeat(nslice/bbslice,0)),[[0.0,0.0],[1.0,0.0]])
        offdiag = kbb*directProd(np.array(offdiag).repeat(nslice/bbslice,0).repeat(nslice/bbslice,1),[[0.0,0.0],[1.0,0.0]])
        tmp1 = spm.vstack([ondiag,offdiag])
        tmp2 = spm.vstack([offdiag,ondiag])
        kmat = spm.hstack([tmp1,tmp2])
        return kmat

    def driftmat(d,nslice,nrings,nbeams):
        dmat = cp_linalg.block_diag(*np.array([[1.0,d],[0.0,1.0]]).reshape(1,2,2).repeat(nslice*nrings*nbeams,0))
        return spm.bsr_matrix(dmat)

    #Define head-on beam-beam matrix
    def hobb(nslice,bbslice,nrings,nbeams,betx,lb,sigs,bbph,xi):
        l0 = spos(bbslice,nrings,lb,sigs,0.0)
        l0 = np.repeat(l0,nslice/bbslice,0)
        dist0 = sdiff(nslice,nrings,l0)
        #Headtail phase shift due to the beam-beam interaction
        #Should it be applied to impedance and damper kick as well??
        #To be checked with tracking
        phin = chromIn(nslice,nrings,nbeams,1,betx,bbph,l0)
        phout = chromOut(nslice,nrings,nbeams,1,betx,bbph,l0)
        ind = indrift(nslice,nrings,nbeams,dist0)
        outd = outdrift(nslice,nrings,nbeams,dist0)
        bbmat = ind
        dtot = -l0[0]
        index = 0
        while dtot!=-l0[(nslice-1)/2]:
            t0 = clock()
            l = spos(bbslice,nrings,lb,sigs,dtot)
            dist2b = sdiff2b(bbslice,nrings,l)
            s = abs(l[index])
            kbb = bb_force(xi,betx,nslice,nrings,s)
            kmat = bbkick(nslice,bbslice,nrings,nbeams,dist2b,kbb)
            d,index = maxi_neg(dist2b)
            d = -d/2.0
            dmat = driftmat(d,nslice,nrings,nbeams)  
            bbmat = dotProd([dmat,kmat,bbmat])
            dtot = dtot+d    
        l = spos(bbslice,nrings,lb,sigs,dtot)
        dist2b = sdiff2b(bbslice,nrings,l)
        s = abs(l[index])
        kbb = bb_force(xi,betx,nslice,nrings,s)
        kmat = bbkick(nslice,bbslice,nrings,nbeams,dist2b,kbb)
        bbmat = dotProd([phin,outd,kmat,bbmat,phout])
        bbmat = spm.csc_matrix(bbmat)
        remove_zeros(bbmat)
        return bbmat
    
    #Allows to move the bunches to the LR IP assuming pi/2 phase advance
    def pi2phaseshift(nslice,nrings,betx,side):
        betmap = [[0.0,betx*side],[-1.0/betx*side,0.0]] 
        b1 = directProd(idMatrix(nslice*nrings),spm.dok_matrix(betmap))
        zeros = spm.dok_matrix((nslice*nrings*2,nslice*nrings*2))
        tmp1 = spm.vstack([b1,zeros])
        tmp2 = spm.vstack([zeros,b1])
        ph = spm.hstack([tmp1,tmp2])
        return ph
    
    def getMatrix(self,beams,pos,basis):
        raise BimBimError('Head-on is not implemented yet');
        #if bunchB1 != None and bunchB2 != None:
        return spm.identity(basis.getSize());
