
import os,sys,gc,pickle,time
from scipy import constants as cst

from BimBim.Matrix import *
from BimBim.Beams import Beams
from BimBim.Basis import Basis
from BimBim.System import System

from BimBim.Action.Transport import Transport
from BimBim.Action.Impedance import Impedance
from BimBim.Action.WakeFromTable import WakeFromTable
from BimBim.Action.Damper import Damper

from matplotlib import pyplot as plt

def getRealTuneShift(tune,qbeta,qs):
    div = (tune-qbeta)/qs
    integ = np.floor(div+0.5)
    diff = div-integ
    return diff*qs,integ

if __name__ == '__main__':
    intensity = 2E11
    outputDir = sys.argv[1]
    studyName = sys.argv[2]
    nSlice = int(sys.argv[3])
    nRing = int(sys.argv[4])
    dGain = float(sys.argv[5])
    nDim = 2
    energy = 6.5E3
    emittance = 2E-6
    sigs = 1.05E-9/4.0*cst.c
    momentumCompaction = 3.483575072011584e-04
    gamma = energy*1E3/cst.value('proton mass energy equivalent in MeV')
    beta=np.sqrt(1.-1./(gamma**2))
    eta = momentumCompaction-1.0/gamma**2
    voltage = 12.0E6
    h = 35640
    p0=cst.m_p*beta*gamma*cst.c
    qs=np.sqrt(cst.e*voltage*eta*h/(2*np.pi*beta*cst.c*p0))
    averageRadius = 26658.883199999/(2*np.pi)
    sigp = qs*sigs/(averageRadius*eta)
    #sigp = 1.129E-4 # mode 0
    qbeta = 0.31
    outputFileName = os.path.join(outputDir,studyName+'.pkl')
#    wakeDefinition = WakeFromTable('/afs/cern.ch/work/x/xbuffat/BimBim_workspace/wakes/wakeforhdtl_PyZbase_Allthemachine_6p5TeV_B1_LHC_ft_6.5TeV_B1_2017_WF.dat')
    wakeDefinition = WakeFromTable('/afs/cern.ch/work/s/sfuruset/08_LHCDipoleEta/wakeforhdtl_PyZbase_Allthemachine_6p5TeV_B1_LHC_ft_6.5TeV_B1_2017_WF.dat')
    fill1 = '1 1 1 0'
    fill2 = '1 0 1 0'

    allocated = False

    chromas = np.arange(-10.0,0.1,1.0)
    chromas = np.arange(0.0,20.1,5.0)
    chromas = [15.0]
    for ichroma,chroma in enumerate(chromas):
        time0 = time.time()
        actionSequence = [None for k in range(2*2)]
        actionSequence[0] = Transport(phaseX=qbeta,betaX=1.0,chromaX=chroma,secondOrderChromaX=0,qs=qs)
        actionSequence[1] = Impedance(wakeDefinition,65.9756,71.5255,quadWake=False,intensityScaling=1.0)
        if dGain > 0.0:
            actionSequence[2] = Damper(dGain)

        beams = Beams(fill1,fill2,energy=energy,intensity = intensity,emittance = emittance,sigs=sigs,sigp=sigp)
        basis = Basis(beams.getNBunchB1(),beams.getNBunchB2(),nSlice,nRing,nDim,equiDistantRing = False)
        system = System(beams,actionSequence,basis)
        oneTurn = system.buildOneTurnMap()
        system = None
        beams = None
        actionSequence = None
        gc.collect()

        eigvals,eigvecs = np.linalg.eig(oneTurn)
        coherentTunes = np.log(eigvals)*1j/(2*np.pi)
        #mask = coherentTunes>0
        #plt.figure(10)
        #plt.plot((np.real(coherentTunes[mask])-qbeta)/qs,np.imag(coherentTunes[mask]),'x')

        if not allocated:
            expectedOutputSize = int(np.shape(eigvecs)[0]/2)
            coherentTuneShifts = np.zeros((np.shape(chromas)[0],expectedOutputSize),dtype=complex)
            sideBands = np.zeros((np.shape(chromas)[0],expectedOutputSize),dtype=float)
            dipoleMoments = np.zeros((np.shape(chromas)[0],expectedOutputSize),dtype=float)
            allocated = True

        i = 0
        for j in range(len(coherentTunes)):
            if np.real(coherentTunes[j])>0:
                realShift,sideBand = getRealTuneShift(np.real(coherentTunes[j]),qbeta,qs)
                coherentTuneShifts[ichroma,i] = realShift+1j*np.imag(coherentTunes[j])
                sideBands[ichroma,i] = sideBand
                eigvec = np.array([eigvecs[k,j] for k in range(np.shape(eigvecs)[0])])
                eigvec*=np.sqrt(2.0*nSlice)# re-normalise such that the max amplitude is 1
                dipoleMoment,dummy = basis.getModeFirstOrderMoment(eigvec)
                dipoleMoments[ichroma,i] = np.abs(dipoleMoment)
                i+=1
        print(chroma,'Times [s]',time.time()-time0)

        myFile = open(outputFileName,'wb')
        pickle.dump([chromas,coherentTuneShifts,sideBands,dipoleMoments],myFile)
        myFile.close()


