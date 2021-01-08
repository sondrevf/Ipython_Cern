import os,pickle
import numpy as np
from matplotlib import pyplot as plt

if __name__=='__main__':
    nSlice = 80
    nRing = 40
    outputDir = './01_Output/LHCDipoleMoments/'
    outputDir = './01_Output/'
    studyName = 'LHCDipoleMoment_nSlice'+str(nSlice)+'_nRing'+str(nRing)
    outputFileName = os.path.join(outputDir,studyName+'.pkl')
    if not os.path.exists(outputFileName):
        print(outputFileName,'does not exist')
        exit()
    myFile = open(outputFileName,'rb')
    [dGains,chromas,coherentTuneShifts,sideBands,dipoleMoments] = pickle.load(myFile)
    myFile.close()

    X,Y = np.meshgrid(chromas,dGains)
    maxDipoleMoments = np.zeros_like(X,dtype=float)
    maxImags = np.zeros_like(X,dtype=float)
    maxReals = np.zeros_like(X,dtype=float)
    maxSideBand = np.zeros_like(X,dtype=float)

    for iGain in range(np.shape(dGains)[0]):
        for iChroma in range(np.shape(chromas)[0]):
            if np.max(np.imag(coherentTuneShifts[iGain,iChroma,:]))>0:
                mask = np.logical_and(np.imag(coherentTuneShifts[iGain,iChroma,:])>0,np.abs(sideBands[iGain,iChroma,:])<nSlice/4)
                tmpCoherentTuneShifts = coherentTuneShifts[iGain,iChroma,mask]
                tmpSideBands = sideBands[iGain,iChroma,mask]
                tmpDipoleMoments = dipoleMoments[iGain,iChroma,mask]
                iMax = np.argmax(np.imag(tmpCoherentTuneShifts))
                maxImags[iGain,iChroma] = np.imag(tmpCoherentTuneShifts[iMax])
                maxReals[iGain,iChroma] = np.real(tmpCoherentTuneShifts[iMax])
                maxDipoleMoments[iGain,iChroma] = tmpDipoleMoments[iMax]
                maxSideBand[iGain,iChroma] = tmpSideBands[iMax]

                #if iChroma == 0:
                #    print(dGains[iGain],tmpCoherentTuneShifts)
            else:
                maxImags[iGain,iChroma] = np.nan
                maxReals[iGain,iChroma] = np.nan
                maxDipoleMoments[iGain,iChroma] = np.nan
                maxSideBand[iGain,iChroma] = np.nan

    plt.figure(0)
    levels = np.arange(0.0,1.0,2E-2)
    plt.contourf(chromas,dGains,-1.0*np.log10(maxDipoleMoments),cmap='cool')
    plt.xlabel('Chromaticity')
    plt.ylabel('Damper gain')
    plt.grid()
    cbar = plt.colorbar()
    cbar.set_label(r'$-log_{10}(\eta)$')

    plt.figure(1)
    levels = np.arange(0.0,1E-5,5E-7)
    plt.contourf(chromas,dGains,maxImags,cmap='cool',levels=levels)
    plt.xlabel('Chromaticity')
    plt.ylabel('Damper gain')
    plt.grid()
    cbar = plt.colorbar()
    cbar.set_label(r'$Max(Im(\Delta Q))$')

    plt.figure(2)
    levels = np.arange(-5.5,6.5)
    plt.contourf(chromas,dGains,maxSideBand,cmap='jet',levels=levels)
    plt.xlabel('Chromaticity')
    plt.ylabel('Damper gain')
    plt.grid()
    cbar = plt.colorbar()
    cbar.set_label(r'Sideband')

    if True:
        for chroma in [-5.0,0.0,5.0,10.0,15.0,20.0]:
            mask = chromas == chroma
            plt.figure(10)
            plt.plot(dGains,maxImags[:,mask],label=str(chroma))
            plt.figure(11)
            plt.plot(dGains,maxDipoleMoments[:,mask],label=str(chroma))
        plt.figure(10)
        plt.xlabel('Damper gain')
        plt.ylabel(r'$Max(Im(\Delta Q))$')
        plt.grid()
        plt.legend()
        plt.figure(11)
        plt.xlabel('Damper gain')
        plt.ylabel(r'Dipole moment')
        plt.grid()
        plt.legend()

        for dGain in [0.0,0.001,0.01]:
            mask = dGains == dGain
            plt.figure(20)
            plt.plot(chromas,np.transpose(maxImags[mask,:]),label=str(dGain))
            plt.figure(21)
            plt.plot(chromas,np.transpose(maxReals[mask,:]),label=str(dGain))
            plt.figure(22)
            plt.plot(chromas,np.transpose(maxDipoleMoments[mask,:]),label=str(dGain))
            plt.figure(23)
            plt.plot(chromas,np.transpose(maxSideBand[mask,:]),label=str(dGain))
        plt.figure(20)
        plt.xlabel('Chromaticity')
        plt.ylabel(r'$Max(Im(\Delta Q))$')
        plt.grid()
        plt.legend()
        plt.figure(21)
        plt.xlabel('Chromaticity')
        plt.ylabel(r'$Re(\Delta Q)$')
        plt.grid()
        plt.legend()
        plt.figure(22)
        plt.xlabel('Chromaticity')
        plt.ylabel(r'Dipole moment')
        plt.grid()
        plt.legend()



    plt.show()
    
