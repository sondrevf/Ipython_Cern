import os,pickle
import numpy as np

if __name__=='__main__':
    nSlice =80 
    nRing = 40
    outputDir = '/afs/cern.ch/work/x/xbuffat/BimBim_workspace/Output/LHCDipoleMoments/'
    outputDir = '/afs/cern.ch/work/s/sfuruset/08_LHCDipoleEta/01_Output/WS2/'
    dGains = np.arange(0.0,4E-2+1E-4,5E-4)
    allocated = False
    for iGain,dGain in enumerate(dGains) :
        studyName = 'LHCDipoleMoment_nSlice'+str(nSlice)+'_nRing'+str(nRing)+'_dGain'+str(dGain)
        outputFileName = os.path.join(outputDir,studyName+'.pkl')
        if not os.path.exists(outputFileName):
            print(outputFileName,'does not exist')
        else:
            myFile = open(outputFileName,'rb')
            [tmpChromas,tmpCoherentTuneShifts,tmpSideBands,tmpDipoleMoments] = pickle.load(myFile)
            myFile.close()
            if not allocated:
                chromas = tmpChromas
                outputShape = (np.shape(dGains)[0],np.shape(chromas)[0],np.shape(tmpCoherentTuneShifts)[1])
                coherentTuneShifts = np.zeros(outputShape,dtype=complex)
                sideBands = np.zeros(outputShape,dtype=float)
                dipoleMoments = np.zeros(outputShape,dtype=float)
                allocated = True
            if np.any(chromas!=tmpChromas):
                print('Chromas don t match in',outputFileName,'skipping it')
            else:
                coherentTuneShifts[iGain,:,:] = tmpCoherentTuneShifts
                sideBands[iGain,:,:] = tmpSideBands
                dipoleMoments[iGain,:,:] = tmpDipoleMoments

    studyName = 'LHCDipoleMoment_nSlice'+str(nSlice)+'_nRing'+str(nRing)
    outputFileName = os.path.join(outputDir,studyName+'.pkl')
    myFile = open(outputFileName,'wb')
    pickle.dump([dGains,chromas,coherentTuneShifts,sideBands,dipoleMoments],myFile)
    myFile.close()
