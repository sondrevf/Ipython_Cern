import os 
import numpy as np
from HTCondor import HTCondorJobLauncher

if __name__=='__main__':
    nSlice = 80
    nRing = 40
    outputDir ='/afs/cern.ch/work/s/sfuruset/08_LHCDipoleEta/01_Output/WS2/'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    for dGain in np.arange(0.0,2E-2+1E-4,1E-2):
        studyName = 'LHCDipoleMoment_nSlice'+str(nSlice)+'_nRing'+str(nRing)+'_dGain'+str(dGain)
        outputFileName = os.path.join(outputDir,studyName+'.pkl')
        if not os.path.exists(outputFileName):
            print('Submitting',studyName)
            job = HTCondorJobLauncher(execFile='/afs/cern.ch/work/s/sfuruset/08_LHCDipoleEta/LHCDipoleMoment_HTC.sh',arguments=outputDir+' '+studyName+' '+str(nSlice)+' '+str(nRing)+' '+str(dGain),outputDir=outputDir,studyName=studyName)
            job.launch()
