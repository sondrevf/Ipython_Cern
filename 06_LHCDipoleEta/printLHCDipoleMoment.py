import os,pickle
import numpy as np
from matplotlib import pyplot as plt

if __name__=='__main__':
    nSlice =80 
    nRing = 40
    outputDir = './01_Output/LHCDipoleMoments/'
    outputDir = './01_Output/mode0'
    outputDir = './01_Output/WS1/'
    studyName = 'LHCDipoleMoment_nSlice'+str(nSlice)+'_nRing'+str(nRing)
    outputFileName = os.path.join(outputDir,studyName+'.pkl')
    if not os.path.exists(outputFileName):
        print(outputFileName,'does not exist')
        exit()
    myFile = open(outputFileName,'rb')
    [dGains,chromas,coherentTuneShifts,sideBands,dipoleMoments]= pickle.load(myFile)
    myFile.close()
    
    i=0
    j=5
    j=0
    ii=[20]  ;  jj=[15,17,20,21,22,23,24,25,35]
    ii=[20]  ;  jj=[30,33,35]
    for i in ii:
        for j in jj:
            print('\ng ',dGains[i])
            print('Q\'',chromas[j])
            for ia,arr in enumerate([coherentTuneShifts,sideBands,dipoleMoments]):
                arrij = arr[i,j]
                #print(np.shape(arrij))
                #print(arrij)
                #print('')

                if ia==0:
                    ind0=ind = np.imag(arrij)>1e-7
                elif ia==1:
                    ind1=ind = (np.abs(arrij)<=3)
                elif ia==2:
                    ind2=ind = arrij>0.004
                #print(np.sum(ind))

            indAll = ind0*ind1*ind2
            print(np.sum(indAll))

            dQ = coherentTuneShifts[i,j,indAll]
            m  = sideBands[i,j,indAll]
            eta= dipoleMoments[i,j,indAll]

            print(dQ)
            print(m)
            print(eta)
            print(eta*dQ.imag)

