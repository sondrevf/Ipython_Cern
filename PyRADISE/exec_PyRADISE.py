# #!/afs/cern.ch/work/s/sfuruset/99_programfiles/anaconda3/bin/python3

import sys,os

pathToStorage='/afs/cern.ch/work/s/sfuruset/07_PyRADISE/'
pltstorage = pathToStorage + '03_Plots/'
pklstorage = pathToStorage + '01_Output/'
pathToPackage='/afs/cern.ch/work/s/sfuruset/07_PyRADISE/'
sys.path.append(pathToStorage)
sys.path.append(pathToPackage)
import numpy as np
import pickle as pkl 
import time 
import PyRADISE

########################################
def checkInput():
    if len(sys.argv)>1:
        filename=sys.argv[1]
    else:
        print('Need filename as input')
        return -1
    

    # Adapt filename        
    if filename.find('.pkl')>0:
        filename=filename[:filename.find('_S')+3]
        print(filename)

    # Find stage of inputfile
    inFilename = pklstorage+filename 
    
    # Check if file exists
    if not os.path.isfile(inFilename+'.pkl'):
        print('%s doesn\'t exists'%(inFilename))
        return -1
    return inFilename
    
########################################
def solve_PSI(inFilename, calc, flag_save_pkl):
    if int(inFilename[-1])==1:
        return calc


    # Solve and postprocess 
    calc.solve(i_dpsidt=1,flagFindalpha=1,flagUpdateReQ=1)
    calc.postProc_Distributions(flag_ProjX=0)
    calc.postProc_Moments(n_moments = 3)

    # Store 
    if flag_save_pkl:
        with open(inFilename[:-3]+'_S1.pkl','wb') as pklfile:
            pkl.dump(calc,pklfile)
    return calc
 
 
def solve_SD(inFilename,calc,scale=None):
    planes = [[0,1],[0]][calc.iCoeff>=3]
    modex = 0+0j if len(calc.M.wmode__DQx)==0 else calc.M.wmode__DQx[0]
    modey = 0+0j if len(calc.M.wmode__DQy)==0 else calc.M.wmode__DQy[0]
    if not (scale==None):
        calc.M.Q.scale(scale)
    
    # Calculate SD for evolving distribution
    calc.SD_calcEvolution(planes=planes,interp_order=calc.interp_order,debug=1,width_ratio=[1,0.07][calc.iCoeff>2])
    
    # Calculate SD for scaled detuning of Psi0
    if len(calc.scales)>0:
        calc.SD_calcScaledPsi0(plane=0,interp_order=calc.interp_order,ind=0,debug=0,nQ=max(150,calc.nQ)) 
        calc.SD_copyScaledPsi0(planeFrom=0) 
    
    # StabilityDiagram Postprocessing
    calc.SD_calcStabilityMargin(plane=0,mode=modex)
    calc.SD_calcStabilityMargin(plane=1,mode=modey)

    calc.SD_calcEffectiveStrength(plane=0,flag_interpStrength=1,mode_R=0,flag_allDQ_R=1,maxmin_ratio=np.nan)
    calc.SD_calcEffectiveStrength(plane=1,flag_interpStrength=1,mode_R=0,flag_allDQ_R=1,maxmin_ratio=np.nan)

    
    # Store 
    addname = '' if scale==None else "_scale%.2f"%scale
    with open(inFilename[:-3]+'_S2%s.pkl'%addname,'wb') as pklfile:
        pkl.dump(calc,pklfile)
        print('Solved PDE and calculated SD for \n %s'%pklfile.name)
    
######################################
def main():
    flag_save_pkl = 0

    # load input 
    inFilename = checkInput()
    if inFilename==-1:
        return    
    with open(inFilename+'.pkl','rb') as pklfile:
        calc = pkl.load(pklfile)    
    print('Loaded initial state')
        
    # Check not already computed
    if os.path.isfile(inFilename[:-2]+'2.pkl'):
        print('Job already done for %s \nABORT'%inFilename)
#        sys.exit()
    
    # If scaled 
    scale = None
    if len(sys.argv)>2:
        scale = float(sys.argv[2])
    
    # Calculate 
    calc = solve_PSI(inFilename,calc,flag_save_pkl)
    #calc.nQ = 300
    solve_SD(inFilename,calc,scale)
        
    

if __name__=="__main__":
    main()


