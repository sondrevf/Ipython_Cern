import numpy as np
from scipy import integrate,sparse
import time 
import sys

from PyRADISE.Coefficients     import DiffCoeffGrid, DriftCoeffGrid, calc_LandauDampedAllModes
from PyRADISE.Plotting         import PlottingClass
from PyRADISE.PostProcessing   import PostProcessingClass
from PyRADISE.StabilityDiagram import StabilityDiagramClass
from PyRADISE.Distribution     import MyDistribution1D,MyDistribution1Dy,MyDistribution2D,MyDistribution2Dy

class SolverClass(PlottingClass,PostProcessingClass,StabilityDiagramClass):
    def __init__(self,grid,machine,iCoeff,tsODE,ntPSImax=20,bool_IC_point=0, 
                    interp_order=1,integrator_epsilon=1e-6,solve_method='BDF',
                    flagFindalpha=1, flagUpdateReQ=1,
                    flag_dpsidt=1,flag_adjustTmax=0):
        self.G = grid
        self.M = machine
        
        self.ND = self.G.ND
        self.iCoeff=iCoeff
        self.bool_IC_point = bool_IC_point


        self.set_time(tsODE)
        self.ntPSImax = ntPSImax
                


                        
        self.solve_method=solve_method
        self.interp_order=interp_order
        self.integrator_epsilon=integrator_epsilon
        
        # For finding damped mode
        self.relstep = [0.5,0.05]
        self.tol = 1e-4
        self.flagFindalpha = [flagFindalpha, [flagFindalpha,flagFindalpha]][np.size(flagFindalpha)==1]
        self.flagUpdateReQ = flagUpdateReQ

        
        #Set initial condition
        self._IC()
        
        #Plotting
        self.initPlotting()
        
        # Solver
        self.flag_dpsidt=flag_dpsidt
        self.flag_adjustTmax=flag_adjustTmax

        err=self.calc_Coeff(self.psi0)
        if err:
            print('Abort calculation')
            sys.exit()
        self.calc_FV_M()
        self.calc_BC_V()
        
    ###################################
    ## Initializing helper functions ##
    def set_time(self,tsODE):
        self.tsODE = tsODE                      # Times used to calculate PSI(T)
        self.tsPSI = tsODE                      # Times at which to store PSI(T)
        self.tmax = np.max(self.tsODE)
        self.ntODE = np.size(self.tsODE)
        self.ntPSI = self.ntODE
        self.indexPSI = np.arange(self.ntODE)
        
        #Set time_scale 
        self.set_tlabels('auto')
        
    ##########################
    ## Physics Coefficients ##
    def calc_InterpDist(self,psi,plane=0):
        nrow = self.G.nrow
        ncol = self.G.ncol
        centersepx = np.diff(self.G.Jxc)
        centersepy = np.diff(self.G.Jyc)
        
        if self.ND==1:
            if plane ==0: Func = MyDistribution1D 
            else:         Func = MyDistribution1Dy
            psi1D = psi.reshape((nrow,ncol))
            dPsi1DdJx   = np.diff(psi1D) / centersepx
            distribution = Func(self.G.Jxbx,self.G.Jxc, psi1D,dPsi1DdJx,interp_order=self.interp_order)
        else:
            if plane ==0: Func = MyDistribution2D 
            else:         Func = MyDistribution2Dy
            psi2D = psi.reshape((nrow,ncol))
            dPsi2DdJx = np.diff(psi2D,axis=1) / centersepx
            dPsi2DdJy = np.diff(psi2D,axis=0) / centersepy[:,np.newaxis]
            distribution = Func(self.G.Jxbx,self.G.Jxc,self.G.Jyby,self.G.Jyc,
                                psi2D,dPsi2DdJx,dPsi2DdJy,interp_order=self.interp_order)
        if plane==0:
            self.interpDistx = distribution
        else:
            self.interpDisty = distribution
        
    def calc_Modes(self,psi=0,**kwargs):        
        # Calculate distribution
        relstep = kwargs.pop('relstep',self.relstep)
        tol     = kwargs.pop('tol',self.tol)
        debug   = kwargs.pop('debug',0)
        flagFindalpha=kwargs.pop('flagFindalpha',self.flagFindalpha)
        flagUpdateReQ=kwargs.pop('flagUpdateReQ',self.flagUpdateReQ)
        for plane in [0,1][:self.ND]:
            self.calc_InterpDist(psi,plane=plane)
            calc_LandauDampedAllModes(self,plane=plane,relstep=relstep,tol=tol,flagFindalpha=flagFindalpha,
                                      flagUpdateReQ=flagUpdateReQ,debug=debug)
    
    def calc_Coeff(self,psi=0,**kwargs):
        if self.M.flag_wmode and self.iCoeff in [3,4]:
            self.calc_Modes(psi,**kwargs)
        
        Diffx, Diffy ,errDiff = DiffCoeffGrid(self.M,self.G,self.iCoeff)
        Driftx,Drifty,errDrift = DriftCoeffGrid(self.M,self.G,self.iCoeff)
        if errDiff+errDrift:
            print('Cannot calculate diffusion or drift coefficients - possibly unstable')
            return errDiff+errDrift
        
        self.DiffE  = np.reshape(Diffx[self.G.sliceE], (self.G.NcTot,1))
        self.DiffW  = np.reshape(Diffx[self.G.sliceW], (self.G.NcTot,1))
        self.DriftE = np.reshape(Driftx[self.G.sliceE], (self.G.NcTot,1))
        self.DriftW = np.reshape(Driftx[self.G.sliceW], (self.G.NcTot,1))
        
        if self.ND >1:
            self.DiffN  = np.reshape(Diffy[self.G.sliceN], (self.G.NcTot,1))
            self.DiffS  = np.reshape(Diffy[self.G.sliceS], (self.G.NcTot,1))
            self.DriftN = np.reshape(Drifty[self.G.sliceN], (self.G.NcTot,1))
            self.DriftS = np.reshape(Drifty[self.G.sliceS], (self.G.NcTot,1))
        return 0
    
    def calc_FV_M(self):
        self.FV_M= ((self.G.MslpE.multiply(self.DiffE) - self.G.MslpW.multiply(self.DiffW))- 
                    (self.G.MavgE.multiply(self.DriftE) - self.G.MavgW.multiply(self.DriftW)))
        if self.ND==2:
            self.FV_M += ((self.G.MslpN.multiply(self.DiffN)  - self.G.MslpS.multiply(self.DiffS))- 
                          (self.G.MavgN.multiply(self.DriftN) - self.G.MavgS.multiply(self.DriftS)))
        return
    
    def _calc_BC_V_1D(self,iBC,K,h,boundary,Diff,Drift,rc):
        if K==0:
            return 0
            print("Homogeneous BC")
        
        #Create Diff and Drift vectors including matrices
        V_Diff = Diff/h**2
        V_Drift= Drift/(2*h)
        if self.G.bool_radial:
            V_Diff /= (2*rc)
            V_Drift/= (rc)
        
        # Return the values
        if   iBC==0:
            return K*boundary * (V_Diff - V_Drift)      #+vertical part
        elif iBC==1:
            return K*boundary * (V_Diff - V_Drift)*h    #+vertical part
        elif iBC>1:
            print("Havent implemented nonhomogneeous BC for BC_xMax>1")
            return 0
    def calc_BC_V(self):
        self.BC_V = 0 
        #Boundary at xmax
        if self.ND==1:
            self.BC_V += self._calc_BC_V_1D(self.G.BC_xMax,self.G.BC_xVal, self.G.hx,
                                       self.G.boundary_E,self.DiffE[:,0],self.DriftE[:,0],self.G.rxc[-1])
        elif self.ND==2:
            self.BC_V += self._calc_BC_V_1D(self.G.BC_xMax,self.G.BC_xVal, self.G.hx,
                                       self.G.boundary_E,self.DiffE[:,0],self.DriftE[:,0],self.G.rxc[-1])
            self.BC_V += self._calc_BC_V_1D(self.G.BC_yMax,self.G.BC_yVal, self.G.hy,
                                       self.G.boundary_N,self.DiffN[:,0],self.DriftN[:,0],self.G.ryc[-1])

    ########################
    ## Initial conditions ##
    def _IC(self):
        self.sigx0 = sigx0 = 1
        bool_point = self.bool_IC_point
        self.G.Jxbx2D[self.G.sliceE]
        if bool_point: self.psi0 =    (1/sigx0**2*np.exp(-[self.G.Jxc1D,.5*self.G.rxc1D**2][self.G.bool_radial]/(sigx0**2)))
        else:          self.psi0 =    (np.exp(-self.G.Jxbx2D[self.G.sliceW]/sigx0**2)-np.exp(-self.G.Jxbx2D[self.G.sliceE]/sigx0**2))/np.diff(self.G.Jxbx2D,axis=1)
        if self.ND==2:
            self.sigy0 = sigy0 = 1
            if bool_point: self.psi0 *=1/sigy0**2*np.exp(-[self.G.Jyc1D,.5*self.G.ryc1D**2][self.G.bool_radial]/(sigy0**2))
            else:          self.psi0 *= (np.exp(-self.G.Jyby2D[self.G.sliceS]/sigy0**2)-np.exp(-self.G.Jyby2D[self.G.sliceN]/sigy0**2))/np.diff(self.G.Jyby2D,axis=0)
        self.psi0 = self.psi0.flatten()
        return
    def set_IC(self,psi0):
        self.psi0 = psi0
        return
    
    
    ###################
    ## Solve the PDE ##
    
    def dpsidtFV_const(self,t,psi):
        return self.FV_M.dot(psi) + self.BC_V

    def dpsidtFV_tdep(self,t,psi):
        # Not relevant time dependence...
        self.M.N.dQxAvg = self.M.Q.dQx(1+3*t/tmax,0)
        
        self.calc_Coeff(psi)
        self.calc_FV_M()
        self.calc_BC_V()
        return self.FV_M.dot(psi) + self.BC_V
    
    def solve(self, **kwargs):
        """
            kwargs include: tsODE, solve_method,debug ,flagFindalpha
        """
        start = time.time()

        if 'tsODE' in kwargs:
            tsODE = kwargs.pop('tsODE',self.tsODE)
            self.set_time(tsODE)
            
        self.solve_method = kwargs.pop('solve_method',self.solve_method)
        self.flagFindalpha= kwargs.get('flagFindalpha',self.flagFindalpha)
        flag_dpsidt    =self.flag_dpsidt  = kwargs.get('flag_dpsidt',self.flag_dpsidt)
        flag_adjustTmax=self.flag_adjustTmax= kwargs.get('flag_adjustTmax',self.flag_adjustTmax)
        
        
        
        # Adjust tmax
        if flag_adjustTmax:
            print("solver.py: Trying to adjust tmax to better fit to the physics \n           set tmax to 2*tmax_estimate")
            dpsidt=lambda t, y: self.dpsidtFV_const(t, y)
            if self.solve_method=='LSODA':
                jac = lambda t, y: self.FV_M.toarray()
                print('LSODA solver does not support sparse matrices in this PyRADISE')
            else:
                jac = lambda t, y: self.FV_M
                
            #Solve for distribution evolution
            newTmax = self.tmax
            self.calc_Modes(self.psi0,**kwargs)
            if self.M.nWmodex>0: 
                growthRates_testx = np.ones((2,self.M.nWmodex))
                growthRates_testx[0,:] = self.M.wmodeLdDQx.imag
            if self.M.nWmodey>0: 
                growthRates_testy = np.ones((2,self.M.nWmodey))
                growthRates_testy[0,:] = self.M.wmodeLdDQy.imag
            while True:
                oldTmax = newTmax
                t_test = oldTmax*[1/2,.1][flag_dpsidt]
                
                solution = integrate.solve_ivp(dpsidt, t_span=[self.tsODE[0],t_test],y0=self.psi0,
                              t_eval=[self.tsODE[0],t_test],
                              method=self.solve_method,
                              jac=jac,
                              vectorized=True
                              )
                psi_test = solution.y.T
                
                # Calc                 
                self.calc_Modes(psi_test[1],**kwargs)
                maxRelChange_testx=maxRelChange_testy=0
                if self.M.nWmodex>0: 
                    growthRates_testx[1,:] = self.M.wmodeLdDQx.imag
                    maxRelChange_testx = np.max(-np.diff(growthRates_testx/growthRates_testx[0],axis=0))
                if self.M.nWmodey>0: 
                    growthRates_testy[1,:] = self.M.wmodeLdDQy.imag
                    maxRelChange_testy = np.max(-np.diff(growthRates_testy/growthRates_testy[0],axis=0))
                
                # Find new tmax
                maxRelChange_test=max(maxRelChange_testx,maxRelChange_testy)
                newTmax = t_test/maxRelChange_test*[1,2*1/4][flag_dpsidt] * 2
                # *3 for margin of estimate, 
                # /4 for assumption of cubic behaviour if flag_dpsidt
                # *2/1.5 for accurate estimate 
                print('Changed tmax=%.1e s -> %.1e s'%(oldTmax,newTmax))
                
                if (oldTmax/newTmax) < 1.1 and (newTmax/oldTmax) < 1.1:
                    break 
            self.tsODE = self.tsODE*newTmax/self.tmax
            self.set_time(self.tsODE)
                
        
        # Do time calculation 
        print('\nSolver.py: Solve diffusion equation.\n')
        if not flag_dpsidt :
            dpsidt=lambda t, y: self.dpsidtFV_const(t, y)
            if self.solve_method=='LSODA':
                jac = lambda t, y: self.FV_M.toarray()
                print('LSODA solver does not support sparse matrices in this PyRADISE')
            else:
                jac = lambda t, y: self.FV_M
            
            #Solve for distribution evolution
            solution = integrate.solve_ivp(dpsidt, t_span=[0,self.tmax],y0=self.psi0,t_eval=self.tsODE,
                              method=self.solve_method,
                              jac=jac, 
                                vectorized=True
                              )
            ts = solution.t
            if not np.all(ts==self.tsODE):
                print('ERROR: Not same input and output times. Biggest offset is %.0e'%(
                        np.max(np.abs(self.tsODE-ts))))
            self.psis=solution.y.T

            # Check the last distribution
            self.calc_Modes(self.psis[-1],**kwargs)
            
            # Calculate growthrates of the first mode 
            if self.M.nWmodex>0 and self.M.flag_wmode: self.SD_calcGrowthRate(plane=0,modeDQ = self.M.wmode__DQx[0])
            if self.M.nWmodey>0 and self.M.flag_wmode: self.SD_calcGrowthRate(plane=1,modeDQ = self.M.wmode__DQy[0])
            
        elif flag_dpsidt:
            nSteps = self.ntODE
            self.psis = np.zeros((nSteps,np.size(self.psi0)))
            self.psis[0] = self.psi0
            
            if self.M.nWmodex>0: self.growthRatex = np.zeros(nSteps)
            if self.M.nWmodey>0: self.growthRatey = np.zeros(nSteps)
            
            for it in range(nSteps):
                #Recalculate matrices
                err = self.calc_Coeff(self.psis[it],**kwargs)
                if self.M.nWmodex>0: self.growthRatex[it] = np.max(self.M.wmodeLdDQx.imag)
                if self.M.nWmodey>0: self.growthRatey[it] = np.max(self.M.wmodeLdDQy.imag)
                if it==nSteps-1:
                    break
                if err:
                    print('Stop calculation')
                    nt=it+1
                    self.ntODE = nt
                    self.psis = self.psis[:nt]
                    self.tsODE = self.tsODE[:nt]
                    
                    # New stability diagram calculations
                    dtSD = int(np.ceil(nt/self.ntSD))
                    self.indexSD = np.arange(0,nt,dtSD)
                    if not (it in self.indexSD):
                        self.indexSD=np.concatenate((self.indexSD,[it]))
                        print('Added the first unstable time to tsSD')
                    ntSD = np.size(self.indexSD)
                    self.ntSD = ntSD
                    self.tsSD = self.tsODE[self.indexSD]
                                       
                    # Report Latency
                    for plane in [0,1]:
                        if [self.M.nWmodex,self.M.nWmodey][plane]==0:
                            continue
                        if plane==0:  growthRate = self.growthRatex
                        elif plane==1:growthRate = self.growthRatey
                        
                        ind = np.argmax(growthRate>0)-1
                        if ind>=0:
                            lat = self.tsODE[ind]-growthRate[ind]/(growthRate[ind+1]-growthRate[ind])*(self.tsODE[ind+1]-self.tsODE[ind])
                            if lat > self.time_scale*1e-1:
                                print('Latency (%s) = %.2f %s = %.6e'%(['x','y'][plane],lat/self.time_scale,self.tunit,lat))
                            else:
                                print('Latency (%s) = %.2e %s = %.6e'%(['x','y'][plane],lat/self.time_scale,self.tunit,lat))
                            if plane==0:
                                self.latencyx=lat 
                            else:
                                self.latencyy=lat
                    break
                print('Timestep %d: %.2es -> %.2es'%(it,self.tsODE[it],self.tsODE[it+1]))
                self.calc_FV_M()
                self.calc_BC_V()
                
                dpsidt=lambda t, y: self.dpsidtFV_const(t, y)
                if self.solve_method=='LSODA':
                    jac = lambda t, y: self.FV_M.toarray()
                    print('LSODA solver does not support sparse matrices in this PyRADISE')
                else:
                    jac = lambda t, y: self.FV_M
                
                #Solve for distribution evolution
                solution = integrate.solve_ivp(dpsidt, t_span=[self.tsODE[it],self.tsODE[it+1]],y0=self.psis[it],
#                              t_eval=[self.tsODE[it],self.tsODE[it+1]],
                              t_eval=[self.tsODE[it+1]],
                              method=self.solve_method,
                              jac=jac, 
                              vectorized=True
                              )
                self.psis[it+1] = solution.y.T[0]
                
            # Reduce the number of stored distributions
            if self.ntSD >0:
                self.indexPSI = self.indexSD
            else:
                dt = int(np.ceil(self.tsODE/self.ntPSImax))
                self.indexPSI = np.arange(0,self.ntODE,dt)
                if not (self.ntODE in self.indexPSI):
                    self.indexPSI = np.concatenate((self.indexPSI,[self.ntODE]))
            self.tsPSI = self.tsPSI[self.indexPSI]
            self.ntPSI = np.size(self.tsPSI)
            self.psis = self.psis[self.indexPSI]
            
            # Set the length of Growth rate 
            self.indexGR = np.arange(self.ntODE)
            if self.M.nWmodex>0: self.growthRatex = self.growthRatex[self.indexGR]
            if self.M.nWmodey>0: self.growthRatey = self.growthRatey[self.indexGR]

                
        else:        
            # Never use this... 
            dpsidt=lambda t, y: self.dpsidtFV_tdep(t,y)
            jac   =None # lambda t, y: self.FV_M
            
            #Solve for distribution evolution
            solution = integrate.solve_ivp(dpsidt, t_span=[0,self.tmax],y0=self.psi0,t_eval=self.tsODE,
                              method=self.solve_method,
                              jac=jac, 
                                vectorized=True
                              )
            ts = solution.t
            if not np.all(ts==self.tsODE):
                print('ERROR: Not same input and output times. Biggest offset is %.0e'%(
                        np.max(np.abs(self.tsODE-ts))))
            self.psis=solution.y.T
         
    
        

        
#        solution = integrate.odeint(dpsidt,y0=self.psi0,t=self.tsODE,tfirst=True)
#        solution = np.zeros((self.ntODE,np.size(self.psi0)))
#        solution[0]=self.psi0
#        for i in  range(self.ntODE-1) :# enumerate(self.tsODE[1:]):
#            solution[i+1] = integrate.BDF(dpsidt,t0=self.tsODE[i],y0=solution[i],t_bound=self.tsODE[i+1],vectorized=True)        
#        self.psis =solution
        
        
        tot = time.time()-start
        print('Solving the PDE took %.2fs'%(tot))

        return self.psis    
        #                                   method='RK45',max_step=dtmax)   
        
    ###########
    ## 
