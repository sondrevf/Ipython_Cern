import numpy as np
import scipy as sc
import multiprocessing as mp
import time,copy
import os 

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm

from PySSD.Integrator      import Integrator #FixedTrapezoidalIntegrator, SimpsonIntegrator, TrapzIntegrator, DblquadIntegrator
from PySSD.Distribution    import Gaussian
from PyRADISE.PySSDHelper  import get_tune_range, findQs,get_tune_range_focused
from PyRADISE.Distribution import MyDistribution1D,MyDistribution1Dy,MyDistribution2D,MyDistribution2Dy
from PyRADISE.Plotting     import sci_not
from PyRADISE.Coefficients import calc_LandauDampedOneMode

## Calculate stability diagram
def calc_Stability_Diagram(args):
    # Take input
    [i, plane, detuning, distribution, nQ, integrator_epsilon, debug, center_tune, width_ratio] =args
    if debug>1: print(i,'entry ok')

    
    ## Set the integrator
    #integrator = TrapzIntegrator(distribution, detuning, maxJ=18)
    #     integrator = SimpsonIntegrator(distribution, detuning, maxJ=18)
    #     integrator = DblquadIntegrator(distribution, detuning, maxJ=18)
    integrator1 = Integrator(distribution, detuning, maxJ=18,epsilon=integrator_epsilon)
    integrator2 = Integrator(distribution, detuning, maxJ=18,epsilon=integrator_epsilon*2)

    ## Start calculation
    t0 = time.time()
    if center_tune==None:
        freeQs = get_tune_range(detuning,margin=2e-5, n_samples=nQ)
    else:
        freeQs = get_tune_range_focused(detuning,margin=2e-5, n_samples=nQ,
                                        center_tune=center_tune, width_ratio=width_ratio)
    cohDQs = np.zeros((len(freeQs)), dtype='complex')
    for k, q in enumerate(freeQs):
#        cohDQs[k] = 2*integrator.integrate(q,epsilon=integrator_epsilon) \
#                    - integrator.integrate(q,epsilon=integrator_epsilon*2)
        cohDQs[k] = 2*integrator1.integrate(q) - integrator2.integrate(q)
#        cohDQs[k] = integrator.integrate(q,epsilon=integrator_epsilon)
#    cohDQs = cohDQs -1j*integrator_epsilon

    # Timing
    t1 = time.time()-t0
    if i<10 or i%10==0:
        print("Elapsed time calculating a stability diagram (%d,%s): %.2fs."%(i,['x','y'][plane],t1))
    return (i,plane,cohDQs,freeQs)


class StabilityDiagramClass(object):
    def initSD(self,indexSD,scales=[],nQ=100,integrator_epsilon=2e-7,interp_order=1,debug=0):
        if np.shape(indexSD)==():
            indexSD = [indexSD]
        self.indexSD= indexSD
        self.scales = scales 
        self.tsSD   = self.tsODE[indexSD]
        self.ntSD   = np.size(self.indexSD)
        self.nQ     = nQ
        self.integrator_epsilon=integrator_epsilon
        self.interp_order=interp_order
        self.debug  = debug

    def _workerSD(self,args):
        [i,ind,plane,interp_order,nQ,integrator_epsilon,debug,detuningScale,center_tune,width_ratio] = args
        # Get the distribution 
        if ind<0:
            distribution=Gaussian()
        else:
            if self.ND==1:
                if plane ==0: Func = MyDistribution1D #( self.G.Jxbx,self.G.Jxc, self.psis[i],self.dPsis1DdJx[i])
                else:         Func = MyDistribution1Dy #(self.G.Jxbx,self.G.Jxc, self.psis[i],self.dPsis1DdJx[i])
                distribution =Func(self.G.Jxbx,self.G.Jxc, self.psis1Dx[ind],self.dPsis1DdJx[ind],interp_order=interp_order)
            else:
                if plane ==0: Func = MyDistribution2D #(self.G.Jxbx,self.G.Jxc,self.G.Jyby,self.G.Jyc,
                                                #self.psis2D[ind],self.dPsis2DdJx[ind],self.dPsis2DdJy[ind],interp_order=interp_order)
                else:         Func = MyDistribution2Dy #(self.G.Jxbx,self.G.Jxc,self.G.Jyby,self.G.Jyc,
                                                #self.psis2D[ind],self.dPsis2DdJx[ind],self.dPsis2DdJy[ind],interp_order=interp_order)
                distribution =Func (self.G.Jxbx,self.G.Jxc,self.G.Jyby,self.G.Jyc,
                                    self.psis2D[ind],self.dPsis2DdJx[ind],self.dPsis2DdJy[ind],interp_order=interp_order)
        if debug>1: print(i,'distribution ok')

        ## Set the detuning - Assume for now linear detuning
        if plane==0: detuning = self.M.Q    # detuning = LinearDetuning(self.M.Q.Q0x, a, b)
        else:        detuning = self.M.Qy   # detuning = LinearDetuning(self.M.Q.Q0y, a, b)
        if detuningScale!=1: 
            detuning = copy.deepcopy(detuning)  
            detuning.scale(detuningScale)

        if debug>1: print(i,'detuning ok')

        args = [i,plane,detuning,distribution,nQ,integrator_epsilon,debug,center_tune, width_ratio]
        return calc_Stability_Diagram(args)

    ## Calculate SD for all distributions calculated with the solver
    def SD_calcEvolution(self,planes=[0,1],interp_order=1,**kwargs):
        """
        kwargs: nQ, indexSD, debug, integrator_epsilon, width_ratio
            width_ratio: if smaller than 1 - calculate a narrow SD
        """
        start = time.time()
        # Calculate all stability diagrams for the distribution
        integrator_epsilon = kwargs.pop('integrator_epsilon',self.integrator_epsilon)
        width_ratio = kwargs.pop('width_ratio',1)
        nQ       = kwargs.pop('nQ',self.nQ)
        debug    = kwargs.pop('debug',self.debug)
        indexSD  = kwargs.pop('indexSD',self.indexSD)
        ntSD     = np.size(indexSD)
        detuningScale = 1
        ## Paralellisation preperation
        nProcs = min(np.size(planes)*np.size(indexSD),mp.cpu_count())
        if os.path.exists('/home/sfuruset/'):
            nProcs = min(nProcs,4)
        print("%d processes (ncpus=%d)"%(nProcs,mp.cpu_count()))
        if nProcs==0:
            print('OBS in SD_calcScaledPsi0: No processes needed')
            return -1
        pool = mp.Pool(processes = nProcs)

        def collect_result(res):
            results.append(res)
        count=0
        procs = []
        results = []   # must be after def collect_result...


        ## Start the calculations
        for plane in planes:
            if plane==0  : 
                self.cohDQsAllx = np.zeros((ntSD,nQ),dtype='complex')
                self.freeQsAllx = np.zeros((ntSD,nQ))
                wmodeLd_Q = self.M.wmodeLdDQx +self.M.Q.Q0x
            elif plane==1: 
                self.cohDQsAlly = np.zeros((ntSD,nQ),dtype='complex')
                self.freeQsAlly = np.zeros((ntSD,nQ))
                wmodeLd_Q = self.M.wmodeLdDQy +self.M.Q.Q0y
            else:
                continue
            # Set the width of SD
            center_tune=None 
            if np.size(wmodeLd_Q)>0 and width_ratio!=1:
                center_tune = wmodeLd_Q[0].real

            # Commit jobs
            for i,ind in enumerate(indexSD):   
                ind = np.argmax(ind==self.indexPSI)         
                args = [i,ind,plane,interp_order,nQ,integrator_epsilon,debug,detuningScale,center_tune,width_ratio]
                p = pool.apply_async(self._workerSD, args = (args,),callback = collect_result)
                procs.append(p)
                count+=1
            for p in procs:
                if debug: p.get()
        ###################
        # Join forces again
        pool.close()
        pool.join()

        ## Extract results
        for j in range(count):
            i,plane,res,res2 = results[j]
            if plane==0: 
                self.cohDQsAllx[i] = res
                self.freeQsAllx[i] = res2
            else:
                self.cohDQsAlly[i] = res
                self.freeQsAlly[i] = res2
        print('SD Evolution: %.2fs'%(time.time()-start))
        return #self.cohDQsAllx,self.freeQsAllx,self.cohDQsAlly,self.freeQsAlly

    ## Calculate SD for IC with various octupole strengths
    def SD_calcScaledPsi0(self,plane=0,interp_order=1,ind=0,**kwargs):
        """
        ind =  >=0: With calculated distribution, <0: With Gaussian()
        kwargs: nQ, debug, integrator_epsilon, scales 
        """
        start = time.time()
        # Load important parameters
        nQ       = kwargs.pop('nQ',self.nQ)
        debug    = kwargs.pop('debug',self.debug)
        integrator_epsilon = kwargs.pop('integrator_epsilon',self.integrator_epsilon)
        scales   = kwargs.pop('scales',self.scales)
        ## Paralellisation preperation
        nSD_scaled = np.size(scales)
        nProcs = min(nSD_scaled,mp.cpu_count())
        if nProcs==0:
            print('OBS in SD_calcScaledPsi0: No processes needed')
            return -1
        print("%d processes (ncpus=%d)"%(nProcs,mp.cpu_count()))
        pool = mp.Pool(processes = nProcs)

        def collect_result(res):
            results.append(res)
        count=0
        procs = []
        results = []   # must be after def collect_result...

        # Full width SD
        center_tune=None
        width_ratio=1
        for i, detuningScale in enumerate(np.concatenate((scales,[0.5]))):
#            args = [i,ind,plane,nQ,debug,integrator_epsilon,scale]
            args = [i,ind,plane,interp_order,nQ,integrator_epsilon*detuningScale,debug,detuningScale,center_tune,width_ratio]
            p = pool.apply_async(self._workerSD, args = (args,),callback = collect_result)
            procs.append(p)
            count+=1
        for p in procs:
            if debug: p.get()

        ###################
        # Join forces again
        pool.close()
        pool.join()

        cohDQs_psi0_scaled=np.zeros((nSD_scaled,nQ),dtype=complex)
        freeQs_psi0_scaled=np.zeros((nSD_scaled,nQ))
        for j in range(count):
            i,_,res1,res2 = results[j]
            if i < nSD_scaled:
                cohDQs_psi0_scaled[i]=res1
                freeQs_psi0_scaled[i]=res2
            else:
                if plane==0:
                    self.cohDQs_halfIx=res1
                else:
                    self.cohDQs_halfIy=res1

        # save in self
        if plane ==0:
            self.cohDQs_psi0_scaledx = cohDQs_psi0_scaled
            self.freeQs_psi0_scaledx = freeQs_psi0_scaled
            self.scalesx = scales
        else:
            self.cohDQs_psi0_scaledy = cohDQs_psi0_scaled
            self.freeQs_psi0_scaledy = freeQs_psi0_scaled
            self.scalesy = scales

        print('SD scaled strengths: %.2fs'%(time.time()-start))
        return cohDQs_psi0_scaled,freeQs_psi0_scaled

    def SD_copyScaledPsi0(self,planeFrom):
        if planeFrom==0:
            self.cohDQs_psi0_scaledy = self.cohDQs_psi0_scaledx
            self.freeQs_psi0_scaledy = self.freeQs_psi0_scaledx
            self.scalesy = self.scalesx
            self.cohDQs_halfIy = self.cohDQs_halfIx
        else:
            self.cohDQs_psi0_scaledx = self.cohDQs_psi0_scaledy
            self.freeQs_psi0_scaledx = self.freeQs_psi0_scaledy
            self.scalesx = self.scalesy
            self.cohDQs_halfIx = self.cohDQs_halfIy

    #######################################################################
    
    def SD_plotSD(self,plane,flag_fillSD=True,flag_savefig=False,savedir='01_Plots/',figname='fig',
                  flag_SD_constituents=False,iTime_step=1,**kwargs):
        if plane==0:
            cohDQsAll = self.cohDQsAllx
            freeQsAll = self.freeQsAllx - self.M.Q.Q0x
        else:
            cohDQsAll = self.cohDQsAlly
            freeQsAll = self.freeQsAlly - self.M.Q.Q0y

        time_scale = self.time_scale
        tlabel= self.tlabel
        ts = self.tsSD
        tmax = kwargs.pop('tmax',self.tmax)
        
        if 'ax' in kwargs:
            ax = kwargs.get('ax')
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(1)#, figsize=(16, 15))
        # col = sns.color_palette("husl", len(tsODE), 0.6)

        # Set colorbar as time measurement:
        cmap = cm.get_cmap(None)
        bounds = ts/time_scale
        if tmax>np.max(ts):
            bounds = np.concatenate((ts,[tmax]))/time_scale
        
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        if 0:
            normed = (np.array(bounds)-bounds[0])/(bounds[-1]-bounds[0])
            listOfCol = [cmap((normed[i]+normed[i+1])*.5) for i in range(len(bounds)-1) ]
            norm= mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(listOfCol))
            cmap = mpl.colors.ListedColormap(listOfCol)

#        norm = mpl.colors.Normalize(vmin=ts.min()/time_scale, vmax=tmax/time_scale)
#        nColor = int(tmax/ts[1]-0.5)
#        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(None,lut=[None,nColor][flag_fillSD]))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cmap.set_array([])

        for i,t in enumerate(ts):
            if i%iTime_step>0: continue
            if flag_fillSD:
                if i==0: continue
                ax.fill(np.concatenate((cohDQsAll[i].real,cohDQsAll[i-iTime_step][::-1].real)),
                        np.concatenate((cohDQsAll[i].imag,cohDQsAll[i-iTime_step][::-1].imag)),
                        color=cmap.to_rgba(0.5*(t+ts[i-iTime_step])/time_scale))
            else :
                ax.plot(cohDQsAll[i].real, cohDQsAll[i].imag, c=cmap.to_rgba(t/time_scale)) 



        ax.set_xlabel("$\operatorname{Re}\{ \Delta Q_\mathrm{coh}\}$")
        ax.set_ylabel("$\operatorname{Im}\{ \Delta Q_\mathrm{coh}\}$")
        ax.grid(True)

        minx = np.min(cohDQsAll[0].real)
        maxx = np.max(cohDQsAll[0].real)
        ax.set_xlim(minx,maxx)

        #finish colorbar
        fac=1.4
        if not 'ax'in kwargs:
            cb=fig.colorbar(cmap,label=tlabel,fraction=0.046/fac,aspect=20*fac, pad=0.02,
                            norm=norm,boundaries=bounds,spacing='proportional') #ticks = np.linspace(0,tmax/time_scale,4),
            tick_locator = mpl.ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()
            fig.tight_layout()

        if flag_savefig: fig.savefig(savedir+'%s.eps'%(figname))

        ####################################################################################
        ####################################################################################
        if flag_SD_constituents:
            fig2,ax2 =plt.subplots(2,1,sharex=True)
            # Set colorbar as time measurement:
            norm = mpl.colors.Normalize(vmin=ts.min()/time_scale, vmax=ts.max()/time_scale)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(None))
            cmap.set_array([])

            for i,t in enumerate(ts):
                Q =  freeQsAll[i]
                ax2[0].plot(Q/self.M.Q.ax,cohDQsAll[i].real-Q, c=cmap.to_rgba(t/time_scale)) 
                ax2[1].plot(Q/self.M.Q.ax,cohDQsAll[i].imag, c=cmap.to_rgba(t/time_scale)) 
            ax2[0].set_ylabel(r'Real $\Delta Q - Q$')
            ax2[1].set_ylabel(r'Imag $\Delta Q$')
            ax2[1].set_xlabel(r'$Q/a$')
            for i in range(2):
                ax2[i].grid(True)
            fig2.tight_layout()
            fig2.subplots_adjust(hspace=0)

        ####################################################################################
        ####################################################################################
            fig2,ax2 =plt.subplots(2,1,sharex=True)
            # Set colorbar as time measurement:
            norm = mpl.colors.Normalize(vmin=ts.min()/time_scale, vmax=ts.max()/time_scale)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(None))
            cmap.set_array([])

            for i,t in enumerate(ts):
                Q =  freeQsAll[i]
                ReDQ = cohDQsAll[i].real
                ImDQ = cohDQsAll[i].imag
                ReInt= -ReDQ / (ReDQ**2 + ImDQ**2)
                ImInt=  ImDQ / (ReDQ**2 + ImDQ**2)
                ax2[0].plot(Q/self.M.Q.ax, ReInt, c=cmap.to_rgba(t/time_scale)) 
                ax2[1].plot(Q/self.M.Q.ax, ImInt, c=cmap.to_rgba(t/time_scale)) 
            ax2[0].set_ylabel('Real integral')
            ax2[1].set_ylabel('Imag integral')
            ax2[1].set_xlabel(r'$Q/a$')
            for i in range(2):
                ax2[i].grid(True)
            fig2.tight_layout()
            fig2.subplots_adjust(hspace=0)


        return fig,ax

    def SD_plotWaterfall(self,plane,basex=None,nlevels=30,flag_logScale=0):
        # Import necessary values
        if plane==0:
            cohDQsAll = self.cohDQsAllx
        else:
            cohDQsAll = self.cohDQsAlly
        time_scale = self.time_scale
        tlabel= self.tlabel
        ts = self.tsSD
        if basex is None:
            minx = np.min(cohDQsAll[0].real)
            maxx = np.max(cohDQsAll[0].real)
            basex = np.linspace(minx,maxx,100)


        # Interpolate each SD to a uniform grid
        nt = np.size(ts)
        nx = np.size(basex)
        AllY = np.zeros((nt,nx))
        for i in range(nt):
            x = cohDQsAll[i].real
            y = cohDQsAll[i].imag
            f = sc.interpolate.interp1d(x,y,kind='linear',bounds_error=False,fill_value=0)
            AllY[i] = f(basex)

        # Find how to show plot
        lims = [0,min(np.max(AllY),2.*np.max(AllY[:,int(nx/2)]))]
        norm = colors.Normalize(vmin = lims[0],vmax=lims[1],clip=False)
        cticks=None
        if np.max(AllY)>lims[1] or flag_logScale :
            exp = int(101.5-np.log10(lims[1]) )-100
            thresh = np.round(lims[1],decimals=exp)
            ratio = 1+np.int(np.log10(np.max(AllY)/thresh))
            nDec = min(2,ratio) 

            norm=colors.SymLogNorm(vmax=thresh*10**nDec,vmin=0,linthresh=thresh,linscale=nDec*3,clip=False)        
            cticks = np.concatenate(([0,np.round(thresh/2,decimals=exp)],thresh*np.logspace(0,nDec,nDec+1)))
        levels=norm.inverse(np.linspace(0,1,nlevels))

        # Create figure
        fig = plt.figure()
        ax=fig.gca()
        cmap = cm.get_cmap(None)
        plt.contourf(AllY,cmap=cmap,levels=levels,norm=norm,
                     extent=(np.min(basex),np.max(basex),0,np.max(ts)/time_scale),extend='both')
        cb = plt.colorbar(label = "$\operatorname{Im}\{ \Delta Q_\mathrm{coh}\}$",extend='both',
                          ticks=cticks)

        # Set Cticklabels
        if (cticks is None):
            tick_locator = mpl.ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()
        else:
            labels = [item.get_text() for item in cb.ax.get_yticklabels()]
            for i in range(len(labels)):
                labels[i] = sci_not(cticks[i],0,flag_ignore1=True)
            #print('c-labels:', cticks,labels)
            cb.ax.set_yticklabels(labels)

        # Fix rest of layout
        ax.set_xlabel("$\operatorname{Re}\{ \Delta Q_\mathrm{coh}\}$")#, fontsize=20)
        ax.set_ylabel(tlabel)
        fig.tight_layout()

        return fig,ax


    ############################
    ## Damping rate of a mode ##
    def SD_calcGrowthRate(self,plane=0,modeDQ = 0 + 0*1j,relstep=[.5,.1],tol=1e-4,debug=0,**kwargs):
        if plane==0:
            Q0   = self.M.Q.Q0x
            detuning = self.M.Q 
        else:
            Q0   = self.M.Q.Q0y
            detuning = self.M.Qy
        self.indexGR = self.indexPSI 
        growthRate = np.zeros(self.ntPSI)
        flagFindalpha = kwargs.pop('flagFindalpha',self.flagFindalpha)
        
        # Calculate stability margin 
        for it in range(self.ntPSI):
            self.calc_InterpDist(self.psis[it],plane=plane)
            if plane==0:
                distribution = self.interpDistx
            else:
                distribution = self.interpDisty
            #integrator = TrapzIntegrator(distribution, detuning, maxJ=18)
#            calc_LandauDampedOneMode(self,integrator,Q0,modeDQ,relstep=[.5,.05],tol=1e-4,flagFindalpha=1,debug=0)
            integrator1 = Integrator(distribution, detuning, maxJ=18,epsilon=self.integrator_epsilon*1)
            integrator2 = Integrator(distribution, detuning, maxJ=18,epsilon=self.integrator_epsilon*2)
            integrator4 = Integrator(distribution, detuning, maxJ=18,epsilon=self.integrator_epsilon*4)
            dampDQ , cnt, relerr = calc_LandauDampedOneMode(self,integrator1,integrator2,integrator4,Q0,modeDQ,relstep=relstep,tol=tol,debug=debug,
                                                flagFindalpha=flagFindalpha)
            print('SD_calcGrowthRate (%.2f %s): Mode: %.2e %s%.2ei -> %.2e %s%.2ei (relerr(%d iterations)=%.1e)'%(
                self.tsPSI[it]/self.time_scale,self.tunit,
                modeDQ.real,['+','-'][np.sign(modeDQ.imag)<0],np.abs(modeDQ.imag),
                dampDQ.real,['+','-'][np.sign(dampDQ.imag)<0],np.abs(dampDQ.imag),cnt,relerr))
            growthRate[it] = dampDQ.imag
        
        # Report if unstable
        if np.any(growthRate>0):
            ind = np.argmax(growthRate>0)-1
            if ind>=0:
                tsGR= self.tsPSI
                lat = tsGR[ind]-growthRate[ind]/(growthRate[ind+1]-growthRate[ind])*(tsGR[ind+1]-tsGR[ind])
                print('Latency (%s) = %.2f %s'%(['x','y'][plane],lat/self.time_scale,self.tunit))
        
        # Store it
        if plane==0:
            self.modex = modeDQ
            self.growthRatex = growthRate
        else:
            self.modey = modeDQ
            self.growthRatey = growthRate
        return
        
    def SD_plotGrowthRate(self,plane=0,flag_normalize=1,label='',c=None,**kwargs):
        """
        kwargs include: ax, iTime_step
        """
        try:
            if plane==0:
                growthRate=self.growthRatex
            else:
                growthRate=self.growthRatey
        except:
            print('ERROR in SD_plotGrowthRate%s: Could not find attributes'%['x','y'][plane])
            return 

        if 'ax' in kwargs:
            ax1 = kwargs.pop('ax')
            ax1.autoscale(axis='y')  # Necessary to update the ylims with the new data!
            fig1= ax1.get_figure()
        else:
            fig1,ax1=plt.subplots(1)

        string_normalize= ''
        labelapp=''

        if flag_normalize:
            growthRate0 = growthRate[0]
            growthRate = growthRate/np.abs(growthRate0)
            string_normalize='Rel. '

        # Find how many GR we have.
        index = [self.indexGR,np.arange(self.ntSD)][np.size(growthRate)==self.ntSD]
        tsGR= self.tsODE[index]
        ax1.plot(tsGR/self.time_scale,growthRate,'o-',c=c,label=label+labelapp)
        ax1.set_ylabel(r'%s$\operatorname{Im}\{ \Delta Q_\mathrm{LD}\}$'%(string_normalize))

        # Layout
        ax1.set_ylim(top=0)
        ax1.set_xlabel(self.tlabel)
        ax1.grid(True)
        if not label=='':
            ax1.legend(loc=0)
        fig1.tight_layout()
        
        # Report if unstable
        if np.any(growthRate>0):
            ind = np.argmax(growthRate>0)-1
            if ind>=0:
                lat = tsGR[ind]-growthRate[ind]/(growthRate[ind+1]-growthRate[ind])*(tsGR[ind+1]-tsGR[ind])
                print('Latency (%s) = %.2f %s'%(['x','y'][plane],lat/self.time_scale,self.tunit))

        return fig1,ax1
    
    #################################
    ## Margin in damping of a mode ##
    def SD_calcStabilityMargin(self,plane=0,mode = 0 + 0*1j,debug=0):
        try:
            if plane==0:
                cohDQsAll = self.cohDQsAllx
                freeQsAll = self.freeQsAllx - self.M.Q.Q0x
            else:
                cohDQsAll = self.cohDQsAlly
                freeQsAll = self.freeQsAlly - self.M.Q.Q0y
        except:
            print('ERROR in SD_calcStabilityMargin%s: Could not find attributes'%['x','y'][plane])
            return 
            
        # The mode components
        mode_R = np.real(mode)
        mode_I = np.imag(mode)

        stabilityMargin = np.zeros(self.ntSD)
        for i in range(self.ntSD):
            freeQ = freeQsAll[i]
            cohDQ = cohDQsAll[i]

            cohDQ_R = np.real(cohDQ)
            cohDQ_I = np.imag(cohDQ)

            # Test input monotonicity
            flag_nonmonotone = np.any(np.diff(cohDQ_R)<0)
            if flag_nonmonotone:
                print("cohDQ_R is not strictly increasing...")
                print(cohDQ_R*1e5-0.31)

            #Find real tune for coherent mode
            right = np.searchsorted(cohDQ_R,mode_R)
            if right==0:
                right+=1
            left = right-1

            dQIdQR = (cohDQ_I[right]-cohDQ_I[left])/(cohDQ_R[right]-cohDQ_R[left])
            cohDQ_I_mode = cohDQ_I[left] + dQIdQR*(mode_R-cohDQ_R[left])

            # Find margin from SD to mode along imaginary axis
            stabilityMargin[i] = cohDQ_I_mode-mode_I
            if debug:
                print(("R(dQ)=%.1e, I(dQ)=%.1e, SD_dQx=%.1e, iL=%d, iR=%d \nx(L,R)=(%.2e,%.2e), dx=%.2e, xi-xL=%.2e\n"+
                      "y_(L,R)=(%.2e,%.2e)\n")%(
                        mode_R,mode_I,cohDQ_I_mode,left,right,cohDQ_R[left],cohDQ_R[right],(cohDQ_R[right]-cohDQ_R[left]),
                        (mode_R-cohDQ_R[left]),cohDQ_I[right],cohDQ_I[left]))

        if plane==0:
            self.modex = mode
            self.stabilityMarginx = stabilityMargin
        else:
            self.modey = mode
            self.stabilityMarginy = stabilityMargin
        return

    def SD_plotStabilityMargin(self,plane=0,flag_relative=1,flag_normalize=1,label='',c=None,flag_plotInscribed=False,**kwargs):
        """
        kwargs include: ax, iTime_step
        """
        try:
            if plane==0:
                mode = self.modex
                stabilityMargin=self.stabilityMarginx
            else:
                mode = self.modey
                stabilityMargin=self.stabilityMarginy
        except:
            print('ERROR in SD_plotStabilityMargin%s: Could not find attributes'%['x','y'][plane])
            return 

        if 'ax' in kwargs:
            ax1 = kwargs.pop('ax')
            ax1.autoscale(axis='y')  # Necessary to update the ylims with the new data!
            fig1= ax1.get_figure()
        else:
            fig1,ax1=plt.subplots(1)

        string_relative = ' - \Delta Q_\mathrm{mode}'
        string_normalize= ''
        labelapp=''
        if not flag_relative:
            stabilityMargin = stabilityMargin + mode.imag
            string_relative = ''
            labelapp = ' thr=%.1e'%mode.imag 
        
        if flag_normalize:
            margin0 = stabilityMargin[0]
            stabilityMargin = stabilityMargin/margin0
            string_normalize='Rel. '
            if len(labelapp)>0:
                labelapp = ' thr=%.2f'%(mode.imag/margin0)
            
#        if flag_relative:
#            ax1.plot(self.tsSD/self.time_scale,stabilityMargin/stabilityMargin[0],'o-',c=c,label=label)
#            ax1.set_ylabel(r'Rel. $\operatorname{Im}\{ \Delta Q_\mathrm{SD}\}$')
#        else:
#            ax1.plot(self.tsSD/self.time_scale,stabilityMargin,'o-',c=c,label=label)
#            ax1.set_ylabel(r'$\operatorname{Im}\{ \Delta Q_\mathrm{SD} - \Delta Q_\mathrm{mode}\}$')

        ax1.plot(self.tsSD/self.time_scale,stabilityMargin,'o-',c=c,label=label+labelapp)
        ax1.set_ylabel(r'%s$\operatorname{Im}\{ \Delta Q_\mathrm{SD}%s\}$'%(string_normalize,string_relative))

        # Layout
        
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel(self.tlabel)
        ax1.grid(True)
        if not label=='':
            ax1.legend(loc=0)
        fig1.tight_layout()

        # Plot mode in stability diagram
        if flag_plotInscribed:
            iTime_step = kwargs.pop('iTime_step',1)
            fig2,ax2=self.SD_plotSD(plane=plane,iTime_step=iTime_step,**kwargs)
            ax2.plot(mode.real,mode.imag,'kx',ms=10,mew=2)
            return fig1,ax1,fig2,ax2
        return fig1,ax1


    ####################################################
    ## Relative change of effective detuning strength ##
    def SD_calcEffectiveStrength(self,plane=0,flag_interpStrength = 1,
                                 mode_R = 0,flag_allDQ_R = 1,maxmin_ratio=np.nan,debug=0):
        try:
            if plane==0:
                scales = self.scalesx
    #            if np.size(scales)==0:
    #                print('No scaled SD in the horizontal plane')
    #                return 
                cohDQsAll = self.cohDQsAllx
                freeQsAll = self.freeQsAllx - self.M.Q.Q0x
                cohDQs_psi0_scaled = self.cohDQs_psi0_scaledx
                freeQs_psi0_scaled = self.freeQs_psi0_scaledx

            else:
                scales = self.scalesy
    #            if np.size(scales)==0:
    #                print('No scaled SD in the vertical plane')
    #                return 
                cohDQsAll = self.cohDQsAlly
                freeQsAll = self.freeQsAlly - self.M.Q.Q0y
                cohDQs_psi0_scaled = self.cohDQs_psi0_scaledy
                freeQs_psi0_scaled = self.freeQs_psi0_scaledy
        except:
            print('ERROR in SD_calcEffectiveStrength%s: Could not find attributes'%['x','y'][plane])
            return 

        kind =['nearest','linear'][1]
        nSD_scaled = np.size(scales)
        effectiveStrength = np.zeros(self.ntSD)

        for i in range(self.ntSD):
            freeQ = freeQsAll[i]
            cohDQ = cohDQsAll[i]

            cohDQ_R = np.real(cohDQ)
            cohDQ_I = np.imag(cohDQ)

            # Test input monotonicity
            flag_nonmonotone = np.any(np.diff(cohDQ_R)<0)
            if flag_nonmonotone:
                print("cohDQ_R is not strictly increasing...")

            # Get the values to plot and test
    #         dQ_R_plot = np.linspace(cohDQ_R[0],cohDQ_R[-1],300)    
            # Interpolate SD at time i
            
            # interp_SD = sc.interpolate.interp1d(cohDQ_R,cohDQ_I,kind=kind,bounds_error=False,fill_value=0)
            interp_SD=sc.interpolate.interp1d(freeQ, np.array([cohDQ_R,cohDQ_I]), 
                                                      kind=kind, axis=1,bounds_error=False,fill_value=0)
            if flag_allDQ_R:
                dQ_R_test,dQ_I_test = interp_SD(np.linspace(freeQ[0],freeQ[-1],10000))
                if np.isnan(maxmin_ratio):
                    ind = dQ_R_test*0<10
                else:
                    ind = np.abs(dQ_I_test) > np.max(dQ_I_test)/maxmin_ratio
#                dQ_R_test = np.linspace(cohDQ_R[ind][0],cohDQ_R[ind][-1],300)    

                dQ_R_test = dQ_R_test[ind]
                dQ_I_test = dQ_I_test[ind]
            else:
                dQ_R_test = mode_R
                if flag_nonmonotone:
                    interp_SD=sc.interpolate.interp1d(freeQ, np.array([cohDQ_R,cohDQ_I]), 
                                                      kind=kind, axis=1,bounds_error=False,fill_value=0)
                    
                    temp = interp_SD(np.linspace(cohDQ_R[ind][0],cohDQ_R[ind][-1],10000))
                    tol = 1e-7
                    for k in range(100):
                        tempInd = np.abs(temp[0]-dQ_R_test)<tol
                        tol*=2
                        if np.any(tempInd):
                            break
                    tempInd = np.abs(temp[0]-dQ_R_test)<tol
                    dQ_I_test = np.min(temp[1][tempInd])
                else:
                    interp_SD = sc.interpolate.interp1d(cohDQ_R,cohDQ_I,kind=kind,bounds_error=False,fill_value=0)
                    dQ_I_test = interp_SD(dQ_R_test)
                    
                        

            
            for j in range(nSD_scaled):
                interp_G  = sc.interpolate.interp1d(cohDQs_psi0_scaled[j].real,cohDQs_psi0_scaled[j].imag,
                                                    kind=kind,bounds_error=False,fill_value=0)
                
                # if margin is always positive, will always be stable
                newmargin = np.min(dQ_I_test-interp_G(dQ_R_test))
                if newmargin>=0:
                    effectiveStrength[i] = scales[j]
                    # if interpolate between the calculated strengths
                    if j>0 and flag_interpStrength:
                        dsdm = (scales[j]-scales[j-1])/(newmargin-oldmargin)
                        effectiveStrength[i] = scales[j] - dsdm * newmargin
                        if debug:
                            print('interp strength %.2e -> %.2e, ds=%.2e, dm=%.2e'%(scales[j],effectiveStrength[i],
                                                                               (scales[j]-scales[j-1]),(newmargin-oldmargin)))
                    break
                # if does not find a margin
                if j==nSD_scaled-1:
                    effectiveStrength[i]=0
                    print("Have not calulated a SD with a weak enough scaled strength!")

                oldmargin = newmargin
        #         print(i,j,newmargin)


        if plane==0:
            self.effectiveStrengthx=effectiveStrength / effectiveStrength[0]
        else:
            self.effectiveStrengthy=effectiveStrength / effectiveStrength[0]
        return effectiveStrength


    def SD_plotEffectiveStrength(self,plane=0,label='',c=None,flag_plotInscribed=False,**kwargs):
        """
        kwargs include: ax, iTime_step,fmt_string,minIntensity
        """
        fmt_string = kwargs.pop('fmt_string','o-')
        minIntensity = kwargs.pop('minIntensity',0)
        ind = self.moments_x[:,0]>minIntensity
        try:
            if plane==0:
                effectiveStrength=self.effectiveStrengthx

            else:
                effectiveStrength=self.effectiveStrengthy

        except:
            print('ERROR in SD_plotEffectiveStrength%s: Could not find attributes'%['x','y'][plane])
            return 
        
        # Possibly load given axes
        if 'ax' in kwargs:
            ax1 = kwargs.pop('ax')
            ax1.autoscale(axis='y')  # Necessary to update the ylims with the new data!
            fig1= ax1.get_figure()
        else:
            fig1,ax1=plt.subplots(1)


        #ax1.set_title("Relative change of Effective octupole current")

        ax1.plot(self.tsSD[ind]/self.time_scale,effectiveStrength[ind],fmt_string,c=c,label=label)
        ax1.set_ylabel(r'Rel. effective detuning')

        # Layout
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel(self.tlabel)
        ax1.grid(True)
        if not label=='':
            ax1.legend(loc=0)
        fig1.tight_layout()

        # Plot inscribed SD with scaled strength
        if flag_plotInscribed:
            iTime_step = kwargs.pop('iTime_step',1)
            fig2,ax2=self.SD_plotSD(plane=plane,iTime_step=iTime_step,**kwargs)
            if plane==0:
                cohDQs_psi0_scaled = self.cohDQs_psi0_scaledx
                scales = self.scalesx
            else:
                cohDQs_psi0_scaled = self.cohDQs_psi0_scaledy
                scales = self.scalesy

            cmap = cm.get_cmap(None)
            for i in range(self.ntSD):
                if i%iTime_step==0:
                    iScale = np.searchsorted(-scales,-effectiveStrength[i],'left',)
                    if iScale==np.size(scales):
                        continue
                    print(r'%s(t=%.2e %s): effectiveScale=%.3f , plotScale=%.3f (iScale=%d)'%(
                            ['H','V'][plane],self.tsSD[i]/3600,'h',
                            effectiveStrength[i],scales[iScale],iScale))
                    if iScale<np.size(scales):
                        ax2.plot(cohDQs_psi0_scaled[iScale].real,cohDQs_psi0_scaled[iScale].imag,
                                 '--',lw=2,c=cmap((i)/(self.ntSD-1)))
            return fig1,ax1,fig2,ax2
        return fig1,ax1

    
