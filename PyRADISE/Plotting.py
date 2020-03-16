import numpy as np
import scipy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.cm     as cm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

from PyRADISE.Coefficients import DiffCoeffJxJy, DriftCoeffJxJy

def sci_not(num,dec,flag_ignore1=False):
    if num>0:
        exp = int(np.log10(num)+100)-100
        num = num/10**exp
        sci = r'$%.*f\!\times \!10^{%d}$'%(dec,num,exp)
        if flag_ignore1 and np.round(num)==1:
            sci = r'$10^{%d}$'%(exp)
    else:
        sci = '$0$'
    return sci

class PlottingClass(object):
    ##############
    ## Plotting ##
    def initPlotting(self,time_scale='auto',plot_rmax=6,flag_fill=1,plot_step=1,
                     figwidth=6,figheight=4.5):
        self.set_tlabels(time_scale)
        self.plot_rmax    = plot_rmax
        self.flag_fill    = flag_fill
        self.plot_step    = plot_step
        self.figwidth     = figwidth
        self.figheight    = figheight
        self.time_scale
        
    def set_tlabels(self,time_scale):
        if time_scale=='auto':
            time_scale = 1    #; tlabel=r'$t$ $[\mathrm{sec}]$' ; tunit = 's'
            if self.tmax<0.1:
                time_scale = 0.001
            if self.tmax>200:
                time_scale = 60   #; tlabel=r'$t$ $[\mathrm{min}]$' ; tunit = 'min'
            if self.tmax > 120*60:
                time_scale = 3600 #; tlabel=r'$t$ $[\mathrm{h}]$' ; tunit = 'h'
            
        self.time_scale = time_scale
        if time_scale==.001:
            self.tunit='ms'
        elif time_scale==1:
            self.tunit='s'
        elif time_scale==60:
            self.tunit='min'
        elif time_scale==3600:
            self.tunit='h'
        self.tlabel=r'$t$ $[\mathrm{%s}]$'%(self.tunit)
        
    
    def plot_1D(self,x,y,ts,time_scale=1,tlabel='t',xlabel='x',ylabel='y',yscale='linear',xlim=[None,None],
                plot_step=1,fignr=None,flag_fill=0,lw=1,ls='-',flag_colorbar=True):
        # Set colorbar as time measurement:
        norm = mpl.colors.Normalize(vmin=ts.min()/time_scale, vmax=ts.max()/time_scale)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(None))
        cmap.set_array([])

        L = np.searchsorted(x,xlim[0],'left')
        R = np.searchsorted(x,xlim[1],'right')
        mask = slice(L,R+1,1)

        # Plotting
        fig, ax = plt.subplots(num=fignr)
        for i , t in enumerate(ts):  
            if i%plot_step==0 and np.size(np.shape(y))>1:
                if not flag_fill:
                    ax.plot(x[mask], y[i,mask], c=cmap.to_rgba(t/time_scale),lw=lw,ls=ls)
                else:
                    if i==0: continue
                    ax.fill_between(x[mask], y[i,mask],y[i-plot_step,mask], color=cmap.to_rgba(t/time_scale))

        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim)
        fig.tight_layout()
        ax.grid(True)

        # Set time label ticks
        if flag_colorbar:
            cb=fig.colorbar(cmap,label=tlabel) #ticks = np.linspace(0,tmax/time_scale,4),
            tick_locator = mpl.ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()

        return fig,ax

    def plot_psis_1D(self,x,psis,ts,iCoord,yscale='linear',D=0,sig0=1,plot_step=1,bool_theory_g0=1,xmax=6,lw=1,ls='-',
                    fignradd=0,flag_fill=0,flag_extDist=0,time_scale=3600,tlabel=r'$t$ $[\mathrm{h}]$'):
        #Fix distribution
        if np.min(x)>0 and flag_extDist:
            x_orig = x
            x = np.concatenate(([0],x_orig))
            psis_orig = psis
            psis = np.zeros((np.size(ts),np.size(x)))
            for i , t in enumerate(ts):  
                if i%plot_step==0:
                    func = sc.interpolate.interp1d(x_orig,psis_orig[i,:],kind='linear',bounds_error=False,fill_value='extrapolate')#(0,0))
                    psis[i] = func(x)


        xlim  =[[0,.5*xmax**2],[0,xmax],[-xmax,xmax]][iCoord]
        xlabel=[r'$J$',r'$r$',r'$x$'][iCoord]
        fig1, ax1 = self.plot_1D(x, psis,ts,time_scale=time_scale,
                            tlabel=tlabel,xlabel=xlabel,ylabel=r'$\Psi$',yscale=yscale,xlim=xlim,
                            plot_step=plot_step,fignr=30+2*iCoord+fignradd,flag_fill=flag_fill,lw=lw,ls=ls)
        fig2, ax2 = self.plot_1D(x, (psis-psis[0])/1,ts,time_scale=time_scale,
                            tlabel=tlabel,xlabel=xlabel,ylabel=r'$\Psi(t)-\Psi(0)$',yscale='linear',xlim=xlim,
                            plot_step=plot_step,fignr=31+2*iCoord+fignradd,flag_fill=flag_fill,lw=lw,ls=ls)

        # Theoretical curve
        if bool_theory_g0:
            tmax = np.max(ts)
            sig2t = sig0**2 + D*tmax
            if iCoord==0:
                psitheory = 1/sig2t*np.exp(-x/sig2t)
            elif iCoord==1:
                psitheory = 1/sig2t*np.exp(-0.5*x**2/sig2t)
            elif iCoord==2:
                psitheory = 1/np.sqrt(2*np.pi*sig2t)*np.exp(-.5*x**2/sig2t)

            ax1.plot(x,psitheory,'--k',lw=4,dashes=[1,5],dash_capstyle='round',label=r'Theory, $g=0$')
            ax1.legend(loc=0)

        return fig1,ax1,fig2,ax2
    
    def plot_psis_1Dx(self, iCoord,yscale = 'linear',lw=1, ls='-', fignradd=0,flag_extDist=0,**kwargs) :
        D=self.M.N.D_kx+self.M.N.D_ibsx
        sig0 = self.sigx0 
        
        plot_rmax = kwargs.get('plot_rmax',self.plot_rmax)
        flag_fill = kwargs.get('flag_fill',self.flag_fill)
        plot_step  = kwargs.get('plot_step', self.plot_step)
        time_scale= kwargs.get('time_scale',self.time_scale)
        tlabel    = kwargs.get('tlabel',self.tlabel)
        bool_theory_g0=kwargs.get('bool_theory_g0',self.M.gx==0)
        
        if iCoord ==0: # Action
            return self.plot_psis_1D(x=self.G.Jxc,psis=self.psis1Dx,ts=self.tsPSI,iCoord=iCoord,yscale=yscale,D=D,
                         sig0=sig0,plot_step=plot_step,bool_theory_g0=bool_theory_g0,xmax=plot_rmax,lw=lw,ls=ls, 
                         fignradd=fignradd, flag_fill=flag_fill,flag_extDist=flag_extDist,
                         time_scale=time_scale,tlabel=tlabel)  
        elif iCoord ==1: # Radial
            return self.plot_psis_1D(x=self.G.rxc,psis=self.psis1Dx,ts=self.tsPSI,iCoord=iCoord,yscale=yscale,D=D,
                         sig0=sig0,plot_step=plot_step,bool_theory_g0=bool_theory_g0,xmax=plot_rmax,lw=lw,ls=ls, 
                         fignradd=fignradd, flag_fill=flag_fill,flag_extDist=flag_extDist,
                         time_scale=time_scale,tlabel=tlabel)  
        elif iCoord ==2: # Radial
            if not self.flag_ProjX:
                print("Had not calculated projection in x and y position -> Doing it now")
                self.postProc_Distributions(flag_ProjX=True,xMax_ProjX=self.xMax_ProjX)
            return self.plot_psis_1D(x=self.cart_x,psis=self.psisPosX,ts=self.tsPSI,iCoord=iCoord,yscale=yscale,D=D,
                         sig0=sig0,plot_step=plot_step,bool_theory_g0=bool_theory_g0,xmax=plot_rmax,lw=lw,ls=ls, 
                         fignradd=fignradd, flag_fill=flag_fill,flag_extDist=flag_extDist,
                         time_scale=time_scale,tlabel=tlabel)
            
    def plot_psis_1Dy(self, iCoord,yscale = 'linear',lw=1, ls='-', fignradd=10,flag_extDist=0,**kwargs) :
        D=self.M.N.D_ky+self.M.N.D_ibsy
        bool_theory_g0=(self.M.gy==0)
        sig0 = self.sigy0 
        
        plot_rmax = kwargs.get('plot_rmax',self.plot_rmax)
        flag_fill = kwargs.get('flag_fill',self.flag_fill)
        plot_step  = kwargs.get('plot_step', self.plot_step)
        time_scale= kwargs.get('time_scale',self.time_scale)
        tlabel    = kwargs.get('tlabel',self.tlabel)
        
        if iCoord ==0: # Action
            return self.plot_psis_1D(x=self.G.Jyc,psis=self.psis1Dy,ts=self.tsPSI,iCoord=iCoord,yscale=yscale,D=D,
                         sig0=sig0,plot_step=plot_step,bool_theory_g0=bool_theory_g0,xmax=plot_rmax,lw=lw,ls=ls, 
                         fignradd=fignradd, flag_fill=flag_fill,flag_extDist=flag_extDist,
                         time_scale=time_scale,tlabel=tlabel)  
        elif iCoord ==1: # Radial
            return self.plot_psis_1D(x=self.G.ryc,psis=self.psis1Dy,ts=self.tsPSI,iCoord=iCoord,yscale=yscale,D=D,
                         sig0=sig0,plot_step=plot_step,bool_theory_g0=bool_theory_g0,xmax=plot_rmax,lw=lw,ls=ls, 
                         fignradd=fignradd, flag_fill=flag_fill,flag_extDist=flag_extDist,
                         time_scale=time_scale,tlabel=tlabel)
        elif iCoord ==2: # Radial
            if not self.flag_ProjX:
                print("Had not calculated projection in x and y position -> Doing it now")
                self.postProc_Distributions(flag_ProjX=True,xMax_ProjX=self.xMax_ProjX)
            return self.plot_psis_1D(x=self.cart_y,psis=self.psisPosY,ts=self.tsPSI,iCoord=iCoord,yscale=yscale,D=D,
                         sig0=sig0,plot_step=plot_step,bool_theory_g0=bool_theory_g0,xmax=plot_rmax,lw=lw,ls=ls, 
                         fignradd=fignradd, flag_fill=flag_fill,flag_extDist=flag_extDist,
                         time_scale=time_scale,tlabel=tlabel)
    
    def plot_dPsis_1Dx(self,order =1 ,flag_JdPsidJ=0,**kwargs):
        yscale = kwargs.get('yscale',['log','linear'][order-1])
        
        time_scale= kwargs.get('time_scale',self.time_scale)
        tlabel    = kwargs.get('tlabel',self.tlabel)
        
        if order ==1:    
            fig,ax = self.plot_1D(self.G.Jxbx[1:-1],-[1,self.G.Jxbx[1:-1]][flag_JdPsidJ]*self.dPsis1DdJx,self.tsPSI,
                             time_scale=time_scale,tlabel=tlabel,
                             xlabel=r'$J$',xlim=[0,.5*self.plot_rmax**2],
                             ylabel=r'$-%s\partial_J(\Psi)$'%['','J'][flag_JdPsidJ],yscale=yscale)
        elif order==2:
            f = -np.diff([1,self.G.Jxbx[1:-1]][flag_JdPsidJ]*self.dPsis1DdJx,axis=1)/np.diff(self.G.Jxbx[1:-1])
            fig,ax = self.plot_1D(self.G.Jxc[1:-1],f,self.tsPSI,
                             time_scale=self.time_scale,tlabel=self.tlabel,
                             xlabel=r'$J$',xlim=[0,.5*self.plot_rmax**2],
                             ylabel=r'$-\partial_J(%s\partial_J(\Psi))$'%['','J'][flag_JdPsidJ],yscale=yscale)
        else:
            print("ERROR plot_dPsi_1Dx: Only implemented derivative of order 1 and 2")
        
        return fig,ax
    
    def plot_moments(self,plane=0,fmt='-o'):
        moments = [self.moments_x,self.moments_y][plane]
        fig,ax= plt.subplots()
        for i in range(self.n_moments):
            ax.plot(self.tsPSI/self.time_scale,moments[:,i],fmt,
                    label=['N',r'$\langle\, J_%s^%d\rangle$'%(['x','y'][plane],i)][i>0])
        ax.set_xlabel(self.tlabel)
        ax.legend(loc=0)
        fig.tight_layout()
        return fig,ax
    
    def plot_intensity(self,fmt='-o'):
        fig,ax= plt.subplots()
        ax.plot(self.tsPSI/self.time_scale,(self.moments_x[:,0]-self.moments_x[0,0])/self.moments_x[0,0],fmt)
        ax.set_ylabel(r'$\Delta N/N_0$')
        ax.set_xlabel(self.tlabel)
        fig.tight_layout()
        return fig,ax

    #######################
    ##### 2D plotting #####

    def plot_2D(self,X,Y,Z,iStyle = 1,xMax=15,yMax=15,vlmin=-1,vlmax=1,nDec=1,
                xlabel=r'$J_x$',ylabel=r'$J_y$',clabel='CLABEL',flag_cbar=1,**kwargs):
        """
            iStyle: 0: pcolormesh (gridded if extra row, col in X,Y)
                    1: contourf with nlevels number of levels
        kwargs include:
            nlevels: number of levels used by contourf
            ax:      axis to plot along
        """
        if 'ax' in kwargs:
            ax = kwargs.get('ax')
            ax.autoscale()  # Necessary to update the ylims with the new data!
            ax.set_aspect('equal')
            fig=plt.gcf()
        else:
            fig = plt.figure(figsize=(self.figwidth,self.figwidth*.8))
            ax=fig.add_subplot(111,aspect='equal')

        # Create norm
        norm = colors.Normalize(vmin=vlmin,vmax=vlmax,clip=False)
        cticks=np.linspace(vlmin,vlmax,5)
        if nDec>0:
            posTicks = np.logspace(np.log10(vlmax),np.log10(vlmax)+nDec,1+nDec)
            cticks = np.concatenate((-posTicks,[0],posTicks))
            if vlmin<0:
                norm = colors.SymLogNorm(linthresh=vlmax, linscale=2,vmin=vlmin*10**nDec,vmax=vlmax*10**nDec,clip=False)
            else:
                norm=colors.SymLogNorm(vmax=vlmax*10**nDec,vmin=0,linthresh=vlmax,linscale=2,clip=False)

        lut = [kwargs.get('nlevels',None) , None][iStyle]
        cmap = cm.get_cmap(None ,lut=lut)
        cmap.set_under('w')
        cmap.set_over('gray')


        if iStyle==0:
    #         im=ax.pcolormesh(X,Y,Z,shading='flat',vmin=-vmax,vmax=vmax)
            if np.size(Z)==np.size(X)*np.size(Y):
                im=ax.pcolormesh(X,Y,Z,shading='gouraud',norm=norm,cmap=cmap)
            else:
                im=ax.pcolormesh(X,Y,Z,shading='flat',norm=norm,cmap=cmap)


        elif iStyle==1:
            nlevels = kwargs.get('nlevels',30) 
            levels = norm.inverse(np.linspace(0,1,nlevels))
            im=ax.contourf(X,Y,Z,extent=[0,xMax,0,yMax],levels=levels,norm=norm,extend='both',cmap=cmap)
            
            for c in im.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)

        # Colorbar
        cb= None
        if flag_cbar:
            cb=fig.colorbar(mappable=im,label=clabel,ticks=cticks,extend='both',
                            fraction=0.04, pad=0.02, aspect=22)

        # Fix layout
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        #Fix ticks
        tick_sep = int(1+xMax/5)
        ax.axis([0,xMax,0,yMax])
        ax.xaxis.set_major_locator(ax.yaxis.get_major_locator())


        plt.tight_layout(pad=1.08)  


    #     ax.set_title(iStyle)
        return fig,ax,(im,cticks,clabel,cb)



    def plot_Dist2D(self,flag_change=0,iTime=0,iStyle=1,interp=1,**kwargs):
        """
        kwargs include:   
            Local:       xMax,yMax,vlmin,vlmax,nDec
            Transferred: nlevels,ax
        """
        xMax = kwargs.pop('xMax',self.plot_rmax**2 /2)
        yMax = kwargs.pop('yMax',xMax)
        if flag_change :
            Z = (self.psis2D[iTime]-self.psis2D[0])/self.psis2D[0]
    #         Z = (self.psis2D[iTime]-self.psis2D[iTime-1])/self.psis2D[iTime-1]
            vlmax = .1 ; vlmin = -vlmax
            vlmax = kwargs.pop('vlmax',vlmax)
            maxZ = max(1e-12,np.max(np.abs(Z)))
            nDec = kwargs.pop('nDec',min(3,int(np.log10(maxZ/vlmax))))
            clabel=r'$\Delta\Psi(t=%.1f\mathrm{%s}) / \Psi_0$'%(self.tsPSI[iTime]/self.time_scale,self.tunit)
        else:
            Z = self.psis2D[iTime]  
            nDec = kwargs.pop('nDec',4)
            vlmin=0 ; vlmax = 1/10**nDec
            clabel = r'$\Psi(t=%.1f\mathrm{%s})$'%(self.tsPSI[iTime]/self.time_scale,self.tunit)

        # kwargs
        vlmin = kwargs.pop('vlmin',vlmin)
        vlmax = kwargs.pop('vlmax',vlmax)
        nDec  = kwargs.pop('nDec',nDec)

        if interp :#and (not i in [2]):
            x2=np.linspace(0,xMax,100)
            y2=np.linspace(0,yMax,100)
            X2,Y2 =np.meshgrid(x2,y2)
            Zfunc= self.postProc_interpDistNri(iTime,interp_order=1,dist=Z)
            Z = Zfunc.getValue(X2,Y2)
            
            out=self.plot_2D(x2,y2,Z,iStyle=iStyle,clabel=clabel,
                             xMax=xMax,yMax=yMax,vlmin=vlmin,vlmax=vlmax,nDec=nDec,**kwargs)

        else:
            X,Y = self.G.Jxbx,self.G.Jyby
            if iStyle==1:
                X,Y = self.G.Jxc,self.G.Jyc

            out=self.plot_2D(X,Y,Z,iStyle=iStyle,clabel=clabel,
                             xMax=xMax,yMax=xMax,vlmin=vlmin,vlmax=vlmax,nDec=nDec,**kwargs)
        return out

    def plot_Coeff2D(self,iCoeffType=0,plane=0,iStyle=1,**kwargs):
        """
            iCoeffType: 0: Diffusion ; 1:Drift
        kwargs: 
            Local:       xMax,yMax,vlmin,vlmax,nDec
            Transferred: nlevels,ax
        """
        xMax = kwargs.pop('xMax',self.plot_rmax**2 /2)
        yMax = kwargs.pop('yMax',xMax)
        X = [self.G.Jxc2D,self.G.Jxn2D][iStyle]
        Y = [self.G.Jyc2D,self.G.Jyn2D][iStyle]
        if iCoeffType==0:
            Z ,err = DiffCoeffJxJy(self.M,1,X,Y,plane,self.iCoeff) #* .6
            vlmin=0
            maxZ = np.max(np.abs(Z))
            vlmax=max(1e-12,np.round(maxZ,decimals=1-int(np.log10(1e-12+maxZ))))
            clabel=r'$D_{%s}$'%(['x','y'][plane])
        else:
            Z ,err = DriftCoeffJxJy(self.M,1,X,Y,plane,self.iCoeff)
            vlmax=max(1e-12,np.max(np.abs(Z)))
            vlmin=-vlmax
            clabel=r'$U_{%s}$'%(['x','y'][plane])

        # kwargs
        vlmin = kwargs.pop('vlmin',vlmin)
        vlmax = kwargs.pop('vlmax',vlmax)
        nDec  = kwargs.pop('nDec',0)

        return self.plot_2D(self.G.Jxn2D,self.G.Jyn2D,Z,iStyle=iStyle,clabel=clabel,
                            xMax=xMax,yMax=yMax,vlmin=vlmin,vlmax=vlmax,nDec=nDec,**kwargs)
                            
    def plot_Coeff1D(self,iCoeffType=0,plane=0,indexY=0,**kwargs):
        """
            iCoeffType: 0: Diffusion ; 1:Drift
        kwargs: 
            Local:       xMax,yscale
        """
        xMax = kwargs.pop('xMax',self.plot_rmax**2 /2)
        yscale=kwargs.pop('yscale','linear')
        X = self.G.Jxbx2D[indexY,:]
        Y = self.G.Jybx2D[indexY,:]
        print('J_y=%.2f'%Y[0])
        if iCoeffType==0:
            Z ,err = DiffCoeffJxJy(self.M,1,X,Y,plane,self.iCoeff) #* .6
            vlmin=0
            maxZ = np.max(np.abs(Z))
            vlmax=max(1e-12,np.round(maxZ,decimals=1-int(np.log10(1e-12+maxZ))))
            clabel=r'$D_{%s}$'%(['x','y'][plane])
        else:
            Z ,err = DriftCoeffJxJy(self.M,1,X,Y,plane,self.iCoeff)
            vlmax=max(1e-12,np.max(np.abs(Z)))
            vlmin=-vlmax
            clabel=r'$U_{%s}$'%(['x','y'][plane])

        # kwargs
        vlmin = kwargs.pop('vlmin',vlmin)
        vlmax = kwargs.pop('vlmax',vlmax)
        nDec  = kwargs.pop('nDec',0)

        fig=plt.figure()
        ax=fig.gca()
        ax.plot(X,Z)
        ax.set_yscale(yscale)
        ax.grid(1)
        plt.xlabel(r'$J_x$')
        plt.ylabel([r'$D_{xx}$',r'$U_{x}$'][iCoeffType])
        fig.tight_layout()

        return fig,ax

