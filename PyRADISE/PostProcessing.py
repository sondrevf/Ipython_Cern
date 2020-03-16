import numpy as np
import scipy as sc
import time
import multiprocessing as mp

from PyRADISE.Distribution import MyDistribution1D,MyDistribution1Dy,MyDistribution2D,MyDistribution2Dy

class PostProcessingClass(object):
    #####################
    ## Postprocessing: ##
    def J2D_proj_J1D(self,Jx1D,Jy1D,f,plane=0):
        if plane ==0:
            # Projection in x
            dy = np.diff(Jy1D)[:,np.newaxis]
            if self.ND==1:
                return f
            else:
                return np.sum(f*dy,axis=plane)
        elif plane ==1: 
            # Projection in y
            dx = np.diff(Jx1D)
            return np.sum(f*dx,axis=plane)
        else: 
            print("ERROR in J2D_proj_J1D:  plane must be 0 (horizontal) or 1 (vertical)")

    def J1D_proj_x1D(self,x,J,psi1D):
        # works well except if use .5*rc^2 for Nx=59, when bool_radial=0
        #always ok for X = Jc (Nx<Nxc)
        func = sc.interpolate.interp1d(J,psi1D,kind='linear',bounds_error=False,fill_value='extrapolate')#(0,0))
        psiX = np.zeros_like(x)
        def integrand(J,x):
            if x**2<2*J:
                return func(J)/(np.pi*np.sqrt(J*2-x**2))
            else:
                return 0
        for j,jx in enumerate(x):
            #psiX[j] += sc.integrate.quad(integrand,0,18,args=(jx,),limit=200,epsrel=1e-4,points=[.5*jx**2])[0]
            psiX[j] += sc.integrate.quad(integrand,.5*jx**2,18,args=(jx,),limit=200,epsrel=1e-4,points=[.5*jx**2])[0]
        return psiX

    
    
    def _worker_psix(self,i):
        return self.J1D_proj_x1D(self.cart_x,self.G.Jxc,self.psis1Dx[i])
    def _worker_psiy(self,i):
        return self.J1D_proj_x1D(self.cart_y,self.G.Jyc,self.psis1Dy[i])
    
    def postProc_Distributions(self,flag_ProjX=0, xMax_ProjX = 3):
        start = time.time()

        nrow = self.G.nrow
        ncol = self.G.ncol
        nt   = self.ntPSI
        
        ## 2D distribution
        self.psis2D = np.zeros((nt,nrow,ncol))
        self.dPsis2DdJx = np.zeros((nt,(nrow),(ncol-1)))
        self.dPsis2DdJy = np.zeros((nt,(nrow-1),(ncol)))
        ## 2D -> 1D projection ##
        self.psis1Dx = np.zeros((nt,ncol))
        self.psis1Dy = np.zeros((nt,nrow))    
        self.dPsis1DdJx = np.zeros((nt,ncol-1))
        self.dPsis1DdJy = np.zeros((nt,nrow-1))
        # 1D radial -> 1D position projection 
        Nx=min(51,self.G.Ncx) ; 
        Ny=min(51,self.G.Ncy) ; 
        self.psisPosX = np.zeros((nt,Nx)) ; 
        self.psisPosY = np.zeros((nt,Ny)) ; 
        self.cart_x = np.linspace(-xMax_ProjX,xMax_ProjX,Nx)
        self.cart_y = np.linspace(-xMax_ProjX,xMax_ProjX,Ny)
        
        # Helper arrays
        centersepx = np.diff(self.G.Jxc)
        centersepy = np.diff(self.G.Jyc)

        # Fill the arrays
        for i in range(nt):
            self.psis2D[i]  = self.psis[i].reshape((nrow,ncol))
            self.psis1Dx[i] = self.J2D_proj_J1D(self.G.Jxbx,self.G.Jyby,self.psis2D[i],0)
            self.psis1Dy[i] = self.J2D_proj_J1D(self.G.Jxbx,self.G.Jyby,self.psis2D[i],1)

            
            self.dPsis2DdJx[i] = np.diff(self.psis2D[i],axis=1) / centersepx
            self.dPsis1DdJx[i] = np.diff(self.psis1Dx[i])       / centersepx
            if self.ND==2:
                self.dPsis2DdJy[i] = np.diff(self.psis2D[i],axis=0) / centersepy[:,np.newaxis]
                self.dPsis1DdJy[i] = np.diff(self.psis1Dy[i])       / centersepy



        # Calculate projection on x and y axis in parallel
        self.flag_ProjX=flag_ProjX
        self.xMax_ProjX=xMax_ProjX
        if flag_ProjX:
            nProcs = min(nt,mp.cpu_count()-2)
            pool = mp.Pool(processes = nProcs)
            self.psisPosX = np.array(pool.map(self._worker_psix,range(nt)))
            if self.ND ==2:
                self.psisPosY = np.array(pool.map(self._worker_psiy,range(nt)))
            pool.close()
            pool.join()

        print('Postprocessing(Distributions) took %.2fs'%(time.time()-start))
        
    def postProc_Moments(self,n_moments=3):
        # store moments
        self.n_moments = n_moments
        self.moments_x = np.zeros((self.ntPSI,n_moments))
        self.moments_y = np.zeros((self.ntPSI,n_moments))
        
        dx = np.diff(self.G.Jxbx)
        dy = np.diff(self.G.Jyby)
        X = self.G.Jxbx
        Y = self.G.Jyby
            
        for i in range(self.ntPSI):

            self.moments_x[i,:]     = [np.sum(self.psis1Dx[i,:]*(X[1:]**(j+1) - X[:-1]**(j+1)))/(j+1) for j in range(n_moments)]
            self.moments_x[i,1:] = self.moments_x[i,1:]/self.moments_x[i,0]
            if self.ND>1:
                self.moments_y[i,:] = [np.sum(self.psis1Dy[i,:]*(Y[1:]**(j+1) - Y[:-1]**(j+1)))/(j+1) for j in range(n_moments)] 
                self.moments_y[i,1:] = self.moments_y[i,1:]/self.moments_y[i,0]
                
    def postProc_interpDistNri(self,i,interp_order=1,plane=0,**kwargs):
        if i>=self.ntPSI:
            print("ERROR (postProc_interpDistNri): Only %d distributions, cannot interpolate nr %d"%(self.ntPSI,i))
        if self.ND==1:
            dist = kwargs.get('dist',self.psis1Dx[i])
            if plane ==0: Func = MyDistribution1D
            else:         Func = MyDistribution1Dy
            self.interpDistNri = Func(self.G.Jxbx,self.G.Jxc,dist,self.dPsis1DdJx[i],
                                      self.G.bool_radial,interp_order)
        elif self.ND==2:
            dist = kwargs.get('dist',self.psis2D[i])
            if plane ==0: Func = MyDistribution2D
            else:         Func = MyDistribution2Dy
            self.interpDistNri = Func(self.G.Jxbx,self.G.Jxc,self.G.Jyby,self.G.Jyc,
                                      dist,self.dPsis2DdJx[i],self.dPsis2DdJy[i],
                                      self.G.bool_radial,interp_order)
        return self.interpDistNri
