import numpy as np
import scipy.sparse as sp


class GridClass(object):
    def __init__(self, Ncx,JxMax,BC_xMax,BC_xVal,bool_radial,
                       Ncy,JyMax,BC_yMax,BC_yVal):
        self.Ncx = Ncx  ;  self.Nbx = Ncx+1
        self.Ncy = Ncy  ;  self.Nby = Ncy+1
        self.NcTot= self.Ncx * self.Ncy
        
        self.ncol = Ncx
        self.nrow = Ncy
        
#        self.JxAvg = JxAvg  ;  self.rxAvg = np.sqrt(2*JxAvg)
        self.JxMax = JxMax  ;  self.rxMax = np.sqrt(2*JxMax)
#        self.JyAvg = JyAvg  ;  self.ryAvg = np.sqrt(2*JyAvg)
        self.JyMax = JyMax  ;  self.ryMax = np.sqrt(2*JyMax)
        
        self.bool_radial = (bool_radial>0)
        if bool_radial:
            self.hx = self.rxMax/Ncx
            self.rxbx = np.linspace(0,self.rxMax,self.Nbx)
            self.Jxbx= .5*self.rxbx**2
            self.hy = self.ryMax/Ncy
            self.ryby = np.linspace(0,self.ryMax,self.Nby)
            self.Jyby = .5*self.ryby**2
            
            #Temp
#            self.rxc = (self.rxbx[1:]+self.rxbx[:-1])*.5
#            self.ryc = (self.ryby[1:]+self.ryby[:-1])*.5
#            self.Jxc = .5*self.rxc**2
#            self.Jyc = .5*self.ryc**2
        else:
            self.hx = JxMax/Ncx
            self.Jxbx= np.linspace(0,self.JxMax,self.Nbx)
            self.rxbx = np.sqrt(2*self.Jxbx)
            self.hy = JyMax/Ncy
            self.Jyby = np.linspace(0,self.JyMax,self.Nby)
            self.ryby = np.sqrt(2*self.Jyby)
                
            #Temp
#            self.Jxc = (self.Jxbx[1:]+self.Jxbx[:-1])*.5
#            self.Jyc = (self.Jyby[1:]+self.Jyby[:-1])*.5
#            self.rxc = np.sqrt(self.Jxc*2)
#            self.ryc = np.sqrt(self.Jyc*2)

        self.rxc = (self.rxbx[1:]+self.rxbx[:-1])*.5
        self.ryc = (self.ryby[1:]+self.ryby[:-1])*.5
        self.Jxc = (self.Jxbx[1:]+self.Jxbx[:-1])*.5
        self.Jyc = (self.Jyby[1:]+self.Jyby[:-1])*.5
        
       
        # Get meshed grids of boundary face points
        self.Jxbx2D , self.Jybx2D = np.meshgrid(self.Jxbx,self.Jyc)
        self.Jxby2D , self.Jyby2D = np.meshgrid(self.Jxc,self.Jyby)
        self.rxbx2D , self.rybx2D = np.meshgrid(self.rxbx,self.ryc)
        self.rxby2D , self.ryby2D = np.meshgrid(self.rxc,self.ryby)
        self.X = [self.Jxbx2D,self.rxbx2D][bool_radial]  
        self.Y = [self.Jyby2D,self.ryby2D][bool_radial]
        
        # Get meshed grids of center-of-cell points
        self.Jxc2D, self.Jyc2D = np.meshgrid(self.Jxc,self.Jyc)
        self.Jxc1D = self.Jxc2D.flatten()  ;  self.Jyc1D = self.Jyc2D.flatten()
        self.rxc2D, self.ryc2D = np.meshgrid(self.rxc,self.ryc)
        self.rxc1D = self.rxc2D.flatten()  ;  self.ryc1D = self.ryc2D.flatten()
        
        # Get meshed grids of corner/node points
        self.Jxn2D, self.Jyn2D = np.meshgrid(self.Jxbx,self.Jyby)
        self.rxn2D, self.ryn2D = np.meshgrid(self.rxbx,self.ryby)       
        
        
        # Create slices
        self.sliceN = [slice(1,None,1),slice(0,None,1)]
        self.sliceS = [slice(0,-1,1),slice(0,None,1)]
        self.sliceE = [slice(0,None,1),slice(1,None,1)]
        self.sliceW = [slice(0,None,1),slice(0,-1,1)]
        
        # Boundary conditions
        self.BC_xMax = BC_xMax
        self.BC_xVal = BC_xVal
        self.BC_yMax = BC_yMax
        self.BC_yVal = BC_yVal
        
        
        
    def get_rx(self):
        return self.rxbx,self.rxc
    def get_Jx(self):
        return self.Jxbx,self.Jxc
    def get_Mavgx(self):
        return self.MavgW,self.MavgE
    def get_Mslpx(self):
        return self.MslpW,self.MslpE
    def get_ry(self):
        return self.ryby,self.ryc
    def get_Jy(self):
        return self.Jyby,self.Jyc
    def get_Mavgy(self):
        return self.MavgS,self.MavgN
    def get_Mslpy(self):
        return self.MslpS,self.MslpN



class Grid1DClass(GridClass):
    def __init__(self, Ncx, JxMax,BC_xMax,BC_xVal,bool_radial,bool_adjustGrid,JxAvg):
        if bool_adjustGrid:
            # Recaclculate the grid points
            if bool_radial:     
                rxAvg = np.sqrt(2*JxAvg)
                hx = np.sqrt(2*JxMax)/Ncx
                hx = (rxAvg)/np.floor(rxAvg/hx)   # adjust so that boundary is at rxAvg
                rxMax0= np.sqrt(2*JxMax)
                print('Change of h from %.3f to %.3f - rMax from %.3f to %.3f'%(
                        rxMax0/Ncx,hx,rxMax0,hx*Ncx))
                rxMax = hx*Ncx
                JxMax = .5*rxMax**2
            else: # bool_radial == False
                hx = JxMax/Ncx
                hx = (JxAvg)/np.floor(JxAvg/hx)   # adjust so that boundary is at JxAvg
                JxMax = hx*Ncx
                print('Change of h from %.3f to %.3f - JxMax from %.3f to %.3f'%(
                        JxMax/Ncx,hx,JxMax,hx*Ncx))
                
        super().__init__(Ncx, JxMax,BC_xMax,BC_xVal,bool_radial,
                         1  , 0    ,0      ,0      )
        self.ND = 1 # 1D grid
                   
        ## Create Stencil Matrices with BC
        self._SpStencilMatrices()
        self._BC()
            
    def _SpStencilMatrices(self):
        print('stencil 1D')
        Nc = self.NcTot
        hx = self.hx
        
        self.MavgW = sp.diags([0.5,0.5] ,[0,-1],shape=(Nc,Nc),format='csr')/(hx)
        self.MavgE = sp.diags([0.5,0.5] ,[0,1 ],shape=(Nc,Nc),format='csr')/(hx)
        self.MslpW = sp.diags([-1,1]    ,[-1,0],shape=(Nc,Nc),format='csr')/(hx*hx)
        self.MslpE = sp.diags([-1,1]    ,[0,1 ],shape=(Nc,Nc),format='csr')/(hx*hx)
        if self.bool_radial:
            #Multiply each row with element: sparse.multiply(numpy[:,np.newaxis])
            #Multiply each col with element: sparse.multiply(numpy)   (faster than making new diag)
            rcinv = 1/self.rxc[:,np.newaxis]
            self.MavgW = self.MavgW.multiply(rcinv).tocsr()    #keep J = .5r^2 in U
            self.MavgE = self.MavgE.multiply(rcinv).tocsr() 
            self.MslpW = self.MslpW.multiply(rcinv/2).tocsr()
            self.MslpE = self.MslpE.multiply(rcinv/2).tocsr() 
            # /2 because D(J) ->D(r)/2
            
    
    def _BC(self):
        print('BC 1D')
        # Mandatory reflective boundary at r=0
        self.MslpW[0,0]=0
        self.MavgW[0,0]=0

        #Vector for nonhomogeneous BC
        self.boundary_E =np.zeros(self.NcTot)
        self.boundary_E[-1]=1 
        
        # BC at xmax
        if  self.BC_xMax==0:
            # Absorbing Dirichlet Psi=0
            None
#            self.MslpE[-1,-1]*=2
        elif self.BC_xMax==1:
            # Reflective boundary at r=rMax
            self.MslpE[-1,-1]=0
            self.MavgE[-1,-1]*=2
            #self.boundary_E *= self.hx # Already in calc_BC_V
        elif self.BC_xMax==2:
            # Robin condition, not yet perfect
            if self.bool_radial:
                self.MslpE[-1,-1]*=   self.hx*self.rxc[-1]
                self.MavgE[-1,-1]*=(2-self.hx*self.rxc[-1])
            else:
                self.MslpE[-1,-1]*=   self.hx
                self.MavgE[-1,-1]*=(2-self.hx)
        elif self.BC_xMax==3:
            # dJPsi_N = dJPsi_N+1
            self.MslpE[-1,-2]=self.MslpE[-1,-1]
            self.MslpE[-1,-1]*=(-1)
            self.MavgE[-1,-2]=-self.MavgE[-1,-1]
            self.MavgE[-1,-1]*=3
        else:
            print('OBS Not implemented BC_xMax > 3!')
    
class Grid2DClass(GridClass):
    def __init__(self, Ncx,  JxMax,  BC_xMax,  BC_xVal,bool_radial,
                       Ncy=1,JyMax=0,BC_yMax=0,BC_yVal=0):
        self.ND = 2
        if Ncy==1:
            # Assume always 1D means horizontal
            self.ND = 1
            JyMax = 0
        super().__init__(Ncx,JxMax,BC_xMax,BC_xVal,bool_radial,
                         Ncy,JyMax,BC_yMax,BC_yVal)
    
            
        # If 2D
        self._SpStencilMatrices()
        self._BC()
        
    def _SpStencilMatrices(self):
        print('stencil 2D')
        Nc = self.NcTot
        col= self.ncol
        hx = self.hx
        hy = self.hy
        
        one0 = np.ones(Nc)
        one1 = np.ones(Nc-1) ; one1[col-1::col]=0
        
        # Create stencil matrices
        self.MavgW = sp.diags([one0,one1] ,[0,-1],shape=(Nc,Nc),format='csr')/(2*hx)
        self.MavgE = sp.diags([one0,one1] ,[0,1 ],shape=(Nc,Nc),format='csr')/(2*hx)
        self.MslpW = sp.diags([-one1,one0],[-1,0],shape=(Nc,Nc),format='csr')/(hx*hx)
        self.MslpE = sp.diags([-one0,one1],[0,1 ],shape=(Nc,Nc),format='csr')/(hx*hx)
        
        if self.ND ==2:
            self.MavgS = sp.diags([1,1]       ,[0,-col],shape=(Nc,Nc),format='csr')/(2*hy)
            self.MavgN = sp.diags([1,1]       ,[0,col] ,shape=(Nc,Nc),format='csr')/(2*hy)
            self.MslpS = sp.diags([-1,1]      ,[-col,0],shape=(Nc,Nc),format='csr')/(hy*hy)
            self.MslpN = sp.diags([-1,1]      ,[0,col] ,shape=(Nc,Nc),format='csr')/(hy*hy)
        
        # Update if radial coordinates
        if self.bool_radial:
            rxcinv = 1/self.rxc1D[:,np.newaxis]
            self.MavgW = self.MavgW.multiply(rxcinv).tocsr()
            self.MavgE = self.MavgE.multiply(rxcinv).tocsr()
            self.MslpW = self.MslpW.multiply(rxcinv/2).tocsr()
            self.MslpE = self.MslpE.multiply(rxcinv/2).tocsr()
            
            if self.ND ==2:
                rycinv = 1/self.ryc1D[:,np.newaxis]
                self.MavgS = self.MavgS.multiply(rycinv).tocsr()
                self.MavgN = self.MavgN.multiply(rycinv).tocsr()
                self.MslpS = self.MslpS.multiply(rycinv/2).tocsr()
                self.MslpN = self.MslpN.multiply(rycinv/2).tocsr()
            
        
    def _BC(self):
        print('BC 2D')
        col = self.ncol
        row = self.nrow
        Nelm= self.NcTot
        elmBE = np.arange(col-1,Nelm,col)
        elmBW = np.arange(0,Nelm,col)
        elmBS = np.arange(0,col,1)
        elmBN = np.arange((row-1)*col,Nelm,1)
        
        # Mandatory reflective boundary at r=0
        self.MslpW[elmBW,elmBW]=0
        self.MavgW[elmBW,elmBW]=0
                   
        #Vector for nonhomogeneous BC
        self.boundary_E =np.zeros(self.NcTot)
        self.boundary_E[elmBE]=1
                
        # BC at xmax
        if  self.BC_xMax==0:
            # Absorbing Dirichlet Psi=0
            None
        elif self.BC_xMax==1:
            # Reflective boundary at r=rMax
            self.MslpE[elmBE,elmBE]=0
            self.MavgE[elmBE,elmBE]*=2
        elif self.BC_xMax>1:
            print("Havent implemented BC in Grid2DClass for iBC>1")
            
        # In Vertical:
        if self.ND ==2:
            # Mandatory reflective boundary at r=0
            self.MslpS[elmBS,elmBS]=0
            self.MavgS[elmBS,elmBS]=0
            
            #Vector for nonhomogeneous BC
            self.boundary_N =np.zeros(self.NcTot)
            self.boundary_N[elmBN]=1
            
            # BC at ymax
            if  self.BC_yMax==0:
                # Absorbing Dirichlet Psi=0
                None
            elif self.BC_yMax==1:
                # Reflective boundary at r=rMax
                self.MslpN[elmBN,elmBN]=0
                self.MavgN[elmBN,elmBN]*=2
            elif self.BC_yMax>1:
                print("Havent implemented BC in Grid2DClass for iBC>1")

        
