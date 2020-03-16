import numpy as np
import scipy as sc 
#from LebedevHelper import *
from PySSD.Integrator      import Integrator #FixedTrapezoidalIntegrator, SimpsonIntegrator, TrapzIntegrator, DblquadIntegrator

#########################
### Physics functions ###
#########################
pi2 = 2*np.pi

def calc_LandauDampedOneMode(self,integrator1,integrator2,integrator4,Q0,modeDQ,relstep=[.5,.05],tol=1e-4,flagFindalpha=[1,1],debug=0):
    absModeDQ = np.abs(modeDQ) # Absolute value of undamped mode - used to find accuracy of mode
    
    # Estimate of damped mode corresponding to free mode
    dampDQ = modeDQ.real + 1j*self.integrator_epsilon
    
    #debug
    if debug:
        dampDQs=[dampDQ]
        modeDQs=[]
    
    cnt=0
    err = 0
    while True and absModeDQ>0:
        if dampDQ.imag<=self.integrator_epsilon*1:
#            tempDQ = integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon)+ \
#                        1j*(dampDQ.imag-self.integrator_epsilon)
#            tempDQ = 2*integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon)  - \
#                       integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon*2)+ \
#                        1j*(dampDQ.imag)
            tempDQ = 2*integrator1.integrate(Q0+dampDQ.real) - 1*integrator2.integrate(Q0+dampDQ.real)+1j*(dampDQ.imag)
            flag_taylor=True
        else:
            #tempDQ=integrator.integrate(Q0+dampDQ.real,epsilon=dampDQ.imag)
            integrator1._detuning += 1j*(self.integrator_epsilon - dampDQ.imag)
            tempDQ=integrator1.integrate(Q0+dampDQ.real)
            integrator1._detuning -= 1j*(self.integrator_epsilon - dampDQ.imag)
            flag_taylor=False
        
        # Calculate error
        errold= err
        err   = tempDQ-modeDQ

        # Break if within tolerance
        if np.abs(err)<absModeDQ*tol:
            break

        ## Update dampDQ (damped mode tune)
        if cnt==0:
            # Simple method
            dDampDQ  =-(err*relstep[0]+errold*relstep[1]) 
        else:
            # Newton's method
            dDampdMode = (dDampDQ)/(err-errold)
            dDampDQ    = -err*dDampdMode #* 0.8**(cnt//10)
        dampDQ = dampDQ + dDampDQ
        
        #debug
        if debug or cnt>40 :
            print("%2d: %11.4e + %11.4ei | %10.2e + %10.2ei - relerr=%.1e"%(
                cnt,dampDQ.real,dampDQ.imag,err.real,err.imag,np.abs(err)/absModeDQ))
            if debug:
                modeDQs+=[tempDQ]
                dampDQs+=[dampDQ]
        
            
        # Break if tried 50 times 
        if cnt>50:
            break
        
        cnt+=1
        if cnt%10==0:
            relstep=[relstep[0]*0.8,relstep[1]*0.8]
            if debug:
                print("Reduced relstep to [%.2f,%.2f]"%(relstep[0],relstep[1]))

    # Calc alpha
    if np.any(np.abs(flagFindalpha)>0) and flag_taylor:
#        alpha = 2j*self.integrator_epsilon/(
#                        integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon*4)-
#                        integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon*2))
        alpha =  2j*self.integrator_epsilon/(
                    integrator4.integrate(Q0+dampDQ.real)-integrator2.integrate(Q0+dampDQ.real))
        alpha = alpha.real*flagFindalpha[0] + 1j*alpha.imag*flagFindalpha[1]
        if flagFindalpha[0]==-1:
            alpha = -1/alpha.real + 1j*alpha.imag
        dampDQold = dampDQ
        dampDQ = dampDQ.real + 1j*alpha*dampDQ.imag
        print('Found alpha!=1, alpha=%.2e %s%.2ej |  dampDQ = %.2e %s%.3ej -> %.2e %s%.3ej'%(
                    alpha.real,['+','-'][np.sign(alpha.imag)<0],np.abs(alpha.imag),
                    dampDQold.real,['+','-'][np.sign(dampDQold.imag)<0],np.abs(dampDQold.imag),
                    dampDQ.real,['+','-'][np.sign(dampDQ.imag)<0],np.abs(dampDQ.imag)))
        
                    
    if debug :
        modeDQs = np.array(modeDQs)
        dampDQs = np.array(dampDQs[:-1])
        print(modeDQs,dampDQs)
    return dampDQ , cnt, np.abs(err)/absModeDQ

def calc_LandauDampedAllModes(self,plane=0,relstep=[.5,.05],tol=1e-4,flagFindalpha=[1,1],debug=0,flagUpdateReQ=1):
    if plane==0:
        wmode__DQ = self.M.wmode__DQx
#        wmodeMgDQ = self.M.wmodeMgDQx
        wmodeLdDQ = self.M.wmodeLdDQx
        wmodeQ0   = self.M.wmodeQ0x
#        wmodeQ0   = self.M.Q.Q0x
        Q0   = self.M.Q.Q0x
        distribution=self.interpDistx
        detuning = self.M.Q 
    else:
        wmode__DQ = self.M.wmode__DQy
#        wmodeMgDQ = self.M.wmodeMgDQy
        wmodeLdDQ = self.M.wmodeLdDQy
        wmodeQ0   = self.M.wmodeQ0y
#        wmodeQ0   = self.M.Q.Q0y
        Q0   = self.M.Q.Q0y
        distribution=self.interpDisty
        detuning = self.M.Qy
        
#    integrator = TrapzIntegrator(distribution, detuning, maxJ=18)
    integrator1 = Integrator(distribution, detuning, maxJ=18,epsilon=self.integrator_epsilon*1)
    integrator2 = Integrator(distribution, detuning, maxJ=18,epsilon=self.integrator_epsilon*2)
    integrator4 = Integrator(distribution, detuning, maxJ=18,epsilon=self.integrator_epsilon*4)
    if np.size(relstep)==1:
        relstep = [relstep,0]
    # Find damped mode
    for i, modeDQ in enumerate(wmode__DQ):
        modeQ0 = wmodeQ0[i]   # irrelevant for this calculation?
        
        # Calc dampDQ0 without alpha
        dampDQ ,cnt, relerr = calc_LandauDampedOneMode(self,integrator1,integrator2,integrator4,Q0,modeDQ,
                                                       relstep,tol=tol,flagFindalpha=flagFindalpha,debug=debug)   
        if relerr>tol:
            print('OBS Consider cancelling due to relerr=%.1e>%.1e'%(relerr, tol))
        
#        # Calc alpha
#        if np.sum(flagFindalpha)>0:
#            alpha = 1j*self.integrator_epsilon/(
#                            integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon*2)-
#                            integrator.integrate(Q0+dampDQ.real,epsilon=self.integrator_epsilon))
#            alpha = alpha.real*flagFindalpha[0] + 1j*alpha.imag*flagFindalpha[1]
#            dampDQold = dampDQ
#            dampDQ = dampDQ.real + alpha*1j*dampDQ.imag
#            print('Found alpha!=1, alpha=%.2e+%.2ej |  dampDQ = %.2e+%.2ej -> %.2e+%.2ej (ext)'%(alpha.real,alpha.imag,
#                        dampDQold.real,dampDQold.imag,dampDQ.real,dampDQ.imag))
                        
        # Set dampDQ 
        if not(flagUpdateReQ or np.abs(wmodeLdDQ[i])==0):
            dampDQ = wmodeLdDQ[i].real + 1j*dampDQ.imag
            
        print('calc_LandauDampedAllModes: Mode: %.2e %s%.2ei -> %.2e %s%.3ei (relerr(%d iterations)=%.1e)'%(
                modeDQ.real,['+','-'][np.sign(modeDQ.imag)<0],np.abs(modeDQ.imag),dampDQ.real,
                ['+','-'][np.sign(dampDQ.imag)<0],np.abs(dampDQ.imag),cnt,relerr))
        wmodeLdDQ[i] = dampDQ
            
    if plane==0:
        self.M.wmodeLdDQx = wmodeLdDQ
    else:
        self.M.wmodeLdDQy = wmodeLdDQ
    return

def find_DiffusionSidebandWeight(sigma_x,maxOrder=50,tol=1e-3,debug=0):
    """
        Function to find the weight of the sideband 
        Taken as average bessel function squared over the longitudinal gaussian distribution
    """    
    orders = np.arange(maxOrder+1)
    factors = np.zeros(maxOrder+1)
    if sigma_x>0:
        xs = np.linspace(0,6,50)*sigma_x
        phi_x = xs/sigma_x**2 * np.exp(-xs**2/sigma_x**2*.5)

        for m in orders:
            ys   = sc.special.jv(m,xs)
            mean = sc.integrate.simps(ys**2*phi_x,x=xs,even='first')
            factors[m]=mean
            if factors[m]/factors[0]< tol:
                factors = factors[:m+1]
                orders = orders[:m+1]
                if debug:
                    print('Need no more than %d sidebands: '%(m+1),factors)
                break
    else:
        factors = np.array([1])
        orders  = np.array([0])
    return factors,orders

###########################
## Lebedev physics
def LL(g2,dmu):
    return (1-g2)**2*(dmu)**2       / (g2**2 + (1-g2)*(dmu)**2)

def dLLdm(g2,dmu):
    return (1-g2)**2*g2**2*2*(dmu)            / (g2**2 + (1-g2)*(dmu)**2)**2





#########################
### Coefficients      ###
#########################


## Diffusion coefficient
def DiffCoeff1(X,D_ibs,D_k):
    # X is the main coordinate (r, J)
    err=0
    return X*(D_ibs) , err
    
def DiffCoeff2(X,D_ibs,D_k,dQ,dQAvg,g):
    # X is the main coordinate (r, J)
    err=0
    g2 = g/2
    dmu = 2*np.pi*(dQ-dQAvg)
    return X*(D_ibs + D_k * LL(g2,dmu)) , err
    
def DiffCoeff3(X,D_ibs,D_k,factors,orders,Qs,incoQ,wmodeQ0,wmode__DQ,wmodeDipm,wmodeLdDQ):
    DD = np.ones_like(incoQ) * D_ibs
    err=0
    for i, modeDQ in enumerate(wmode__DQ):
#         if i!=0: continue
        dampDQ = wmodeLdDQ[i]
        if dampDQ.imag>0:
            print('Instability - stop! growthrate=%.1e'%dampDQ.imag)
            DD *=0
            err =1
            break
            
        dipm   = wmodeDipm[i]
        modeQ0 = wmodeQ0[i]        
        newQ2  =  (modeQ0+dampDQ.real)**2 - (dampDQ.imag)**2
        newQIR =  (modeQ0+dampDQ.real)*dampDQ.imag
        absModeDQ2 = np.abs(modeDQ)**2
                
        correction = (modeQ0*(modeQ0+modeDQ.real) + absModeDQ2/4)/(modeQ0+dampDQ.real)**2
        # Add diffusion coefficient from different sidebands
        for m in orders:
            for sign in [-1,1][m==0:]:
                incoQm = incoQ + sign*m*Qs
                B = 1/(1 + (newQ2 - (incoQm)**2)**2/(4*(newQIR)**2))
                DD += factors[m] * X * D_k * dipm**2 * absModeDQ2/(dampDQ.imag)**2 * B * correction
                
        if np.abs(correction-1)>1e-3:
            print('Correction of mode %i: 1+(%.1e)'%(i,correction-1))
#        DD += X * D_k * dipm**2 * absModeDQ2/(dampDQ.imag)**2 * B * correction
#        print('Mode: %.2e + %.2ei -> %.2e + %.2ei '%(modeDQ.real,modeDQ.imag,dampDQ.real,dampDQ.imag))
#        print(r'|modedQ|^2=%.2e , newQ2=%.2e , Q0modeQR=%.2e , dampQR2=%.2e , |modedQ|/dampDQ.imag=%.2e'%(
#            absModeDQ2,newQ2,Q0*(Q0+modeDQ.real),(Q0+dampDQ.real)**2,absModeDQ2**.5/dampDQ.imag))
    return DD ,err
    
def DiffCoeffJxJy(Mach,Xp,JX,JY,plane,iCoeff):
    if iCoeff ==1:
        if plane==0:
            DD,err = DiffCoeff1(Xp,Mach.N.D_ibsx,Mach.N.D_kx)
        elif plane==1:
            DD,err = DiffCoeff1(Xp,Mach.N.D_ibsy,Mach.N.D_ky)
    elif iCoeff==2:
        if plane==0:
            DD,err = DiffCoeff2(Xp, Mach.N.D_ibsx, Mach.N.D_kx,
                            Mach.Q.dQx(JX,JY), Mach.N.dQxAvg, Mach.gx)
        elif plane==1:
            DD,err = DiffCoeff2(Xp, Mach.N.D_ibsy, Mach.N.D_ky,
                            Mach.Q.dQy(JX,JY), Mach.N.dQyAvg, Mach.gy)
    elif iCoeff==3:
        if plane==0:
            DD,err = DiffCoeff3(Xp, Mach.N.D_ibsx, Mach.N.D_kx,Mach.factorsSBx,Mach.ordersSBx,Mach.Qs,
                            Mach.Q.Q0x+Mach.Q.dQx(JX,JY), Mach.wmodeQ0x, Mach.wmode__DQx,Mach.wmodeDipmx,Mach.wmodeLdDQx)
        elif plane==1:
            DD,err = DiffCoeff3(Xp, Mach.N.D_ibsy, Mach.N.D_ky,Mach.factorsSBy,Mach.ordersSBy,Mach.Qs,
                            Mach.Q.Q0y+Mach.Q.dQy(JX,JY), Mach.wmodeQ0x, Mach.wmode__DQy,Mach.wmodeDipmy,Mach.wmodeLdDQy)
    elif iCoeff==4:
        ans2 = DiffCoeffJxJy(Mach,Xp,JX,JY,plane,2)
        ans3 = DiffCoeffJxJy(Mach,Xp,JX,JY,plane,3)
        ans1 = DiffCoeffJxJy(Mach,Xp,JX,JY,plane,1)
        DD  = ans2[0]+ans3[0]-ans1[0]
        err = ans2[1]+ans3[1]+ans1[1]
    else: 
        print('ERROR (DiffCoeff): iCoeff in [1,2] only')
    return DD,err

def DiffCoeffGrid(Mach,Grid,iCoeff):
    # Noise: Mach.N
    # Detuning: Mach.Q
    DDx,errx = DiffCoeffJxJy(Mach,Grid.X,Grid.Jxbx2D,Grid.Jybx2D,plane=0,iCoeff=iCoeff)
    DDy,erry = DiffCoeffJxJy(Mach,Grid.Y,Grid.Jxby2D,Grid.Jyby2D,plane=1,iCoeff=iCoeff)
    return DDx, DDy, errx+erry
   
   
   

## Drift coefficient
def DriftCoeff2(Jb,D_k,alpha,dQ,dQAvg,dQdJ,g):
    err=0
    g2 = g/2
    dmu = 2*np.pi*(dQ-dQAvg)        
    return -Jb*((1-alpha)*D_k * dLLdm(g2,dmu) * pi2*dQdJ) , err
    
def DriftCoeffJxJy(Mach,JP,JX,JY,plane,iCoeff):
    err=0
    if iCoeff in [1,3,4]:
        U = np.zeros_like(JX)
    elif iCoeff==2:
        if plane==0:
            U,err = DriftCoeff2(JP,Mach.N.D_kx,Mach.N.alpha,
                            Mach.Q.dQx(JX,JY),Mach.N.dQxAvg,Mach.Q.dQxdJx(JX,JY),Mach.gx)
        elif plane==1:
            U,err = DriftCoeff2(JP,Mach.N.D_ky,Mach.N.alpha,
                            Mach.Q.dQy(JX,JY),Mach.N.dQxAvg,Mach.Q.dQydJy(JX,JY),Mach.gy)
    else: 
        print('ERROR (DriftCoeff): iCoeff in [1,2] only')
    return U , err
def DriftCoeffGrid(Mach,Grid,iCoeff):
    # Noise: Mach.N
    # Detuning: Mach.Q  
    Ux,errx = DriftCoeffJxJy(Mach,Grid.Jxbx2D,Grid.Jxbx2D,Grid.Jybx2D,plane=0,iCoeff=iCoeff)
    Uy,erry = DriftCoeffJxJy(Mach,Grid.Jyby2D,Grid.Jxby2D,Grid.Jyby2D,plane=1,iCoeff=iCoeff)
    return Ux,Uy,errx+erry

