import numpy as np
import sys, time
sys.path.append('../')
import matplotlib.pyplot as plt


# PySSD
from PySSD.Distribution import Distribution,Gaussian
from PySSD.Detuning import LinearDetuning
from PySSD.Integrator import Integrator



class empty():
    # Similar as the one I use in PyRADISE
    pass

def calc_CapitalOmega(distribution,detuning,integrator_epsilon,Q0,modeDQ,tol=1e-4,flagFindalpha=[1,1],debug=0):    
    absModeDQ = np.abs(modeDQ) # Absolute value of undamped mode - used to find accuracy of mode
    
    # Additional integrators
    integrator1 = Integrator(distribution, detuning, maxJ=18,epsilon=integrator_epsilon*1)
    integrator2 = Integrator(distribution, detuning, maxJ=18,epsilon=integrator_epsilon*2)
    integrator4 = Integrator(distribution, detuning, maxJ=18,epsilon=integrator_epsilon*4)
    
    # Estimate of damped mode corresponding to free mode
    dampDQ = modeDQ.real + 1j*integrator_epsilon
    
    #debug
    if debug:
        dampDQs=[dampDQ]
        modeDQs=[]
    
    cnt = 0
    err = 0
    while True and absModeDQ>0:
        if dampDQ.imag<=integrator_epsilon*1:
            # Taylor approach to getting stabilized Omega
            tempDQ = 2*integrator1.integrate(Q0+dampDQ.real) - 1*integrator2.integrate(Q0+dampDQ.real)+1j*(dampDQ.imag)
            flag_taylor=True
        else:
            # Original approach to getting unstable Omega
            integrator1._detuning += 1j*(integrator_epsilon - dampDQ.imag)
            tempDQ=integrator1.integrate(Q0+dampDQ.real)
            integrator1._detuning -= 1j*(integrator_epsilon - dampDQ.imag)
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
            dDampDQ  =-(err*.5+errold*.05) 
        else:
            # Newton's method
            dDampdMode = (dDampDQ)/(err-errold)
            dDampDQ    = -err*dDampdMode 
        dampDQ = dampDQ + dDampDQ
        
        #debug
        if debug or cnt>40 :
            print("%2d:dampDQ=%11.4e + %11.4ei | err=%10.2e + %10.2ei - relerr=%.1e"%(
                cnt,dampDQ.real,dampDQ.imag,err.real,err.imag,np.abs(err)/absModeDQ))
            if debug:
                modeDQs+=[tempDQ]
                dampDQs+=[dampDQ]
        
            
        # Break if tried 50 times 
        if cnt>50:
            break
        
        cnt+=1

    # Calc alpha
    if np.any(np.abs(flagFindalpha)>0) and flag_taylor:
        # Calculate alpha 
        alpha =  2j*integrator_epsilon/(
                    integrator4.integrate(Q0+dampDQ.real)-integrator2.integrate(Q0+dampDQ.real))
        # Use the parts of alpha as given by flagFindalpha
        alpha = alpha.real*flagFindalpha[0] + 1j*alpha.imag*flagFindalpha[1]
        dampDQold = dampDQ
        dampDQ = dampDQ.real + 1j*alpha*dampDQ.imag
        print('Found alpha!=1, alpha=%.2e %s%.2ej |  dampDQ = %.2e %s%.3ej -> %.2e %s%.3ej'%(
                    alpha.real,['+','-'][np.sign(alpha.imag)<0],np.abs(alpha.imag),
                    dampDQold.real,['+','-'][np.sign(dampDQold.imag)<0],np.abs(dampDQold.imag),
                    dampDQ.real,['+','-'][np.sign(dampDQ.imag)<0],np.abs(dampDQ.imag)))
        
        
    if debug :
        modeDQs = np.array(modeDQs)
        dampDQs = np.array(dampDQs[:-1])
        print('modeDQ',modeDQs)
        print('dampDQ',dampDQs[1:])
    return dampDQ , cnt, np.abs(err)/absModeDQ


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

if __name__ =="__main__":
    f_rev =   1# 11.2455e3
    sigma_k = 1# 1e-4
    Qs = 0.002
    
    ax = 1.2e-4
    bx = 0
    Qx = 0.31
    wmode__DQ = 0+1e-4j
    wmodeQ0 = Qx
    wmodeDipm= 1
    integrator_epsilon = 1e-6 * ax/1e-4 *[1,4][bx==0]
    flagFindalpha=[1,1]
    detuning = LinearDetuning(Qx,ax,bx)
    distribution = Gaussian()
    tol = 1e-4
    debug = 1

    wmodeLdDQ,cnt,relerr=calc_CapitalOmega(distribution,detuning,integrator_epsilon,
                                        Qx,wmode__DQ,tol=tol,flagFindalpha=flagFindalpha,debug=debug)
    print('\nAfter %d steps: dampDQ = %.2e %s%.3ej | relerr=%.2e\n'%(cnt,
                                                   wmodeLdDQ.real,['+','-'][np.sign(wmodeLdDQ.imag)<0],np.abs(wmodeLdDQ.imag),
                                                   relerr))
                                                   
    J = np.linspace(0,10,200)
    incoQ = Qx*0 + detuning(J,0)   
    D_ibs = 0 
    D_k = f_rev*sigma_k**2/2       
    D_wake,err = DiffCoeff3(J,D_ibs,D_k,[1],[0],Qs,incoQ,[wmodeQ0],[wmode__DQ],[wmodeDipm] ,[wmodeLdDQ])
    
    plt.figure()
    plt.plot(J,D_wake/J,label='Wake diffusion')
    plt.plot(J,D_k*np.ones_like(J),label='Direct diffusion')
    plt.plot(J,D_k+D_wake/J,label='Total diffusion')
    plt.legend(loc=0)
    plt.xlabel('J')
    plt.ylabel('D')
    plt.show()
    
    
