
class NoiseClass(object):
    def __init__(self,f_rev, sigma_kx,dQxAvg,sigma_ky,dQyAvg,alpha,
                             sigma_ibsx,sigma_ibsy,
                             A_hx,dQ_hx,A_hy,dQ_hy):
        # White noise (Lebedev)
        self.sigma_kx = sigma_kx
        self.sigma_ky = sigma_ky
        self.dQxAvg = dQxAvg
        self.dQyAvg = dQyAvg
        self.alpha  = alpha
        self.D_kx = f_rev*self.sigma_kx**2/2
        self.D_ky = f_rev*self.sigma_ky**2/2
        # IBS, incoherent noise
        self.sigma_ibsx = sigma_ibsx
        self.sigma_ibsy = sigma_ibsy
        self.D_ibsx = f_rev*sigma_ibsx**2/2
        self.D_ibsy = f_rev*sigma_ibsy**2/2
        # Harmonic excitation
        self.A_hx  = A_hx
        self.dQ_hx = dQ_hx
        self.A_hy  = A_hy
        self.dQ_hy = dQ_hy
    
