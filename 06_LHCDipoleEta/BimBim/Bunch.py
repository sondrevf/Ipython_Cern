import numpy as np


class Bunch:
    _number = -1;
    _energy = 0;
    _gamma = 0;
    _beta = 0;
    _emittance = 0;
    _sigs = 0;
    _sigp = 0;
    _intensity = 0;
    _particleMass = 0;
    _particleCharge = 0;
    _r0 = 1.535e-18;
    _spos = 0;
    
    def __init__(self,number,energy,intensity,emittance,sigs,sigp,spos,mass=0.93827231,charge=1):
        self._number = number;
        self._energy = energy;
        self._particleMass = mass;
        self._gamma = 1.0 + self._energy/self._particleMass;
        self._beta = np.sqrt(1.0 - 1.0/self._gamma**2)
        self._particleCharge = charge;
        self._emittance = emittance;
        self._intensity = intensity;
        #TODO _r0 = ...
        self._sigs = sigs;
        self._sigp = sigp;
        self._spos = spos;
        
    def getNumber(self):
        return self._number;
        
    def getEmittance(self):
        return self._emittance;
        
    def getIntensity(self):
        return self._intensity;
        
    def setIntensity(self,intensity):
        self._intensity = intensity;
        
    def getGamma(self):
        return self._gamma;
    
    def getBeta(self):
        return self._beta;

    def getR0(self):
        return self._r0;
        
    def getSPosition(self):
        return self._spos;
        
    def getSigS(self):
        return self._sigs;

    def getSigP(self):
        return self._sigp;

    def getMass(self):
        return self._particleMass;
