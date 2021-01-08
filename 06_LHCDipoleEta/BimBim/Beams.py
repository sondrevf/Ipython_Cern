from BimBim.Bunch import Bunch

class Beams:
    _bunchConfigB1 = [];
    _bunchConfigB2 = [];
    _nBunchB1 = 0;
    _nBunchB2 = 0;
    _singleBeam = False;
    _bunchSpacing = 0;
    
    def _parseStringRep(self,rep,energy = 0,emittance=0,sigs=0,sigp=0,intensity=0):
        tokens = rep.split(' ');
        if len(tokens)%2 > 0:
            print("ERROR parsing filling scheme",rep);
            return [];
        retVal = [];
        bunchNumber = 0;
        spos = 0.0;
        for i in range(0,len(tokens),2):
             if tokens[i+1] == '0':
                bunch = False;
             else:
                bunch = True;
             for j in range(int(tokens[i])):
                if bunch:
                    retVal.append(Bunch(bunchNumber,energy,intensity,emittance,sigs,sigp,spos));
                    retVal.append(None);
                    bunchNumber += 1;
                else:
                    retVal.append(None);
                    retVal.append(None);
                spos += self._bunchSpacing;
        return retVal,bunchNumber;
    
    def __init__(self,stringRepB1,stringRepB2=None,energy=4000.0,emittance = 0,sigs=0,sigp=0,intensity = 0,bunchSpacing = 0):
        self._bunchSpacing = bunchSpacing;
        self._bunchConfigB1,self._nBunchB1 = self._parseStringRep(stringRepB1,energy = energy,emittance=emittance,sigs=sigs,sigp=sigp,intensity=intensity);
        
        #reverse beam 1
        for i in range(1,int(len(self._bunchConfigB1)/2)):
            tmp = self._bunchConfigB1[i];
            self._bunchConfigB1[i] = self._bunchConfigB1[-i];
            self._bunchConfigB1[-i] = tmp;

        
        #print 'B1 : ',[self._bunchConfigB1[k].getNumber() if self._bunchConfigB1[k] != None else 'None' for k in range(len(self._bunchConfigB1))];
        if stringRepB2 != None:
            self._bunchConfigB2,self._nBunchB2 = self._parseStringRep(stringRepB2,energy = energy,emittance=emittance,sigs=sigs,sigp=sigp,intensity=intensity);
            if len(self._bunchConfigB1) > len(self._bunchConfigB2):
                while len(self._bunchConfigB1) > len(self._bunchConfigB2):
                    self._bunchConfigB2.append(None);
            else:
                while len(self._bunchConfigB2) > len(self._bunchConfigB1):
                    self._bunchConfigB1.append(None);
            #print 'B2 : ',[self._bunchConfigB2[k].getNumber() if self._bunchConfigB2[k] != None else 'None' for k in range(len(self._bunchConfigB2))];
        else:
            self._singleBeam = True;
    
    def getBunchConfigB1(self):
        return self._bunchConfigB1;
            
    def getBunchConfigB2(self):
        return self._bunchConfigB2;

    def getBunchConfig(self,beam=1):
        if beam==1:
            return self._bunchConfigB1;
        else:
            return self._bunchConfigB2;

    def getBunchB1(self,pos):
        return self._bunchConfigB1[pos];

    def getBunchB2(self,pos):
        return self._bunchConfigB2[pos];

    def getBunch(self,pos,beam=1):
        if beam==1:
            return self._bunchConfigB1[pos];
        else:
            return self._bunchConfigB2[pos];

    def getNBunchB1(self):
        return self._nBunchB1;
            
    def getNBunchB2(self):
        return self._nBunchB2;

    def getNBunch(self,beam=1):
        if beam==1:
            return self._nBunchB1;
        else:
            return self._nBunchB2;

    def isSingleBeam(self):
        return self._singleBeam;
        
    def getNBeam(self):
        if self.isSingleBeam():
            return 1;
        else:
            return 2;
        
    def step(self):
        self._bunchConfigB1.insert(0,self._bunchConfigB1[-1]);
        self._bunchConfigB1.pop();
        if not self.isSingleBeam():
            self._bunchConfigB2.append(self._bunchConfigB2[0]);
            self._bunchConfigB2.remove(self._bunchConfigB2[0]);
            
