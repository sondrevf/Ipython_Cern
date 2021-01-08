from BimBim.Error import BimBimError
import scipy.sparse as spm

class System:
    _beams = None;
    _actionSequence = None;
    _basis = None;
    _npos = 0;
    _returnSparse = False
    
    def __init__(self,beams,actionSequence,basis,returnSparse=False):
        self._beams = beams;
        self._actionSequence = actionSequence;
        self._basis = basis;
        self._npos = len(actionSequence);
        self._returnSparse = returnSparse
        self.checkValidity();
        
    def checkValidity(self):
        if len(self._actionSequence) != self._npos:
            raise BimBimError('System error : action sequence length should be equal to the number of position '+str(len(self._actionSequence))+', '+str(self._npos));
        if not self._basis.getNBeam() > 1:
            if len(self._beams.getBunchConfigB1()) != len(self._beams.getBunchConfigB2()):
                raise BimBimError('System error : Bunch configurations must have equal size ('+str(len(self._beams.getBunchConfigB1()))+', '+str(len(self._beams.getBunchConfigB1())));
        if len(self._beams.getBunchConfigB1()) != self._npos:
            raise BimBimError('System error : Bunch configuration length should be identical to action sequence length ('+str(len(self._beams.getBunchConfigB1()))+'/'+str(self._npos)+')');
        
    def getNStep(self):
        return len(self._actionSequence);
        
    def getNBunch(self):
        return self._nBunch;
        
    def getNBeam(self):
        return self._nBeam;
    
    def step(self):
        self._beams.step();
    
    def getAction(self,pos):
        return self._actionSequence[pos];
    
    def getMatrix(self):
        #print 'Computing step matrix for',nSlice,'slices and',nRing,'rings';
        stepMatrix = spm.identity(self._basis.getSize(),format="dok");
        for pos in range(self._npos):
            #print 'position,',pos,',bunch1',self._beams.getBunchesB1()[pos],',bunch2',self._beams.getBunchesB2()[pos];
            if self._actionSequence[pos] != None:
                if self._beams.getBunchB1(pos) != None or self._beams.getBunchB2(pos) != None:
                    #print(self._actionSequence[pos])
                    try:
                        posMatrix = self._actionSequence[pos].getMatrix(self._beams,pos,self._basis);
                    except AttributeError:
                        posMatrix = spm.identity(self._basis.getSize(),format="dok");
                        for action in self._actionSequence[pos]:
                            posMatrix = action.getMatrix(self._beams,pos,self._basis).dot(posMatrix)
                    stepMatrix = posMatrix.dot(stepMatrix);
        return stepMatrix;
        
    def buildOneTurnMap(self):
        oneTurn = spm.identity(self._basis.getSize(),format="coo");
        for step in range(self.getNStep()):
            #print(step)
            stepMatrix = self.getMatrix();
            oneTurn = stepMatrix.dot(oneTurn);
            self.step();
        if self._returnSparse:
            return oneTurn
        else:
            return oneTurn.todense();
