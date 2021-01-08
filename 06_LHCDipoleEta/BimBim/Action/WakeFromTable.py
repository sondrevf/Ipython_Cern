
def binarySearch(array,value,upper,lower=0,nStep=0):
    pivot = int((lower+upper)/2)
    if (upper-lower)<=1:
        #print nStep
        return lower
    if value < array[pivot]:
        upper = pivot
        nStep+=1
        return binarySearch(array,value,upper,lower,nStep)
    elif value > array[pivot]:
        lower = pivot
        nStep+=1
        return binarySearch(array,value,upper,lower,nStep)
    else:
        #print nStep
        return pivot

class WakeFromTable:

    def __init__(self,wakeFileName):
        self._wakeUnitConversion = 1E15
        self.parseWakeTable(wakeFileName)

    def parseWakeTable(self,filename):
        wakefile = open(filename,'r')
        lines = wakefile.readlines()
        row = len(lines)
        col = len(lines[0].split())
        self._wakeTableT = [0.0 for i in range(row)]
        self._wakeTable = [[0.0 for i in range(col-1)] for j in range(row)]
        for i in range(row):
            thisline = lines[i].split()
            self._wakeTableT[i] = float(thisline[0])
            for j in range(col-1):
                self._wakeTable[i][j]=float(thisline[j+1])*self._wakeUnitConversion
        wakefile.close()

    #Generate map of indexes in impedance table
    #def getIndexMap(self,basis):
    #    indmat = [[0 for j in range(basis.getNSlice()*basis.getNRing())] for i in range(basis.getNSlice()*basis.getNRing())]
    #    for i in range(basis.getNSlice()*basis.getNRing()):
    #        for j in range(basis.getNSlice()*basis.getNRing()):
    #            if distmap[i][j]>0.0:
    #                index = getTabindex(distmap[i][j])
    #            else:
    #                index = 0
    #            indmat[i][j]=index
    #    return indmat

    #get the wake for a given position
    #def getWakeTab(pos,ipos):
    #    dipx = self._wakeTable[ipos-1][1]+(self._wakeTable[ipos][1]-self._wakeTable[ipos-1][1])*(abs(pos)-self._wakeTable[ipos-1][0])/(self._wakeTable[ipos][0]-self._wakeTable[ipos-1][0])
    #    dipy = self._wakeTable[ipos-1][2]+(self._wakeTable[ipos][2]-self._wakeTable[ipos-1][2])*(abs(pos)-self._wakeTable[ipos-1][0])/(self._wakeTable[ipos][0]-self._wakeTable[ipos-1][0])
    #    quadx = self._wakeTable[ipos-1][3]+(self._wakeTable[ipos][3]-self._wakeTable[ipos-1][3])*(abs(pos)-self._wakeTable[ipos-1][0])/(self._wakeTable[ipos][0]-self._wakeTable[ipos-1][0])
    #    quady = self._wakeTable[ipos-1][4]+(self._wakeTable[ipos][4]-self._wakeTable[ipos-1][4])*(abs(pos)-self._wakeTable[ipos-1][0])/(self._wakeTable[ipos][0]-self._wakeTable[ipos-1][0])
    #    return dipx,dipy,quadx,quady

    def interpolateWake(self,distance,wakeIndex,index):
        return self._wakeTable[index][wakeIndex] + (distance-self._wakeTableT[index])*(self._wakeTable[index+1][wakeIndex]-self._wakeTable[index][wakeIndex])/(self._wakeTableT[index+1]-self._wakeTableT[index])

    # distance in ns
    def getWake(self,distance):
        dipx = 0.0
        dipy = 0.0
        quadx = 0.0
        quady = 0.0
        i = binarySearch(self._wakeTableT,distance,len(self._wakeTableT))
        dipx = self.interpolateWake(distance,0,i)
        dipy = self.interpolateWake(distance,1,i)
        if len(self._wakeTable[0]) > 2:
            quadx = self.interpolateWake(distance,2,i)
            quady = self.interpolateWake(distance,3,i)
        return dipx,dipy,quadx,quady
