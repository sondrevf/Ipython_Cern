import os

class HTCondorJobLauncher:
    def __init__(self,execFile,arguments,outputDir,studyName='HTC'):
        if not os.path.exists(outputDir):
            raise(Exception(outputDir+' does not exist'))
        self._fileName = os.path.join(outputDir,studyName+'.job')
        myFile = open(self._fileName,'w')
        myFile.write(
                'executable            ='+ execFile+'\n'
                +'arguments             ='+ arguments+'\n'
                +'output                ='+ os.path.join(outputDir,studyName+'.out')+'\n'
                +'error                 ='+ os.path.join(outputDir,studyName+'.err')+'\n'
                +'log                   ='+ os.path.join(outputDir,studyName+'.log')+'\n'
                +'+MaxRunTime           = 600000\n'
                +'queue')
        myFile.close()



    def launch(self):
        os.system('condor_submit '+self._fileName)
