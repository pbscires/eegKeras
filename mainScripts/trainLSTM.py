import sys
from Models import StackedLSTM
import json
import os

class TrainingConfigReader(object):
    '''
    Contains methods to read the *config.json files
    '''
    def __init__(self, configFile):
        '''
        Read the given config file into a json dictionary
        '''
        self.configFile = configFile
        f = open(configFile, 'r')
        self.jsonData = json.load(f)
        f.close()
    
    def getTrainingDataDir(self):
        return self.jsonData['trainingDataTopDir']
    
    def getModelOutputDir(self):
        return self.jsonData['modelOutputDir']
    
    def getSavedModelPrefix(self):
        return self.jsonData['savedModelFilePrefix']
    
    def getTrainingFiles(self):
        return self.jsonData['trainingFiles']

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Error! Invalid number of arguments")
        exit (-1)
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    cfgReader = TrainingConfigReader(configFile)
    trainingDataTopDir = cfgReader.getTrainingDataDir()
    trainingFiles = cfgReader.getTrainingFiles()
    modelOutputDir = cfgReader.getModelOutputDir()
    savedModelFilePrefix = cfgReader.getSavedModelPrefix()
    print ("trainingDataTopDir = ", trainingDataTopDir)
    print ("trainingFiles = ", trainingFiles)
    print ("modelOutputDir = ", modelOutputDir)
    print ("savedModelFilePrefix = ", savedModelFilePrefix)

    numFeatures = 19
    stackedLSTM = StackedLSTM.StackedLSTM("encoder_decoder_sequence", 30, 10, numFeatures)

    if (len(trainingFiles) == 1):
        trainingFile = trainingFiles[0]
        print ("trainingFile = ", trainingFile)
        stackedLSTM.prepareDataset_1file(os.path.join(trainingDataTopDir, trainingFile))
        stackedLSTM.fit()
        stackedLSTM.save(modelOutputDir, savedModelFilePrefix)