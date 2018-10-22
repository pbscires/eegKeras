import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from Models import eegLSTM
import sys
import os
import json


class TestConfigReader(object):
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
    
    def getTestDataDir(self):
        return self.jsonData['testDataTopDir']
    
    def getSavedModelFile(self):
        return self.jsonData['savedModelFilePath']
    
    def getSavedWeightsFile(self):
        return self.jsonData['savedWeightsFilePath']
    
    def getTestFiles(self):
        return self.jsonData['testFiles']

    def getSeqLens(self):
        inSeqLen, outSeqLen = int(self.jsonData['inSeqLen']), int(self.jsonData['outSeqLen'])
        return (inSeqLen, outSeqLen)

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Error! Invalid number of arguments")
        exit (-1)
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    cfgReader = TestConfigReader(configFile)


    # Load the saved model and try to evaluate on new data
    # load json and create model
    modelFile = cfgReader.getSavedModelFile()
    weightsFile = cfgReader.getSavedWeightsFile()
    inSeqLen, outSeqLen = cfgReader.getSeqLens()

    # evaluate loaded model on test data
    testFiles = cfgReader.getTestFiles()
    testDataTopDir = cfgReader.getTestDataDir()

    print ("modelFile = {}, weightsFile = {}".format(modelFile, weightsFile))
    print ("inSeqLen = {}, outSeqLen = {}".format(inSeqLen, outSeqLen))

    lstmObj = eegLSTM.eegLSTM("encoder_decoder_sequence")
    numFeatures = 168
    lstmObj.loadModel(modelFile, weightsFile, inSeqLen, outSeqLen, numFeatures)
    print("Loaded model from disk")
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LL_PreIctal/chb04.csv'
    for testFile in testFiles:
        testFilePath = os.path.join(testDataTopDir, testFile)
        print ("testFilePath = ", testFilePath)
        lstmObj.prepareDataset_1file(testFilePath)
        # dataset = np.loadtxt(testFile, delimiter=',')
        # X = dataset[:,:19]
        # y = dataset[:,19]
        lstmObj.evaluate()

        # score = loaded_model.evaluate(X, y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))