import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from Models import StackedLSTM
from Models import DNN
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
    
    def getSavedModelLSTM(self):
        return self.jsonData['savedModelLSTM']
    
    def getSavedWeightsLSTM(self):
        return self.jsonData['savedWeightsLSTM']

    def getSavedModelDNN(self):
        return self.jsonData['savedModelDNN']

    def getSavedWeightsDNN(self):
        return self.jsonData['savedWeightsDNN']
    
    def getTestDataDir(self):
        return self.jsonData['testDataTopDir']
    
    def getTestFiles(self):
        return self.jsonData['testFiles']

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Error! Invalid number of arguments")
        exit (-1)
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    cfgReader = TestConfigReader(configFile)


    # Load the saved model and try to evaluate on new data
    # load json and create model
    lstmModelFile = cfgReader.getSavedModelLSTM()
    lstmWeightsFile = cfgReader.getSavedWeightsLSTM()
    dnnModelFile = cfgReader.getSavedModelDNN()
    dnnWeightsFile = cfgReader.getSavedWeightsDNN()

    print ("lstmModel={}, lstmWeights={}, dnnModel={}, dnnWeights={}"
            .format(lstmModelFile, lstmWeightsFile, dnnModelFile, dnnWeightsFile))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    testFiles = cfgReader.getTestFiles()
    testDataTopDir = cfgReader.getTestDataDir()

    lstmModel = StackedLSTM.StackedLSTM("encoder_decoder_sequence")
    lstmModel.loadModel(lstmModelFile, lstmWeightsFile, 30, 10)
    dnnModel = DNN.DNN("Classifier_3layers")
    dnnModel.loadModel(dnnModelFile, dnnWeightsFile)
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LL_PreIctal/chb04.csv'
    for testFile in testFiles:
        testFilePath = os.path.join(testDataTopDir, testFile)
        print ("testFilePath = ", testFilePath)
        # lstmModel.prepareDataset_1file(testFilePath)
        # lstmModel.evaluate()


        # score = loaded_model.evaluate(X, y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))