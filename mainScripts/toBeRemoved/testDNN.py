import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
    
    def getTestDataDir(self):
        return self.jsonData['testDataTopDir']
    
    def getSavedModelFile(self):
        return self.jsonData['savedModelFilePath']
    
    def getSavedWeightsFile(self):
        return self.jsonData['savedWeightsFilePath']
    
    def getTestFiles(self):
        return self.jsonData['testFiles']

def calculateMetrics(predictedValues, actualValues):
    predictedPositive = predictedNegative = 0
    actualPositive = actualNegative = 0
    truePositives = falsePositives = trueNegatives = falseNegatives = 0
    numTotal = predictedValues.shape[0]
    threshold = 0.5
    for i in range(numTotal):
        if (abs(actualValues[i] - 1.0) < 0.1):
            actualPositive += 1
        elif (actualValues[i] < 0.1):
            actualNegative += 1
        if (abs(predictedValues[i] - 1.0) < threshold):
            predictedPositive += 1
            if (abs(predictedValues[i] - actualValues[i]) < threshold):
                truePositives += 1
            else:
                falsePositives += 1
        elif (predictedValues[i] < threshold):
            predictedNegative += 1
            if (abs(predictedValues[i] - actualValues[i]) < threshold):
                trueNegatives += 1
            else:
                falseNegatives += 1
        else:
            print ("predictedValues[{}] = {}".format(i, predictedValues[i]))

    if ((actualPositive + actualNegative) != numTotal):
        print ("Something is wrong!! numbers do not match")

    # don't bother to print anything if there are no positive values in the actual dataset
    if (actualPositive <= 0):
        print ("Nothing to report:  This testcase did not have any actual positives")
        return

    print ("truePositives = {}, falsePositives = {}, trueNegatives = {}, falseNegatives = {}"
            .format(truePositives, falsePositives, trueNegatives, falseNegatives))
    print ("actualPositive = {}, actualNegative = {}".format(actualPositive, actualNegative))
    print ("Precision = ", float((truePositives + trueNegatives) / (actualPositive + actualNegative)))
    if (actualPositive > 0):
        print ("Recall = ", float(truePositives / actualPositive))
    else:
        print ("Recall = N.A. (actualPositive == 0)")

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
    print ("modeFile = {}, weightsFile = {}".format(modelFile, weightsFile))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    testFiles = cfgReader.getTestFiles()
    testDataTopDir = cfgReader.getTestDataDir()

    dnnModel = DNN.DNN("Classifier_3layers")
    dnnModel.loadModel(modelFile, weightsFile)
    print (dnnModel.getModel().summary())
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LL_PreIctal/chb04.csv'
    for testFile in testFiles:
        testFilePath = os.path.join(testDataTopDir, testFile)
        print ("testFilePath = ", testFilePath)
        dnnModel.prepareDataset_1file(testFilePath)
        # dnnModel.evaluate()
        dataset = np.loadtxt(testFilePath, delimiter=',')
        numRows = dataset.shape[0]
        numFeatures = dataset.shape[1] - 1

        predictedSeizureValues = np.empty((numRows))
        for i in range(numRows):
            predictedSeizureValues[i] = dnnModel.getModel().predict(np.expand_dims(dataset[i, :numFeatures], axis=0))

        calculateMetrics(predictedSeizureValues, dataset[:, numFeatures])
        # score = loaded_model.evaluate(X, y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))