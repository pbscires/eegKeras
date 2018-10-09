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

    def getTimestepsToPredict(self):
        return int(self.jsonData['timesteps_to_predict'])

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
        # print ("Nothing to report:  This testcase did not have any actual positives")
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
    lstmModelFile = cfgReader.getSavedModelLSTM()
    lstmWeightsFile = cfgReader.getSavedWeightsLSTM()
    dnnModelFile = cfgReader.getSavedModelDNN()
    dnnWeightsFile = cfgReader.getSavedWeightsDNN()

    print ("lstmModel={}, lstmWeights={}, dnnModel={}, dnnWeights={}"
            .format(lstmModelFile, lstmWeightsFile, dnnModelFile, dnnWeightsFile))

    timeStepsToPredict = cfgReader.getTimestepsToPredict()

    # evaluate loaded model on test data
    testFiles = cfgReader.getTestFiles()
    testDataTopDir = cfgReader.getTestDataDir()

    lstmModel = StackedLSTM.StackedLSTM("encoder_decoder_sequence")
    inSeqLen = 30
    outSeqLen = 10
    lstmModel.loadModel(lstmModelFile, lstmWeightsFile, inSeqLen, outSeqLen)
    dnnModel = DNN.DNN("Classifier_3layers")
    dnnModel.loadModel(dnnModelFile, dnnWeightsFile)
    print ("LSTM model = ", lstmModel.getModel().summary())
    print ("DNN model = ", dnnModel.getModel().summary())
    print("Loaded model from disk")

    for testFile in testFiles:
        testFilePath = os.path.join(testDataTopDir, testFile)
        print ("testFilePath = ", testFilePath)

        # 1. Identify the number of time steps available in the test file.
        # 2. Read the first inSeqLen number of entries from the test file.
        # 3. Predict outSeqLen entries using LSTM
        # 4. Predict the seizure/non-seizure for outSeqLen entries using DNN
        # 5. If the prediction was accurate, move the inSequence pointer by outSeqLen
        #     i.e inSeq pointer += outSeqLen
        # 6. If (number of remaining time steps > inSeqLen + outSeqLen) in the test file,
        #       go back to step 2.

        dataset = np.loadtxt(testFilePath, delimiter=',')
        numFeatures = dataset.shape[1] - 1
        numRowsNeededForTest = inSeqLen + outSeqLen
        numRows = dataset.shape[0]
        while (numRows > numRowsNeededForTest):
            numRemainingRows = min (numRows, (inSeqLen+timeStepsToPredict))
            # print ("numRows={}, numFeatures={}, numRemainingRows={}"
            #         .format(numRows, numFeatures, numRemainingRows))

            predictedDataset = np.empty((1, (inSeqLen+timeStepsToPredict), numFeatures))
            predictedSeizureValues = np.empty((inSeqLen+timeStepsToPredict))
            inputRowStart = 0
            inputRowEnd = inputRowStart + inSeqLen
            outputRowStart = inputRowEnd
            outputRowEnd = outputRowStart + outSeqLen
            # print ("inputRowStart={}, inputRowEnd={}, outputRowStart={}, outputRowEnd={}"
            #         .format(inputRowStart, inputRowEnd, outputRowStart, outputRowEnd))
            predictedDataset[0, inputRowStart:inputRowEnd,:numFeatures] = dataset[inputRowStart:inputRowEnd, :numFeatures]
            predictedSeizureValues[inputRowStart:inputRowEnd] = dataset[inputRowStart:inputRowEnd, numFeatures]
            while (numRemainingRows >= numRowsNeededForTest):
                predictedDataset[:, outputRowStart:outputRowEnd, :] = \
                    lstmModel.getModel().predict(predictedDataset[:, inputRowStart:inputRowEnd, :])
                for i in range(outSeqLen):
                    predictedSeizureValues[outputRowStart+i] = dnnModel.getModel().predict(predictedDataset[:, outputRowStart+i, :])
                
                inputRowStart += outSeqLen
                inputRowEnd = inputRowStart + inSeqLen
                outputRowStart = inputRowEnd
                outputRowEnd = outputRowStart + outSeqLen
                numRemainingRows -= outSeqLen
                # print ("inputRowStart={}, inputRowEnd={}, outputRowStart={}, outputRowEnd={}"
                #         .format(inputRowStart, inputRowEnd, outputRowStart, outputRowEnd))

            calculateMetrics(predictedSeizureValues[inSeqLen:], dataset[inSeqLen:,numFeatures])
            dataset = np.delete(dataset, [0,1,2,3,4], axis=0)
            numRows = dataset.shape[0]
