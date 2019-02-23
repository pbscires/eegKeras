import sys
import os
import re
import json
from Models.eegDNN import eegDNN
from Models.eegLSTM import eegLSTM
from DataSets.TUHdataset import TUHdataset
from DataSets.CHBdataset import CHBdataset
import numpy as np
import pandas as pd


class ConfigReader(object):
    def __init__(self, configFile, modelName):
        '''
        Read the given config file into a json dictionary
        '''
        self.configFile = configFile
        f = open(configFile, 'r')
        self.jsonRoot = json.load(f)
        self.csvModels = self.jsonRoot['CSV_MODELS']
        self.edfModels = self.jsonRoot['EDF_MODELS']
        self.jsonData = self.jsonRoot[modelName]
        f.close()

    def getTestingDataDir(self):
        return self.jsonData['testingDataTopDir']
    
    def getSavedLSTMModelFile(self):
        # This method is used for HYBRID model testing
        return self.jsonData['savedLSTMModelFile']
    
    def getSavedLSTMWeightsFile(self):
        # This method is used for HYBRID model testing
        return self.jsonData['savedLSTMWeightsFile']
    
    def getTimeStepsToPredict(self):
        # This method is used for HYBRID model testing
        return self.jsonData['timesteps_to_predict']
    
    def getSavedDNNModelFile(self):
        # This method is used for HYBRID model testing
        return self.jsonData['savedDNNModelFile']
    
    def getSavedDNNWeightsFile(self):
        # This method is used for HYBRID model testing
        return self.jsonData['savedDNNWeightsFile']
    
    def getRecordInfoFile(self):
        return self.jsonData['recordInfoJsonFile']
    
    def getCSVsummaryJsonFile(self):
        return self.jsonData['csvSummaryJsonFile']
        
    def getTestingRecords(self):
        return self.jsonData['testingRecords']
    
    def getEpochSeconds(self):
        return self.jsonData['epochSeconds']
    
    def getSlidingWindowSeconds(self):
        return self.jsonData['SlidingWindowSeconds']
    
    def getSaveLSTMOutputToCSV(self):
        return self.jsonData['saveLSTMOutputToCSV']
    
    def getSavedLSTMOutputSuffix(self):
        return self.jsonData['savedLSTMOutputSuffix']

def getCSVfilesForRecords(allRecords, csvRecordInfo):
    allFiles = {}
    for recordID in allRecords:
        if (recordID in csvRecordInfo.keys()):
            allFiles[recordID] = csvRecordInfo[recordID]['CSVpath']
    return (allFiles)

def verifyAndGetNumFeaturesEDF(datasetObj, allRecords):
    # Verify that all the records have same features
    features = datasetObj.recordInfo[allRecords[0]]['channelLabels']
    featuresSet = set(features)
    for recordID in allRecords:
        tmpSet = set(datasetObj.recordInfo[recordID]['channelLabels'])
        xorSet = featuresSet.symmetric_difference(tmpSet)
        if (len(xorSet) > 0):
            print ("features are not common between", allRecords[0], "and", recordID)
            numFeatures = -1
            return (numFeatures)
    print ("features are common between all the records!")
    numFeatures = len(featuresSet)
    return (numFeatures)

def verifyAndGetNumFeaturesCSV(csvRecordInfo, allRecords):
    # Verify that all the records have same features
    n_features_1 = csvRecordInfo[allRecords[0]]['numFeatures']
    for recordID in allRecords:
        n_features = csvRecordInfo[recordID]['numFeatures']
        if (n_features != n_features_1):
            print ("features are not common between", allRecords[0], "and", recordID)
            numFeatures = -1
            return (numFeatures)
    print ("features are common between all the records!")
    numFeatures = n_features_1
    return (numFeatures)

def testWithLSTMAndSave(modelFile, weightsFile, numFeaturesInTestFiles, allFiles, lstmCsvSuffix):
    lstmObj = eegLSTM("encoder_decoder_sequence")
    lstmObj.loadModel(modelFile, weightsFile)
    print("Loaded LSTM model from disk")
    numFeatures = lstmObj.numFeatures
    inSeqLen = lstmObj.inSeqLen
    outSeqLen = lstmObj.outSeqLen
    if (numFeaturesInTestFiles != lstmObj.numFeatures):
        print ("number of features in testfiles ", numFeaturesInTestFiles, 
            "!= number of feature in loaded model ", lstmObj.numFeatures)
    for testFilePath in allFiles.values():
        print ("testFilePath = ", testFilePath)
        dataset = pd.read_csv(testFilePath)
        columnNames = dataset.columns.values.tolist()
        columnNames = columnNames[1:]
        print ("columnNames[{}] = {}".format(len(columnNames), columnNames))
        dataset = dataset.values # Convert to a numpy array from pandas dataframe
        # Remove the index column (unfortunately we have do do this explicitly)
        dataset = dataset[:,1:]
        # print ("dataset = ", dataset)
        numRows = dataset.shape[0]
        numLSTMruns = (numRows - inSeqLen) // outSeqLen
        print ("inSeqLen={}, outSeqLen={}, numFeatures={}, numRows={}, numLSTMruns={}".format(
            inSeqLen, outSeqLen, numFeatures, numRows, numLSTMruns))
        predictedDataset = np.empty(dataset.shape)
        predictedDataset[:inSeqLen,:] = dataset[:inSeqLen,:]
        inputRowStart = 0
        inputRowEnd = inputRowStart + inSeqLen
        for j in range(numLSTMruns):
            outputRowStart = inputRowEnd
            outputRowEnd = outputRowStart + outSeqLen
            intputDataset = np.expand_dims(dataset[inputRowStart:inputRowEnd, :numFeatures], axis=0)
            predictedDataset[outputRowStart:outputRowEnd,:numFeatures] = \
                    lstmObj.getModel().predict(intputDataset)
            predictedDataset[outputRowStart:outputRowEnd, numFeatures] = \
                    dataset[outputRowStart:outputRowEnd, numFeatures]
            inputRowStart += outSeqLen
            inputRowEnd = inputRowStart + inSeqLen
        predictedDataset = pd.DataFrame(predictedDataset, columns=columnNames)
        outputFilename = os.path.basename(testFilePath).replace('.csv', '.'+lstmCsvSuffix+'.csv')
        (head, tail) = os.path.split(testFilePath)
        outputFilePath = os.path.join(head, outputFilename)
        print ("Saving the predicted LSTM output to ", outputFilePath)
        predictedDataset.to_csv(outputFilePath)

    return

def testWithLSTM(modelFile, weightsFile, numFeaturesInTestFiles, allFiles):
    lstmObj = eegLSTM("encoder_decoder_sequence")
    lstmObj.loadModel(modelFile, weightsFile)
    print("Loaded LSTM model from disk")
    if (numFeaturesInTestFiles != lstmObj.numFeatures):
        print ("number of features in testfiles ", numFeaturesInTestFiles, 
            "!= number of feature in loaded model ", lstmObj.numFeatures)
    timeStepsToPredict = 1
    for testFilePath in allFiles.values():
        print ("testFilePath = ", testFilePath)
        # lstmObj.prepareDataset_fullfile(testFilePath)
        # lstmObj.evaluate()
        dataset = pd.read_csv(testFilePath)
        dataset = dataset.values # Convert to a numpy array from pandas dataframe
        dataset = dataset[:,1:]
        print ("dataset = ", dataset)
        numFeatures = lstmObj.numFeatures
        inSeqLen = lstmObj.inSeqLen
        outSeqLen = lstmObj.outSeqLen
        numRowsNeededForTest = max((inSeqLen + outSeqLen), (inSeqLen+timeStepsToPredict))
        numRows = dataset.shape[0]
        print ("inSeqLen={}, outSeqLen={}, numFeatures={}, numRows={}, numRowsNeededForTest={}".format(
            inSeqLen, outSeqLen, numFeatures, numRows, numRowsNeededForTest
        ))
        # lstmObj.prepareDataset_fullfile(testFilePath)
        while (numRows > numRowsNeededForTest):
            numRemainingRows = min (numRows, (inSeqLen+timeStepsToPredict))
            # print ("numRows={}, numFeatures={}, numRemainingRows={}"
            #         .format(numRows, numFeatures, numRemainingRows))

            predictedDataset = np.empty((1, (inSeqLen+timeStepsToPredict), numFeatures))
            inputRowStart = 0
            inputRowEnd = inputRowStart + inSeqLen
            outputRowStart = inputRowEnd
            outputRowEnd = outputRowStart + outSeqLen
            # print ("inputRowStart={}, inputRowEnd={}, outputRowStart={}, outputRowEnd={}"
            #         .format(inputRowStart, inputRowEnd, outputRowStart, outputRowEnd))
            predictedDataset[0, inputRowStart:inputRowEnd,:numFeatures] = dataset[inputRowStart:inputRowEnd, :numFeatures]
            while (numRemainingRows >= numRowsNeededForTest):
                predictedDataset[:, outputRowStart:outputRowEnd, :] = \
                    lstmObj.getModel().predict(predictedDataset[:, inputRowStart:inputRowEnd, :])
                
                inputRowStart += outSeqLen
                inputRowEnd = inputRowStart + inSeqLen
                outputRowStart = inputRowEnd
                outputRowEnd = outputRowStart + outSeqLen
                numRemainingRows -= outSeqLen
                # print ("inputRowStart={}, inputRowEnd={}, outputRowStart={}, outputRowEnd={}"
                #         .format(inputRowStart, inputRowEnd, outputRowStart, outputRowEnd))

            print ("predictedDataset = ", predictedDataset[0, inSeqLen:min (numRows, (inSeqLen+timeStepsToPredict)), :numFeatures])
            print ("actual dataset = ", dataset[inSeqLen:min (numRows, (inSeqLen+timeStepsToPredict)), :numFeatures])
            calculateLSTMMetrics(predictedDataset[0, inSeqLen:min (numRows, (inSeqLen+timeStepsToPredict)), :numFeatures],
                dataset[inSeqLen:min (numRows, (inSeqLen+timeStepsToPredict)), :numFeatures])
            dataset = np.delete(dataset, list(range(timeStepsToPredict)), axis=0)
            numRows = dataset.shape[0]

    return

def testWithDNN(modelFile, weightsFile, numFeaturesInTestFiles, allFiles):
    dnnObj = eegDNN("Classifier_3layers")
    dnnObj.loadModel(modelFile, weightsFile)
    print("Loaded DNN model from disk")
    if (numFeaturesInTestFiles != dnnObj.numFeatures):
        print ("number of features in testfiles ", numFeaturesInTestFiles, 
            "!= number of feature in loaded model ", dnnObj.numFeatures)
    for testFilePath in allFiles.values():
        print ("testFilePath = ", testFilePath)
        dnnObj.prepareDataset_fullfile(testFilePath)
        # dataset = np.loadtxt(testFile, delimiter=',')
        # X = dataset[:,:19]
        # y = dataset[:,19]
        dnnObj.evaluate()
        # Now predict the seizure values one at a time and calculate the metrics
        (X, y) = dnnObj.get_X_y()
        numRows = X.shape[0]
        print ("numRows = ", numRows)
        X = np.expand_dims(X, axis=0)
        y_hat = np.empty((numRows))
        for i in range(numRows):
            y_hat[i] = dnnObj.getModel().predict(X[:, i, :])
        (precision, recall) = calculateDNNMetrics(y_hat, y)
        if (recall > 0):
            print ("precision = {}, recall = {}".format(precision, recall))

    
def testWithHybridModel(lstmModelFile, lstmWeightsFile, dnnModelFile, 
                    dnnWeightsFile, numFeaturesInTestFiles, allFiles,
                    timeStepsToPredict):
    lstmObj = eegLSTM("encoder_decoder_sequence")
    lstmObj.loadModel(lstmModelFile, lstmWeightsFile)
    print("Loaded LSTM model from disk")
    dnnObj = eegDNN("Classifier_3layers")
    dnnObj.loadModel(dnnModelFile, dnnWeightsFile)
    print("Loaded DNN model from disk")
    if (numFeaturesInTestFiles != lstmObj.numFeatures):
        print ("number of features in testfiles ", numFeaturesInTestFiles, 
            "!= number of feature in loaded model ", lstmObj.numFeatures)
    if (numFeaturesInTestFiles != dnnObj.numFeatures):
        print ("number of features in testfiles ", numFeaturesInTestFiles, 
            "!= number of feature in loaded model ", dnnObj.numFeatures)

    numFeatures = lstmObj.numFeatures
    inSeqLen = lstmObj.inSeqLen
    outSeqLen = lstmObj.outSeqLen
    if (timeStepsToPredict % outSeqLen != 0):
        print ("Error! timeStepsToPredict should be a multiple of outSeqLen")
        exit (-1)
    print ("inSeqLen={}, outSeqLen={}, numFeatures={}".format(inSeqLen, outSeqLen, numFeatures))

    precisions = []
    recalls = []
    f1_scores = []

    for testFilePath in allFiles.values():
        print ("testFilePath = ", testFilePath)
        dataset = pd.read_csv(testFilePath)
        dataset = dataset.values # Convert to a numpy array from pandas dataframe
        # Remove the index column (unfortunately we have do do this explicitly)
        dataset = dataset[:,1:]
        # print ("dataset[column 0] = ", dataset[:,0])
        numLSTMruns = timeStepsToPredict // outSeqLen  # "//"" is integer division
        # numRowsNeededForTest = max((inSeqLen + outSeqLen), (inSeqLen+timeStepsToPredict))
        numRows = dataset.shape[0]
        numHybridRuns = numRows - (inSeqLen + timeStepsToPredict)
        print ("numRows={}, numLSTMruns={}, numHybridRuns={}".format(
            numRows, numLSTMruns, numHybridRuns))
        predictedDataset = np.empty((1, dataset.shape[0], numFeatures))
        predictedSeizureValues = np.empty((dataset.shape[0]))
        for i in range(numHybridRuns):
            inputRowStart = i
            inputRowEnd = inputRowStart + inSeqLen
            # print ("inputRowStart={}, inputRowEnd={}, outputRowStart={}, outputRowEnd={}"
            #         .format(inputRowStart, inputRowEnd, outputRowStart, outputRowEnd))
            predictedDataset[0, inputRowStart:inputRowEnd,:numFeatures] = dataset[inputRowStart:inputRowEnd, :numFeatures]
            predictedSeizureValues[inputRowStart:inputRowEnd] = dataset[inputRowStart:inputRowEnd, numFeatures]
            for j in range(numLSTMruns):
                outputRowStart = inputRowEnd
                outputRowEnd = outputRowStart + outSeqLen
                predictedDataset[:, outputRowStart:outputRowEnd, :] = \
                    lstmObj.getModel().predict(predictedDataset[:, inputRowStart:inputRowEnd, :])                
                inputRowStart += outSeqLen
                inputRowEnd = inputRowStart + inSeqLen
                # print ("inputRowStart={}, inputRowEnd={}, outputRowStart={}, outputRowEnd={}"
                #         .format(inputRowStart, inputRowEnd, outputRowStart, outputRowEnd))

            predictedRowStart = i+inSeqLen
            predictedRowEnd = predictedRowStart + timeStepsToPredict - 1
            for k in range(predictedRowStart, predictedRowEnd):
                predictedSeizureValues[k] = dnnObj.getModel().predict(predictedDataset[:, k, :])
                if (dataset[k, numFeatures] >= 0.5):
                    print ("predictedDataset[", k, "] = ", predictedDataset[:, k, :], 
                            ",", predictedSeizureValues[k])
                    print (dataset[k, :])
                    print ("-----------------------------------------------------------------------")

            (precision, recall, f1_score) = calculateDNNMetrics(predictedSeizureValues[predictedRowStart:predictedRowEnd], 
                                                        dataset[predictedRowStart:predictedRowEnd,numFeatures])
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            # if (rc > 0):
            #     print ("predictedDataset = ", predictedDataset[0, inSeqLen:min (numRows, (inSeqLen+timeStepsToPredict)), :numFeatures])
            #     print ("actual dataset = ", dataset[inSeqLen:min (numRows, (inSeqLen+timeStepsToPredict)), :numFeatures])
            # dataset = np.delete(dataset, list(range(timeStepsToPredict)), axis=0)
            # numRows = dataset.shape[0]

    # print ("precisions = ", precisions)
    # print ("recalls = ", recalls)

    sum = 0
    numPrecisions = 0
    for i in range(len(precisions)):
        if precisions[i]>=0:
            sum += precisions[i]
            numPrecisions += 1
    avg_precision = sum / numPrecisions

    sum = 0
    numRecalls = 0
    for i in range(len(recalls)):
        if recalls[i]>=0:
            sum +=recalls[i]
            numRecalls += 1
    avg_recall = sum / numRecalls

    sum = 0
    num_f1_scores = 0
    for i in range(len(f1_scores)):
        if f1_scores[i]>=0:
            sum += f1_scores[i]
            num_f1_scores += 1
    avg_f1_score = sum / num_f1_scores

    print ("avg_precision = ", avg_precision)
    print ("avg_recall = ", avg_recall)
    print ("avg_F1_score = ", avg_f1_score)
    return

def calculateLSTMMetrics(predictedValues, actualValues):
    divergence = np.empty((predictedValues.shape[0], predictedValues.shape[1]))
    for i in range(predictedValues.shape[0]):
        for j in range(predictedValues.shape[1]):
            divergence[i, j] = (predictedValues[i, j] - actualValues[i, j]) / actualValues[i, j]
    
    print ("divergence = ", divergence)


def calculateDNNMetrics(predictedValues, actualValues):
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
        else:
            print ("actualValues[{}] = {}".format(i, actualValues[i]))
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
        print ("actualPositive = {}, actualNegative = {}, numTotal = {}".format(
            actualPositive, actualNegative, numTotal
        ))
        print ("Something is wrong!! numbers do not match")

    # don't bother to print anything if there are no positive values in the actual dataset
    # if (actualPositive <= 0):
    #     print ("Nothing to report:  This testcase did not have any actual positives")
    #     return (-1, -1)

    if(actualPositive > 0):
        print ("truePositives = {}, falsePositives = {}, trueNegatives = {}, falseNegatives = {}"
            .format(truePositives, falsePositives, trueNegatives, falseNegatives))
        print ("actualPositive = {}, actualNegative = {}".format(actualPositive, actualNegative))
    # if((truePositives+falsePositives)<=0):
    #     precision = -1
    # else:
    precision = float((truePositives + trueNegatives) / (actualNegative + actualPositive))
    if (actualPositive > 0):
        recall = float(truePositives / actualPositive)
    else:
        recall = -1
        if(actualPositive > 0):
            print ("Recall = N.A. (= -1, because actualPositive == 0)")
    if(actualPositive > 0):
        print ("Precision = {}, Recall = {}".format(precision, recall))
    
    if(recall>0) and (precision>0):
        f1_score = (2.0*precision*recall)/(precision+recall)
    else:
        f1_score = -1
    
    return (precision ,recall, f1_score)

if __name__ == "__main__":
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    modelName = sys.argv[2]
    print ("model to train = ", modelName)
    cfgReader = ConfigReader(configFile, modelName)

    testingDataTopDir = cfgReader.getTestingDataDir()
    testingRecords = cfgReader.getTestingRecords()
    recordInfoFile = cfgReader.getRecordInfoFile()

    print ("testingDataTopDir =", testingDataTopDir)
    print ("recordInfoFile = ", recordInfoFile)

    # Initlaize the variables to null values
    dataSource = '' # CHB or TUH
    dataFormat = '' # EDF or CSV
    modelType = ''  # LSTM, DNN, or HYBRID
    testingRecordsScope = '' # ALL, ITERATE_OVER_PATIENTS, SINGLE_PATIENT, SPECIFIC_RECORDS

    if (modelName in cfgReader.csvModels):
        epochSeconds = cfgReader.getEpochSeconds()
        slidingWindowSeconds = cfgReader.getSlidingWindowSeconds()
        csvSummaryJsonFile = cfgReader.getCSVsummaryJsonFile()
        print ("Epoch Seconds = ", epochSeconds)
        print ("Sliding Window Seconds = ", slidingWindowSeconds)
        print ("csvSummaryJsonFile = ", csvSummaryJsonFile)
        dataFormat = "CSV"
        f = open(csvSummaryJsonFile, 'r')
        csvRecordInfo = json.load(f)
        f.close()
        del (csvRecordInfo['EOFmarker'])
    elif (modelName in cfgReader.edfModels):
        dataFormat = "EDF"
        csvRecordInfo = None
        print ("Currently testing directly from the EDF file is not supported :(")
        exit (-1)
    else:
        print ("Error! Unknown data format!!")
        exit (-1)

    if (re.search('LSTM', modelName) != None):
        modelType = "LSTM"
        print ("modelType = ", modelType)
    elif (re.search('DNN', modelName) != None):
        modelType = "DNN"
        print ("modelType = ", modelType)
    elif (re.search('HYBRID', modelName) != None):
        modelType = "HYBRID"
        print ("modelType = ", modelType)
    else:
        print ("Error! Unknown model type!!")
        exit (-1)

    if (re.search('CHB', modelName) != None):
        dataSource = "CHB"
        # Do not provide the seizures.json file; we will load the seizure info from the recordInfo json file
        datasetObj = CHBdataset(testingDataTopDir, '')
        # loadJsonFile() will initialize the object
        datasetObj.loadJsonFile(recordInfoFile)
    elif (re.search('TUH', modelName) != None):
        dataSource = "TUH"
        # Do not provide the xlsx file; we will load the seizure info from the json file
        datasetObj = TUHdataset(testingDataTopDir, '')
        # loadJsonFile() will initialize the object
        datasetObj.loadJsonFile(recordInfoFile)
    else:
        print ("Error! Unknown data source!!")
        exit (-1)

    allRecords = []
    allFiles = {}
    if (testingRecords[0] == "all"):
        testingRecordsScope = 'ALL'
        allRecords = list(datasetObj.recordInfo.keys())
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, csvRecordInfo)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    elif (re.search("records for patient (\w+)", testingRecords[0]) != None):
        testingRecordsScope = 'SPECIFIC_PATIENT'
        m = re.match("records for patient (\w+)", testingRecords[0])
        patientID = m.group(1)
        print ("finding records for patient ID", patientID)
        allRecords = datasetObj.patientInfo[patientID]['records']
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, csvRecordInfo)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    elif (re.search("one patient at a time", testingRecords[0]) != None):
        testingRecordsScope = 'ITERATE_OVER_PATIENTS'
        # Load the saved model and try to evaluate on new data
        # load json and create model
        if (modelType == "LSTM"):
            modelFile = cfgReader.getSavedLSTMModelFile()
            weightsFile = cfgReader.getSavedLSTMWeightsFile()
            print ("modelFile = {}, weightsFile = {}".format(modelFile, weightsFile))
            modelFilePerPatient = {}
            weightsFilePerPatient = {}
        elif (modelType == "DNN"):
            modelFile = cfgReader.getSavedDNNModelFile()
            weightsFile = cfgReader.getSavedDNNWeightsFile()
            print ("modelFile = {}, weightsFile = {}".format(modelFile, weightsFile))
            modelFilePerPatient = {}
            weightsFilePerPatient = {}
        elif (modelType == "HYBRID"):
            lstmModelFile = cfgReader.getSavedLSTMModelFile()
            lstmWeightsFile = cfgReader.getSavedLSTMWeightsFile()
            dnnModelFile = cfgReader.getSavedDNNModelFile()
            dnnWeightsFile = cfgReader.getSavedDNNWeightsFile()
            print ("lstmModelFile = {}, lstmWeightsFile = {}, dnnModelFile = {}, dnnWeightsFile = {}".format(
                lstmModelFile, lstmWeightsFile, dnnModelFile, dnnWeightsFile
            ))
            lstmModelFilePerPatient = {}
            lstmWeightsFilePerPatient = {}
            dnnModelFilePerPatient = {}
            dnnWeightsFilePerPatient = {}
        else:
            print ("Invalid modelType ", modelType)
            exit (-1)
        filesPerPatient = {}
        for patientID in datasetObj.patientInfo.keys():
            filesPerPatient[patientID] = getCSVfilesForRecords(datasetObj.patientInfo[patientID]['records'], csvRecordInfo)
            if (modelType != "HYBRID"):
                modelFilePerPatient[patientID] = modelFile.replace("<PATIENT_ID>", patientID)
                weightsFilePerPatient[patientID] = weightsFile.replace("<PATIENT_ID>", patientID)
            else:
                lstmModelFilePerPatient[patientID] = lstmModelFile.replace("<PATIENT_ID>", patientID)
                lstmWeightsFilePerPatient[patientID] = lstmWeightsFile.replace("<PATIENT_ID>", patientID)
                dnnModelFilePerPatient[patientID] = dnnModelFile.replace("<PATIENT_ID>", patientID)
                dnnWeightsFilePerPatient[patientID] = dnnWeightsFile.replace("<PATIENT_ID>", patientID)
    else:
        testingRecordsScope = 'SPECIFIC_RECORDS'
        allRecords = testingRecords
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, csvRecordInfo)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)

    if (testingRecordsScope != 'ITERATE_OVER_PATIENTS'):
        if (dataFormat == "EDF"):
            numFiles = len(allRecords)
        else:
            numFiles = len(allFiles)
        print ("Number of files to test the model on = ", numFiles)

        if (dataFormat == "CSV"):
            numFeaturesInTestFiles = verifyAndGetNumFeaturesCSV(csvRecordInfo, allRecords)
        else:
            numFeaturesInTestFiles = verifyAndGetNumFeaturesEDF(datasetObj, allRecords)

        if (numFeaturesInTestFiles <= 0):
            print ("Error in numFeaturesInTestFiles value!")
            exit (-1)
        if (modelType == "LSTM"):
            modelFile = cfgReader.getSavedLSTMModelFile()
            weightsFile = cfgReader.getSavedLSTMWeightsFile()
            if cfgReader.getSaveLSTMOutputToCSV():
                lstmCsvSuffix = cfgReader.getSavedLSTMOutputSuffix()
                testWithLSTMAndSave(modelFile, weightsFile, numFeaturesInTestFiles, allFiles, lstmCsvSuffix)
            else:
                testWithLSTM(modelFile, weightsFile, numFeaturesInTestFiles, allFiles)
        elif (modelType == "DNN"):
            modelFile = cfgReader.getSavedDNNModelFile()
            weightsFile = cfgReader.getSavedDNNWeightsFile()
            testWithDNN(modelFile, weightsFile, numFeaturesInTestFiles, allFiles)
        elif (modelType == "HYBRID"):
            lstmModelFile = cfgReader.getSavedLSTMModelFile()
            lstmWeightsFile = cfgReader.getSavedLSTMWeightsFile()
            dnnModelFile = cfgReader.getSavedDNNModelFile()
            dnnWeightsFile = cfgReader.getSavedDNNWeightsFile()
            timeStepsToPredict = cfgReader.getTimeStepsToPredict()
            testWithHybridModel(lstmModelFile, lstmWeightsFile, dnnModelFile, 
                                dnnWeightsFile, numFeaturesInTestFiles, allFiles,
                                timeStepsToPredict)
        else:
            print ("modelType ", modelType, "is not yet supported")
            exit (-1)
    else:
        # Iterate testing over each patient
        for patientID in filesPerPatient.keys():
            allRecords = datasetObj.patientInfo[patientID]['records']
            if (dataFormat == "CSV"):
                numFeaturesInTestFiles = verifyAndGetNumFeaturesCSV(csvRecordInfo, allRecords)
            else:
                numFeaturesInTestFiles = verifyAndGetNumFeaturesEDF(datasetObj, allRecords)
            if (numFeaturesInTestFiles <= 0):
                # Move on to the next patient
                print ("Skipping patient", patientID, "because the number of features has some issue")
                continue
            allFiles = filesPerPatient[patientID]
            modelFile = modelFilePerPatient[patientID]
            weightsFile = weightsFilePerPatient[patientID]
            print ("PatientID = {}, numFeaturesInTestFiles = {}, modelFile = {}, weightsFile = {}".format(
                patientID, numFeaturesInTestFiles, modelFile, weightsFile
            ))
            if (modelType == "LSTM"):
                if cfgReader.getSaveLSTMOutputToCSV():
                    lstmCsvSuffix = cfgReader.getSavedLSTMOutputSuffix()
                    testWithLSTMAndSave(modelFile, weightsFile, numFeaturesInTestFiles, allFiles, lstmCsvSuffix)
                else:
                    testWithLSTM(modelFile, weightsFile, numFeaturesInTestFiles, allFiles)
            elif (modelType == "DNN"):
                testWithDNN(modelFile, weightsFile, numFeaturesInTestFiles, allFiles)
            else:
                print ("modelType ", modelType, "is not yet supported")
                exit (-1)
