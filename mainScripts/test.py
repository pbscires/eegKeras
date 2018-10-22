import sys
import os
import re
import json
from Models.eegDNN import eegDNN
from Models.eegLSTM import eegLSTM
from DataSets.TUHdataset import TUHdataset
from DataSets.CHBdataset import CHBdataset


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
    
    def getSavedModelFile(self):
        return self.jsonData['savedModelFile']
    
    def getSavedWeightsFile(self):
        return self.jsonData['savedWeightsFile']
    
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

def getCSVfilesForRecords(allRecords, testingDataTopDir):
    allFiles = {}
    for root, dirs, files in os.walk(testingDataTopDir):
        for filename in files:
            if (re.search("\.csv$", filename) != None):
                recordID = os.path.splitext(os.path.basename(filename))[0]
                if (recordID in allRecords):
                    allFiles[recordID] = os.path.join(root, filename)
    return (allFiles)

def verifyAndGetNumFeatures(datasetObj, allRecords):
    # Verify that all the records have same features
    features = datasetObj.recordInfo[allRecords[0]]['channelLabels']
    featuresSet = set(features)
    for recordID in allRecords:
        tmpSet = set(datasetObj.recordInfo[recordID]['channelLabels'])
        xorSet = featuresSet.symmetric_difference(tmpSet)
        if (len(xorSet) > 0):
            print ("features are not common between", allRecords[0], "and", recordID)
            exit (-1)
    numFeatures = len(featuresSet)
    print ("features are common between all the records! numFeatures = ", numFeatures)
    return (numFeatures)

if __name__ == "__main__":
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    modelName = sys.argv[2]
    print ("model to train = ", modelName)
    cfgReader = ConfigReader(configFile, modelName)

    testingDataTopDir = cfgReader.getTestingDataDir()
    testingRecords = cfgReader.getTestingRecords()
    # Load the saved model and try to evaluate on new data
    # load json and create model
    modelFile = cfgReader.getSavedModelFile()
    weightsFile = cfgReader.getSavedWeightsFile()
    recordInfoFile = cfgReader.getRecordInfoFile()

    print ("testingDataTopDir = {}", testingDataTopDir)
    print ("modelFile = {}, weightsFile = {}".format(modelFile, weightsFile))
    print ("recordInfoFile = ", recordInfoFile)

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
        print ("Currently testing directly from the EDF file is not supported :(")
        exit (-1)
    else:
        print ("Error! Unknown data format!!")
        exit (-1)

    if (re.search('LSTM', modelName) != None):
        modelType = "LSTM"
        print ("modelType = {}", modelType)
    elif (re.search('DNN', modelName) != None):
        modelType = "DNN"
        print ("modelType = {}", modelType)
    else:
        print ("Error! Unknown model type!!")
        exit (-1)

    if (re.search('CHB', modelName) != None):
        dataSource = "CHB"
        print ("CHB data source is currently not yet implemented :(")
        exit (-1)
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
        savedModelFilePrefix = modelName + "_all"
        allRecords = list(datasetObj.recordInfo.keys())
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, testingDataTopDir)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    elif (re.search("records for patient (\w+)", testingRecords[0]) != None):
        m = re.match("records for patient (\w+)", testingRecords[0])
        patientID = m.group(1)
        print ("finding records for patient ID", patientID)
        allRecords = datasetObj.patientInfo[patientID]['records']
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, testingDataTopDir)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    else:
        allRecords = testingRecords
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, testingDataTopDir)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)

    if (dataFormat == "EDF"):
        numFiles = len(allRecords)
    else:
        numFiles = len(allFiles)
    print ("Number of files to test the model on = ", numFiles)

    numFeaturesInTestFiles = verifyAndGetNumFeatures(datasetObj, allRecords)
    if (modelType == "LSTM"):
        lstmObj = eegLSTM("encoder_decoder_sequence")
        # numFeatures = 168
        # lstmObj.loadModel(modelFile, weightsFile, inSeqLen, outSeqLen, numFeatures)
        lstmObj.loadModel(modelFile, weightsFile)
        print("Loaded model from disk")
        if (numFeaturesInTestFiles != lstmObj.numFeatures):
            print ("number of features in testfiles ", numFeaturesInTestFiles, 
                "!= number of feature in loaded model ", lstmObj.numFeatures)
        for testFilePath in allFiles.values():
            print ("testFilePath = ", testFilePath)
            lstmObj.prepareDataset_fullfile(testFilePath)
            # dataset = np.loadtxt(testFile, delimiter=',')
            # X = dataset[:,:19]
            # y = dataset[:,19]
            lstmObj.evaluate()
    elif (modelType == "DNN"):
        dnnObj = eegDNN("Classifier_3layers")
        dnnObj.loadModel(modelFile, weightsFile)
        print("Loaded model from disk")
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
    else:
        print ("modelType ", modelType, "is not yet supported")
        exit (-1)
