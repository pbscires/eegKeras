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

    def getTrainingDataDir(self):
        return self.jsonData['trainingDataTopDir']
    
    def getModelOutputDir(self):
        return self.jsonData['modelOutputDir']
    
    def getRecordInfoFile(self):
        return self.jsonData['recordInfoJsonFile']
    
    def getCSVsummaryJsonFile(self):
        return self.jsonData['csvSummaryJsonFile']
        
    def getTrainingRecords(self):
        return self.jsonData['trainingRecords']
    
    def getDataSubsetString(self):
        return self.jsonData['data_subset']
    
    def getDNNLayers(self):
        return self.jsonData['dnn_layers']
    
    def getLSTMlayers(self):
        return self.jsonData['lstm_layers']
    
    def getInputSeqLen(self):
        return self.jsonData['inSeqLen']
    
    def getOutputSeqLen(self):
        return self.jsonData['outSeqLen']
    
    def getTrainingEpochs(self):
        return self.jsonData['training_epochs']
    
    def getTrainingBatchSize(self):
        return self.jsonData['training_batchsize']
    
    def getValidationSplit(self):
        return self.jsonData['validation_split']
    
    def getEpochSeconds(self):
        return self.jsonData['epochSeconds']
    
    def getSlidingWindowSeconds(self):
        return self.jsonData['SlidingWindowSeconds']

def getCSVfilesForRecords(allRecords, trainingDataTopDir):
    allFiles = {}
    for root, dirs, files in os.walk(trainingDataTopDir):
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
    print ("features are common between all the records!")
    numFeatures = len(featuresSet)
    return (numFeatures)

def getPriorAndPostSeconds(dataSubset):
    if (dataSubset == "fulldata"):
        print ("Will be training on full data")
        priorSeconds = postSeconds = -1
    elif (re.search("seizure\-(\d+), seizure\+(\d+)", dataSubset) != None):
        m = re.match("seizure\-(\d+), seizure\+(\d+)", dataSubset)
        priorSeconds = int(m.group(1))
        postSeconds = int(m.group(2))
        print ("data subset = [seizure-" + str(priorSeconds), ", seizure+" + str(postSeconds) + "]")
    
    return (priorSeconds, postSeconds)

if __name__ == "__main__":
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    modelName = sys.argv[2]
    print ("model to train = ", modelName)
    cfgReader = ConfigReader(configFile, modelName)

    trainingDataTopDir = cfgReader.getTrainingDataDir()
    trainingRecords = cfgReader.getTrainingRecords()
    modelOutputDir = cfgReader.getModelOutputDir()
    recordInfoFile = cfgReader.getRecordInfoFile()
    dataSubset = cfgReader.getDataSubsetString()
    trainingEpochs = cfgReader.getTrainingEpochs()
    trainingBatchSize = cfgReader.getTrainingBatchSize()
    validation_split = cfgReader.getValidationSplit()

    print ("trainingDataTopDir = ", trainingDataTopDir)
    print ("trainingRecords = ", trainingRecords)
    print ("modelOutputDir = ", modelOutputDir)
    print ("data subset string = ", dataSubset)
    print ("recordInfoFile = ", recordInfoFile)
    print ("trainingEpochs = ", trainingEpochs)
    print ("trainingBatchSize = ", trainingBatchSize)
    print ("validation_split = ", validation_split)

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
        lstm_layers = cfgReader.getLSTMlayers()
        inSeqLen = cfgReader.getInputSeqLen()
        outSeqLen = cfgReader.getOutputSeqLen()
        print ("modelType = {}, lstm_layers = {}, inSeqLen = {}, outSeqLen = {}".format(
            modelType, lstm_layers, inSeqLen, outSeqLen))
    elif (re.search('DNN', modelName) != None):
        modelType = "DNN"
        dnn_layers = cfgReader.getDNNLayers()
        print ("modelType = {}, dnn_layers = {}".format(modelType, dnn_layers))
    else:
        print ("Error! Unknown model type!!")
        exit (-1)
    
    if (re.search('CHB', modelName) != None):
        dataSource = "CHB"
        print ("CHB data source is currently not yet implemented :(")
        exit (-1)
        # Do not provide the seizures.json file; we will load the seizure info from the recordInfo json file
        datasetObj = CHBdataset(trainingDataTopDir, '')
        # loadJsonFile() will initialize the object
        datasetObj.loadJsonFile(recordInfoFile)
    elif (re.search('TUH', modelName) != None):
        dataSource = "TUH"
        # Do not provide the xlsx file; we will load the seizure info from the json file
        datasetObj = TUHdataset(trainingDataTopDir, '')
        # loadJsonFile() will initialize the object
        datasetObj.loadJsonFile(recordInfoFile)
    else:
        print ("Error! Unknown data source!!")
        exit (-1)


    allRecords = []
    allFiles = {}
    if (trainingRecords[0] == "all"):
        savedModelFilePrefix = modelName + "_all"
        allRecords = list(datasetObj.recordInfo.keys())
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, trainingDataTopDir)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    elif (re.search("records for patient (\w+)", trainingRecords[0]) != None):
        m = re.match("records for patient (\w+)", trainingRecords[0])
        patientID = m.group(1)
        print ("finding records for patient ID", patientID)
        allRecords = datasetObj.patientInfo[patientID]['records']
        savedModelFilePrefix = modelName + "_" + patientID
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, trainingDataTopDir)        
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    else:
        savedModelFilePrefix = modelName + "_customRecords"
        allRecords = trainingRecords
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, trainingDataTopDir)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    
    print ("savedModelFilePrefix = ", savedModelFilePrefix)
    numFeatures = verifyAndGetNumFeatures(datasetObj, allRecords)
    
    if (dataFormat == "EDF"):
        numFiles = len(allRecords)
    else:
        numFiles = len(allFiles)
    print ("Number of files to train the model on = ", numFiles)

    (priorSeconds, postSeconds) = getPriorAndPostSeconds(dataSubset)
    if (priorSeconds == -1 or postSeconds == -1):
        print ("Training the model with full file is not yet implemented")
        exit (-1)

    if (modelType == "LSTM"):
        lstmObj = eegLSTM("encoder_decoder_sequence")
        # lstmObj = eegLSTM("stacked_LSTM")
        lstmObj.createModel(inSeqLen, outSeqLen, numFeatures, lstm_layers)
        if (dataFormat == "CSV"):
            lstmObj.prepareDataSubset_fromCSV(datasetObj, allRecords, csvRecordInfo, (epochSeconds*1000), (slidingWindowSeconds*1000), priorSeconds, postSeconds)
        else:
            print ("LSTM model for EDF format is not yet implemented in this version")
            exit(-1)
        lstmObj.fit(trainingEpochs, trainingBatchSize)
        lstmObj.saveModel(modelOutputDir, savedModelFilePrefix)
    elif (modelType == "DNN"):
        dnnObj = eegDNN("Classifier_3layers")
        dnnObj.createModel(numFeatures, dnn_layers)
        if (dataFormat == "CSV"):
            dnnObj.prepareDataSubset_fromCSV(datasetObj, allRecords, csvRecordInfo, (epochSeconds*1000), (slidingWindowSeconds*1000), priorSeconds, postSeconds)
        else:
            print ("DNN model for EDF format is not yet implemented in this version")
            exit(-1)
        dnnObj.fit(trainingEpochs, trainingBatchSize, validation_split)
        dnnObj.saveModel(modelOutputDir, savedModelFilePrefix)
        # print ("DNN model is not yet implemented!!")
        # exit (-1)
    else:
        print ("Invalid model type!!")
        exit (-1)

    # curFileNum = 0
    # for filename in allRecords:
    #     curFileNum += 1
    #     print ("Currently training file {} of {} ...".format(curFileNum, numFiles))
    #     filePrefix = os.path.splitext(os.path.basename(filename))[0]
    #     filePrefix = savedModelFilePrefix + filePrefix
    #     dnnModel = eegDNN("Classifier_3layers")
    #     filePath = os.path.join(trainingDataTopDir, filename)
    #     print ("trainingFilePath = ", filePath)
    #     dnnModel.prepareDataset_1file(filePath)
    #     numFeatures = dnnModel.numFeatures
    #     dnnModel.createModel(numFeatures)
    #     dnnModel.fit(trainingEpochs, trainingBatchSize, validation_split)
    #     dnnModel.saveModel(modelOutputDir, filePrefix)