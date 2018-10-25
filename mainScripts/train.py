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
    print(csvRecordInfo)
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

def trainWithLSTM(params):
    inSeqLen = params['inSeqLen']
    outSeqLen = params['outSeqLen']
    numFeatures = params['numFeatures']
    lstm_layers = params['lstm_layers']
    datasetObj = params['datasetObj']
    allRecords = params['allRecords']
    csvRecordInfo = params['csvRecordInfo']
    epochSeconds = params['epochSeconds']
    slidingWindowSeconds = params['slidingWindowSeconds']
    priorSeconds = params['priorSeconds']
    postSeconds = params['postSeconds']
    trainingEpochs = params['trainingEpochs']
    trainingBatchSize = params['trainingBatchSize']
    validation_split = params['validation_split']
    modelOutputDir = params['modelOutputDir']
    savedModelFilePrefix = params['savedModelFilePrefix']

    lstmObj = eegLSTM("encoder_decoder_sequence")
    # lstmObj = eegLSTM("stacked_LSTM")
    lstmObj.createModel(inSeqLen, outSeqLen, numFeatures, lstm_layers)
    if (dataFormat == "CSV"):
        numSamples = lstmObj.prepareDataSubset_fromCSV(datasetObj, allRecords, 
            csvRecordInfo, (epochSeconds*1000), (slidingWindowSeconds*1000), 
            priorSeconds, postSeconds)
        # If the number of samples is too low, there is no point in training with this dataset
        if (numSamples <= 10):
            print ("numSamples ({}) is too low! Returning without creating a saved model!".format(numSamples))
            return
    else:
        print ("LSTM model for EDF format is not yet implemented in this version")
        exit(-1)
    lstmObj.fit(trainingEpochs, trainingBatchSize, validation_split)
    lstmObj.saveModel(modelOutputDir, savedModelFilePrefix)
    return

def trainWithDNN(params):
    numFeatures = params['numFeatures']
    dnn_layers = params['dnn_layers']
    datasetObj = params['datasetObj']
    allRecords = params['allRecords']
    csvRecordInfo = params['csvRecordInfo']
    epochSeconds = params['epochSeconds']
    slidingWindowSeconds = params['slidingWindowSeconds']
    priorSeconds = params['priorSeconds']
    postSeconds = params['postSeconds']
    trainingEpochs = params['trainingEpochs']
    trainingBatchSize = params['trainingBatchSize']
    validation_split = params['validation_split']
    modelOutputDir = params['modelOutputDir']
    savedModelFilePrefix = params['savedModelFilePrefix']

    dnnObj = eegDNN("Classifier_3layers")
    dnnObj.createModel(numFeatures, dnn_layers)
    if (dataFormat == "CSV"):
        numSamples = dnnObj.prepareDataSubset_fromCSV(datasetObj, allRecords, 
            csvRecordInfo, (epochSeconds*1000), (slidingWindowSeconds*1000), 
            priorSeconds, postSeconds)
        # If the number of samples is too low, there is no point in training with this dataset
        if (numSamples <= 10):
            print ("numSamples ({}) is too low! Returning without creating a saved model!".format(numSamples))
            return
    else:
        print ("DNN model for EDF format is not yet implemented in this version")
        exit(-1)
    dnnObj.fit(trainingEpochs, trainingBatchSize, validation_split)
    dnnObj.saveModel(modelOutputDir, savedModelFilePrefix)
    # print ("DNN model is not yet implemented!!")
    # exit (-1)
    return

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

    # Initlaize the variables to null values
    dataSource = '' # CHB or TUH
    dataFormat = '' # EDF or CSV
    modelType = ''  # LSTM, DNN, or HYBRID
    trainingRecordsScope = '' # ALL, ITERATE_OVER_PATIENTS, SINGLE_PATIENT, SPECIFIC_RECORDS

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
        print ("Currently training directly from the EDF file is not supported :(")
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
        # print ("CHB data source is currently not yet implemented :(")
        # exit (-1)
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
        trainingRecordsScope = 'ALL'
        savedModelFilePrefix = modelName + "_all"
        print ("savedModelFilePrefix = ", savedModelFilePrefix)
        allRecords = list(datasetObj.recordInfo.keys())
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, csvRecordInfo)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    elif (re.search("records for patient (\w+)", trainingRecords[0]) != None):
        trainingRecordsScope = 'SPECIFIC_PATIENT'
        m = re.match("records for patient (\w+)", trainingRecords[0])
        patientID = m.group(1)
        print ("finding records for patient ID", patientID)
        allRecords = datasetObj.patientInfo[patientID]['records']
        savedModelFilePrefix = modelName + "_" + patientID
        print ("savedModelFilePrefix = ", savedModelFilePrefix)
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, csvRecordInfo)        
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    elif (re.search("one patient at a time", trainingRecords[0]) != None):
        trainingRecordsScope = 'ITERATE_OVER_PATIENTS'
        filesPerPatient = {}
        savedFilePrefixPerPatient = {}
        for patientID in datasetObj.patientInfo.keys():
            filesPerPatient[patientID] = getCSVfilesForRecords(datasetObj.patientInfo[patientID]['records'], csvRecordInfo)
            savedFilePrefixPerPatient[patientID] = modelName + "_" + patientID
    else:
        trainingRecordsScope = 'SPECIFIC_RECORDS'
        savedModelFilePrefix = modelName + "_customRecords"
        print ("savedModelFilePrefix = ", savedModelFilePrefix)
        allRecords = trainingRecords
        # We need to gather the list of files only if the file format is CSV;
        # No need to gather the files list for EDF files as the file path is 
        # given in the recordInfo.json file
        if (dataFormat == "CSV"):
            allFiles = getCSVfilesForRecords(allRecords, csvRecordInfo)
        else:
            print ("Invalid data format ", dataFormat)
            exit (-1)
    
    (priorSeconds, postSeconds) = getPriorAndPostSeconds(dataSubset)
    if (priorSeconds == -1 or postSeconds == -1):
        print ("Training the model with full file is not yet implemented")
        exit (-1)

    params = {}
    params['datasetObj'] = datasetObj
    params['csvRecordInfo'] = csvRecordInfo
    params['epochSeconds'] = epochSeconds
    params['slidingWindowSeconds'] = slidingWindowSeconds
    params['priorSeconds'] = priorSeconds
    params['postSeconds'] = postSeconds
    params['trainingEpochs'] = trainingEpochs
    params['trainingBatchSize'] = trainingBatchSize
    params['validation_split'] = validation_split
    params['modelOutputDir'] = modelOutputDir

    if (trainingRecordsScope != 'ITERATE_OVER_PATIENTS'):
        params['allRecords'] = allRecords
        if (dataFormat == "CSV"):
            numFeatures = verifyAndGetNumFeaturesCSV(csvRecordInfo, allRecords)
        else:
            numFeatures = verifyAndGetNumFeaturesEDF(datasetObj, allRecords)
        if (numFeatures <= 0):
            print ("Error in numFeatures value!")
            exit (-1)

        params['numFeatures'] = numFeatures
        params['savedModelFilePrefix'] = savedModelFilePrefix
        
        if (dataFormat == "EDF"):
            numFiles = len(allRecords)
        else:
            numFiles = len(allFiles)
        print ("Number of files to train the model on = ", numFiles)

        if (modelType == "LSTM"):
            params['inSeqLen'] = inSeqLen
            params['outSeqLen'] = outSeqLen
            params['lstm_layers'] = lstm_layers
            trainWithLSTM(params)
        elif (modelType == "DNN"):
            params['dnn_layers'] = dnn_layers
            trainWithDNN(params)
        else:
            print ("Invalid model type!!")
            exit (-1)
    else:
        # Iterate training over each patient
        for patientID in filesPerPatient.keys():
            allRecords = datasetObj.patientInfo[patientID]['records']
            if (dataFormat == "CSV"):
                numFeatures = verifyAndGetNumFeaturesCSV(csvRecordInfo, allRecords)
            else:
                numFeatures = verifyAndGetNumFeaturesEDF(datasetObj, allRecords)
            if (numFeatures <= 0):
                # Move on to the next patient
                print ("Skipping patient", patientID, "because the number of features has some issue")
                continue
            allFiles = filesPerPatient[patientID]
            if (dataFormat == "EDF"):
                numFiles = len(allRecords)
            else:
                numFiles = len(allFiles)
            print ("Number of files to train the model on = ", numFiles)
            params['allRecords'] = allRecords
            params['numFeatures'] = numFeatures
            params['savedModelFilePrefix'] = savedFilePrefixPerPatient[patientID]
            if (modelType == "LSTM"):
                params['inSeqLen'] = inSeqLen
                params['outSeqLen'] = outSeqLen
                params['lstm_layers'] = lstm_layers
                trainWithLSTM(params)
            elif (modelType == "DNN"):
                params['dnn_layers'] = dnn_layers
                trainWithDNN(params)
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