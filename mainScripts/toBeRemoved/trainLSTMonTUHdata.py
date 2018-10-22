import numpy as np
import pyedflib
import re
import sys
import os
from util.TrainingConfigReader import TrainingConfigReader
from DataSets.TUHdataset import TUHdataset
from Models.eegLSTM import eegLSTM

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Error! Invalid number of arguments")
        exit (-1)
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    cfgReader = TrainingConfigReader(configFile)
    trainingDataTopDir = cfgReader.getTrainingDataDir()
    trainingRecords = cfgReader.getTrainingRecords()
    modelOutputDir = cfgReader.getModelOutputDir()
    lstmLayers = cfgReader.getLSTMLayers()
    inSeqLen, outSeqLen = cfgReader.getSeqLens()
    dataSubset = cfgReader.get_datasubset()
    epochs = cfgReader.getEpochs()
    batchsize = cfgReader.getBatchsize()
    recordInfoJson = cfgReader.getRecordInfoJsonFile()

    print ("trainingDataTopDir = ", trainingDataTopDir)
    if (trainingRecords[0] == "all"):
        print ("all the files will be used for training")
    else:
        print ("training files = ", trainingRecords)
    print ("modelOutputDir = ", modelOutputDir)
    print ("LSTM layers = ", lstmLayers)
    print ("inSeqLen = {}, outSeqLen = {}".format(inSeqLen, outSeqLen))
    print ("dataSubset = ", dataSubset)
    print ("epochs = ", epochs)
    print ("batchsize = ", batchsize)
    print ("recordInfoJsonFile = ", recordInfoJson)

    allRecords = []
    tuhd = TUHdataset(trainingDataTopDir, '')
    # tuhd.summarizeDatset()
    tuhd.loadJsonFile(recordInfoJson)

    if (trainingRecords[0] == "all"):
        allRecords = list(tuhd.recordInfo.keys())
    elif (re.search("records for patient (\d+)", trainingRecords[0]) != None):
        m = re.match("records for patient (\d+)", trainingRecords[0])
        patientID = m.group(1)
        print ("finding records for patient ID", patientID)
        allRecords = tuhd.patientInfo[patientID]['records']
    else:
        allRecords = trainingRecords

    print ("Number of records to use for training = ", len(allRecords))

    if (dataSubset == "fulldata"):
        print ("Will be training on full data")
        priorSeconds = postSeconds = -1
    elif (re.search("seizure\-(\d+), seizure\+(\d+)", dataSubset) != None):
        m = re.match("seizure\-(\d+), seizure\+(\d+)", dataSubset)
        priorSeconds = int(m.group(1))
        postSeconds = int(m.group(2))
        print ("data subset = [seizure-" + str(priorSeconds), ", seizure+" + str(postSeconds) + "]")
    
    # Verify that all the records have same features
    features = tuhd.recordInfo[allRecords[0]]['channelLabels']
    featuresSet = set(features)
    for recordID in allRecords:
        tmpSet = set(tuhd.recordInfo[recordID]['channelLabels'])
        xorSet = featuresSet.symmetric_difference(tmpSet)
        if (len(xorSet) > 0):
            print ("features are not common between", allRecords[0], "and", recordID)
            exit (-1)
    print ("features are common between all the records!")
    numFeatures = len(featuresSet)
    lstmObj = eegLSTM("encoder_decoder_sequence")
    # lstmObj = eegLSTM("stacked_LSTM")
    lstmObj.createModel(inSeqLen, outSeqLen, numFeatures, lstmLayers)
    lstmObj.prepareDataset_fromTUHedf(tuhd, allRecords, priorSeconds, postSeconds)
    lstmObj.fit(epochs, batchsize)
    lstmObj.saveModel(modelOutputDir, recordID+"LSTM")
