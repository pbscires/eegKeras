import sys
from Models.eegDNN import eegDNN
from Models.eegLSTM import eegLSTM
import json
import os
import re

class ConfigReader(object):
    def __init__(self, configFile, modelName):
        '''
        Read the given config file into a json dictionary
        '''
        self.configFile = configFile
        f = open(configFile, 'r')
        self.jsonRoot = json.load(f)
        self.jsonData = self.jsonRoot[modelName]
        f.close()

    def getTrainingDataDir(self):
        return self.jsonData['trainingDataTopDir']
    
    def getModelOutputDir(self):
        return self.jsonData['modelOutputDir']
        
    def getTrainingFiles(self):
        return self.jsonData['trainingFiles']

def getFileNamesForTraining_CHB(trainingRecords):
    allRecords = []
    if (trainingRecords[0] == "all"):
        allRecords = list(tuhd.recordInfo.keys())
    elif (re.search("records for patient (\d+)", trainingRecords[0]) != None):
        m = re.match("records for patient (\d+)", trainingRecords[0])
        patientID = m.group(1)
        print ("finding records for patient ID", patientID)
        allRecords = tuhd.patientInfo[patientID]['records']
    else:
        allRecords = trainingRecords

    return (allRecords)

if __name__ == "__main__":
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    modelName = sys.argv[2]
    print ("model to train = ", modelName)
    cfgReader = ConfigReader(configFile, modelName)

    trainingDataTopDir = cfgReader.getTrainingDataDir()
    trainingFiles = cfgReader.getTrainingFiles()
    modelOutputDir = cfgReader.getModelOutputDir()
    savedModelFilePrefix = modelName
    print ("trainingDataTopDir = ", trainingDataTopDir)
    print ("trainingFiles = ", trainingFiles)
    print ("modelOutputDir = ", modelOutputDir)
    print ("part of the savedModelFilePrefix = ", savedModelFilePrefix)

    allFiles = []
    if (trainingFiles[0] == "all"):
        for root, dirs, files in os.walk(trainingDataTopDir):
            for filename in files:
                if (re.search("\.csv$", filename) != None):
                    allFiles.append(filename)
    
    numFiles = len(allFiles)
    print ("Number of files to train the model on = ", numFiles)

    curFileNum = 0
    for filename in allFiles:
        curFileNum += 1
        print ("Currently training file {} of {} ...".format(curFileNum, numFiles))
        filePrefix = os.path.splitext(os.path.basename(filename))[0]
        filePrefix = savedModelFilePrefix + filePrefix
        dnnModel = eegDNN("Classifier_3layers")
        filePath = os.path.join(trainingDataTopDir, filename)
        print ("trainingFilePath = ", filePath)
        dnnModel.prepareDataset_1file(filePath)
        numFeatures = dnnModel.numFeatures
        dnnModel.createModel(numFeatures)
        dnnModel.fit()
        dnnModel.saveModel(modelOutputDir, filePrefix)