import json

class TrainingConfigReader(object):
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
    
    def getTrainingDataDir(self):
        return self.jsonData['trainingDataTopDir']
    
    def getModelOutputDir(self):
        return self.jsonData['modelOutputDir']
    
    def getSavedModelPrefix(self):
        return self.jsonData['savedModelFilePrefix']
    
    def getTrainingFiles(self):
        return self.jsonData['trainingFiles']
    
    def getLSTMLayers(self):
        return ([int(i) for i in self.jsonData['lstm_layers']])
    
    def getSeqLens(self):
        inSeqLen, outSeqLen = int(self.jsonData['inSeqLen']), int(self.jsonData['outSeqLen'])
        return (inSeqLen, outSeqLen)
    
    def getTrainingRecords(self):
        return self.jsonData['trainingRecords']
    
    def get_datasubset(self):
        return self.jsonData['data_subset']
    
    def getEpochs(self):
        return int(self.jsonData['epochs'])
    
    def getBatchsize(self):
        return int(self.jsonData['batchsize'])

    def getRecordInfoJsonFile(self):
        return self.jsonData['recordInfoJsonFile']