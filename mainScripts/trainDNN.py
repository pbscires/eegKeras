import sys
from Models import DNN
import json
import os

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

