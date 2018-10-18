from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sys
import json
import os
import re

class ConfigReader(object):
    def __init__(self, configFile):
        '''
        Read the given config file into a json dictionary
        '''
        self.configFile = configFile
        f = open(configFile, 'r')
        self.jsonData = json.load(f)
        f.close()
    
    def getFromCSVDir(self):
        return self.jsonData['from_csv_dir']
    
    def getToCSVDir(self):
        return self.jsonData['to_csv_dir']
    
    def getNumComponents(self):
        return int(self.jsonData['n_components'])
    


def loadCSV(filePath):
    df = pd.read_csv(filePath)
    print (df.shape)
    # print (df.head)
    X = df.iloc[:,range(df.shape[1]-1)]
    y = df.iloc[:,[df.shape[1]-1]]
    # print (X, y)
    return (X, y)

def applyPCA(X, n_components):
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    if (n_components < X.shape[1]):
        pca = PCA(n_components)
        X = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        print ("explained_variance = ", explained_variance)
        # print (X)
    return (X)

def saveCSVwithPCA(X, y, filePath):
    df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    # print ("df.shape = ", df.shape)
    # print (df.head)
    df.to_csv(filePath)


if __name__ == "__main__":
    configFile = sys.argv[1]
    print ("configFile = ", configFile)
    cfgReader = ConfigReader(configFile)
    fromDir = cfgReader.getFromCSVDir()
    toDir = cfgReader.getToCSVDir()
    n_components = cfgReader.getNumComponents()
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(fromDir):
        for filename in files:
            if (re.search("\.csv$", filename) != None):
                from_filePath = os.path.join(root, filename)
                to_filePath = os.path.join(toDir, filename)
                print ("from_filePath = {}, to_filePath = {}".format(from_filePath, to_filePath))
                if (os.path.exists(to_filePath)):
                    print (to_filePath, "already created!")
                    continue

                (X, y) = loadCSV(from_filePath)
                X = applyPCA(X, n_components)
                saveCSVwithPCA(X, y, to_filePath)