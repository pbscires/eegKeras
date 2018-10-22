import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import json
import os

class eegDNN(object):
    '''
        Constructor
    '''
    def __init__(self, modelName):
        self.modelName = modelName
    
    def createModel(self, numFeatures, dnn_layers):
        self.numFeatures = numFeatures

        model = Sequential()
        if (self.modelName == "Classifier_3layers"):
            model.add(Dense(dnn_layers[0], input_dim=numFeatures, activation='relu')) 
            for i in range(1, len(dnn_layers)-1):
                model.add(Dense(dnn_layers[i], activation='relu'))
            # model.add(Dense(20, activation='relu')) 
            # model.add(Dense(10, activation='relu')) 
            # model.add(Dense(1, activation='softmax'))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = model

    def loadModel(self, modelFile, weightsFile):
        if (self.modelName == "Classifier_3layers"):
            json_file = open(modelFile, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(weightsFile)
            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = loaded_model
            self.numFeatures = loaded_model.layers[0].get_input_at(0).get_shape().as_list()[1]
            print ("numFeatures=", self.numFeatures)

    def saveModel(self, outputDir, filePrefix):
        outFilename_model = filePrefix + '.json'
        outFilepath = os.path.join(outputDir, outFilename_model)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(outFilepath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        outFilename_weights = filePrefix + '.h5'
        outFilepath = os.path.join(outputDir, outFilename_weights)
        self.model.save_weights(outFilepath)
        print("Saved model to disk")

    def getModel(self):
        return (self.model)

    def prepareDataset_fullfile(self, filePath):
        # dataset = np.loadtxt(filePath, delimiter=',')
        dataset = pd.read_csv(filePath)
        numRows = dataset.shape[0]
        numFeatures = dataset.shape[1] - 2
        print ("numRows = ", numRows, ", numFeatures = ", numFeatures)
        self.dataset = dataset
        self.numRows = numRows
        self.numFeatures = numFeatures
        self.X = dataset.iloc[:,range(numFeatures)]
        self.y = dataset.iloc[:,[numFeatures]]
        # self.X = dataset[:,:numFeatures]
        # self.y = dataset[:,numFeatures]
        print ("X.shape = {}, y.shape = {}".format(self.X.shape, self.y.shape))

    def prepareDataSubset_fromCSV(self, datasetObj, recordIDs, csvRecordInfo, epochLen, slidingWinLen, priorSeconds, postSeconds):
        '''
        This method creates self.X and self.y matrices from all the given
        records.
        self.X is of shape (numSamples, numInputTimeSteps, numFeatures)
        self.y is of shape (numSamples, numOutputTimeSteps, numFeatures)
        Each record corresponds to an CSV file with 0 or more seizures.
        The Time step duration is determined by sample frequency when the
          EDF file was recorded.
        This method supports selectively including only those sequences that
        lead to a seizure.

        Input Parameters:
        recordIDs -- list of records from which samples have to be created
        priorSeconds, postSeconds -- determine which rows from the dataset
               will be used for training the model. e.g., if the seizure
               occured from 200 to 220 seconds and priorSeconds = 60, 
               postSeconds = 10, then the data from 140th (200-60) second to 
               230th (220+10) second will be used for training the model.

        '''
        recordsWithSeizures = []
        for recordID in recordIDs:
           if (datasetObj.recordContainsSeizure(recordID)):
                recordsWithSeizures.append(recordID)
        print ("total number of records = ", len(recordIDs))
        print ("Number of records with seizures = ", len(recordsWithSeizures))
        totalSamples = 0
        seizureVectors = {}
        for recordID in recordsWithSeizures:
            numRows = csvRecordInfo[recordID]['numRows']
            curVector = datasetObj.getExtendedSeizuresVectorCSV(recordID, epochLen, slidingWinLen, numRows, priorSeconds, postSeconds)
            seizureVectors[recordID] = curVector
            curNumRows = sum(curVector)
            print ("recordID={}, curNumRows={}".format(recordID, curNumRows))
            if (curNumRows > 0):
                totalSamples += curNumRows

        print ("total number of Samples = ", totalSamples)
        if (totalSamples <= 0):
            return (totalSamples)
        numFeatures = self.numFeatures
        allRecords_X = np.empty([totalSamples, numFeatures])
        allRecords_y = np.empty([totalSamples])
        curInd = 0
        for recordID in recordsWithSeizures:
            filePath = csvRecordInfo[recordID]['CSVpath']
            dataset = pd.read_csv(filePath)
            dataset = datasetObj.getCSVDataSubset(recordID, dataset, seizureVectors[recordID])
            dataset = dataset.values # Convert to a numpy array from pandas dataframe
            print ("dataset.shape = ", dataset.shape)
            numSamples = dataset.shape[0]
            endInd = curInd + numSamples
            allRecords_X[curInd:endInd] = dataset[:,1:numFeatures+1]
            allRecords_y[curInd:endInd] = dataset[:,numFeatures+1]
            curInd = endInd
        
        self.X = allRecords_X
        self.y = allRecords_y
        print ("self.X.shape = ", self.X.shape, ",self.y.shape = ", self.y.shape)
        return (totalSamples)

    def fit(self, epochs=50, batch_size=10, validation_split=0.33):
        self.model.fit(self.X, self.y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=2)

    def evaluate(self):
        score = self.model.evaluate(self.X, self.y, verbose=2)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

