import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.models import model_from_json
from keras.layers import Bidirectional
import keras.backend as K
import json
import os
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class eegLSTM(object):
    '''
        Constructor
    '''
    def __init__(self, modelName):
        self.modelName = modelName
    
    def createModel(self, inSeqLen, outSeqLen, numFeatures, lstm_units):

        self.inSeqLen = inSeqLen
        self.outSeqLen = outSeqLen
        self.numFeatures = numFeatures

        assert (len(lstm_units) >= 2), "Invalid number of LSTM Layers"
        # define LSTM
        model = Sequential()

        if (self.modelName == "encoder_decoder_sequence"):
            model.add(Bidirectional(LSTM(lstm_units[0], input_shape=(inSeqLen, numFeatures), bias_initializer='ones', name='FirstLayer')))
            # model.add(LSTM(lstm_units[0], input_shape=(inSeqLen, numFeatures)))
            model.add(RepeatVector(outSeqLen))
            for i in range(1, len(lstm_units)-1):
                model.add(LSTM(lstm_units[i], return_sequences=True))
            model.add(LSTM(lstm_units[-1], return_sequences=True)) # For the last LSTM layer
            model.add(TimeDistributed(Dense(numFeatures)))
            model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=[self.numNear, self.numFar])
            # model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
        if (self.modelName == "stacked_LSTM"):
            model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=(inSeqLen, numFeatures)))
            for i in range(1, len(lstm_units)-1):
                model.add(LSTM(lstm_units[i], return_sequences=True))
            model.add(LSTM(lstm_units[-1]))  # Last layer
            model.add(Dense(outSeqLen))
            model.compile(loss='mean_absolute_percentage_error', optimizer='sgd', metrics=['accuracy'])
            # model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])

        self.model = model
        # print (model.summary())
    
    def loadModel(self, modelFile, weightsFile):
        if (self.modelName == "encoder_decoder_sequence"):
            json_file = open(modelFile, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(weightsFile)
            loaded_model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=[self.numNear, self.numFar ])
            # loaded_model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
            self.model = loaded_model
            # self.inSeqLen = inSeqLen
            # self.outSeqLen = outSeqLen
            # self.numFeatures = numFeatures
            self.inSeqLen = loaded_model.layers[0].get_input_at(0).get_shape().as_list()[1]
            self.outSeqLen = loaded_model.layers[-1].get_output_at(0).get_shape().as_list()[1]
            self.numFeatures = loaded_model.layers[0].get_input_at(0).get_shape().as_list()[2]
            print ("inSeqLen={}, outSeqLen={}, numFeatures={}".format(self.inSeqLen, self.outSeqLen, self.numFeatures))

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
        dataset = pd.read_csv(filePath)
        dataset = dataset.values # Convert to a numpy array from pandas dataframe
        # dataset = np.loadtxt(filePath, delimiter=',')
        # discard the last column which represents the occurrence of seizure
        dataset = dataset[:,:self.numFeatures]
        # scaler = MinMaxScaler(feature_range=(0,1))
        # scaler = scaler.fit(dataset)
        # # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # dataset = scaler.transform(dataset)
        print (dataset)
        numRows = dataset.shape[0]
        # numFeatures = dataset.shape[1]
        numFeatures = self.numFeatures
        print ("numRows = ", numRows, ", numFeatures = ", numFeatures)
        self.dataset = dataset
        self.numRows = numRows
        # self.numFeatures = numFeatures

        inSeqLen = self.inSeqLen
        outSeqLen = self.outSeqLen
        numSamples = numRows - (inSeqLen + outSeqLen)
        X = np.empty([numSamples, inSeqLen, numFeatures])
        y = np.empty([numSamples, outSeqLen, numFeatures])
        for i in range(numSamples):
            inSeqEnd = i + inSeqLen
            outSeqEnd = inSeqEnd + outSeqLen
            try:
                # X[i] = np.flipud(dataset[i:inSeqEnd,:])
                X[i] = dataset[i:inSeqEnd,:]
                y[i] = dataset[inSeqEnd:outSeqEnd,:]
            except ValueError:
                print ("i = {}, inSeqEnd = {}, outSeqEnd = {}".format(i, inSeqEnd, outSeqEnd))
        
        print ("X.shape = {}, y.shape = {}".format(X.shape, y.shape))
        self.X = X
        self.y = y

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
        inSeqLen = self.inSeqLen
        outSeqLen = self.outSeqLen
        numFeatures = self.numFeatures
        allRecords_X = np.empty([totalSamples, inSeqLen, numFeatures])
        allRecords_y = np.empty([totalSamples, outSeqLen, numFeatures])
        curInd = 0
        for recordID in recordsWithSeizures:
            filePath = csvRecordInfo[recordID]['CSVpath']
            dataset = pd.read_csv(filePath)
            dataset = datasetObj.getCSVDataSubset(recordID, dataset, seizureVectors[recordID])
            dataset = dataset.values # Convert to a numpy array from pandas dataframe
            numRows = dataset.shape[0]
            numFeatures = self.numFeatures
            numSamples = numRows - (inSeqLen + outSeqLen)
            try:
                X = np.empty([numSamples, inSeqLen, numFeatures])
                y = np.empty([numSamples, outSeqLen, numFeatures])
            except ValueError:
                print ("numSamples={}, inSeqLen={}, numFeatures={}".format(numSamples, inSeqLen, numFeatures))
                print ("numRows={}, filePath={}".format(numRows, filePath))
                exit (-1)
            for i in range(numSamples):
                inSeqEnd = i + inSeqLen
                outSeqEnd = inSeqEnd + outSeqLen
                try:
                    # X[i] = np.flipud(dataset[i:inSeqEnd,:])
                    X[i] = dataset[i:inSeqEnd,1:numFeatures+1]
                    y[i] = dataset[inSeqEnd:outSeqEnd,1:numFeatures+1]
                except ValueError:
                    print ("Error occurred while trying to slice dataset")
                    print ("i = {}, inSeqEnd = {}, outSeqEnd = {}, numFeatures={}".format(i, inSeqEnd, outSeqEnd, numFeatures))
                    print (sys.exc_info())
                    exit (-1)
            
            # print ("X.shape = {}, y.shape = {}".format(X.shape, y.shape))
            endInd = curInd + X.shape[0]
            allRecords_X[curInd:endInd] = X
            allRecords_y[curInd:endInd] = y
            curInd = endInd
        self.X = allRecords_X
        self.y = allRecords_y
        print ("self.X.shape = ", self.X.shape, ",self.y.shape = ", self.y.shape)

    def prepareDataSubset_fromEDF(self, datasetObj, recordIDs, priorSeconds, postSeconds):
        '''
        This method creates self.X and self.y matrices from all the given
        records.
        self.X is of shape (numSamples, numInputTimeSteps, numFeatures)
        self.y is of shape (numSamples, numOutputTimeSteps, numFeatures)
        Each record corresponds to an EDF file with at most 1 seizure.
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
        totalSamples = 0
        recordsWithSeizures = []
        for recordID in recordIDs:
            curNumRows = datasetObj.countRowsForEDFDataSubset(recordID, priorSeconds, postSeconds)
            if (curNumRows > 0):
                totalSamples += curNumRows
                recordsWithSeizures.append(recordID)
        print ("total number of Samples = ", totalSamples)
        print ("total number of records = ", len(recordIDs))
        print ("Number of records with seizures = ", len(recordsWithSeizures))
        if (totalSamples <= 0):
            return (totalSamples)
        inSeqLen = self.inSeqLen
        outSeqLen = self.outSeqLen
        numFeatures = self.numFeatures
        allRecords_X = np.empty([totalSamples, inSeqLen, numFeatures])
        allRecords_y = np.empty([totalSamples, outSeqLen, numFeatures])
        curInd = 0

        for recordID in recordsWithSeizures:
            dataset = datasetObj.getEDFDataSubset(recordID, priorSeconds, postSeconds)
            numRows = dataset.shape[0]
            numFeatures = self.numFeatures
            numSamples = numRows - (inSeqLen + outSeqLen)
            X = np.empty([numSamples, inSeqLen, numFeatures])
            y = np.empty([numSamples, outSeqLen, numFeatures])
            for i in range(numSamples):
                inSeqEnd = i + inSeqLen
                outSeqEnd = inSeqEnd + outSeqLen
                try:
                    # X[i] = np.flipud(dataset[i:inSeqEnd,:])
                    X[i] = dataset[i:inSeqEnd,:]
                    y[i] = dataset[inSeqEnd:outSeqEnd,:]
                except ValueError:
                    print ("i = {}, inSeqEnd = {}, outSeqEnd = {}".format(i, inSeqEnd, outSeqEnd))
            
            print ("X.shape = {}, y.shape = {}".format(X.shape, y.shape))
            endInd = curInd + X.shape[0]
            allRecords_X[curInd:endInd] = X
            allRecords_y[curInd:endInd] = y
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
        # y_hat = self.model.predict(self.X)
        # for i in range(self.y.shape[0]):
        #     print (y_hat[i].transpose(), self.y[i].transpose(), (y_hat[i] - self.y[i]).transpose())
            # print ("y_hat[0] = ", y_hat[0])
            # print ("y = ", self.y)
    
    def numNear(self, y_true, y_pred):
        # if ((abs(y_true - y_pred) / y_true) < 0.1):
        #     count = 1
        return (K.sum(K.cast(K.less_equal(K.abs((y_true-y_pred) / y_pred), 0.1), 'int32')))

    def numFar(self, y_true, y_pred):
        # if ((abs(y_true - y_pred) / y_true) < 0.1):
        #     count = 1
        return (K.sum(K.cast(K.greater_equal(K.abs((y_true-y_pred) / y_pred), 0.5), 'int32')))
