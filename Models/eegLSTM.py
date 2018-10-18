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
            model.add(Bidirectional(LSTM(lstm_units[0], input_shape=(inSeqLen, numFeatures), bias_initializer='ones')))
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
    
    def loadModel(self, modelFile, weightsFile, inSeqLen, outSeqLen, numFeatures):
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
            self.inSeqLen = inSeqLen
            self.outSeqLen = outSeqLen
            self.numFeatures = numFeatures

    def saveModel(self, outputDir, filePrefix):
        outFilename_model = filePrefix + '_LSTM.json'
        outFilepath = os.path.join(outputDir, outFilename_model)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(outFilepath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        outFilename_weights = filePrefix + '_LSTM.h5'
        outFilepath = os.path.join(outputDir, outFilename_weights)
        self.model.save_weights(outFilepath)
        print("Saved model to disk")

    def getModel(self):
        return (self.model)
    
    def prepareDataset_1file(self, filePath):
        dataset = np.loadtxt(filePath, delimiter=',')
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

    def _getNumRows(self, tuhd, recordID, priorSeconds, postSeconds):
        if ('seizureStart' in tuhd.recordInfo[recordID].keys()):
            (seizureStart, seizureEnd) = tuhd.getSeizureStartEndTimes(recordID)
        else:
            return (0) # This record has no seizure data

        numRows = tuhd.recordInfo[recordID]['numSamples']
        numFeatures = tuhd.recordInfo[recordID]['numChannels']
        print ("numRows = ", numRows, ", numFeatures = ", numFeatures)
        startSec = seizureStart - priorSeconds
        endSec = seizureEnd + postSeconds
        startRowNum = int(startSec * tuhd.recordInfo[recordID]['sampleFrequency'])
        endRowNum = int(endSec * tuhd.recordInfo[recordID]['sampleFrequency'])
        if (startRowNum < 0):
            startRowNum = 0
        if (endRowNum > numRows):
            endRowNum = numRows
        numRows = endRowNum - startRowNum + 1
        print ("numRows = ", numRows)
        return (numRows)

    def _getDataset(self, tuhd, recordID, priorSeconds, postSeconds):
        dataset = tuhd.getRecordData(recordID)
        print (dataset)
        if ('seizureStart' in tuhd.recordInfo[recordID].keys()):
            (seizureStart, seizureEnd) = tuhd.getSeizureStartEndTimes(recordID)
        else:
            return (None) # This record has no seizure data
        
        numRows = dataset.shape[0]
        numFeatures = self.numFeatures
        print ("numRows = ", numRows, ", numFeatures = ", numFeatures)
        startSec = seizureStart - priorSeconds
        endSec = seizureEnd + postSeconds
        startRowNum = int(startSec * tuhd.recordInfo[recordID]['sampleFrequency'])
        endRowNum = int(endSec * tuhd.recordInfo[recordID]['sampleFrequency'])
        if (startRowNum < 0):
            startRowNum = 0
        if (endRowNum > numRows):
            endRowNum = numRows
        numRows = endRowNum - startRowNum + 1
        print ("numRows = ", numRows)
        dataset = dataset[startRowNum:endRowNum+1]
        return (dataset)

    def prepareDataset_fromTUHedf(self, tuhd, recordIDs, priorSeconds, postSeconds):
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
            curNumRows = self._getNumRows(tuhd, recordID, priorSeconds, postSeconds)
            if (curNumRows > 0):
                totalSamples += curNumRows
                recordsWithSeizures.append(recordID)
        print ("total number of Samples = ", totalSamples)
        print ("total number of records = ", len(recordIDs))
        print ("Number of records with seizures = ", len(recordsWithSeizures))
        inSeqLen = self.inSeqLen
        outSeqLen = self.outSeqLen
        numFeatures = self.numFeatures
        allRecords_X = np.empty([totalSamples, inSeqLen, numFeatures])
        allRecords_y = np.empty([totalSamples, outSeqLen, numFeatures])
        curInd = 0

        for recordID in recordsWithSeizures:
            dataset = self._getDataset(tuhd, recordID, priorSeconds, postSeconds)
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
    
    def applyPCAToDataset(self, tuhd, recordID):
        '''
        This method was created to test the application of PCA to raw data.
        This is not the right way to do PCA; it is better to do PCA after the
        features are extracted.
        So, this method may eventually be removed.
        '''
        dataset = tuhd.getRecordData(recordID)
        seizuresVec = tuhd.getSeizuresVectorEDF(recordID)
        X_train, X_test, y_train, y_test = train_test_split(dataset, seizuresVec, 
                    test_size=0.2, random_state=0)
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        pca = PCA(n_components=4)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        print ("explained_variance = ", explained_variance)
        print (X_train)
   
    def fit(self, epochs=50, batch_size=10):
        self.model.fit(self.X, self.y, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=2)

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
