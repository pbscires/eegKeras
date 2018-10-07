import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import os

class StackedLSTM(object):
    '''
        Constructor
    '''
    def __init__(self, modelName, inSeqLen, outSeqLen, numFeatures):

        self.modelName = modelName
        self.inSeqLen = inSeqLen
        self.outSeqLen = outSeqLen
        self.numFeatures = numFeatures

        # define LSTM
        model = Sequential()

        if (modelName == "encoder_decoder_sequence"):
            model.add(LSTM(175, input_shape=(inSeqLen, numFeatures)))
            model.add(RepeatVector(outSeqLen))
            model.add(LSTM(150, return_sequences=True))
            model.add(TimeDistributed(Dense(numFeatures)))
            model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])

        self.model = model
    
    def getModel(self):
        return (self.model)
    
    def prepareDataset_1file(self, filePath):
        dataset = np.loadtxt(filePath, delimiter=',')
        # discard the last column which represents the occurrence of seizure
        dataset = dataset[:,:-1]
        numRows = dataset.shape[0]
        numFeatures = dataset.shape[1]
        print ("numRows = ", numRows, ", numFeatures = ", numFeatures)
        self.dataset = dataset
        self.numRows = numRows
        self.numFeatures = numFeatures

        inSeqLen = self.inSeqLen
        outSeqLen = self.outSeqLen
        numSamples = numRows - (inSeqLen + outSeqLen)
        X = np.empty([numSamples, inSeqLen, numFeatures])
        y = np.empty([numSamples, outSeqLen, numFeatures])
        for i in range(numSamples):
            inSeqEnd = i + inSeqLen
            outSeqEnd = inSeqEnd + outSeqLen
            try:
                X[i] = dataset[i:inSeqEnd,:]
                y[i] = dataset[inSeqEnd:outSeqEnd,:]
            except ValueError:
                print ("i = {}, inSeqEnd = {}, outSeqEnd = {}".format(i, inSeqEnd, outSeqEnd))
        
        print ("X.shape = {}, y.shape = {}".format(X.shape, y.shape))
        self.X = X
        self.y = y
    
    def fit(self):
        # # checkpoint
        # chkp_filepath="lstm_LL_keras_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        # checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # callbacks_list = [checkpoint]

        # self.model.fit(X, y, validation_split=0.33, epochs=50, batch_size=10, verbose=0, callbacks=callbacks_list)
        self.model.fit(self.X, self.y, validation_split=0.33, epochs=50, batch_size=10, verbose=0)

    def save(self, outputDir, filePrefix):
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



