import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import json
import os

class DNN(object):
    '''
        Constructor
    '''
    def __init__(self, modelName):
        self.modelName = modelName
    
    def createModel(self, numFeatures):
        self.numFeatures = numFeatures

        model = Sequential()
        if (self.modelName == "Classifier_3layers"):
            model.add(Dense(12, input_dim=numFeatures, activation='relu')) 
            model.add(Dense(8, activation='relu')) 
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

    def saveModel(self, outputDir, filePrefix):
        outFilename_model = filePrefix + '_DNN.json'
        outFilepath = os.path.join(outputDir, outFilename_model)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(outFilepath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        outFilename_weights = filePrefix + '_DNN.h5'
        outFilepath = os.path.join(outputDir, outFilename_weights)
        self.model.save_weights(outFilepath)
        print("Saved model to disk")

    def getModel(self):
        return (self.model)

    def prepareDataset_1file(self, filePath):
        dataset = np.loadtxt(filePath, delimiter=',')
        numRows = dataset.shape[0]
        numFeatures = dataset.shape[1] - 1
        print ("numRows = ", numRows, ", numFeatures = ", numFeatures)
        self.dataset = dataset
        self.numRows = numRows
        self.numFeatures = numFeatures
        self.X = dataset[:,:numFeatures]
        self.y = dataset[:,numFeatures]
        print ("X.shape = {}, y.shape = {}".format(self.X.shape, self.y.shape))

    def fit(self):
        self.model.fit(self.X, self.y, validation_split=0.33, epochs=50, batch_size=10, verbose=2)

    def evaluate(self):
        score = self.model.evaluate(self.X, self.y, verbose=2)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

