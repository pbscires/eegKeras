import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class DNN(object):
    '''
        Constructor
    '''
    def __init__(self, modelName, numFeatures):
        self.modelName = modelName
        self.numFeatures = numFeatures

        model = Sequential()
        if (modelName == "DNN_3layers"):
            model.add(Dense(12, input_dim=numFeatures, activation='relu')) 
            model.add(Dense(8, activation='relu')) 
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = model

    def getModel(self):
        return (self.model)