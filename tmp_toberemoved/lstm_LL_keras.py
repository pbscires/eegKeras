import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.callbacks import ModelCheckpoint
import sys

# Function to create model, required for KerasClassifier
def create_model(numFeatures, inSeqLen, outSeqLen):
    # define LSTM
    model = Sequential()
    # input sequence length = X_numbers
    # number of features = 1
    model.add(LSTM(175, input_shape=(inSeqLen, numFeatures)))
    model.add(RepeatVector(outSeqLen))
    model.add(LSTM(150, return_sequences=True))
    model.add(TimeDistributed(Dense(numFeatures)))
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'
    # filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LL_PreIctal/chb03.csv'
    dataset = np.loadtxt(filePath, delimiter=',')

    # discard the last column which represents the occurrence of seizure
    dataset = dataset[:,:-1]
    numRows = dataset.shape[0]
    numFeatures = dataset.shape[1]
    print ("numRows = ", numRows, ", numFeatures = ", numFeatures)

    inSeqLen = int(sys.argv[1])
    outSeqLen = int(sys.argv[2])
    print ("inSeqLen = {}, outSeqLen = {}".format(inSeqLen, outSeqLen))

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

    # checkpoint
    chkp_filepath="lstm_LL_keras_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # create model
    model = create_model(numFeatures, inSeqLen, outSeqLen)
    model.fit(X, y, validation_split=0.33, epochs=50, batch_size=10, verbose=0, callbacks=callbacks_list)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_lstm_LL_keras.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_lstm_LL_keras.h5")
    print("Saved model to disk")
