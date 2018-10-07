import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector


# This program will read a .csv file, e.g.: LineLength.chb03_04.edf.csv,
#  and try to create an LSTM-based model to detect the seizures.

# 1. 

filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'

# generate the train data set
def generate_train_and_test_data(filePath, trainPct):
    # 80% train, 20% test data
    dataset = np.loadtxt(filePath, delimiter=',')
    X = dataset[:,:19]
    y = dataset[:,19]
    numSamples = X.shape[0]
    print ("numSamples = ", numSamples)
    numTrain = int (trainPct * numSamples)
    numTest = numSamples - numTrain
    print ("numTrain = ", numTrain, ", numTest = ", numTest, ", number of lines = ", numSamples)
    X_train = X[:numTrain]
    X_test = X[numTrain:]
    y_train = y[:numTrain]
    y_test = y[numTrain:]
    return (X_train, y_train, X_test, y_test)

def createModel_LSTM_DNN(numTimeSteps_1, numTimeSteps_2, numFeatures):
    # define LSTM
    model = Sequential()
    # input sequence length = numTimeSteps
    model.add(LSTM(150, input_shape=(numTimeSteps_1, numFeatures)))
    model.add(RepeatVector(numTimeSteps_1))
    model.add(LSTM(100, return_sequences=True))
    # model.add(LSTM(25, return_sequences=True))
    # model.add(TimeDistributed(Dense(numFeatures)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return (model)

def createModel_DNN(numSamples, numFeatures):
    # define LSTM
    model = Sequential()
    # input sequence length = numTimeSteps
    model.add(Dense(32, input_shape=(numFeatures, ), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return (model)

if __name__ == '__main__':
    (X_train, y_train, X_test, y_test) = generate_train_and_test_data(filePath, 0.8)
    print ("X_train.shape = ", X_train.shape)
    print ("y_train.shape = ", y_train.shape)
    print ("X_test.shape = ", X_test.shape)
    print ("y_test.shape = ", y_test.shape)
    # X_train = X_train[:1400, :]
    # X_train = np.reshape(X_train, (28, 50, 19))
    # # print (X_train)
    # print ("X_train.shape = ", X_train.shape)
    # y_train = y_train[:1400]
    # y_train = np.reshape(y_train, (28, 50, 1))
    # model = createModel_LSTM_DNN(50, 25, X_train.shape[2])
    
    # Fit the model
    model = createModel_DNN(20, 19)
    model.fit(X_train, y_train, epochs=50, batch_size=10)

    # evaluate the model
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    correct = 0
    # X_test = X_test[:350, :]
    # X_test = np.reshape(X_test, (7, 50, 19))
    # y_test = y_test[:350]
    y_hat = model.predict(X_test)
    print ("shape of y_hat = ", y_hat.shape)
    # print ("y_hat = ", y_hat)
    # print ("y_test = ", y_test)
    