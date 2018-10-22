import sys
from Models import eegLSTM
import json
import os
from util.TrainingConfigReader import TrainingConfigReader

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Error! Invalid number of arguments")
        exit (-1)
    configFile = sys.argv[1]
    print ("ConfigFile = {}".format(configFile))
    cfgReader = TrainingConfigReader(configFile)
    trainingDataTopDir = cfgReader.getTrainingDataDir()
    trainingFiles = cfgReader.getTrainingFiles()
    modelOutputDir = cfgReader.getModelOutputDir()
    savedModelFilePrefix = cfgReader.getSavedModelPrefix()
    lstmLayers = cfgReader.getLSTMLayers()
    inSeqLen, outSeqLen = cfgReader.getSeqLens()

    print ("trainingDataTopDir = ", trainingDataTopDir)
    print ("trainingFiles = ", trainingFiles)
    print ("modelOutputDir = ", modelOutputDir)
    print ("savedModelFilePrefix = ", savedModelFilePrefix)
    print ("LSTM layers = ", lstmLayers)
    print ("inSeqLen = {}, outSeqLen = {}".format(inSeqLen, outSeqLen))

    # numFeatures = 19
    numFeatures = 168
    lstmObj = eegLSTM.eegLSTM("encoder_decoder_sequence")
    # lstmObj = eegLSTM.eegLSTM("stacked_LSTM")
    lstmObj.createModel(inSeqLen, outSeqLen, numFeatures, lstmLayers)

    if (len(trainingFiles) == 1):
        trainingFile = trainingFiles[0]
        print ("trainingFile = ", trainingFile)
<<<<<<< HEAD
        print ("Preparing training file")
        trainingSections = np.empty([len(trainingFile), len(trainingFile[0])])
        for row in trainingFile:
            if row[]
        stackedLSTM.prepareDataset_1file(os.path.join(trainingDataTopDir, trainingFile))
        stackedLSTM.fit()
        stackedLSTM.saveModel(modelOutputDir, savedModelFilePrefix)
=======
        lstmObj.prepareDataset_1file(os.path.join(trainingDataTopDir, trainingFile))
        lstmObj.fit(epochs=20, batch_size=10)
        lstmObj.saveModel(modelOutputDir, savedModelFilePrefix)
>>>>>>> 678ec5e1b7a3c2f688a3aa133dfeda85477de0ed
