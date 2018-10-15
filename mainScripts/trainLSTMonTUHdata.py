import numpy as np
import pyedflib
import re
import sys
import os

class EEGRecord(object):
    '''
    This class represents a single EEG record. i.e. one .edf file wich contains
    values for 23 channels over a period of 1 hour. 
    '''


    def __init__(self, dirPath, subjectName):
        '''
        Constructor
        '''
        self.dirPath = dirPath
        # self.edfFilePath = os.path.join(dirPath, )
        self.subjectName = subjectName
    
    def loadFile(self):
        f = pyedflib.EdfReader(self.filePath)
        self.numChannels = f.signals_in_file
        print ("number of signals in file = ", self.numChannels)
        self.signal_labels = f.getSignalLabels()
        print ("signal labels = ", self.signal_labels)
        # EEG data is valid only when the signal label has '-REF' at the end
        self.other_labels = []
        columnsToDel = []
        for i in np.arange(self.numChannels):
            if (re.search('\-REF', self.signal_labels[i]) == None):
                self.other_labels.append(self.signal_labels[i])
                columnsToDel.append(i)
                self.numChannels -= 1
        self.signal_labels = np.delete(self.signal_labels, columnsToDel, axis=0)

        # numSamples = 3600 * 256 = 921,600
        self.numSamples = f.getNSamples()[0]
        # sampleFrequency = 256
        self.sampleFrequency = f.getSampleFrequency(0)
        print ("numSample = {}, sampleFrequency = {}".format(self.numSamples, self.sampleFrequency))
        self.sigbufs = np.zeros((self.numChannels, self.numSamples))
        for i in np.arange(self.numChannels):
            try:
                self.sigbufs[i, :] = f.readSignal(i)
            except ValueError:
                print ("Failed to channel {} with name {}".format(i, self.signal_labels[i]))
        # sigbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        self.sigbufs = self.sigbufs.transpose()
        print (self.sigbufs)

if __name__ == '__main__':
    filePath = sys.argv[1]
    subjectName = os.path.basename(filePath)
    print ("filePath = {}, subjectName = {}".format(filePath, subjectName))
    tuhEegRec = EEGRecord(sys.argv[1], subjectName)
    tuhEegRec.loadFile()