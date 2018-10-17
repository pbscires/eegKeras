'''

'''
from scipy import signal
import numpy as np
import pandas as pd
from util.ElapsedTime import ElapsedTime
from multiprocessing import Pool


class FFT(object):
    '''
    Extract Fast Fourier Transform feature from the edf files
    '''


    def __init__(self, epochLength=10000, slidingWindowLen=2000, 
                 startingFreq=0.5, endingFreq=25.0, numDivisions=8):
        '''
        Constructor
        '''
        # Sliding window length should be < epocLength
        print ("epochLength = ", epochLength, "slidingWindowLen = ", slidingWindowLen)
        if (slidingWindowLen > epochLength):
            print ("Invalid values for sliding window length and/or epoch length")
            exit(-1)
        self.epochLength = epochLength
        self.slidingWindowLen = slidingWindowLen
        self.startingFreq = startingFreq
        self.endingFreq = endingFreq
        self.numDivisions = numDivisions
        self.freqBands = []
        freqPerDiv = (endingFreq-startingFreq)/numDivisions
        for i in range(numDivisions):
            freq_strt = startingFreq + (i*freqPerDiv)
            freq_end = freq_strt + freqPerDiv
            self.freqBands.append((freq_strt, freq_end))
        print ("freqBands = ", self.freqBands)

    def calculateFFTperEpoch(self, i):
        epochStart = i * self.numSamplesPerWindow
        epochEnd = epochStart + self.numSamplesPerEpoch
        fftArr = np.zeros(self.numChannels * self.numDivisions)
        for j in range(self.numChannels):
            oneEpoch = self.sigbufs[epochStart:epochEnd, j]
            (f, Pxx) = signal.periodogram(oneEpoch, self.sampleFrequency)
            for k in range(len(f)):
                for l in range(self.numDivisions):
                    if ((f[k] >= self.freqBands[l][0]) and (f[k] < self.freqBands[l][1])):
                        fftArr[(j*self.numDivisions)+l] += Pxx[k]
        return (fftArr)

        
    def extractFeatureMultiProcessing(self, sigbufs, signal_labels, sampleFrequency):
        '''
        Extract the Line length feature from the given input file to the given output file.
        Input file is expected to be in the EDF file format.
        Output file is CSV file with 2 values per row -- epoch number and LineLength value
        '''
        self.sigbufs = sigbufs
        self.sampleFrequency = sampleFrequency
        numSamples = sigbufs.shape[0]
        numChannels = sigbufs.shape[1]
        self.numChannels = numChannels
        numSamplesPerEpoch = int(sampleFrequency * self.epochLength / 1000)
        print ("numSamples = ", numSamples, ", sampleFrequency = ", sampleFrequency)
#         oneEpoch = np.delete(sigbufs, list(range(1,numChannels)), axis=1)
#         numSamplesPerEpoch = 128
        numSamplesPerWindow = int(sampleFrequency * self.slidingWindowLen / 1000)
        self.numSamplesPerWindow = numSamplesPerWindow
        self.numSamplesPerEpoch = numSamplesPerEpoch

        numEpochWindows = int (numSamples / numSamplesPerWindow)
        while (((numEpochWindows * numSamplesPerWindow) + numSamplesPerEpoch) > numSamples):
            numEpochWindows -= 1
        self.numEpochWindows = numEpochWindows
        timer1 = ElapsedTime()
        timer1.reset()
        fftMat = np.zeros((numEpochWindows, numChannels * self.numDivisions))
        p = Pool()
        fftArrList = p.map(self.calculateFFTperEpoch, list(range(numEpochWindows)))
        p.close()
        print ("len of fftArrList = ", len(fftArrList))
        i = 0
        for arr in fftArrList:
            fftMat[i,:] = arr
            i += 1
        # self.fftMat = fftMat
        print ("fftMat shape = ", fftMat.shape, "len(signal_labels) = ", len(signal_labels))
        self.fftDf = pd.DataFrame(data = fftMat)
        print ("Time taken for processing one file = ", timer1.timeDiff())


    def extractFeature(self, sigbufs, signal_labels, sampleFrequency):
        '''
        Extract the Line length feature from the given input file to the given output file.
        Input file is expected to be in the EDF file format.
        Output file is CSV file with 2 values per row -- epoch number and LineLength value
        '''
        numSamples = sigbufs.shape[0]
        numChannels = sigbufs.shape[1]
        self.numChannels = numChannels
        numSamplesPerEpoch = int(sampleFrequency * self.epochLength / 1000)
        print ("numSamples = ", numSamples, ", sampleFrequency = ", sampleFrequency)
#         oneEpoch = np.delete(sigbufs, list(range(1,numChannels)), axis=1)
#         numSamplesPerEpoch = 128
        numSamplesPerWindow = int(sampleFrequency * self.slidingWindowLen / 1000)

        numEpochWindows = int (numSamples / numSamplesPerWindow)
        while (((numEpochWindows * numSamplesPerWindow) + numSamplesPerEpoch) > numSamples):
            numEpochWindows -= 1
        self.numEpochWindows = numEpochWindows

        timer1 = ElapsedTime()
        timer1.reset()
        fftMat = np.zeros((numEpochWindows, numChannels * self.numDivisions))
        for i in range(numEpochWindows):
            epochStart = i * numSamplesPerWindow
            epochEnd = epochStart + numSamplesPerEpoch
            for j in range(numChannels):
                oneEpoch = sigbufs[epochStart:epochEnd, j]
#                 print ("shape of oneEpoch = ", oneEpoch.shape, ", oneEpoch = ", oneEpoch)
                (f, Pxx) = signal.periodogram(oneEpoch, sampleFrequency)
#                 print ("f.shape = ", f.shape, ", Pxx.shape = ", Pxx.shape)
#                 print ("f = ", f)
#                 print ("Pxx = ", Pxx)
#                 new_Pxx = 10 * np.log10(Pxx)
                for k in range(len(f)):
                    for l in range(self.numDivisions):
                        if ((f[k] >= self.freqBands[l][0]) and (f[k] < self.freqBands[l][1])):
                            fftMat[i, (j*self.numDivisions)+l] += Pxx[k]
#                             fftMat[i, j, l] += new_Pxx[k]
            if (i % 100 == 0):
                print ("i = ", i, ",elapsed time = ", timer1.timeDiff())
#                 for i in range(len(f)):
#                     if (Pxx[i] > 10.0):
#                         print ("f[", i, "] = ", f[i], ", Pxx[", i, "] = ", new_Pxx[i])
        # self.fftMat = fftMat
        self.fftDf = pd.DataFrame(data = fftMat, columns = signal_labels)
        print (self.fftDf.shape)

    def saveFFTWithSeizureInfo(self, outFilePath, tuhDataObj, recordID):
        numEpochs = self.fftDf.shape[0]
        numChannels = self.fftDf.shape[1]
        print ("numEpochs = ", numEpochs, ", numChannels = ", numChannels)
        fileToWrite = outFilePath
        seizuresVector = tuhDataObj.getSeizuresVector(recordID, self.epochLength, self.slidingWindowLen, numEpochs)
        self.fftDf['SeizurePresent'] = seizuresVector
        self.fftDf.to_csv(fileToWrite)
