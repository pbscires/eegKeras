import numpy as np
import pyedflib
import re
import sys
import os

class TUHdataset(object):
    '''
    This class represents a single EEG record. i.e. one .edf file wich contains
    values for 23 channels over a period of 1 hour. 
    '''


    def __init__(self, rootDir, csvFilePath):
        '''
        Constructor
        '''
        self.rootDir = rootDir
        print ("Top level directory for the dataset = ", rootDir)
        self.csvFilePath = csvFilePath
        print ("CSV File = ", csvFilePath)
        # self.edfFilePath = os.path.join(dirPath, )
        # self.subjectName = subjectName
    
    def summarizeDatset(self):
        '''
        Print various summary information about the dataset, based on the root directory.

        Filename:
            edf/dev_test/01_tcp_ar/002/00000258/s002_2003_07_21/00000258_s002_t000.edf

            Components:
            edf: contains the edf data

            dev_test: part of the dev_test set (vs.) train

            01_tcp_ar: data that follows the averaged reference (AR) configuration,
                        while annotations use the TCP channel configutation

            002: a three-digit identifier meant to keep the number of subdirectories
                in a directory manageable. This follows the TUH EEG v1.1.0 convention.

            00000258: official patient number that is linked to v1.1.0 of TUH EEG

            s002_2003_07_21: session two (s002) for this patient. The session
                            was archived on 07/21/2003.

            00000258_s002_t000.edf: the actual EEG file. These are split into a series of
                        files starting with t000.edf, t001.edf, ... These
                        represent pruned EEGs, so the original EEG is 
                        split into these segments, and uninteresting
                        parts of the original recording were deleted
                        (common in clinical practice).

            The easiest way to access the annotations is through the spreadsheet
            provided (_SEIZURES_*.xlsx). This contains the start and stop time
            of each seizure event in an easy to understand format. Convert the
            file to .csv if you need a machine-readable version.
        '''
        # First summarize from the directory
        numPatients = 0
        numPatientSessions = 0
        numEdfs = 0
        numEdfs = 0
        patientDirs = {}
        patientSessions = {}
        sessionEdfs = {}
        edfInfo = {}
        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(self.rootDir):
            pathComponents = root.split(os.sep)
            # print ("root = ", root)
            if (re.search("^\d\d\d$", pathComponents[-1])):
                # print ("List of patients = ", dirs)
                numPatients += len(dirs)
                for patientID in dirs:
                    patientSessions[patientID] = []
            if pathComponents[-1] in patientSessions.keys():
                patientID = pathComponents[-1]
                for sessionID in dirs:
                    patientSessions[patientID].append(os.path.join(root, sessionID))
                    numPatientSessions += 1
            for filename in files:
                if (re.search("\.edf$", filename) != None):
                    edfFilePath = os.path.join(root, filename)
                    patientID = pathComponents[-2]
                    sessionID = pathComponents[-1]
                    edfInfo[edfFilePath] = {}
                    edfInfo[edfFilePath]['patientID'] = patientID
                    edfInfo[edfFilePath]['sessionID'] = sessionID
                    numEdfs += 1
                    if (patientID not in sessionEdfs.keys()):
                        sessionEdfs[patientID] = [edfFilePath]
                    else:
                        sessionEdfs[patientID].append(edfFilePath)
            # print((len(pathComponents) - 1) * '---', os.path.basename(root))
            # for file in files:
            #     print(len(pathComponents) * '---', file) 
        print ("Total number of patients = ", numPatients)
        print ("number of unique patients = ", len(patientSessions))
        print ("number of patient sessions = ", numPatientSessions)
        print ("number of session edfs = ", len(sessionEdfs))
        print ("Total number of EDFs = ", numEdfs)

        self.numPatients = numPatients
        self.patientSessions = patientSessions
        self.sessionEdfs = sessionEdfs
        self.numEdfs = numEdfs
        self.edfInfo = edfInfo
        for filePath in self.edfInfo.keys():
            self.getEdfSummary(filePath)
        
        for patientID in sessionEdfs.keys():
            for edfFilePath in sessionEdfs[patientID]:
                print ("EDF summary for patient {} and edfFilePath {} is:".format(patientID, edfFilePath))
                print (self.edfInfo[edfFilePath])
    
    def getEdfSummary(self, filePath):
        f = pyedflib.EdfReader(filePath)
        numChannels = f.signals_in_file
        channelLabels = f.getSignalLabels()
        otherLabels = []
        columnsToDel = []
        for i in np.arange(numChannels):
            if (re.search('\-REF', channelLabels[i]) == None):
                otherLabels.append(channelLabels[i])
                columnsToDel.append(i)
                numChannels -= 1
        self.edfInfo[filePath]['channelLabels'] = np.delete(channelLabels, columnsToDel, axis=0)
        self.edfInfo[filePath]['other_labels'] = otherLabels
        self.edfInfo[filePath]['numChannels'] = numChannels
        self.edfInfo[filePath]['numSamples'] = f.getNSamples()[0]
        self.edfInfo[filePath]['sampleFrequency'] = f.getSampleFrequency(0)

    def loadFile(self, filePath):
        f = pyedflib.EdfReader(filePath)
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
