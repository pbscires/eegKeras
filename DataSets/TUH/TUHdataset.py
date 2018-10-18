import numpy as np
import pandas as pd
from pandas import DataFrame
import pyedflib
import re
import sys
import os
import math
import json

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
        patientInfo = {}
        recordInfo = {}
        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(self.rootDir):
            pathComponents = root.split(os.sep)
            # print ("root = ", root)
            if (re.search("^\d\d\d$", pathComponents[-1])):
                # print ("List of patients = ", dirs)
                numPatients += len(dirs)
                for patientID in dirs:
                    patientInfo[patientID] = {}
                    patientInfo[patientID]['sessions'] = []
                    patientInfo[patientID]['records'] = []
            if pathComponents[-1] in patientInfo.keys():
                patientID = pathComponents[-1]
                for sessionID in dirs:
                    patientInfo[patientID]['sessions'].append(sessionID)
                    numPatientSessions += 1
            for filename in files:
                if (re.search("\.edf$", filename) != None):
                    edfFilePath = os.path.join(root, filename)
                    patientID = pathComponents[-2]
                    sessionID = pathComponents[-1]
                    recordID = os.path.basename(edfFilePath)
                    recordID = os.path.splitext(recordID)[0]
                    patientInfo[patientID]['records'].append(recordID)
                    recordInfo[recordID] = {}
                    recordInfo[recordID]['edfFilePath'] = edfFilePath
                    recordInfo[recordID]['patientID'] = patientID
                    recordInfo[recordID]['sessionID'] = sessionID
                    numEdfs += 1
            # print((len(pathComponents) - 1) * '---', os.path.basename(root))
            # for file in files:
            #     print(len(pathComponents) * '---', file) 
        print ("Total number of patients = ", numPatients)
        print ("number of unique patients = ", len(patientInfo))
        print ("number of patient sessions = ", numPatientSessions)
        print ("Total number of EDFs = ", numEdfs)

        self.numPatients = numPatients
        self.patientInfo = patientInfo
        self.numEdfs = numEdfs
        self.recordInfo = recordInfo
        for recordID in self.recordInfo.keys():
            self.getEdfSummary(recordID)
        
        # for patientID in patientInfo.keys():
        #     for recordID in patientInfo[patientID]['records']:
        #         print (self.recordInfo[recordID])
        
    def getEdfSummary(self, recordID):
        filePath = self.recordInfo[recordID]['edfFilePath']
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
        self.recordInfo[recordID]['channelLabels'] = np.delete(channelLabels, columnsToDel, axis=0).tolist()
        self.recordInfo[recordID]['other_labels'] = otherLabels
        self.recordInfo[recordID]['numChannels'] = np.int32(numChannels).item()
        self.recordInfo[recordID]['numSamples'] = np.int32(f.getNSamples()[0]).item()
        self.recordInfo[recordID]['sampleFrequency'] = np.int32(f.getSampleFrequency(0)).item()

    def getRecordData(self, recordID):
        filePath = self.recordInfo[recordID]['edfFilePath']
        numChannels = self.recordInfo[recordID]['numChannels']
        numSamples = self.recordInfo[recordID]['numSamples']
        channelLabels = self.recordInfo[recordID]['channelLabels']
        sigbufs = np.zeros((numChannels, numSamples))
        f = pyedflib.EdfReader(filePath)
        for i in np.arange(numChannels):
            try:
                sigbufs[i, :] = f.readSignal(i)
            except ValueError:
                print ("Failed to read channel {} with name {}".format(i, channelLabels[i]))
        # sigbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        sigbufs = sigbufs.transpose()
        f._close()
        del(f)
        # print (sigbufs)
        return (sigbufs)
    
    def getSeizuresSummary(self):
        '''
        Read the CSV file and summarize the seizure information on per-record basis
        '''
        with open(self.csvFilePath, 'rb') as f:
            df_out = pd.read_excel(f, sheet_name='train', usecols="A:O", dtype=object)
        
        print (df_out)
        filenames = df_out['Filename']
        filenameCol = df_out.columns.get_loc('Filename')
        seizureStartCol = filenameCol + 1
        seizureEndCol = seizureStartCol + 1
        seizureTypeCol = seizureEndCol + 1
        print ("seizureStartCol = {}, seizureEndCol = {}".format(seizureStartCol, seizureEndCol))
        recordIDs = []
        rowIndex = -1
        for filename in filenames:
            rowIndex += 1
            if (not isinstance(filename, str)):
                continue
            # print ("filename = ", filename)
            if (re.search('\.tse$', filename) != None):
                recordID = os.path.basename(filename)
                recordID = os.path.splitext(recordID)[0]
                # try:
                seizureStartTime = df_out.iloc[rowIndex,seizureStartCol]
                seizureEndTime = df_out.iloc[rowIndex, seizureEndCol]
                seizureType = df_out.iloc[rowIndex, seizureTypeCol]
                if (not math.isnan(seizureStartTime)):
                    self.recordInfo[recordID]['seizureStart'] = np.float32(seizureStartTime).item()
                    self.recordInfo[recordID]['seizureEnd'] = np.float32(seizureEndTime).item()
                    self.recordInfo[recordID]['seizureType'] = seizureType
                # except:
                #     print ("rowIndex = ", rowIndex)
                # recordIDs.append(recordID)
        # print (recordIDs)
        # Get the column index for 'Seizure Time'
        # for patientID in self.patientInfo.keys():
        #     for recordID in self.patientInfo[patientID]['records']:
        #         print (self.recordInfo[recordID])
    
    def saveToJsonFile(self, filePath):
        print ("Saving to the json file ", filePath)

        with open(filePath, 'w') as f:
            f.write("{\n")
            for patientID in self.patientInfo.keys():
                for recordID in self.patientInfo[patientID]['records']:
                    try:
                        f.write("\"" + recordID + "\" : ")
                        f.write(json.dumps(self.recordInfo[recordID]))
                        f.write(",\n")
                    except TypeError:
                        print ("Record = ", self.recordInfo[recordID])
            f.write("\"EOFmarker\" : \"EOF\" }\n")

    def loadJsonFile(self, filePath):
        print ("Loading from json file ", filePath)
        f = open(filePath, 'r')
        self.recordInfo = json.load(f)
        f.close()
        del (self.recordInfo['EOFmarker'])

        # Build self.patientInfo
        self.numEdfs = len(self.recordInfo)
        patientInfo = {}
        for recordID in self.recordInfo.keys():
            patientID = self.recordInfo[recordID]['patientID']
            sessionID = self.recordInfo[recordID]['sessionID']
            if (patientID not in patientInfo.keys()):
                patientInfo[patientID] = {}
                patientInfo[patientID]['records'] = [recordID]
                patientInfo[patientID]['sessions'] = [sessionID]
            else:
                patientInfo[patientID]['records'].append(recordID)
                patientInfo[patientID]['sessions'].append(sessionID)
        #     print ("self.recordInfo[" + recordID + "][\'numChannels\'] = ", self.recordInfo[recordID]['numChannels'])
        self.patientInfo = patientInfo
        self.numPatients = len(self.patientInfo)

    def isSeizurePresent(self, recordID, epochNum, epochLen, slidingWindowLen):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return False
        # Convert epochNum to start and end datetime objects
#         print ("recordID = ", recordID, ", epochNum =", epochNum, ", epochLen = ", epochLen,
#                ", slidingWindowLen = ", slidingWindowLen)
        epochStart = float(epochNum * slidingWindowLen / 1000)
        epochEnd = epochStart + float(epochLen / 1000)

        seizureStart = self.recordInfo[recordID]['seizureStart']
        seizureEnd = self.recordInfo[recordID]['seizureEnd']
        # seizureType = self.recordInfo[recordID]['seizureType']

        if (( (epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
            ( (epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
            return True

        return False
    
    def getSeizuresVector(self, recordID, epochLen, slidingWindowLen, numEpochs):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''

        seizuresVector = np.zeros((numEpochs), dtype=np.int32)
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return seizuresVector
        else:
            seizureStart = self.recordInfo[recordID]['seizureStart']
            seizureEnd = self.recordInfo[recordID]['seizureEnd']
            # seizureType = self.recordInfo[recordID]['seizureType']
            for i in range(numEpochs):
                epochStart = float(i * slidingWindowLen / 1000)
                epochEnd = epochStart + float(epochLen / 1000)
                if (( (epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
                    ( (epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
                        seizuresVector[i] = 1
            return seizuresVector
    
    def getSeizureStartEndTimes(self, recordID):
        seizureStart = self.recordInfo[recordID]['seizureStart']
        seizureEnd = self.recordInfo[recordID]['seizureEnd']
        return (seizureStart, seizureEnd)
