import numpy as np
import pandas as pd
from pandas import DataFrame
import pyedflib
import re
import sys
import os
import math
import json
from DataSets.BaseDataset import BaseDataset

class CHBdataset(BaseDataset):
    '''
    This class contains several methods that can process the EDF files in the CHB dataset.
    It serves 2 purposes:
    1) to create (and later load) a json file that contains the metadata for the entire TUH data.
    2) to read a single EDF file and extract signal values from that file.
    '''
    def __init__(self, rootDir, seizureFilePath):
        '''
        Constructor
        '''
        self.rootDir = rootDir
        print ("Top level directory for the dataset = ", rootDir)
        self.seizureFilePath = seizureFilePath
        print ("Seizures JSON File = ", seizureFilePath)

    def summarizeDatset(self):
        '''
        Print various summary information about the dataset, based on the root directory.
        '''
        # First summarize from the directory
        numPatients = 0
        numEdfs = 0
        patientInfo = {}
        recordInfo = {}
        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(self.rootDir):
            # print ("root = ", root)
            for filename in files:
                if (re.search("^chb\d\d_\d\d\.edf$", filename)):
                    # print ("List of patients = ", dirs)
                    edfFilePath = os.path.join(root, filename)
                    m = re.match("^(chb\d\d)_(\d\d)\.edf$", filename)
                    patientID = m.group(1)
                    sessionID = m.group(2)
                    recordID = patientID + '_' + sessionID
                    print ("recordID = {}, patientID = {}, sessionID = {}, edfFilePath = {}".format(
                        recordID, patientID, sessionID, edfFilePath))
                    recordInfo[recordID] = {}
                    recordInfo[recordID]['edfFilePath'] = edfFilePath
                    recordInfo[recordID]['patientID'] = patientID
                    recordInfo[recordID]['sessionID'] = sessionID
                    if (patientID not in patientInfo.keys()):
                        patientInfo[patientID] = {}
                        patientInfo[patientID]['sessions'] = [sessionID]
                        patientInfo[patientID]['records'] = [recordID]
                        numPatients += 1
                    else:
                        patientInfo[patientID]['sessions'].append(sessionID)
                        patientInfo[patientID]['records'].append(recordID)
                    numEdfs += 1

        print ("Total number of patients = ", len(patientInfo))
        print ("Total number of records = ", len(recordInfo))
        self.numPatients = numPatients
        self.patientInfo = patientInfo
        self.numEdfs = numEdfs
        self.recordInfo = recordInfo
        for recordID in self.recordInfo.keys():
            self.getEdfSummary(recordID)

    def getEdfSummary(self, recordID):
        filePath = self.recordInfo[recordID]['edfFilePath']
        f = pyedflib.EdfReader(filePath)
        numChannels = f.signals_in_file
        channelLabels = f.getSignalLabels()
        otherLabels = []
        columnsToDel = []
        for i in np.arange(numChannels):
            if (re.search('\w+\d+-\w+\d+', channelLabels[i]) == None):
                otherLabels.append(channelLabels[i])
                columnsToDel.append(i)
                numChannels -= 1
        self.recordInfo[recordID]['channelLabels'] = np.delete(channelLabels, columnsToDel, axis=0).tolist()
        self.recordInfo[recordID]['other_labels'] = otherLabels
        self.recordInfo[recordID]['numChannels'] = np.int32(numChannels).item()
        self.recordInfo[recordID]['numSamples'] = np.int32(f.getNSamples()[0]).item()
        self.recordInfo[recordID]['sampleFrequency'] = np.int32(f.getSampleFrequency(0)).item()

    def getSeizuresSummary(self):
        '''
        Read the seizures.json file and summarize the seizure information on per-record basis
        '''
        f = open(self.seizureFilePath, 'r')
        self.jsonRoot = json.load(f)
        f.close()
        patientIDs = self.jsonRoot.keys()
        for patientID in patientIDs:
            patientJsonDataList = self.jsonRoot[patientID]
            for sessionJsonData in patientJsonDataList:
                filename = sessionJsonData["FileName"]
                m = re.match("^(chb\d\d)_(\d\d)\.edf$", filename)
                if (m == None):
                    print ("Error! the file {} does not correspond to a patient record".format(filename))
                else:
                    patientID = m.group(1)
                    sessionID = m.group(2)
                    recordID = patientID + '_' + sessionID
                    print ("recordID = {}, patientID = {}, sessionID = {}".format(
                        recordID, patientID, sessionID))
                    self.recordInfo[recordID]['seizureStart'] = sessionJsonData["SeizureStartTimes"]
                    self.recordInfo[recordID]['seizureEnd'] = sessionJsonData["SeizureEndTimes"]
                    self.recordInfo[recordID]['seizureType'] = "FNCZ"  # To be verified

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

        seizureStarts = self.recordInfo[recordID]['seizureStart']
        seizureEnds = self.recordInfo[recordID]['seizureEnd']
        # seizureType = self.recordInfo[recordID]['seizureType']

        for (seizureStart, seizureEnd) in zip(seizureStarts, seizureEnds):
            if (( (epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
                ( (epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
                return True

        return False

    def recordContainsSeizure(self, recordID):
        '''
        Returns True if there is at least one seizure entry in the entire record
                False otherwise
        '''
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return False
        else:
            return True
    
    def getSeizuresVectorCSV(self, recordID, epochLen, slidingWindowLen, numEpochs):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''

        seizuresVector = np.zeros((numEpochs), dtype=np.int32)
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return seizuresVector
        else:
            seizureStarts = self.recordInfo[recordID]['seizureStart']
            seizureEnds = self.recordInfo[recordID]['seizureEnd']
            # seizureType = self.recordInfo[recordID]['seizureType']
            for i in range(numEpochs):
                epochStart = float(i * slidingWindowLen / 1000)
                epochEnd = epochStart + float(epochLen / 1000)
                for (seizureStart, seizureEnd) in zip(seizureStarts, seizureEnds):
                    if (( (epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
                        ( (epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
                            seizuresVector[i] = 1
                            break # break from the inner loop only
            return seizuresVector
