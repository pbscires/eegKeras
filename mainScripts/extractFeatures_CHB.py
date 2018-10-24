import sys
import os
import re
from DataSets.CHBdataset import CHBdataset
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT
import time
import json
import pandas as pd

# This script gathers data from the seizures.json file and the EDF files and
# creates a recordInfo.json file.

def createRecordInfoFromSeizureJsonFile():
    rootDir = sys.argv[1]
    seizuresJsonFile = sys.argv[2]
    print ("rootDir = {}, seizuresJsonFile = {}".format(rootDir, seizuresJsonFile))

    tuhd = CHBdataset(rootDir, seizuresJsonFile)
    tuhd.summarizeDatset()
    tuhd.getSeizuresSummary()
    jsonFilePath = sys.argv[3]
    print ("json file path = ", jsonFilePath)
    tuhd.saveToJsonFile(jsonFilePath)

def createLLcsvs():
    rootDir = sys.argv[1]
    seizuresJsonFile = sys.argv[2]
    print ("rootDir = {}, seizuresJsonFile = {}".format(rootDir, seizuresJsonFile))

    chbd = CHBdataset(rootDir, seizuresJsonFile)
    chbd.summarizeDatset()
    chbd.getSeizuresSummary()
    csvDirPath = sys.argv[3]
    print ("csvDirPath = {}".format(csvDirPath))
    numRecords = len(chbd.recordInfo)
    print ("numRecords = ", numRecords)
    curRecordNum = 0
    for recordID in chbd.recordInfo.keys():
        curRecordNum += 1
        print ("Processing record ", recordID, "(", curRecordNum, " of ", numRecords, ")")
        filePath = os.path.join(csvDirPath, recordID+'.csv')
        if (os.path.exists(filePath)):
            print ("Already processed the record", recordID)
            continue
        llObj = LineLength()
        sigbufs = chbd.getRecordData(recordID)
        llObj.extractFeature(sigbufs, chbd.recordInfo[recordID]['channelLabels'], chbd.recordInfo[recordID]['sampleFrequency'])
        print ("Saving Line Length feature values to csv file ", filePath)
        llObj.saveLLdfWithSeizureInfo(filePath, chbd, recordID)
    return

def createFFTcsvs():
    rootDir = sys.argv[1]
    seizuresJsonFile = sys.argv[2]
    print ("rootDir = {}, seizuresJsonFile = {}".format(rootDir, seizuresJsonFile))

    chbd = CHBdataset(rootDir, seizuresJsonFile)
    chbd.summarizeDatset()
    chbd.getSeizuresSummary()
    csvDirPath = sys.argv[3]
    print ("csvDirPath = {}".format(csvDirPath))
    numRecords = len(chbd.recordInfo)
    print ("numRecords = ", numRecords)
    curRecordNum = 0
    for recordID in chbd.recordInfo.keys():
        curRecordNum += 1
        print ("Processing record ", recordID, "(", curRecordNum, " of ", numRecords, ")")
        filePath = os.path.join(csvDirPath, recordID+'.csv')
        if (os.path.exists(filePath)):
            print ("Already processed the record", recordID)
            continue
        fftObj = FFT()
        sigbufs = chbd.getRecordData(recordID)
        fftObj.extractFeature(sigbufs, chbd.recordInfo[recordID]['channelLabels'], chbd.recordInfo[recordID]['sampleFrequency'])
        print ("Saving Line Length feature values to csv file ", filePath)
        fftObj.saveFFTWithSeizureInfo(filePath, chbd, recordID)
    return

def _saveToJsonFile(recordInfo, filePath):
    print ("Saving to the json file ", filePath)

    with open(filePath, 'w') as f:
        f.write("{\n")
        for recordID in recordInfo.keys():
            try:
                f.write("\"" + recordID + "\" : ")
                f.write(json.dumps(recordInfo[recordID]))
                f.write(",\n")
            except TypeError:
                print ("Record = ", recordInfo[recordID])
        f.write("\"EOFmarker\" : \"EOF\" }\n")

def summarizeCSVs():
    rootDir = sys.argv[1]
    summaryJsonFile = sys.argv[2]
    print ("rootDir={}, summaryJsonFile={}".format(rootDir, summaryJsonFile))
    recordInfo = {}
    for root, dirs, files in os.walk(rootDir):
        for filename in files:
            if (re.search("\.csv$", filename) != None):
                recordID = os.path.splitext(os.path.basename(filename))[0]
                m = re.match('\w+\.(chb\d+_\d+)\.edf', recordID)
                if (m != None):
                    recordID = m.group(1)
                recordInfo[recordID] = {}
                recordInfo[recordID]['CSVpath'] = os.path.join(root, filename)
    
    # Get the information for each csv file
    for recordID in recordInfo.keys():
        filePath = recordInfo[recordID]['CSVpath']
        dataset = pd.read_csv(filePath)
        # print ("dataset.shape = ", dataset.shape)
        numRows = dataset.shape[0]
        # To get numFeatures, subtract 2 from dataset.shape[1] so that 
        # the index column and the seizuresPresent Column are not counted.
        numFeatures = dataset.shape[1] - 2 
        # print ("recordID={}, filePath={}, numRows={}, numFeatures={}".format(recordID,
        #     filePath, numRows, numFeatures))
        # real_data = dataset.values
        # real_data = real_data[:,1:numFeatures+1]
        # print ("Shape of real_data = ", real_data.shape)
        recordInfo[recordID]['numRows'] = numRows
        recordInfo[recordID]['numFeatures'] = numFeatures
        recordInfo[recordID]['containsHeaderRow'] = True
        recordInfo[recordID]['containsIndexCol'] = True
    _saveToJsonFile(recordInfo, summaryJsonFile)


if __name__ == '__main__':
    # Uncomment one of the following code blocks

    # -------------------------------------------------
    # Create a JSON file summary of teh TUH EDF files
    # Inputs: <top-directory-of-TUH-files> 
    #         <Filepath of the JSON file listing seizures>
    #     The JSON file was hand-created last year under
    #      workspace\eegAnalysis\Configuration\seizures.json
    #  Note:  createJsonFile() needs to be run only once.
    #     After the json file summarizing all the records
    #     is succesfully created, it does not have to be 
    #     invoked again.
    # ------uncomment beginning at the line below------------
    # createRecordInfoFromSeizureJsonFile()
    # ------uncomment ending at the line above---------------

    # -------------------------------------------------
    # Test a previously created json file by loading it
    # Inputs: <path-to-the-json-file>
    # This is a one time test.  It does not need to be
    #   uncommented after the initial test.
    # ------uncomment beginning at the line below------------
    # tuhd = TUHdataset('', '')
    # tuhd.loadJsonFile(sys.argv[1])
    # ------uncomment ending at the line above---------------

    # -------------------------------------------------
    # Create a unique CSV file for each of the EDF files in the TUH dataset.
    # The CSV file contains the Line Length feature values for each of the 
    # channels.  Optionally Principal Component Analysis can be done
    # on the raw data, which results in fewer components (linear combination 
    # of channels) that account for most of the variance in the values.
    # Inputs: <top-directory-of-CHB-files> 
    #         <Filepath of the seizures.json file listing seizures>
    #         <path to the directory where csv files have to be stored>
    # ------uncomment beginning at the line below------------
    # createLLcsvs()
    # ------uncomment ending at the line above---------------

    # -------------------------------------------------
    # Create a unique CSV file for each of the EDF files in the TUH dataset.
    # The CSV file contains the FFT feature values for each of the 
    # channels.  Optionally Principal Component Analysis can be done
    # on the raw data, which results in fewer components (linear combination 
    # of channels) that account for most of the variance in the values.
    # Inputs: <top-directory-of-CHB-files> 
    #         <Filepath of the seizures.json file listing seizures>
    #         <path to the directory where csv files have to be stored>
    # ------uncomment beginning at the line below------------
    # createFFTcsvs()
    # ------uncomment ending at the line above---------------

    # -------------------------------------------------
    # Summarize all the CSV files in a given directory into a json file.
    # Inputs: <top-directory-ofCSV-files> <summaryJsonFile>
    # ------uncomment beginning at the line below------------
    summarizeCSVs()
    # ------uncomment ending at the line above---------------
