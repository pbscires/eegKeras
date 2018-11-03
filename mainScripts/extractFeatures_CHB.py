import sys
import os
import re
from DataSets.CHBdataset import CHBdataset
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT
import time
import json
import pandas as pd
import numpy as np

# This script gathers data from the seizures.json file and the EDF files and
# creates a recordInfo.json file.

def createRecordInfoFromSeizureJsonFile(rootDir, seizuresJsonFile, jsonFilePath):

    chbd = CHBdataset(rootDir, seizuresJsonFile)
    chbd.summarizeDatset()
    chbd.getSeizuresSummary()
    chbd.saveToJsonFile(jsonFilePath)

def convert_numpycsv_to_pandascsv(rootDir):
    filePathsTobeConverted = []
    # The 2-stage procedure was written to handle case of a directory with
    # some csv files in pandas dataframe format and some in numpy array format.
    # The csvfile with pandas data frame format has the index column (1..numRows)
    for root, dirs, files in os.walk(rootDir):
        for filename in files:
            if (re.search("\.csv$", filename) != None):
                # print ("Converting the file ", filename)
                filePath = os.path.join(root, filename)
                # dataset_arr = np.loadtxt(filePath, delimiter=',')
                dataset_df = pd.read_csv(filePath)
                dataset_arr = dataset_df.values
                for i in range(1, dataset_arr.shape[0]):
                    try:
                        if (int(dataset_arr[i,0]) != i):
                            # print ("dataset_arr[{},0] = {}".format( i, dataset_arr[i,0]))
                            break
                    except ValueError:
                        print ("i = ", i)
                        exit(-1)
                if (i >= dataset_arr.shape[0]-1):
                    print ("file ", filename, " is already converted to pd csv")
                else:
                    filePathsTobeConverted.append(filePath)

    for filePath in filePathsTobeConverted:
        print ("Converting the file ", filePath)
        dataset_arr = np.loadtxt(filePath, delimiter=',')
        df = pd.DataFrame(dataset_arr)
        df.to_csv(filePath)

def createLLcsvs(rootDir, seizuresJsonFile, csvDirPath):
    chbd = CHBdataset(rootDir, seizuresJsonFile)
    chbd.summarizeDatset()
    chbd.getSeizuresSummary()
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

def createFFTcsvs(rootDir, seizuresJsonFile, csvDirPath):

    chbd = CHBdataset(rootDir, seizuresJsonFile)
    chbd.summarizeDatset()
    chbd.getSeizuresSummary()
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
        fftObj.extractFeatureMultiProcessing(sigbufs, chbd.recordInfo[recordID]['channelLabels'], chbd.recordInfo[recordID]['sampleFrequency'])
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

def summarizeCSVs(rootDir, summaryJsonFile):
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
    command = sys.argv[1]

    if (command == "CreateRecordInfo"):
        # -------------------------------------------------
        # Create a JSON file summary of teh CHB EDF files
        # Inputs: <top-directory-of-CHB-files> 
        #         <Filepath of the JSON file listing seizures>
        #     The JSON file was hand-created last year under
        #      workspace\eegAnalysis\Configuration\seizures.json
        #  Note:  createJsonFile() needs to be run only once.
        #     After the json file summarizing all the records
        #     is succesfully created, it does not have to be 
        #     invoked again.
        rootDir = sys.argv[2]
        seizuresJsonFile = sys.argv[3]
        outputJsonFilePath = sys.argv[4]
        print ("rootDir = {}, seizuresJsonFile = {}, outputFile = {}".format(
            rootDir, seizuresJsonFile, outputJsonFilePath))
        createRecordInfoFromSeizureJsonFile(rootDir, seizuresJsonFile, outputJsonFilePath)

    # -------------------------------------------------
    # Test a previously created json file by loading it
    # Inputs: <path-to-the-json-file>
    # This is a one time test.  It does not need to be
    #   uncommented after the initial test.
    if (command == "loadRecordInfo"):
        chbd = CHBdataset('', '')
        chbd.loadJsonFile(sys.argv[2])

    # -------------------------------------------------
    # Create a unique CSV file for each of the EDF files in the CHB dataset.
    # The CSV file contains the Line Length feature values for each of the 
    # channels.  Optionally Principal Component Analysis can be done
    # on the raw data, which results in fewer components (linear combination 
    # of channels) that account for most of the variance in the values.
    # Inputs: <top-directory-of-CHB-files> 
    #         <Filepath of the seizures.json file listing seizures>
    #         <path to the directory where csv files have to be stored>
    if (command == "CreateLLcsvs"):
        rootDir = sys.argv[2]
        seizuresJsonFile = sys.argv[3]
        csvDirPath = sys.argv[4]
        print ("rootDir = {}, seizuresJsonFile = {}, csvDirPath = {}".format(
            rootDir, seizuresJsonFile, csvDirPath))
        createLLcsvs(rootDir, seizuresJsonFile, csvDirPath)

    # -------------------------------------------------
    # Create a unique CSV file for each of the EDF files in the CHB dataset.
    # The CSV file contains the FFT feature values for each of the 
    # channels.  Optionally Principal Component Analysis can be done
    # on the raw data, which results in fewer components (linear combination 
    # of channels) that account for most of the variance in the values.
    # Inputs: <top-directory-of-CHB-files> 
    #         <Filepath of the seizures.json file listing seizures>
    #         <path to the directory where csv files have to be stored>
    if (command == "CreateFFTcsvs"):
        rootDir = sys.argv[2]
        seizuresJsonFile = sys.argv[3]
        csvDirPath = sys.argv[4]
        print ("rootDir = {}, seizuresJsonFile = {}, csvDirPath = {}".format(
            rootDir, seizuresJsonFile, csvDirPath))
        createFFTcsvs(rootDir, seizuresJsonFile, csvDirPath)

    # -------------------------------------------------
    # Summarize all the CSV files in a given directory into a json file.
    # Inputs: <top-directory-ofCSV-files> <summaryJsonFile>
    if (command == "summarizeCSVs"):
        rootDir = sys.argv[2]
        summaryJsonFile = sys.argv[3]
        print ("rootDir={}, summaryJsonFile={}".format(rootDir, summaryJsonFile))
        summarizeCSVs(rootDir, summaryJsonFile)

    # -------------------------------------------------
    # Convert the previosuly create csv file from numpy format to 
    # Pandas dataframe format
    if (command == "convertToPandas"):
        rootDir = sys.argv[2]
        print ("rootDir = ", rootDir)
        convert_numpycsv_to_pandascsv(rootDir)