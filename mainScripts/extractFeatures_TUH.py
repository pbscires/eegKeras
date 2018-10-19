import sys
import os
from DataSets.TUHdataset import TUHdataset
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT
import time

def createJsonFile():
    rootDir = sys.argv[1]
    xlsxFilePath = sys.argv[2]
    print ("rootDir = {}, xlsxFilePath = {}".format(rootDir, xlsxFilePath))

    tuhd = TUHdataset(rootDir, xlsxFilePath)
    tuhd.summarizeDatset()
    tuhd.getSeizuresSummary()
    jsonFilePath = sys.argv[3]
    print ("json file path = ", jsonFilePath)
    tuhd.saveToJsonFile(jsonFilePath)

def createLLcsvs():
    rootDir = sys.argv[1]
    xlsxFilePath = sys.argv[2]
    print ("rootDir = {}, xlsxFilePath = {}".format(rootDir, xlsxFilePath))

    tuhd = TUHdataset(rootDir, xlsxFilePath)
    tuhd.summarizeDatset()
    tuhd.getSeizuresSummary()
    csvDirPath = sys.argv[3]
    print ("csvDirPath = {}".format(csvDirPath))
    numRecords = len(tuhd.recordInfo)
    print ("numRecords = ", numRecords)
    curRecordNum = 0
    for recordID in tuhd.recordInfo.keys():
        curRecordNum += 1
        print ("Processing record ", recordID, "(", curRecordNum, " of ", numRecords, ")")
        filePath = os.path.join(csvDirPath, recordID+'.csv')
        if (os.path.exists(filePath)):
            print ("Already processed the record", recordID)
            continue
        llObj = LineLength()
        sigbufs = tuhd.getRecordData(recordID)
        llObj.extractFeature(sigbufs, tuhd.recordInfo[recordID]['channelLabels'], tuhd.recordInfo[recordID]['sampleFrequency'])
        print ("Saving Line Length feature values to csv file ", filePath)
        llObj.saveLLdfWithSeizureInfo(filePath, tuhd, recordID)
    return

def createFFTcsvs():
    rootDir = sys.argv[1]
    xlsxFilePath = sys.argv[2]
    print ("rootDir = {}, xlsxFilePath = {}".format(rootDir, xlsxFilePath))

    tuhd = TUHdataset(rootDir, xlsxFilePath)
    tuhd.summarizeDatset()
    tuhd.getSeizuresSummary()
    csvDirPath = sys.argv[3]
    print ("csvDirPath = {}".format(csvDirPath))
    numRecords = len(tuhd.recordInfo)
    print ("numRecords = ", numRecords)
    curRecordNum = 0
    for recordID in tuhd.recordInfo.keys():
        curRecordNum += 1
        print ("Processing record ", recordID, "(", curRecordNum, " of ", numRecords, ")")
        filePath = os.path.join(csvDirPath, recordID+'.csv')
        if (os.path.exists(filePath)):
            print ("Already processed the record", recordID)
            continue
        fftObj = FFT()
        sigbufs = tuhd.getRecordData(recordID)
        fftObj.extractFeatureMultiProcessing(sigbufs, tuhd.recordInfo[recordID]['channelLabels'], tuhd.recordInfo[recordID]['sampleFrequency'])
        print ("Saving Line Length feature values to csv file ", filePath)
        fftObj.saveFFTWithSeizureInfo(filePath, tuhd, recordID)
        if (curRecordNum % 20 == 0):
            time.sleep(60) # Sleep for a minute
    return

if __name__ == '__main__':
    # Uncomment one of the following code blocks

    # -------------------------------------------------
    # Create a JSON file summary of teh TUH EDF files
    # Inputs: <top-directory-of-TUH-files> 
    #         <Filepath of the xlsx file listing seizures>
    #     The xlsx file is in the top level directory of 
    #      the TUH data with the name "seizures_v30r.xlsx"
    #  Note:  createJsonFile() needs to be run only once.
    #     After the json file summarizing all the records
    #     is succesfully created, it does not have to be 
    #     invoked again.
    # ------uncomment beginning at the line below------------
    # createJsonFile()
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
    # Inputs: <top-directory-of-TUH-files> 
    #         <Filepath of the xlsx file listing seizures>
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
    # Inputs: <top-directory-of-TUH-files> 
    #         <Filepath of the xlsx file listing seizures>
    #         <path to the directory where csv files have to be stored>
    # ------uncomment beginning at the line below------------
    createFFTcsvs()
    # ------uncomment ending at the line above---------------
