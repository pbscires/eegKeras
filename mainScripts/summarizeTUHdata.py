import sys
import os
from DataSets.TUH.TUHdataset import TUHdataset
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT
import time

def createLLcsvs():
    rootDir = sys.argv[1]
    xlsxFilePath = sys.argv[2]
    print ("rootDir = {}, xlsxFilePath = {}".format(rootDir, xlsxFilePath))

    tuhd = TUHdataset(rootDir, xlsxFilePath)
    tuhd.summarizeDatset()
    tuhd.getSeizuresSummary()
    csvDirPath = sys.argv[3]
    print ("csvDirPath = {}".format(csvDirPath))
    for recordID in tuhd.recordInfo.keys():
        llObj = LineLength()
        sigbufs = tuhd.getRecordData(recordID)
        llObj.extractFeature(sigbufs, tuhd.recordInfo[recordID]['channelLabels'], tuhd.recordInfo[recordID]['sampleFrequency'])
        filePath = os.path.join(csvDirPath, recordID+'.csv')
        print ("Saving Line Length feature values to csv file ", filePath)
        llObj.saveLLdfWithSeizureInfo(filePath, tuhd, recordID)

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
if __name__ == '__main__':
    createFFTcsvs()