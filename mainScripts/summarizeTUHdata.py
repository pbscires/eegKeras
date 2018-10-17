import sys
import os
from DataSets.TUH.TUHdataset import TUHdataset
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT

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
    for recordID in tuhd.recordInfo.keys():
        llObj = LineLength()
        sigbufs = tuhd.getRecordData(recordID)
        llObj.extractFeature(sigbufs, tuhd.recordInfo[recordID]['channelLabels'], tuhd.recordInfo[recordID]['sampleFrequency'])
        filePath = os.path.join(csvDirPath, recordID+'.csv')
        print ("Saving Line Length feature values to csv file ", filePath)
        llObj.saveLLdfWithSeizureInfo(filePath, tuhd, recordID)

if __name__ == '__main__':
    createFFTcsvs()