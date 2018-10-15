import sys
from DataSets.TUH.TUHdataset import TUHdataset

if __name__ == '__main__':
    rootDir = sys.argv[1]
    csvFilePath = sys.argv[2]
    print ("rootDir = {}, csvFilePath = {}".format(rootDir, csvFilePath))

    tuhd = TUHdataset(rootDir, csvFilePath)
    tuhd.summarizeDatset()
