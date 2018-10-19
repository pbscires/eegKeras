import sys
import os
from DataSets.CHBdataset import CHBdataset
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT
import time

# This script gathers data from the seizures.json file and the EDF files and
# creates a recordInfo.json file.

def createJsonFile():
    rootDir = sys.argv[1]
    seizuresJsonFile = sys.argv[2]
    print ("rootDir = {}, seizuresJsonFile = {}".format(rootDir, seizuresJsonFile))

    tuhd = CHBdataset(rootDir, seizuresJsonFile)
    tuhd.summarizeDatset()
    tuhd.getSeizuresSummary()
    jsonFilePath = sys.argv[3]
    print ("json file path = ", jsonFilePath)
    tuhd.saveToJsonFile(jsonFilePath)


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
    createJsonFile()
    # ------uncomment ending at the line above---------------
