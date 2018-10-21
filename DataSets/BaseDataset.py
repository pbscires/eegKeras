class BaseDataset(object):
    def countRowsForDataSubset(self, recordID, priorSeconds, postSeconds):
        pass
    def getDataSubset(self, recordID, priorSeconds, postSeconds):
        pass
    def recordContainsSeizure(self, recordID):
        pass