'''

'''
from datetime import datetime

class ElapsedTime(object):
    '''
    Keeps track of the elapsed time for performance analysis
    '''

    def __init__(self):
        '''
        Reset the timer
        '''
        self.startTime = datetime.now()
    
    def reset(self):
        self.startTime = datetime.now()
        
    def timeDiff(self):
        return(datetime.now() - self.startTime)
    
    def __str__(self):
        return (self.timeDiff())