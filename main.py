# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:29:08 2021
@author: Debanjan

Main Function
"""

import sys
import datetime

# Import the libraries
from source import fileRead, preProcessing, modelling, prediction

from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)


def main():
    data_raw = fileRead.import_data()

    data = preProcessing.processing_step1(data_raw)

    modelling.model_randomForest(data)

    prediction.predict_func(data)


if __name__ == "__main__":
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open('./logs/logfile'+str(datetime.datetime.now())+'.log', "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            pass


    sys.stdout = Logger()
    start_time = datetime.datetime.now()
    print('Main execution started at:', start_time)
    main()
    end_time = datetime.datetime.now()
    print('Main execution ended at: \
        {}.\nTotal execution duration: {}.'.format(end_time, end_time - start_time))

print("**************************************************************************************************")
print("**************************************************************************************************")
