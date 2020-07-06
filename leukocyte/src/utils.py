import numpy as np
import pandas as pd
import os,sys
sys.path.append(os.path.dirname(__file__))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from PATH import *

###########################
#######    read    ########
##########################

def read_leukocyte(data_path,separate=False,csv=None):
    """
    read leukocyte rate, return the ALL.csv by default,
    when set `separate=True` will return a dict containing 10 kinds cancer
    """
    if separate:
        if csv is None:
            pds= {}
            csvs = os.listdir(leukocyte_path)
            csvs = list(filter(lambda x : '.csv' in x,csvs))
            for csv in csvs:
                if csv == 'ALL.csv':
                    continue              # skip ALL.csv
                fn = csv.split('.')[0]
                pds[fn.upper()] = pd.read_csv(os.path.join(data_path,'leukocyte_ratio',csv))
        else:
            assert(csv in os.listdir(data_path+'/leukocyte_ratio'))
            pds = pd.read_csv(os.path.join(data_path,'leukocyte_ratio',csv))
    else:
        pds = pd.read_csv(os.path.join(data_path,'leukocyte_ratio','ALL.csv'))
    
    return pds