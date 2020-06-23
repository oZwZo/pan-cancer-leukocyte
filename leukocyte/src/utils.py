import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

###########################
#######    read    ########
##########################

def read_leukocyte(data_path,separate=False):
    """
    read leukocyte rate, return the ALL.csv by default,
    when set `separate=True` will return a dict containing 10 kinds cancer
    """
    if separate:
        pds= {}
        for csv in os.listdir(data_path+'/leukocyte_ratio'):
            if csv == 'ALL.csv':
                continue              # skip ALL.csv
            fn = csv.split('.')[0]
            pds[fn.upper()] = pd.read_csv(os.path.join(data_path,'leukocyte_ratio',csv))
    else:
        pds = pd.read_csv(os.path.join(data_path,'leukocyte_ratio','ALL.csv'))
    
    return pds