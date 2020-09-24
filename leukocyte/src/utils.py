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

global name_dict
my_cancer_type=['TNBC','NSCLC','HNSCC','CERVICAL','STOMACH','KIC2','BLADDER','MELANOMA','COLON','LIVER']
formal_name=['TNBC','NSCLC','HNSC','CESC','STAD','KIC','BLAC','SKCM','COREAD','LIHC']
name_dict=dict(zip(my_cancer_type,formal_name))

def read_leukocyte(leukocyte_path,separate=False,csv=None):
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
                pds[fn.upper()] = pd.read_csv(os.path.join(leukocyte_path,csv))
        else:
            assert(csv in os.listdir(leukocyte_path))
            pds = pd.read_csv(os.path.join(leukocyte_path,csv))
    else:
        pds = pd.read_csv(os.path.join(leukocyte_path,'ALL.csv'))
    
    return pds


def read_color():
    return pd.read_csv(os.path.join(top_path,'leukocyte','color_leukocyte.csv'),index_col=0)

def read_survival():
    return pd.read_csv(os.path.join(data_path,'label','survival_time.csv')).drop(['Unnamed: 3'],axis=1) 

def get_cluster_survival_intersect(survival_info,):
    if survival_info is None:
        survival