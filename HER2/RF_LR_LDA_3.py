import os,sys

import pandas as pd
import numpy as np
from PATH import data_path
import joblib
from matplotlib import pyplot as plt
import argparse
import warnings

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils import *
from logger import *
from Model import *

pj = os.path.join
warnings.filterwarnings('ignore')

# ===============  parser   =============== 

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,help='the model to use : can choose from : LDA, LR, RF, SVC')
parser.add_argument('--marker',type=str,default='HER2',help='the marker state to predict')
parser.add_argument('--repeat',default=1000,type=int,help='times of spliting and training')
args = parser.parse_args()

model = {'LDA':LDA, 'LR':LR,'RF':RF, 'SVC':SVC}[args.model]
model_arg = {'LDA':{}, 'LR':LR_arg,'RF':RF_arg, 'SVC':{}}[args.model]

out_dir = os.path.join('{}_result/'.format(args.marker),'{}_result'.format(args.model))

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    
# =============== loading data by function in utils  =============== 

data_DF,y=get_known_sample(data_path,args.marker)

data = data_DF.values

# ===============  scale   =============== 


# by gene : function: `scaling_by_gene`  
data_erbb2 = scaling_by_gene(data,'ERBB2',column=data_DF.columns)
# norm 
data_norm = preprocessing.scale(data)
data_log = preprocessing.scale(np.log(data+0.01))

datas = [data,data_log,data_norm,data_erbb2]
#datas += [PCA.fit_transform(data) for data in datas]


val_metrics = []
test_metrics = []
seed_ls = []

#. ================. SPLIT DATA =================

for repeat in range(args.repeat):
    
    seed=np.random.randint(low=10000000)
    print('\n             |================|               ')
    print('               the {} repeat               '.format(repeat))
    print('               seed : {}                '.format(seed))
    print('             |================|               \n')
    
    set_ls,idx_ls = spliting_datas_by_the_same(datas,y,True,**{'idx':y,'random_state':seed})
    
#. ================. TRAINING .=================

    RF_ls = [single_model(model=model,DataSet=Set,with_test=True,**model_arg) for Set in set_ls]
    
    local_val = []
    local_test = []
    for i in range(len(RF_ls)):
        RF_ls[i].get_metrics(test=0)
        val_metrics.append([RF_ls[i].f1,RF_ls[i].auc])
        local_val.append(RF_ls[i].f1+RF_ls[i].auc)

        RF_ls[i].get_metrics(test=1)
        test_metrics.append([RF_ls[i].f1,RF_ls[i].auc])
        local_test.append([RF_ls[i].f1,RF_ls[i].auc])
    
    
    vobosing(val_metrics[-4:],test_metrics[-4:])

#. ================. VALIDATION .=================

    max_id = np.argmax(local_val)
    if (np.sum(local_test[max_id]) >= np.max(np.sum(test_metrics,axis=1))) & (local_test[max_id][0] > 0.6):
        seed_ls.append([seed,max_id])
        #joblib.dump(set_ls,os.path.join(out_dir,'the_{}_setls.SetList'.format(repeat)))
        #joblib.dump(RF_ls[max_id],os.path.join(out_dir,'the_{}_{}.Model'.format(repeat,args.model)))
        
np.save(pj(out_dir,'predict_{}_{}_seed.npy'.format(args.marker,args.model)),np.array(seed_ls))
print("END OF PROGRAME")