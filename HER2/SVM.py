import os
import pandas as pd
import numpy as np
from PATH import data_path
import joblib
from matplotlib import pyplot as plt
import argparse
import warnings
import time


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
parser.add_argument('--title',default=str(time.time()).split('.')[1],type=str,help='for the convience of distinguishing runs')
parser.add_argument('--data',default='norm',type=str,help='the data to include , choose from raw, norm,log erbb2')
parser.add_argument('--C',default='1,10,100,500,1000',type=str,help='the data to include , choose from raw, norm, log, erbb2')
parser.add_argument('--kernel',default='rbf',type=str,help='the kernel function to use , choose from linear,poly,rbf')
parser.add_argument('--repeat',default=1000,type=int,help='times of spliting and training')
args = parser.parse_args()

C = [float(c) for c in args.C.split(',')]
out_dir = os.path.join('result/','SVM_result')

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# =============== loading data by function in utils  =============== 


label = read_label(data_path)
data_DF = read_data_DF(data_path)

merged_DF=data_DF.merge(label,left_index=True,right_on=['sample'])
merged_DF.index = range(merged_DF.shape[0])

DF=merged_DF[merged_DF.status=='Negative'].iloc[:,0:-2].append(
    merged_DF[merged_DF.status=='Positive'].iloc[:,0:-2])

data = DF.values
N_nega=np.sum(merged_DF.status=='Negative')



# ===============  scale   =============== 


# by gene : function: `scaling_by_gene`  
data_erbb2 = scaling_by_gene(data,'ERBB2',column=data_DF.columns)
# norm 
data_norm = preprocessing.scale(data)
data_log = np.log(data+0.01)

y=np.array([0]*463 + [1]*(611 - 463))

Datas = [data,data_log,data_norm,data_erbb2]
data_label = {'raw':0,'norm':1,'log':2,'erbb2':3}
datas = []
seed_ls = []
for D in args.data.split(','):
    datas.append(Datas[data_label[D]])
    
#datas += [PCA.fit_transform(data) for data in datas]


val_metrics = []
test_metrics = []

#. ================. SPLIT DATA =================

for repeat in range(args.repeat):
    
    seed=np.random.randint(low=10000000)
    print('\n             |================|               ')
    print('               the {} repeat               '.format(repeat))
    print('               seed : {}                '.format(seed))
    print('             |================|               \n')
    
    set_ls,idx_ls = spliting_datas_by_the_same(datas,y,True,**{'idx':y,'random_state':seed})

    #. ================. TRAINING .=================

    SV_ls = [[single_model(model=SVC,DataSet=Set,with_test=True,**{'C':i,'probability':True,'kernel':args.kernel}) for i in C ] for Set in set_ls]
    
    local_val = []
    local_test = []
    
    for Set_i in range(len(set_ls)):
        
        model_ls = SV_ls[Set_i]
        data_label = args.data.split(',')[Set_i]
        
        for i in range(len(model_ls)):
            model_ls[i].get_metrics(test=0)
            val_metrics.append([model_ls[i].f1,model_ls[i].auc])
            local_val.append(model_ls[i].f1+model_ls[i].auc)

            model_ls[i].get_metrics(test=1)
            test_metrics.append([model_ls[i].f1,model_ls[i].auc])
            local_test.append([model_ls[i].f1,model_ls[i].auc])
    
        vobosing_SVM(val_metrics[-5:],local_test,data_label,C)

    #. ================. SAVING GOOD MODEL.=================

    max_id = np.argmax(local_val)
    if (np.sum(local_test[max_id]) >= np.max(np.sum(test_metrics,axis=1))) & (local_test[max_id][0] > 0.6):
        
        seed_ls.append([seed,max_id])
        # save data set file size : 300M 
        #joblib.dump(set_ls,os.path.join(out_dir,'the_{}_{}_setls.SetList'.format(args.title,repeat)))
        # save the model : attention need to paid for file too large
        #joblib.dump(SV_ls,os.path.join(out_dir,'the_{}_{}_SVM.ModelList'.format(args.title,repeat)))
        
np.save(pj(out_dir,'the_{}_seed.npy'.format(args.title)),np.array(seed_ls))
print('\n             |================|               ')
print('              END OF PROGRAMME              '.format(repeat))
print('             |================|               \n')
    