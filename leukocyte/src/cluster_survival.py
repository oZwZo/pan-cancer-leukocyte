import os,sys,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from PATH import *
import utils
import leukocyte_cluster as LEUC
from lifelines import KaplanMeierFitter, CoxPHFitter

#========================== read data , preprocess ======================
#  --------   read immune cell info   --------
ALL = utils.read_leukocyte(leukocyte_path2)
DF_filtered, statistics=LEUC.statistics_preprocess(ALL,False)
data = DF_filtered.iloc[:,1:].values

#  --------   read survival info      --------
OS_EFS = pd.read_csv(os.path.join(PATH.data_path,'label','survival_time.csv')).drop(['Unnamed: 3'],axis=1) 


#==========================        cluster         ======================
labels,metrics = LEUC.cluster3(data,n_clusters=range(4,9))
LEUC.metrics_curve(metrics)

#  --------     save cluster label     --------
DF_3 = DF_filtered.copy()
for method in ['ac','km','gmm']:
    for i,n_cluster in enumerate(range(4,9)):
        DF_3.loc[:,method+str(n_cluster)] = labels[method][i]

# seems like the id of OS EFS csv is only the first 12 letters
DF_3.loc[:,'A0_Samples'] = DF_3.id.apply(lambda x: x[:12])
interset_DF=OS_EFS.merge(DF_3.iloc[:,21:],right_on=['A0_Samples'],left_on=['A0_Samples']) 
# merge the DF to have the intersect
interset_DF=interset_DF.dropna()

#=========================  visualize cluster result ====================
ALL_tsnes = LEUC.t_sne_tuning(data,'ALL.npy')  # get tsne data
# 0,1,2 : 4,5,6
for n_cluster in [0,1,2]:
    #  --------     scatter plot     --------
    LEUC.t_sne_label(ALL_tsnes,labels,metrics,manual_vote=n_cluster)
    
    #  --------     heatmap plot     --------
    fig=plt.figure(figsize=(15,4))
    axs = fig.subplots(1,3)
    for i,method in  enumerate(['ac','km','gmm']):
        sorted_data,label = LEUC.sort_data_by_label(data,labels,metrics,method=method,manual_vote=n_cluster)
        LEUC.tree_heatmap(sorted_data,None,ax=axs[i])
        axs[i].set_title("n_cluster {} ,{}".format(range(4,9)[n_cluster],method),fontsize=12)



