import os,sys,re
import pandas as pd
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from matplotlib import cm
from src import PATH, utils,my_metrics
from src import leukocyte_cluster as LEUC

"""read data"""
ALL = utils.read_leukocyte(PATH.leukocyte_path)
DF_filtered, statistics=LEUC.statistics_preprocess(ALL,False)
data = DF_filtered.iloc[:,1:-5].values

"""clustering"""
r_labels,r_metrics_dict = LEUC.cluster3(data,n_clusters=range(2,11))
for method in ['ac','km','gmm']:
    r_pctg_exp_vars = [my_metrics.explain_variance(data,label) for label in r_labels[method]]
    plt.plot(range(2,11),r_pctg_exp_vars,'-o',label=method)
plt.xlabel("# of clusters",fontsize=14)
plt.ylabel("ratio of explained variance",fontsize=14)
plt.legend()
# zoom in to km method
r_pctg_exp_vars = [my_metrics.explain_variance(data,label) for label in r_metrics_dict['km']]
plt.plot(range(2,11),r_pctg_exp_vars,'-o',label='km');
plt.xlabel("# of clusters",fontsize=14)
plt.ylabel("ratio of explained variance",fontsize=14)
plt.legend()

"""create a new DF to store the clustering result"""
DF_reclustering = DF_filtered.iloc[:,:-3]
tsne_1 = pd.read_csv("/home/wergillius/Project/pan-cancer-leukocyte/leukocyte/t_sne/ALL_tsne_p30.csv")
km_label = km_label.merge(tsne_1[['id','cancer_type']],right_on=['id'],left_on=['id'])

DF_reclustering.loc[:,'km3'] = rkm_labels['km'][1]
DF_reclustering.loc[:,'km6'] = rkm_labels['km'][4]

DF_reclustering.to_csv("cluster_report/20210310_km_label.csv",index=False)


# concatenate clinical infomation
stage_info = pd.read_csv(os.path.join(PATH.data_path,'label','patient_stage.csv'))
stage_info = stage_info.iloc[:,:25]
stage_info=stage_info.drop([ 'km', 'ac', 'id.1', 'cluster5'],axis=1)

DF_stage = DF_reclustering[['id','km3', 'km6']].merge(stage_info,right_on=['id'],left_on=['id'],how='inner')

DF_stage.to_csv(PATH.data_path+"/label/20210310_recluster_stage_info.csv",index=False)