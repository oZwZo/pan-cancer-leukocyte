import os,sys,utils, my_metrics
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.cluster import AffinityPropagation,AgglomerativeClustering,KMeans,MeanShift
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import normalize,scale
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage,dendrogram,distance 
from PATH import *
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
from IPython.core.pylabtools import figsize

global cancer_types
# all the cancer type 
cancer_types = [file.split('.csv')[0] for file in os.listdir(os.path.join(data_path,'leukocyte_ratio')) if '.csv' in file]

def heatmap_with_text(data,statistics):
    '''
    plotting func
    '''
    valfmt = StrMethodFormatter('{x:.3f}')
    fig,ax = plt.subplots(figsize=(17,10))

    im = ax.imshow(data,cmap=cm.Blues);
    cbar = ax.figure.colorbar(im,shrink=0.6);

    ax.set_title('statistics of the ratio of the 22 leukocytes\n',size=16)

    ax.xaxis.set_tick_params(pad=0,rotation=-45) 
    ax.yaxis.set_ticks(range(0,8));
    ax.set_yticklabels(statistics.index[1:],size=12)
    ax.xaxis.set_ticks(range(0,22));
    ax.set_xticklabels(statistics.columns,ha='left',size=12);

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            font_color = ['black','w'][(data[i,j] > 0.5)]
            ax.text(j-0.35,i+0.08,valfmt(data[i,j],None),color=font_color,size=9)

def violin(data,statistics):
    '''
    plotting func
    '''
    fig,ax = plt.subplots(figsize=(13,7))
    violin = ax.violinplot(data,showmeans=True,widths=0.7,showextrema=True,)

    ax.xaxis.set_tick_params(pad=0,rotation=-30) 
    ax.xaxis.set_ticks(range(1,23));
    ax.set_xticklabels(statistics.columns,ha='left',size=10); 
                                
def data_filtering(DF,statistics):
    threshold = statistics.loc['95%'].values
    v_data = np.array(DF.values[:,1:],dtype=np.float32)          
    gtr_sum = np.sum(v_data > threshold,axis=1)
    return DF[gtr_sum < 4]
                                
def statistics_preprocess(DF):
    """
    for a leukoctye dataframe compute the statitical quantile,
    - visualize with heatmap and violin plot
    - return filtered data
    """
    
    # create statistics quantile 
    statistics = DF.describe(percentiles=[0.25,0.5,0.75,0.95,0.99])
    
    # statistics and quantile heatmap
    data = statistics.iloc[1:,:].values
    heatmap_with_text(data,statistics)
                            
    # screen out data with 4 or more outlines feature
    filtered_DF = data_filtering(DF,statistics)

    return filtered_DF

def t_sne_tuning(data,file_name):
    """
    A new DF should be used, perform T_SNE decomposition and save to npy file 
    ...file_name : cancer_type + '.npy'
    """
    assert(file_name.split('.npy')[0] in cancer_types)   # make sure the naming is correct
    npy = os.path.join(tsne_path,file_name)
    if not os.path.exists(npy):  
        # will skip this part if already computed
        tsne_datas = [TSNE(perplexity=perplexity,n_jobs=8,n_iter=8000).fit_transform(data) 
                      for perplexity in tqdm([5,10,15,20,25,30,35,40,45])]   # TSNE
        tsne_datas = np.stack(tsne_datas)                          # list to array

        np.save(npy,tsne_datas)
        print('/n T_SNE data save to ',npy)
    else:
        print('/n T_SNE data exists \n')
        tsne_datas = np.load(npy)
        
    fig = plt.figure(figsize=(16,12))
    ax = fig.subplots(3,3)
    i = 0
    for k in range(3):
        for j in range(3):
            ax[k,j].scatter(tsne_datas[i][:,0],tsne_datas[i][:,1],s=1)
            i += 1

def hierarchical_tree(data,metric='euclidean',p_start=5,p_step=1):
    disMat=distance.pdist(data,metric=metric)
    Z=linkage(disMat,method='average') 
    p = p_start
    fig = plt.figure(figsize=(16,12))
    axs = fig.subplots(3,3)
    for i in range(3):
        for j in range(3):
            ax = axs[i,j]
            tree = dendrogram(Z,p=p,truncate_mode='level',no_labels=True,distance_sort=True,ax=ax)
            ax.set_title('P = {}  N Clusters = {}'.format(p, len(tree['leaves'])))
            p += p_step
    
    
def cluster3(data,n_clusters = range(5,20)):
    '''
    cluster data with 3 different method : agglomerative , K-means, GMM
    and generate its cluster metrics 
    '''
    labels = {}
    metrics = {}
    methods = ['ac','km','gmm']

    # clustering
    labels['ac'] = [AgglomerativeClustering(n_clusters=i).fit_predict(data) for i in n_clusters]
    labels['km'] = [KMeans(n_clusters=i,n_jobs=8).fit_predict(data) for i in n_clusters]
    labels['gmm'] = [GMM(n_components=i,max_iter=400).fit_predict(data) for i in  n_clusters]

    # metrics class
    for method in methods:
        metrics[method] = my_metrics.cluster_metrics(data,labels[method])
    
    return labels,metrics

def metrics_curve(metrics):
    methods = metrics.keys()
    
    figall = plt.figure(figsize=(16,16))
    ax = figall.subplots(4,2)

    for method in methods:          
        metrics[method].all_in_one(fig=figall,ax=ax,**{'label':method})
    for ls in ax:
        for axx in ls:
            axx.legend()
    plt.show()