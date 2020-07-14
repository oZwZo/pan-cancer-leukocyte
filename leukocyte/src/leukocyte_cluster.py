import os,sys
sys.path.append(os.path.dirname(__file__))
import utils, my_metrics
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.cluster import AffinityPropagation,AgglomerativeClustering,KMeans,MeanShift
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import normalize,scale
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise
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
    """
    screen data by row and by columns
    """
    threshold1 = statistics.loc['95%'].values
    threshold2 = statistics.loc['99%'].values
    v_data = np.array(DF.values[:,1:],dtype=np.float64)          
    row_screen = (np.sum(v_data > threshold1,axis=1) <4) | (np.sum(v_data > threshold2,axis=1) <2)
    column_screen = np.where(np.logical_and(statistics.loc['std'].values > 0.01,threshold2 > 0.01))[0] 
    column_screen = [0] + list(column_screen + 1) # move right 1 columns for 'id'
    return DF[row_screen].iloc[:,column_screen]

def statistics_preprocess(DF,plotting=True):
    """
    for a leukoctye dataframe compute the statitical quantile,
    - visualize with heatmap and violin plot
    - return filtered data
    v.3
    """
    
    # create statistics quantile 
    statistics = DF.describe(percentiles=[0.25,0.5,0.75,0.95,0.99])
    
    # statistics and quantile heatmap
    data = statistics.iloc[1:,:].values
    if plotting:
        heatmap_with_text(data,statistics)
                            
    # screen out data with 4 or more outlines feature
    filtered_DF = data_filtering(DF,statistics)

    return filtered_DF,statistics

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
            ax[k,j].scatter(tsne_datas[i][:,0],tsne_datas[i][:,1],s=2)
            i += 1
    return tsne_datas[5]

def t_sne_label(tsne_data,labels,metrics):
    fig = plt.figure(figsize=(18,5))
    axs = fig.subplots(1,3)
    i = 0
    for method in labels.keys():
        vote = metrics[method].vote()
        label = labels[method][vote]
        axs[i].scatter(tsne_data[:,0],tsne_data[:,1],c=label,s=3)
        axs[i].set_title('{} cluster with optimal k  {}'.format(method,metrics[method]._k_ls[vote]))
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
    return Z
    
def cluster3(data,n_clusters = range(5,10)):
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
    labels['gmm'] = [GMM(n_components=i,covariance_type='tied',max_iter=400).fit_predict(data) for i in  n_clusters]

    # metrics class
    for method in methods:
        metrics[method] = my_metrics.cluster_metrics(data,labels[method])
    
    return labels,metrics

def metrics_curve(metrics):
    methods = metrics.keys()
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.subplots(3,2)

    for method in methods:          
        metrics[method].all_in_one(fig=fig,ax=ax,**{'label':method})
    for ls in ax:
        for axx in ls:
            axx.legend()
            
def sort_data_by_label(n_data,labels,metrics,method='km',direct_label=None):
    """
    sort data for heatmap and stack-bar plot
    """
    if direct_label is not None:
        label = direct_label
    else:
        label = labels[method][metrics[method].vote()]
    label_df=pd.DataFrame({'x':range(n_data.shape[0]),'label': label}).sort_values('label')
    index=label_df['x']
    n_data = n_data[index]
    return n_data,label
    
def stack_barplot(data,labels,cmap=cm.Spectral,**kwarg):
    """
    given the data(sorted)  with label,and Z, plot the stacked bar-plot
    """
    print('this will take several minutes')
   

    label_num = np.bincount(labels)
    
    figsize = (45,15) if data.shape[0] > 2000 else (12,5)
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_axes([0,0,0.8,0.7]) 
    ax2 = fig.add_axes([0,0.7-0.01,0.8,0.05])   # left,bottom,width,height
#     ax3 = fig.add_axes([1,0,0.3,0.7])       # left,bottom,width,height
    
    N = np.arange(data.shape[0])
    
    LEFT = 0
    for num in label_num:
        ax2.barh(0,num,left=LEFT)
        ax2.set_xlim(0,data.shape[0]-1)
        ax2.axis('off')
        LEFT += num
    
#     ax3.axis('off')
#     tree = dendrogram(Z,p=15,truncate_mode='level',no_labels=True,above_threshold_color='black',color_threshold=0.1,ax=ax3)
    if isinstance(cmap,type(cm.Spectral)):
        colour = cmap(0/data.shape[1])
    elif isinstance(cmap,np.ndarray):
        colour = cmap[0]
    ax.bar(N,height=data[:,0],color=colour,**kwarg)
    for i in range(data.shape[1]):
        if isinstance(cmap,type(cm.Spectral)):
            colour = cmap(i/data.shape[1])
        elif isinstance(cmap,np.ndarray):
            colour = cmap[i]
        ax.bar(N,height=data[:,i],bottom=data[:,:i].sum(axis=1),color=colour,width=1,label=cell_label[i],**kwarg)
        ax.set_xlim(0,data.shape[0])
        ax.axis('off')
    fig.legend(bbox_to_anchor=(0.81,0.7),loc='upper left',fontsize=20)
    fig.show()
    
def tree_heatmap(n_data,Z):
    """
    hierarchical clustering and heatmap, data needed to sort
    """
    n_data = normalize(n_data)

    # data for heat map
    Dist_M = pairwise.pairwise_distances(n_data)
    Sim_M = np.max(Dist_M) - Dist_M   # make distance Matrix a similarity Matrix


    fig = plt.figure(figsize = (16,10))
    ax = fig.add_axes([0,0,1,0.8])    #left,bottom,width,height
    ax.axis('off')
    ax.imshow(Sim_M,cmap=cm.Blues);

    ax2 = fig.add_axes([0.25,0.8,0.5,0.2]) #left,bottom,width,height
    ax2.axis('off')
    tree = dendrogram(Z,p=15,truncate_mode='level',no_labels=True,above_threshold_color='black',color_threshold=0.1,ax=ax2)
    fig.show()
    
def cluter_in_all(DF,labels_dict,cancer_type):

    all_tsne_df = pd.read_csv(os.path.join(tsne_path,'ALL_tsne_p30.csv'))
    global_tsne = all_tsne_df.iloc[:,1:].values
    
    fig,ax = plt.subplots(1,3,figsize=(15,5))
    ii = 0 
    
    for method in labels_dict.keys():
        # extract info by merging DF
        DF.loc[:,'label'] = labels_dict[method]
        merge_df = DF[['id','label']].merge(
            all_tsne_df,
            left_on=['id'],
            right_on=['id'],
            how='inner'
        ).sort_values('label')
        local_tsne = merge_df.loc[:,['axis0','axis1']].values
        local_label = merge_df.label
        # plot data
        ax[ii].set_xlabel('tsne-1',fontsize=15)
        ax[ii].set_ylabel('tsne-2',fontsize=15)
        ax[ii].set_title('{} cluter result of type {} in all data \n'.format(method,cancer_type),fontsize=17)
        # scatter
        ax[ii].scatter(global_tsne[:,0],global_tsne[:,1],s=10,color='gray',alpha=0.3);  # ALL
        ax[ii].scatter(local_tsne[:,0],local_tsne[:,1],s=10,c=local_label);   # specific cancer type 
        ii += 1
        
def cluter_in_all_no_label(DF,cancer_type,ax,**kwarg):

    all_tsne_df = pd.read_csv(os.path.join(tsne_path,'ALL_tsne_p30.csv'))
    global_tsne = all_tsne_df.iloc[:,1:].values


#     extract info by merging DF
#     DF.loc[:,'label'] = labels_dict[method]
    merge_df = DF[['id']].merge(
        all_tsne_df,
        left_on=['id'],
        right_on=['id'],
        how='inner'
    )
    local_tsne = merge_df.loc[:,['axis0','axis1']].values
#     local_label = merge_df.label
    # plot data
    ax.set_xlabel('tsne-1',fontsize=12)
    ax.set_ylabel('tsne-2',fontsize=12)
    ax.set_title(' {}  data '.format(cancer_type),fontsize=14)
    # scatter
    ax.scatter(global_tsne[:,0],global_tsne[:,1],s=10,color='gray',alpha=0.3);  # ALL
    ax.scatter(local_tsne[:,0],local_tsne[:,1],s=10,**kwarg);   # specific cancer type 


