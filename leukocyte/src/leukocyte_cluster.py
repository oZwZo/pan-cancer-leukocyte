import numpy as np
import pandas as pd
import utils
from PATH import data_path

from sklearn.cluster import AffinityPropagation,AgglomerativeClustering,KMeans,MeanShift
from sklearn.preprocessing import normalize,scale
from scipy.cluster.hierarchy import linkage,dendrogram,distance 

from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
from IPython.core.pylabtools import figsize

def heatmap_with_text(data,statistics):
    '''
    plotting func
    '''
    valfmt = StrMethodFormatter('{x:.3f}'
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
    data = statistics[1:,:].values
    heatmap_with_text(data,statistics)

    # violin plot
    v_data = np.array(DF.values[:,1:],dtype=np.float32)
    v_data = v_data - np.mean(v_data,axis=0)
    violin(v_data,statistics)
                                
    # screen out data with 4 or more outlines feature
    filtered_DF = data_filtering(DF,statistics)

    return filtered_DF


DF = utils.read_leukocyte(data_path,separate=True)

