import numpy as np
from numba import jit
from sklearn.metrics import pairwise
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
import warnings

def SSw_SSb(data,labels):
    """
    SSb is between group, SSw is within group
    """
    SSw = 0
    SSb = 0
    mu = np.mean(data,axis=0)
    for cluster in np.unique(labels):
        cluster_index = np.where(labels == cluster)[0]
        cluster_data = data[labels == cluster]
        cluster_mean = np.mean(cluster_data,axis=0)
        
        SSw += np.sum((cluster_data - cluster_mean)**2)
        SSb += np.sum( (mu - cluster_mean)**2 )*len(cluster_index)
    return SSw,SSb

def compute_wk(X,labels,KL=False):
    """
    Wk stands for the sum of euclidean distance of data points to its cooresponding cluster center
    arguments:
    ...X : (n_samples,n_feature)
    ...labels : (n_samples,) dtype = int
    ...KL : recale by k and data dimension p , should set to 'True' when later used to 
    """
    p = X.shape[1]
    n_cluster = len(np.unique(labels))
    # get v
    v = np.stack([np.mean(X[labels==c],axis=0) for c in np.unique(labels)])

    # compute si
    wk = np.sum([np.sum(pairwise.pairwise_distances(X[labels ==c],v[c].reshape(1,-1))) for c in np.unique(labels)])
    ## 
    if KL:
        wk = (n_cluster)**(2/p)*wk
    
    return wk

def pairwise_dist(X,Y=None):
    """
    compute the element-wise distance of matrix X and Y by shape (X.shape[0],Y.shape[0])
    if Y is not given, compute the pairwise distance between each sample of X
    """
    if Y is not None:
        assert(X.shape[1] == Y.shape[1])
    else:
        Y = X
    x_samples = X.shape[0]
    y_samples = Y.shape[0]
    X = np.stack([X]*y_samples).transpose((1,0,2))       # ie. (100,100,22)
    Y = np.stack([Y]*x_samples)
    
    dist_M = np.sqrt(np.sum((X - Y)**2,axis=2))   # ie. (100,100)
    return dist_M

def ZwZ_calinski_harabasz_score(data,label):
    r"""```CH = \frac{SS_B}{SS_W} \times \frac{N-k}{k-1}```
    """
    n_samples = data.shape[0]
    n_labels = len(np.unique(label))
    intra_disp, extra_disp  = SSw_SSb(data,label)
    if intra_disp == 0:
        CH =1
    else:
        CH = extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.))
    return CH



# %load -s my_silhouette my_metrics.py
def my_silhouette(X,label,metric='euclidean'):
    r"""math:`S_i = \frac{\beta_i - \alpha_i}{\max(\alpha_i,\beta_i)}`
    """
    
    S = np.zeros(X.shape[0])
    alpha = np.zeros(X.shape[0])
    beta = np.zeros(X.shape[0])
    cluster_data = {}
    for cluster in np.unique(label):
         cluster_data[cluster] = X[label == cluster]            # ie. (100,22)
        
    for cluster in np.unique(label):
        c_index = np.where(label == cluster)[0]
        # ----------- compute a_i of the current class ------------
        X_c = cluster_data[cluster]
        alpha_dist = pairwise.pairwise_distances(X_c)
        #alpha = np.mean(alpha_dist,axis=1)
        alpha[c_index] = np.sum(alpha_dist,axis=1)/(alpha_dist.shape[1]-1)
        
        # ----------- compute b_i of the current class ------------        
        beta_ls = []
        for other in np.unique(label):
            if cluster == other:
                continue
            X_nc = cluster_data[other]                           # ie. (300,22)
            beta_dist = pairwise.pairwise_distances(X_c,X_nc)    # ie. (100,300)
            beta_ls.append(np.mean(beta_dist,axis=1))
        beta[c_index] = np.min(np.stack(beta_ls),axis=0)
        
        #x_nc = X[label != cluster]
        #beta_dist = pairwise.pairwise_distances(X_c,X_nc)
        #beta = np.mean(beta_dist,axis=1)
        
        # ----------- compute S_i of the current class ------------
                
        S[c_index] = (beta[c_index] - alpha[c_index])/np.maximum(beta[c_index],alpha[c_index])

    return S



def gap_statistics(X,labels,n_random=100):
    r""" compute the Gap(k) and sk for current data with label
    refer: 
    ... https://blog.csdn.net/baidu_17640849/article/details/70769555?fps=1&locationNum=2
    """
    n_cluster = len(np.unique(labels))
    
    # compute Wk
    cluster_center = np.stack([np.mean(X[labels==c],axis=0) for c in np.unique(labels)])
    assert(cluster_center.shape == (n_cluster,X.shape[1]))
    # Wk is sum
    Wk = np.sum([
         np.sum(pairwise.pairwise_distances(X[labels == c],cluster_center[c].reshape(1,-1)))
                 for c in np.unique(labels)])
    
    # random sample and compute Wkb
    W_prims = []
    data_range = np.matrix(np.diag(X.max(axis=0) - X.min(axis=0)))

    for r in range(n_random):
        refs = np.random.random(X.shape)*data_range + X.min(axis=0)    # ie. (3194,22)
        refs_center,refs_label = kmeans2(refs,n_cluster)        # ie. (5,22) ,(3194,)
        refs_center = refs_center.take(refs_label,axis=0)              # ie. (3194,22)
        Wkb = pairwise.paired_distances(refs,refs_center)    #  sum ( (3194,) ) -> R
        assert(Wkb.shape == (X.shape[0],))
        W_prims.append(np.log(np.sum(Wkb)))
    
    W_prim = np.mean(W_prims)
    
    # compute sd_k
    sd_k = np.sqrt(np.mean((np.array(W_prims) - W_prim)**2))
    s_k = np.sqrt((1+n_random)*sd_k/n_random )
    
    # compute Gap(k)
    Gap_k = W_prim - np.log(Wk)
    return Gap_k , s_k


def DB_score(X,labels):
    
    n_cluster = len(np.unique(labels))
    # get v
    v = np.stack([np.mean(X[labels==c],axis=0) for c in np.unique(labels)])

    # d_ij
    d_ij = pairwise.pairwise_distances(v)
    for i in range(d_ij.shape[0]):
        d_ij[i,i] = np.inf

    # compute si
    s_i = np.stack([np.mean(pairwise.pairwise_distances(X[labels ==c],v[c].reshape(1,-1))) for c in np.unique(labels)])
    s_i_bc = np.broadcast_to(s_i,(n_cluster,n_cluster))

    # compute R
    R_ij = np.divide((s_i_bc.T + s_i_bc),d_ij)
    R_i = np.max(R_ij,axis=1)

    # compute DB
    DB = np.mean(R_i)

    return DB

def jump_of_k(X,labels):
    p = X.shape[1]
    Centoid = np.stack([np.mean(X[labels == cluster],axis=0) for cluster in np.unique(labels)])
    dist_M = pairwise.pairwise_distances(X,Centoid,metric='mahalanobis')
    dk =  np.min(np.mean(dist_M,axis=0))/p
    return dk

def KL_score(X,label_ls):
    '''
    when given data X and a list containing cluster labels with different n_clusters setting
    argument:
    ...X : (n_samples,n_feature)
    ...label_ls : list 
    outpur:
    ... KL_k : list , length : len(label_ls) - 2
    '''
    k_ls = [len(np.unique(labels)) for labels in label_ls]
    wk_ls = [compute_wk(X,labels,KL=True) for labels in label_ls]
    
    # len : ks -1
    DIFF_k = [wk_ls[i-1] - wk_ls[i] for i in range(1,len(label_ls))]
    #
    KL_k = [np.abs(np.divide(DIFF_k[i],DIFF_k[i+1])) for i in range(0,len(DIFF_k)-1)]
    return KL_k

def Hartigan_score(X,label_ls):
    """
    the calculation of Hartigan score  require Wk and Wk+1 
    """
    n = X.shape[0]
    k_ls = [len(np.unique(labels)) for labels in label_ls]
    
    # remember to set KL=True , for Wk share the same definition with that in KL method
    wk_ls = [compute_wk(X,labels,KL=True) for labels in label_ls]
    
    # compute Hk , require W_{k} and W_{k+1}
    Hk_ls = [(wk_ls[i]/wk_ls[i+1] -1)*(n-k_ls[i]-1) for i in range(0,len(k_ls)-1)]
    
    return Hk_ls



class cluster_metrics(object):
    
    def __init__(self,X,label_ls):
        warnings.filterwarnings("ignore")
        self._k_ls = [len(np.unique(labels)) for labels in label_ls][:-1]

        # single mapping metrics
        # CH , Silhoutte , Davies-Doulbin
        self._CSD = np.array([
                (calinski_harabasz_score(X,labels),silhouette_score(X,labels),davies_bouldin_score(X,labels)) 
            for labels in label_ls
        ])
        self.CH_score = self._CSD[:-1,0]
        self.SH_score = self._CSD[:-1,1]
        self.DB_score = self._CSD[:-1,2]

        # gap_k , sd_k
        # will return ks - 1 gap statistics only
        self._Gap_sdk = np.array([gap_statistics(X,labels) for labels in label_ls])
        self._Gapstat = self._Gap_sdk[:-1,0]
        self._Gap_scale = np.sum(self._Gap_sdk,axis = 1)[1:]    # gap(k) + s_k
        self.Gap_diff = self._Gapstat - self._Gap_scale
        
        # KL_score
#         self.KL_score = KL_score(X, label_ls)
        
        # Hartigan score
        self.Hartigan = Hartigan_score(X, label_ls)
        
        # Jump method
        self._dk_ls = np.array([jump_of_k(X,labels) for labels in label_ls])
        self.Jump_score =  self._dk_ls[1:] - self._dk_ls[:-1]
    
    def vote(self):
        to_pass = lambda x : x[0] != '_'                              # metrics are not starting with '_'
        attrs = list(filter(to_pass,dir(self)))                       # those metrics attr
        self._metrics = [self.__getattribute__(attr) for attr in  attrs]    # list of metrics 
        
        max_indexs = np.array([np.argmax(metric) for metric in self._metrics])  # max for each metric of a cluster method
        
        return np.argmax(np.bincount(max_indexs))                     #  the most 
        
    def all_in_one(self,fig=None,ax=None,**kwarg):
        if fig is None:
            fig = plt.figure(figsize=(12,8))
        if fig is None:
            ax = fig.subplots(3,2)
        
        ax[0,0].plot(self._k_ls,self.SH_score,**kwarg);
        ax[0,0].set_title('Silhouttee score')
        
        ax[0,1].plot(self._k_ls,self.DB_score,**kwarg);
        ax[0,1].set_title('DB score')
        
        ax[1,0].plot(self._k_ls,self.CH_score,**kwarg);
        ax[1,0].set_title('CH score')
        
        # will return ks - 1 gap statistics only
        ax[1,1].plot(self._k_ls,self.Gap_diff,**kwarg);
        ax[1,1].set_title('Gap statistics')
        
        ax[2,0].plot(self._k_ls,self.Hartigan,**kwarg)
        ax[2,0].set_title('Hartigan score')
        
        ax[2,1].plot(self._k_ls,self.Jump_score,**kwarg)
        ax[2,1].set_title('Jump score')
        
#         ax[3,0].plot(self._k_ls[1:-1],self.KL_score,**kwarg)
#         ax[3,0].set_title('KL score')