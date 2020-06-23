import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from sklearn.model_selection import train_test_split

# models 
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

# metrics
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,confusion_matrix,precision_recall_curve


global xtiscks 
global LR_arg
global RF_arg

LR_arg={'penalty':'l2','dual':False,'tol':0.0001,'C':1.0,'fit_intercept':True,
   'intercept_scaling':1,'class_weight':None,'random_state':None,'solver':'lbfgs',
   'max_iter':500,'multi_class':'auto','verbose':0,'warm_start':False,'n_jobs':4,'l1_ratio':None}

RF_arg={'n_estimators':10,'criterion':'gini','max_depth':None,'min_samples_split':2,'min_samples_leaf':1,
'min_weight_fraction_leaf':0.0,'max_features':'auto','max_leaf_nodes':None,'min_impurity_decrease':0.0,
'min_impurity_split':None,'bootstrap':True,'oob_score':False,'n_jobs':4,'random_state':None,'verbose':0,
'warm_start':False,'class_weight':None}
    
def tuning_C_given(SV_ls):
    """
    given model list of last training , delve into the peak range and search better performance
    """
    best_C=np.argmax([model.auc for model in SV_ls])
    C_ls = [model.C for model in SV_ls]
    if best_C != 0:
        lower=C_ls[best_C-1]
    else:
        lower=C_ls[best_C]
    if best_C != len(C_ls) -1:
        upper=C_ls[best_C+1]
    else:
        upper=C_ls[best_C]

    return np.linspace(lower,upper,10)

def printting_training_result(SV_ls):
    print('-----------------------')
    print('   C \t \t  AUC')
    print('-----------------------')
    for model in SV_ls:
        print(' %.3f \t \t %.4f'%(model.C,model.auc))
    print('-----------------------')
    
def load_SVM_model(seed,kernel,C,data=None):
    if data is None:
        data = read_known_DF(data_path).values
    y=np.array([0]*463 + [1]*(611 - 463))
    set_ls,idx_ls = spliting_datas_by_the_same([data],y,True,**{'idx':y,'random_state':seed})
    SVM=single_model(model=SVC,DataSet=set_ls[0],with_test=True,**{'C':C,'probability':True,'kernel':kernel})
    return SVM
    
# ================================================
#               PLOTTING
# ================================================
xtiscks = {'ticks':range(0,8),
           'labels':['raw data','log data','norm data','erbb2 data',
                     'raw_PCA','log_PCA','norm_PCA','erbb2_PCA']}

def plot_PR_curve(model_ls,test=0,label=xtiscks['labels']):
    
    
    for i in range(len(model_ls)):
        model_ls[i].get_metrics(test)
        plt.plot(model_ls[i].pr[1],model_ls[i].pr[0],label=label[i])
        
    plt.ylabel('precision');
    plt.xlabel('recall');
    plt.legend()
    
def plot_auc_f1(model_ls,test=0,**kwarg):
    for i in range(len(model_ls)):
        model_ls[i].get_metrics(test)
    plt.plot([model.auc for model in model_ls],label='Roc Auc');
    plt.plot([model.f1 for model in model_ls],label='f1');
    plt.xticks(**kwarg)
    plt.legend()
    
    
# ================================================
#               single model
# ================================================

class single_model(object):
    """ 
    within a Dataset , model and **kwarg of model ,we can auto matically compute metrics and plot graph
    """
    def __init__(self,DataSet,model,with_test=True,**kwarg):
        if with_test:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = DataSet
        else:
             self.X_train, self.X_val, self.y_train, self.y_val = DataSet
        self._model = model(**kwarg).fit(self.X_train,self.y_train)
        #self._model.fit(X_train,y_train)

    #  == validation set == 
        # predict
    def get_metrics(self,test=0,verbose=False):
        """ return in the order of AUC F1 CM PR ROC
        """
        X = self.X_val if test == 0 else self.X_test
        y = self.y_val if test == 0 else self.y_test
        
        proba_dict = {'y_true':y,'y_score':self._model.predict_proba(X)[:,1]}
        probas_dict = {'y_true':y,'probas_pred':self._model.predict_proba(X)[:,1]}
        pred_dict =  {'y_true':y,'y_pred':self._model.predict(X)}
        # metrics of validation set y_true,
    
        self.cm = confusion_matrix(**pred_dict)
        self.f1 = f1_score(**pred_dict)
        
        self.pr = precision_recall_curve(**probas_dict)
        
        self.roc = roc_curve(**proba_dict)
        self.auc = roc_auc_score(**proba_dict)
        
        if verbose:
            return self.auc,self.f1,self.cm,self.pr,self.roc

# ================================================
#               LDA related
# ================================================
''' 
def LDA_model(DataSet):
    
    X_train,X_val,X_test,y_train,y_val,y_test = DataSet
    model = LDA()
    model.fit(X_train,y_train)
    
    #  == validation set == 
    # predict
    model.y_proba=model.predict_proba(X_val)
    model.pred_val =  model.predict(X_val)
    # metrics of validation set
    model.cm = confusion_matrix(y_val,model.pred_val)
    model.f1 = f1_score(y_val,model.predict(X_val))
    model.pr = precision_recall_curve(y_true=y_val,probas_pred=model.y_proba[:,1])
    model.roc = roc_curve(y_true=y_val,y_score=model.y_proba[:,1])
    model.auc = roc_auc_score(y_true=y_val,y_score=model.y_proba[:,1])
    
    #  == test set == 
    # predict
    model.y_proba_test=model.predict_proba(X_test)
    model.pred_test =  model.predict(X_test)
    # metrics of validation set
    model.cm_test = confusion_matrix(y_test,model.pred_test)
    model.f1_test = f1_score(y_test,model.pred_test)
    model.pr_test = precision_recall_curve(y_true=y_test,probas_pred=model.y_proba_test[:,1])
    model.roc_test = roc_curve(y_true=y_test,y_score=model.y_proba_test[:,1])
    model.auc_test = roc_auc_score(y_true=y_test,y_score=model.y_proba_test[:,1])
    return model
'''    

# ================================================
#               SVM related
# ================================================

def SVC_model(DataSet,with_test=True,**kwarg):
    """ 
    given Training set and validation set, return a model with y_proba,roc curve,ROC_AUC
    arguments:    
        X_train, np array
        X_val,   np array
        y_train, shape (n,)
        y_val,   shape (n,)
        **kwarg  : the key word arguments to class SVC()
    """
    if with_test:
        X_train,X_val,X_test,y_train,y_val,y_test = DataSet
    else:
        X_train,X_val,y_train,y_val = DataSet
    model = SVC(probability=True,**kwarg)
    model.fit(X_train,y_train)
    
    #  == validation set == 
    # predict
    model.y_proba=model.predict_proba(X_val)
    model.pred_val =  model.predict(X_val)
    # metrics of validation set
    model.cm = confusion_matrix(y_val,model.pred_val)
    model.f1 = f1_score(y_val,model.predict(X_val))
    model.pr = precision_recall_curve(y_true=y_val,probas_pred=model.y_proba[:,1])
    model.roc = roc_curve(y_true=y_val,y_score=model.y_proba[:,1])
    model.auc = roc_auc_score(y_true=y_val,y_score=model.y_proba[:,1])
    
    if with_test:
        #  == test set == 
        # predict
        model.y_proba_test=model.predict_proba(X_test)
        model.pred_test =  model.predict(X_test)
        # metrics of validation set
        model.cm_test = confusion_matrix(y_test,model.pred_test)
        model.f1_test = f1_score(y_test,model.pred_test)
        model.pr_test = precision_recall_curve(y_true=y_test,probas_pred=model.y_proba_test[:,1])
        model.roc_test = roc_curve(y_true=y_test,y_score=model.y_proba_test[:,1])
        model.auc_test = roc_auc_score(y_true=y_test,y_score=model.y_proba_test[:,1])
    return model

def AUC_trace(model_ls,formatt='%.2f'):
    """
    input a list of SVM model, plot the roc_auc trace
    """
    C_ls = [model.C for model in model_ls]
    auc_ls = [model.auc for model in model_ls]
    plt.figure(figsize=(0.6*len(C_ls),6))
    plt.plot(auc_ls);
    plt.title('auc to C',fontsize=18);
    plt.ylabel('Area Under Curve',fontsize=12);
    plt.xlabel('hyper-parameter : C',fontsize=12);
    plt.xticks(np.arange(0,len(auc_ls)),[formatt%f for f in C_ls],fontsize=11);
    plt.yticks(fontsize=12);
    

vec_aa_dict = {0: 'A', 1: 'V', 2: 'I', 3: 'L', 4: 'M', 5: 'F', 6: 'Y', 7: 'W', 8: 'S', 9: 'T', 10: 'N', 11: 'Q', 12: 'R', 13: 'H', 14: 'K', 15: 'D', 16: 'E', 17: 'C', 18: 'U', 19: 'G', 20: 'P', 21: '-',22:'&'}

aa_vec_dict = {'A': 0, 'V': 1, 'I': 2, 'L': 3, 'M': 4, 'F': 5, 'Y': 6, 'W': 7, 'S': 8, 'T': 9, 'N': 10, 'Q': 11, 'R': 12, 'H': 13, 'K': 14, 'D': 15, 'E': 16, 'C': 17, 'U': 18, 'G': 19, 'P': 20, '-': 21, 'B': 22, 'J': 22, 'X': 22, 'Z': 22}

def flip_dict(dic):
    key=list(dic.keys())
    values=list(dic.values())
    if len(values) != len(np.unique(values)):
        return dict(zip(values,key))
    else:
        raise ValueError('flipped dict key has multiple pointer')

ratio_of = lambda y : np.sum(y==1)/len(y)        

class SVM_omega(object):
    global aa_vec_dict
    def __init__(self,coef,encoder_dict=aa_vec_dict,threshold='scale',keep2raw=None):  
        global aa_vec_dict
        global vec_aa_dict
        self.coef = coef[0]
        self.n_alphabet = len(np.unique(list(encoder_dict.values())))
        self.seq_len = len(self.coef)/self.n_alphabet
        self.threshold = np.quantile(np.abs(self.coef[np.where(self.coef != 0)[0]]),0.85) if threshold=='scale' else threshold
        self.aa2int = encoder_dict
        self.int2aa = vec_aa_dict if encoder_dict is aa_vec_dict else flip_dict(self.aa2int)
        self.important_sites = np.where(np.abs(self.coef)>self.threshold)[0]
        self.sites_weight = self.coef[self.important_sites]
        self.sites_in_seq = self.important_sites%self.seq_len
        self.sites_position = self.important_sites//self.seq_len       
        self.sites_aa = [self.int2aa[i] for i in self.sites_position]
        self.DF = self.dataframe()
        
        if len(self.important_sites) == 0:
            self.non_zero_hist()
            raise ValueError('Invalid threshold, please rechoose a threshold or try "scale"')
        
        if keep2raw is not None:
            self.raw_sites=np.array(keep2raw)[self.sites_in_seq.astype(int)]
            self.DF = self.dataframe(keep2raw=keep2raw)
                        
    def non_zero_hist(self,bins=20,**kwarg):
        plt.figure(figsize=(8,6))
        plt.hist(self.coef[np.where(self.coef != 0)[0]],bins=bins,**kwarg)
        
    def dataframe(self,columns=['sites_in_seq','sites_aa', 'sites_weight','sites_position'],keep2raw=None):
        if keep2raw is not None:
            columns=['raw_sites','sites_aa', 'sites_weight','sites_position']
        dataframe=pd.DataFrame(dict(zip(columns,[self.__getattribute__(x) for x in columns]))).sort_values(columns[0])
        dataframe.index = range(dataframe.shape[0])
        return dataframe
    
    def letter_plot(self,cmap=cm.cubehelix,**kwarg):
        figsize=(self.DF.shape[0]/4,3)
        plt.figure(figsize=figsize)
        plt.plot(range(self.DF.shape[0]),[1]*self.DF.shape[0],color='white')
        for i in range(self.DF.shape[0]):
            plt.text(i,1,self.DF.iloc[i,1],size=13+2*self.DF.iloc[i,2],color=cmap(self.DF.iloc[i,3]/22),**kwarg)
        