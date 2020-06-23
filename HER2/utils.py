import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

###########################
#######    read    ########
##########################

def read_combot():
    print("""whole = read_data_DF(data_path)
known_df = read_known_DF(data_path)
to_predict=read_predict_DF(data_path)""")
    
def read_all_label(data_path):
    """
    given the dir,read xlsx file and rename the columns 
    """
    rename_dict = {'A0_Samples': 'sample',
                   'breast_carcinoma_estrogen_receptor_status': 'ER',
                   'breast_carcinoma_progesterone_receptor_status': 'PR',
                   'HER2 status_IHC YC': 'HER2'}
    return pd.read_excel(os.path.join(data_path,'BRACA_clinic.xlsx')).rename(rename_dict,axis=1)

def read_label(data_path):
    return pd.read_csv(os.path.join(data_path,'TCGA_BRCA_Her2.csv')).rename({'A0_Samples':'sample', 'HER2 status_IHC YC':'status'},axis=1)

def read_data_DF(data_path):
    return pd.read_csv(os.path.join(data_path,'tumor_only_clean.csv')).rename({'Unnamed: 0':'ID'},axis=1).set_index('ID')

def read_predict_DF(data_path):
    return pd.read_csv(os.path.join(data_path,'to_pred_DF.csv')).set_index('sample')

def read_known_DF(data_path):
    return pd.read_csv(os.path.join(data_path,'KNOWN_DF.csv')).set_index('sample')


def read_leukocyte(data_path,separate=False):
    """
    read leukocyte rate, return the ALL.csv by default,
    when set `separate=True` will return a dict containing 10 kinds cancer
    """
    if separate:
        pds= {}
        for csv in os.listdir(data_path+'/leukocyte_ratio'):
            if csv == 'ALL.csv':
                continue              # skip ALL.csv
            fn = csv.split('.')[0]
            pds[fn.upper()] = pd.read_csv(os.path.join(data_path,'leukocyte_ratio',csv))
    else:
        pds = pd.read_csv(os.path.join(data_path,'leukocyte_ratio','ALL.csv'))
    
    return pds

def save_labels2DF(labels,name,pds):
    sample_name = pds.id
    column_name = ['TCGA_id']+[str(i)+'_clusters' for i in range(5,20)]
    data = np.stack([sample_name.values] + labels)
    DF = pd.DataFrame(data.T,columns=column_name)
    DF.to_csv(os.path.join(data_path,name),index=False)

# -------merged func-----------

def get_known_sample(data_path,status):
    """given data_path, read the `all_label` and `tumor_only` table, and return DF and y
    ...done by merged them and return whose status = "Negative" status = "Postive"
    """
    label = read_all_label(data_path)
    data_DF = read_data_DF(data_path) # tumor only

    merged_DF=label.merge(data_DF,right_index=True,left_on=['sample'])
    merged_DF.index = range(merged_DF.shape[0])

    DF=merged_DF[merged_DF[status]=='Negative'].iloc[:,4:].append(
        merged_DF[merged_DF[status]=='Positive'].iloc[:,4:])

    N_nega=np.sum(merged_DF[status]=='Negative')
    y=np.array([0]*N_nega + [1]*(DF.shape[0] - N_nega))
    return DF,y

def get_unknown_sample(data_path,status):
    """given data_path, read the `all_label` and `tumor_only` table, and return DF and y
    ...done by merged them and return whose status = "Negative" status = "Postive"
    """
    label = read_all_label(data_path)
    data_DF = read_data_DF(data_path) # tumor only

    merged_DF=label.merge(data_DF,right_index=True,left_on=['sample'])
    merged_DF.index = range(merged_DF.shape[0])

    DF=merged_DF[(merged_DF[status]!='Negative')&(merged_DF[status]!='Positive')].set_index('sample').iloc[:,3:]

    return DF

def read_expression_matrix(path):
    ret=[]
    gene=[]
    with open(path,'r') as f:
        header = f.readline().strip('\n').split('\t')
        rest = f.readlines()
        f.close()

    for line in rest:
        split_ls=line.strip('\n').split('\t')
        gene.append(split_ls[0])
        ret.append(split_ls)

    ret=np.array(ret)#,dtype=np.float32)
    return ret,header,gene

################################
####    preprocess data      ###
################################
def triple_preprocess(DF):
    """
    will return in the order of raw log norm erbb2
    """
    data = DF.values
    data_norm = scale(data)
    data_log = scale(np.log(data+0.01))
    data_erbb2 = scaling_by_gene(data,'ERBB2',DF.columns)
    return [data,data_log,data_norm,data_erbb2]

################################
####      spliting data      ###
################################
ratio_of = lambda y : np.sum(y ==1 )/len(y)

def split_check_ratio(X,y,idx=None,val=True,size1=0.3,size2=0.1,**kwarg):
    """
    return `X_train, X_val, X_test, y_train, y_val, y_test` when val=True
    """
    # split test from others
    X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=size1,**kwarg)
    if not val:
        
        # ratios
        if idx is not None:
            ratios = [ratio_of(x) for x in [idx[y],idx[y_1],idx[y_test]]]
        else:
            ratios = [ratio_of(x) for x in [y,y_1,y_test]]
        # printting
        for Set, ratio in zip(['whole','train','test'],ratios):
            print('the ratio of {}\tset is {}'.format(Set,ratio))
        return X_1, X_test, y_1, y_test
    else:
        # split train from val
        X_train, X_val, y_train, y_val = train_test_split(X_1, y_1, test_size=size2,**kwarg)
        
        if idx is not None:
            ratios = [ratio_of(x) for x in [idx[y],idx[y_train],idx[y_val],idx[y_test]]]
        else:
            ratios = [ratio_of(x) for x in [y,y_train,y_val,y_test]]
        # printting
        print('------------------------------------')
        print('Data Set    Raio of positive Sample')
        print('------------------------------------')
        for Set, ratio in zip(['whole','train','val','test'],ratios):
            print(' {}\t | \t {}'.format(Set,ratio))
        print('------------------------------------')
        return X_train, X_val, X_test, y_train, y_val, y_test

def spliting_datas_by_the_same(datas,y,val=True,**kwargs):
    '''
     given a List of data, split them by func ` split_check_ratio ` in the same index
    '''
    idx = range(datas[0].shape[0])
    if val:
        # use idx = y , for ratios need y[idx_train] .etc
        X_train,X_val,X_test,idx_train,idx_val,idx_test = split_check_ratio(datas[0],idx,**kwargs) 
        set_ls = [(data[idx_train],data[idx_val],data[idx_test],y[idx_train],y[idx_val],y[idx_test]) for data in datas]
    else:
        X_train,X_test,idx_train,idx_test = split_check_ratio(datas[0],idx,**kwargs) 
        set_ls = [(data[idx_train],data[idx_test],y[idx_train],y[idx_test]) for data in datas]
    return set_ls,[idx_train,idx_val,idx_test]

def scaling_by_gene(x,gene,column):
    gene_idx = np.where(column == gene)[0][0]
    mean = np.mean(x,axis=0)
    x = x - mean
    gene_relative = x[:,gene_idx]
    for i in range(x.shape[0]):
        x[i,:] = x[i,:] / gene_relative[i]
    print('scale by gene {} in column {}'.format(gene,gene_idx))
    return x

def count_freq(Array):
    """given a list or np array, return a DataFrame counting occurency of each element  
    """
    Array = np.array(Array)
    item = np.unique(Array)
    freq = [np.sum(Array == i) for i in item]
    return pd.DataFrame({'item':item,'frequency':freq})

def keep_identical(List):
    List = np.array(List)
    counts = count_freq(List)

    dup=counts[counts.frequency != 1].item.values

    discard=np.array([np.where(List == item)[0] for item in dup])[:,1]

    keep = [i for i in range(len(List)) if i not in discard]

    return keep,discard