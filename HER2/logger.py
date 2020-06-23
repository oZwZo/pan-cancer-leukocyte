import re
import numpy as np
import pandas as pd

match_ratio = lambda x :list(re.match(r'\s*(\w{,10})\s*\|\s*([\d,\.]*)\s',x).groups())
match_performance = lambda x :list(re.match(r'\s{0,1}(\w*\s\w*)[\s,\|]*'+r' ([\d,\.]{,10})[\s,\|]*'*4,x).groups())

class Mlog(object):
    """
    a class to analyze log generate during training model
    """
    def __init__(self,log_path,interval=25,seed=True,compute_all=True):
        with open(log_path,'r') as f:
            log = f.readlines()[2:]  # the first 10 lines are unnaccessary info like posi of erbb2 and primary split
            f.close()
        self.log = log
        self._interval=interval   # may changed according to model
        if compute_all:
            self.all_perform=np.stack([self.performance_of(i).values for i in range(self.__len__())])
            self.val_f1auc_sum = self.all_perform[:,:,0] + self.all_perform[:,:,1]
            self.test_f1auc_sum = self.all_perform[:,:,2] + self.all_perform[:,:,3]
        if seed:
            self.all_seed = np.array([self.seed_of(i) for i in range(self.__len__())])
        
    def __len__(self):
        return len(self.log)//self._interval
    
    def __getitem__(self,index):
        start=self._interval*index
        end = self._interval*(index+1)
        return self.log[start:end]
    
    def seed_of(self,index,place=2):
        """return the seed of a single repeat
        arg:
        ...place: the position of line where seed is recorded
        """
        posi = self._interval*index + place 
        line = self.log[posi]
        rgx = re.match('\s*(seed \: )(\d{0,10})\s*',line)
        try:
            seed = int(rgx.group(2))
        except:
            raise ValueError('reg-exp not match')
        return seed
    
    def ratio_of(self,index,matcher=match_ratio,Columns=['SET','RATIO_OF_POSI']):
        """a method to return ratio of a single repeat (index) of the log, 
        args:
        ...matcher: the re matcher to recognize pattern and return a list
        ...Columns: the columns name of dataframe
        """
        log_i = self.__getitem__(index)
        ratio_ls = log_i[8:12]
        ratio = np.stack([matcher(line) for line in ratio_ls])
        return pd.DataFrame(ratio,columns=Columns).set_index('SET').astype(float)
        
    def performance_of(self,index,matcher=match_performance,Columns=['PREPROCESS','V_F1','V_AUCROC','T_F1','T_AUCROC']):
        """a method to return ratio of a single repeat (index) of the log, 
        args:
        ...matcher: the re matcher to recognize pattern and return a list
        ...Columns: the columns name of dataframe
        """
        log_i = self.__getitem__(index)
        start = log_i.index('     \t \t |  f1  \t auc \t|  f1  \t  auc \t  |\n')
        perform_ls = log_i[start+2:start+6]
        perform = np.stack([matcher(line) for line in perform_ls])
        return pd.DataFrame(perform,columns=Columns).set_index("PREPROCESS").astype(float)
    
    
################################
####      for verbose        ###
################################

def vobosing(val,test):
    Labels=['raw data', 'log data', 'norm data', 'erbb2 data']
    print('\n')
    print('-----------------------------------------------------------')
    print('data \t \t |   \t val  \t \t|      test \t')
    print('     \t \t |  f1  \t auc \t|  f1  \t  auc \t  |')
    print('-----------------------------------------------------------')
    for i in range(4):
        print('%s \t | %.5f   %.5f  \t| %.5f  %.5f'%(Labels[i],val[i][0],val[i][1],test[i][0],test[i][1]))
    print('-----------------------------------------------------------')
    
def vobosing_SVM(vvv,ttt,data,C):
    vvv = np.array(vvv)
    ttt = np.array(ttt)
    print('  DATA : %s'%data)
    print('----'*22)
    print(' C \t | {} \t|   {} \t|    {}  \t|   {}  \t|   {} \t'.format(C[0],C[1],C[2],C[3],C[4]))
    print('----'*22)
    for i in range(2):
        print('%s \t | %.5f \t| %5.f \t| %.5f \t| %.5f \t|  %.5f \t'%('v:'+[' f1',' auc'][i],vvv[0,i],vvv[1,i],vvv[2,i],vvv[3,i],vvv[4,i]))
    print('----'*22)
    for i in range(2):
        print('%s \t | %.5f \t| %5.f \t| %.5f \t| %.5f \t|  %.5f \t'%('t:'+[' f1',' auc'][i],ttt[0,i],ttt[1,i],ttt[2,i],ttt[3,i],ttt[4,i]))
    print('----'*22)