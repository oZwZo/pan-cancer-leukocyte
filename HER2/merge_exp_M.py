import numpy as np
import pandas as pd
import os
from PATH import data_path
import argparse
from tqdm import tqdm
# ---------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser('the python script to merge all expression txt file into single csv file')
parser.add_argument('--dir',type=str,default=None,help='should set to None for ST server,default dir is the pan... under the data_path')
parser.add_argument('--out',type=str,default=None,help='should set to None for ST server, otherwise should pass a abs path of .csv tail')
args = parser.parse_args()

# ---------------------------------------------------------------------------------------------------------

#                         ------ set path ------
path = os.path.join(data_path,'pan-cancer-EXPM') if (args.dir is None) else args.dir
out = os.path.join(data_path,'pan_cancer.csv') if (args.out is None) else args.out
#                           ----------------
    
def read_exp_table(tables,paths=path):
    table_path = os.path.join(paths,tables)
    table = pd.read_table(table_path).set_index('id')
    return table.sort_index().transpose()
    
tables = list(filter(lambda x: 'txt' in x,os.listdir(path)))

print('read data .....\n \n \n')
matrix_s = [read_exp_table(table) for table in tables]

# -------- append all the matrix togather --------

merged_df = matrix_s[0]  
print('merging table ...')
for i in tqdm(range(1,len(matrix_s))):
    table = matrix_s[i]
    assert(table.shape[1] == merged_df.shape[1])
    assert(np.sum(np.array(table.columns) == np.array(merged_df.columns)) == len(merged_df.columns) )
    merged_df = merged_df.append(table)  # remember to reassign 
    
# [table.shape[1] for table in matrix_s]
# [np.sum(np.array(table.columns) == np.array(merged_df.columns)) == len(merged_df.columns) for table in matrix_s]

# ---------- write to csv file ----------------


print( '\n writing ..... \n')

merged_df.to_csv(out,index=True)

print('\t writed !!! ')
print('\t save to %s' %out)