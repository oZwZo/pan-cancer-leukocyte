from PATH import *
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import my_metrics
import argparse
from leukocyte_cluster import *
from matplotlib.backends.backend_pdf import PdfPages

#                               ------------ parser ------------
parser = argparse.ArgumentParser("script for automatic analysis and reporting cluster result")
parser.add_argument('--csv',type=str,default=None,help='the cancer type to use, will open all file seperatively if not given')
args = parser.parse_args()
#                               ------------------------------

DF = utils.read_leukocyte(data_path,True,csv=args.csv)
print('\n read {} data  \n'.format(args.csv))
pdf = PdfPages(os.path.join(report_path,args.csv.replace('csv','pdf')))

# heatmap violin and data filtering
DF = statistics_preprocess(DF)
print('\n preprocesing ... \n')
pdf.savefig()

# violin plot
data = DF.iloc[:,1:].values
v_data = data - np.mean(data,axis=0)
violin(v_data,DF)
pdf.savefig()

if args.csv is not None:
    print('\n start T-SNE decomposition')
    t_sne_tuning(data,args.csv.replace('csv','npy'))
    pdf.savefig()
    
print('\n HC cluster \n')
hierarchical_tree(data) 
pdf.savefig()


print('\n agglomerative, K-means and GMM \n')
labels,metrics = cluster3(data)

metrics_curve(metrics)
pdf.savefig()
pdf.close()
print(' pdf saved \n End of analysis')
