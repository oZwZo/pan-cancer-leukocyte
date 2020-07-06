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

def main(csv):
    DF = utils.read_leukocyte(data_path,True,csv=csv)
    DF = DF.fillna(0)
    print('---------------------------------------------------------')
    print('\n read {} data '.format(csv))
    pdf = PdfPages(os.path.join(report_path,csv.replace('csv','pdf')))

    # heatmap violin and data filtering
    DF,statistics = statistics_preprocess(DF)
    DF = DF.fillna(0)
    print('\n preprocesing ... ')
    pdf.savefig()

    # violin plot
    data = DF.iloc[:,1:].values
    v_data = data - np.mean(data,axis=0)
    violin(v_data,statistics)
    pdf.savefig()

    if csv is not None:
        print('\n start T-SNE decomposition')
        t_sne = t_sne_tuning(data,csv.replace('csv','npy'))
        pdf.savefig()

    print('\n HC cluster')
    Z = hierarchical_tree(data)       # Z is for later use
    pdf.savefig()

    print('\n agglomerative, K-means and GMM')
    labels,metrics = cluster3(data)

    metrics_curve(metrics)
    pdf.savefig()

    t_sne_label(t_sne,labels,metrics)  # there's vote inside
    pdf.savefig()

    label_vote = {}
    for method in labels.keys():
        label_vote[method] = labels[method][metrics[method].vote()]
        
       
    cluter_in_all(DF,label_vote,csv.replace('.csv',''))
    pdf.savefig()
    
    #sort data 
    sorted_data,labels = sort_data_by_label(data,labels,metrics,method='km')
    
    stack_barplot(sorted_data,labels,Z)
    pdf.savefig()
    
    tree_heatmap(sorted_data,Z)
    pdf.savefig()
    
    pdf.close()
    
    for method in ['ac','km']:
        DF.loc[:,method] = label_vote[method]
    DF[['id','ac','km']].to_csv(os.path.join(report_path,"cluster_label_"+csv))
    
    print(' pdf saved \n End of analysis')
    print('---------------------------------------------------------')

parser = argparse.ArgumentParser('script to do cluster analysis ')
parser.add_argument('--csv',type=str,default='ALL.csv',help='pls pass file name of csv under leukocyte_ratio')
args = parser.parse_args()

if args.csv == 'iter':
    csvs = os.listdir(os.path.join(data_path,'leukocyte_ratio/'))
    csvs = list(filter(lambda x: 'csv' in x, csvs))
    for csv in csvs:
        main(csv)
else:
    main(args.csv)