import os,sys
sys.path.append(os.path.dirname(__file__))
global data_path
global top_path
global tsne_path
global report_path
global leukocyte_path

data_path='/home/ZwZ/database/BRACA'
top_path = '/home/ZwZ/script/HER2_prediction'
tsne_path = '/home/ZwZ/script/HER2_prediction/leukocyte/t_sne'
report_path = '/home/ZwZ/script/HER2_prediction/leukocyte/cluster_report'
leukocyte_path = os.path.join(data_path,'leukocyte_ratio')

sys.path.append(top_path)


for path in [data_path,top_path,tsne_path]:
    if not os.path.exists(path):
        os.path.mkdir(path) # correct code ?