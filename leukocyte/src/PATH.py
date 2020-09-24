import os,sys
sys.path.append(os.path.dirname(__file__))
global data_path
global top_path
global tsne_path
global report_path
global leukocyte_path
global leukocyte_path2

top_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path=os.path.join(top_path,'data')
leukocyte_path = os.path.join(data_path,'leukocyte_ratio')
leukocyte_path2 = os.path.join(data_path,'leukocyte_ratio2')
tsne_path = os.path.join(top_path,'leukocyte','t_sne')
report_path = os.path.join(top_path,'leukocyte','report')

sys.path.append(top_path)


# for path in [data_path,top_path,tsne_path]:
#     if not os.path.exists(path):
#         os.path.mkdir(path) # correct code ?