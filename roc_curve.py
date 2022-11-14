from re import X
from turtle import title
import pandas as pd
import matplotlib.pyplot as plt
import json
from pprint import pprint
import numpy as np
import seaborn as sns
import os 
from ntpath import join

def plot_curve(dataset):
    relevants = np.load(f'recall_precision/{dataset}/relevants.npy')
    idxs_resnet = np.load(f'recall_precision/{dataset}/idxs.npy')
    idxs_veteb = np.load(f'recall_precision/{dataset}CLIPBASE/idxs.npy')
    pprint(idxs_resnet.shape)
    pprint(idxs_veteb.shape)

    recall_aux = []
    precision_aux = []
    recall_aux_veteb = []
    precision_aux_veteb = []
    for i, unordered_relevant in enumerate(relevants):

        # ResNet-50
        ids = idxs_resnet[i]
        relevant = unordered_relevant[ids]
        r_aux = []
        p_aux = []
        n = len(relevant[relevant == 1])
        n_rel = 0
        for j, r in enumerate(relevant):
            if r:
                n_rel += 1
                r_aux.append( n_rel / n)
                p_aux.append( n_rel / (j + 1) )
        if n_rel != 0:
            recall_aux.append(r_aux)
            precision_aux.append(p_aux)

        # VETE-B
        ids_veteb = idxs_veteb[i]
        relevant_veteb = unordered_relevant[ids_veteb]
        r_aux_veteb = []
        p_aux_veteb = []
        n_veteb = len(relevant_veteb[relevant_veteb == 1])
        n_rel_veteb = 0
        for k, r in enumerate(relevant_veteb):
            if r:
                n_rel_veteb += 1
                r_aux_veteb.append( n_rel_veteb / n_veteb)
                p_aux_veteb.append( n_rel_veteb / (k + 1) )
        if n_rel_veteb != 0:
            recall_aux_veteb.append(r_aux_veteb)
            precision_aux_veteb.append(p_aux_veteb)


    recall_aux = np.array(recall_aux)
    precision_aux = np.array(precision_aux)

    recall = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    precision = []
    for i in range(len(precision_aux)):
        this_precision = []
        this_recall_aux = np.array(recall_aux[i])
        this_precision_aux = np.array(precision_aux[i])
        for r in recall:
            indices = np.where(this_recall_aux >= r)
            #pprint(this_precision_aux)
            #pprint(indices[0])
            this_precision.append(np.max(this_precision_aux[indices[0]]))
        
        precision.append(this_precision)
    precision = np.mean(np.array(precision), axis=0)

    recall_aux_veteb = np.array(recall_aux_veteb)
    precision_aux_veteb = np.array(precision_aux_veteb)

    precision_veteb = []
    for i in range(len(precision_aux_veteb)):
        this_precision_veteb = []
        this_recall_aux_veteb = np.array(recall_aux_veteb[i])
        this_precision_aux_veteb = np.array(precision_aux_veteb[i])
        for r in recall:
            indices = np.where(this_recall_aux_veteb >= r)
            #pprint(this_precision_aux_veteb)
            #pprint(indices[0])
            this_precision_veteb.append(np.max(this_precision_aux_veteb[indices[0]]))
        
        precision_veteb.append(this_precision_veteb)
    precision_veteb = np.mean(np.array(precision_veteb), axis=0)

    pprint(precision)
    pprint(precision_veteb)
    pprint(recall)

    df1 = pd.DataFrame({'x': recall, 'y': precision, 'model': 'ResNet-50'})
    df2 = pd.DataFrame({'x': recall, 'y': precision_veteb, 'model': 'VETE-B'})

    df = pd.concat([df1, df2])
    sns.set_theme()

    #ax = sns.lineplot(data=df, x='x', y='y', hue='model')
    ax = sns.lineplot(data=df, x='x', y='y', hue='model')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision vs Recall')
    plt.show()

if __name__ == '__main__':
    dataset = 'Pepeganga'

    if "Pepeganga" in dataset:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            real_queries_path = "real_queries_2"
            which_realq = "RealQ_2"
    else:
        real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
        real_queries_path = "real_queries"
        which_realq = "RealQ_3"


    eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
    eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

    with open(f'{dataset}_real_images.txt', 'a') as the_file:
        for f in eval_files:
            the_file.write(f'{f}\n')
    
    # plot_curve(dataset)



