from genericpath import exists
from ntpath import join
from re import search
import sys
from tkinter.tix import Tree
from scipy import spatial
#Please, change the following path to where convnet2 can be located
sys.path.append("F:\Documentos Universidad\MEMORIA\convnet_visual_attributes\convnet2")
import io
import tensorflow as tf
import datasets.data as data
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.io as io
import skimage.transform as trans
import os
import argparse
import numpy as np
import pandas as pd
import statistics
import umap
import umap.plot
import hdbscan
import pathlib
import torch
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import textsearch
from pprint import pprint
from visual_text_parameters import parameters, test_parameters, best_parameters
from data_utils import prepare_dataset
from bpm_parameters import *
from visual_text_nn import VTNN, VTNN8dim, VTNN128dim
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageDraw, ImageFont
from clip_ssearch import CLIPSSearch
from pprint import pprint

#DATA_DIR = /home/vision/smb-datasets/VisualAttributes
#SEARCH_DIR = /home/vision/smb-datasets/VisualAttributes

class SSearch :
    def __init__(self, config_file, model_name):
        
        self.configuration = conf.ConfigurationFile(config_file, model_name)
        #defiing input_shape                    
        self.input_shape =  (self.configuration.get_image_height(), 
                             self.configuration.get_image_width(),
                             self.configuration.get_number_of_channels())                       
        #loading the model
        model = tf.keras.applications.ResNet50(include_top=True, 
                                               weights='imagenet', 
                                               input_tensor=None, 
                                               input_shape =self.input_shape, 
                                               pooling=None, 
                                               classes=1000)
        #redefining the model to get the hidden output
        #self.output_layer_name = 'conv4_block6_out'
        self.output_layer_name = 'avg_pool'
        output = model.get_layer(self.output_layer_name).output
        #output = tf.keras.layers.GlobalAveragePooling2D()(output)                
        self.sim_model = tf.keras.Model(model.input, output)        
        model.summary()
        #self.sim_model.summary()
        
        #defining image processing function
        #self.process_fun =  imgproc.process_image_visual_attribute
        self.process_fun =  imgproc.process_image
        #loading catalog
        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), 'ssearch')
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt')        
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file)
        print('loading catalog ...')
        self.load_catalog(catalog_file)
        print('loading catalog ok ...')
        self.enable_search = False        
        
    #read_image
    def read_image(self, filename):      
        #print("Reading {}".format(filename))  
        im = self.process_fun(data.read_image(filename, self.input_shape[2]), (self.input_shape[0], self.input_shape[1]))        
        #for resnet
        im = tf.keras.applications.resnet50.preprocess_input(im)    
        return im
    
    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")                        
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype = np.int32)
        self.features = np.fromfile(fvs_file, dtype = np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')
        
    def load_catalog(self, catalog):
        with open(catalog) as f_in :
            self.filenames = [filename.strip() for filename in f_in ]
        self.data_size = len(self.filenames)    
            
    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]
        
    def compute_features(self, image, expand_dims = False):
        #image = image - self.mean_image
        if expand_dims :
            image = tf.expand_dims(image, 0)        
        fv = self.sim_model.predict(image)            
        return fv
    
    def normalize(self, data) :
        """
        unit normalization
        """
        norm = np.sqrt(np.sum(np.square(data), axis = 1))
        norm = np.expand_dims(norm, 0)  
        #print(norm)      
        data = data / np.transpose(norm)
        return data
    
    def square_root_norm(self, data) :
        return self.normalize(np.sign(data)*np.sqrt(np.abs(data)))

    def adjust_query_embedding(self, query, original_embeddings, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(query[0]))

        if decide:
            r_filenames = self.get_filenames(idx_sorted[:top])
            categories = []
            for i, file in enumerate(r_filenames):
                base = os.path.basename(file)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                try:
                    category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "GlobalCategoryEN"].values[0]
                except:
                    category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == name_and_productid[1]), "GlobalCategoryEN"].values[0]
                categories.append(category)
            
            adjust = all(x == categories[0] for x in categories)
            if adjust:
                #print("Decided to adjust")
                return new_query
            else:
                #print("Decided to NOT adjust")
                return query

        return new_query


    def adjust_query_embedding_sim(self, query, original_embeddings, text_model, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        #print(query.shape)
        #print("len(query[0]): ", len(query[0]))
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(query[0]))

        if decide:
            r_filenames = self.get_filenames(idx_sorted[:top])
            categories = []
            for i, file in enumerate(r_filenames):
                base = os.path.basename(file)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                try:
                    product_description = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "ProductDescriptionEN"].values[0]
                except:
                    product_description = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == name_and_productid[1]), "ProductDescriptionEN"].values[0]
                #print("Description {}: {}".format(i, product_description))
                if text_model.model_name == "clip-base":
                    text_input = text_model.tokenizer(product_description, truncate=True).to(text_model.device)
                    with torch.no_grad():
                        text_features = text_model.model.encode_text(text_input)
                    text_features = text_features.cpu().numpy()[0]
                    data = text_features.astype(np.float32)
                    categories.append(data)
                elif text_model.model_name == "roberta":
                    data = text_model.model.encode(product_description)
                    categories.append(data)
                
            cos_sim_1 = 1 - spatial.distance.cosine(categories[0], categories[1])
            cos_sim_2 = 1 - spatial.distance.cosine(categories[0], categories[2])
            cos_sim_3 = 1 - spatial.distance.cosine(categories[1], categories[2])
            cos_sim_list = [cos_sim_1, cos_sim_2, cos_sim_3]
            adjust = all(cos_sim >= 0.8 for cos_sim in cos_sim_list)
            #adjust = True
            if adjust:
                #print("Decided to adjust")
                return new_query
            else:
                #print("NOT adjusting")
                return query
                #return new_query
        return new_query

 
    def search(self, im_query, metric = 'l2', norm = 'None', top=90, reducer=None, vtnn=None, adjust_query=False, adjust_query_sim=False, original_embeddings=None, df=None, text_model=None):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims = True)
        if adjust_query:
            q_fv = self.adjust_query_embedding(query=q_fv, original_embeddings=original_embeddings, top=3, df=df)
        if adjust_query_sim:
            q_fv = self.adjust_query_embedding_sim(query=q_fv, original_embeddings=original_embeddings, text_model=text_model, top=3, df=df)
        if vtnn is not None:
            q_fv = torch.tensor(q_fv)
            vtnn.eval()
            with torch.no_grad():
                q_fv = q_fv.to('cuda')
                q_fv = q_fv.view(-1, 2048)
                q_fv = vtnn(q_fv).cpu().numpy()
        #print("EMBEDDING SIZE: {}".format(len(q_fv[0])))
        #it seems that Euclidean performs better than cosine
        if metric == 'l2':
            if reducer is not None:
                data = reducer.transform(self.features)
                query = reducer.transform(q_fv)
            else:
                data = self.features
                query = q_fv
            if norm == 'square_root':
                data = self.square_root_norm(data)
                query = self.square_root_norm(query)
            #print("Query features:", query.shape)
            #print("data features:", data.shape)
            d = np.sqrt(np.sum(np.square(data - query[0]), axis = 1))
            idx_sorted = np.argsort(d)
            d_sorted = np.sort(d)
        elif metric == 'cos' : 
            if norm == 'square_root':
                self.features = self.square_root_norm(self.features)
                q_fv = self.square_root_norm(q_fv)
            sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
            sim = np.reshape(sim, (-1))            
            idx_sorted = np.argsort(-sim)
            d_sorted = -np.sort(-sim)
            #print(sim[idx_sorted][:20])
        #print("idx_sorted: ", idx_sorted[:top])
        if top is not None:
            return idx_sorted[:top], d_sorted[:top], q_fv, data
        return idx_sorted, d_sorted, q_fv, data
                                
    def compute_features_from_catalog(self):
        n_batch = self.configuration.get_batch_size()        
        images = np.empty((self.data_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = np.float32)
        for i, filename in enumerate(self.filenames) :
            if i % 1000 == 0:
                print('reading {}'.format(i))
                sys.stdout.flush()
            images[i, ] = self.read_image(filename)        
        n_iter = np.int(np.ceil(self.data_size / n_batch))
        result = []
        for i in range(n_iter) :
            print('iter {} / {}'.format(i, n_iter))  
            sys.stdout.flush()             
            batch = images[i*n_batch : min((i + 1) * n_batch, self.data_size), ]
            result.append(self.compute_features(batch))
        fvs = np.concatenate(result)    
        print('fvs {}'.format(fvs.shape))    
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)       
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def draw_result(self, filenames, write_data=False, similarity=None, distance=None):
        w = 1000
        h = 1000
        #w_i = np.int(w / 10)
        w_i = int(w / 10)
        #h_i = np.int(h / 10)
        h_i = int(h / 10)
        image_r = np.zeros((w,h,3), dtype = np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames) :
            pos = (i * w_i)
            x = pos % w
            #y = np.int(np.floor(pos / w)) * h_i
            y = int(np.floor(pos / w)) * h_i
            image = data.read_image(filename, 3)
            
            if write_data:
                ### Add text with the product id
                try:
                    base = os.path.basename(filename)
                    filename = os.path.splitext(base)[0]
                    name_and_productid = filename.rsplit('_', 1)
                    font = ImageFont.truetype("arial.ttf", 30)
                    PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
                    draw = ImageDraw.Draw(PIL_image)
                    if (similarity is None and distance is None) or (i == 0):
                        draw.text((0, 0), "id: {}".format(name_and_productid[1]), font=font, fill='rgb(0, 0, 0)')
                    elif similarity is not None:
                        draw.text((0, 0), "id: {} / sim: {}".format(name_and_productid[1], round(similarity[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
                    elif distance is not None:
                        draw.text((0, 0), "id: {} / dist: {}".format(name_and_productid[1], round(distance[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
                except:
                    #print("Could not write id for product.")
                    pass
                image = np.array(PIL_image)

            image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
            image_r[y:y+h_i, x : x +  w_i, :] = image              
        return image_r    
                    
#unit test  

def get_ordered_relevants(r_filenames, dataframe, real_df=None):
    from pprint import pprint
    #print("Empezando")
    df = dataframe
    #pprint(r_filenames)
    relevants = []
    pprint(r_filenames[:3])

    base = os.path.basename(r_filenames[0])
    filename = os.path.splitext(base)[0]
    name_and_productid = filename.rsplit('_', 1)

    base_gc = real_df[real_df['Title'] == name_and_productid[0]]["GlobalCategoryEN"].values[0]
    dataframe['relevant'] = dataframe.apply(lambda x: 1 if base_gc == x['GlobalCategoryEN'] else 0, axis=1)
    for _, file in enumerate(r_filenames[1:]):
        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]
        name_and_productid = filename.rsplit('_', 1)
        #pprint(name_and_productid)
        # try:
        #     print('try')
        #     gc = df[(df['Title'] == name_and_productid[0]) & (str(df['ProductId']) == name_and_productid[1])]['relevant'].values[0]
        #     relevants.append(gc)
        # except:
        #print('except')
        #gc = df[(df['Title'] == name_and_productid[0])  & (df['ProductId'] == name_and_productid[1])]['relevant'].values[0] #Homy
        gc = df[(df['Title'] == name_and_productid[0])  & (df['ProductId'] == int(name_and_productid[1]))]['relevant'].values[0] # Pepeganga
        #df = df.drop([gc.index[0]])
        #print(len(df))
        relevants.append(gc)
    #print("Terminooo")
    return relevants


def get_product_and_category(r_filenames, dataframe, real_df=None):
    df = dataframe
    products = []
    base_category = ""
    for i, file in enumerate(r_filenames):
        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]
        name_and_productid = filename.rsplit('_', 1)
        #print("name_and_productid: {}".format(name_and_productid))
        #real_datasets = ["Pepeganga", "Homy", "WorldMarket", "IKEA", "WorldMarket"]
        #if any(rq_dataset in dataset for rq_dataset in real_datasets):
        if real_df is not None:
        #if use_real_queries and (dataset == "Pepeganga" or dataset == "Homy" or dataset == "WorldMarket" or dataset == "IKEA" or dataset == "WorldMarket2" or dataset == "WorldMarket3"):
            #real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_RealQ/data/questions_dataset.xlsx") # REALQ
            #if "Pepeganga" in dataset:
            #    real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            #else:
            #    real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
            try:
                #category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "GlobalCategoryEN"].iloc[0]
                categories = df.loc[(df['Title'] == name_and_productid[0]) & (str(df['ProductId']) == name_and_productid[1]), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            except:
                try: 
                    #category = df.loc[df['Title'] == name_and_productid[0], category_field].iloc[0]
                    categories = df.loc[df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
                except:
                    categories = real_df.loc[real_df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            if i == 0:
                #base_category = category
                #print("BASE CATEGORY: ", base_category)
                base_categories = categories
                #print("BASE CATEGORY: ", base_categories)
            else:
                #file_info = [filename, category]
                file_info = [filename, categories[0], categories[1], categories[2]]
                #print("Product {}: {}, {} (GC), {} (CT)".format(i, file_info[0], file_info[1], file_info[2]))
                products.append(file_info)

        else:
            try:
                #category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "GlobalCategoryEN"].iloc[0]
                categories = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            except: 
                try:
                    categories = df.loc[(df['Title'] == name_and_productid[0]) & (str(df['ProductId']) == name_and_productid[1]), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
                except:
                    #print("Two exceptions on name_and_productid: {}".format(name_and_productid))
                    #category = df.loc[df['Title'] == name_and_productid[0], category_field].iloc[0]
                    categories = df.loc[df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            if i == 0:
                #base_category = category
                #print("BASE CATEGORY: ", base_category)
                base_categories = categories
                #print("BASE CATEGORY: ", base_categories)
            else:
                #file_info = [filename, category]
                file_info = [filename, categories[0], categories[1], categories[2]]
                #print("Product {}: {}, {} (GC), {} (CT)".format(i, file_info[0], file_info[1], file_info[2]))
                products.append(file_info)


    return base_categories, products


def recall_precision(relevants):
    n_relevant = 0

    recall = []
    precision = []
    n = len(relevants)

    pos = 1 
    for i, r in enumerate(relevants):
        if r:
            n_relevant += 1
            recall.append((i + 1)/n)
            precision.append(n_relevant/pos)
        pos += 1


    return recall, precision


def avg_precision(y, y_pred, use_all_categories=False, homy=False):
    p = 0
    n_relevant = 0
    pos = 1

    if homy:
        for product in y_pred:
            contains = any(i in y.split(",") for i in product[1].split(","))
            if contains:
            #if product[1] == y:
                n_relevant += 1
                p += n_relevant / pos
            pos += 1
    
        if n_relevant != 0:
            ap = p / n_relevant
        else:
            ap = 0 

        return ap       

    if use_all_categories:
        p_tree = 0
        n_relevant_tree = 0
        pos_tree = 1
        p_sub = 0
        n_relevant_sub = 0
        pos_sub = 1
        for product in y_pred:
            if product[1] == y[0]:
                n_relevant += 1
                p += n_relevant / pos
            pos += 1
            if product[2] == y[1]:
                n_relevant_tree += 1
                p_tree += n_relevant_tree / pos_tree
            pos_tree += 1
            if product[3] == y[2]:
                n_relevant_sub += 1
                p_sub += n_relevant_sub / pos_sub
            pos_sub += 1
        
        if n_relevant != 0:
            ap = p / n_relevant
        else:
            ap = 0 

        if n_relevant_tree != 0:
            ap_tree = p_tree / n_relevant_tree
        else:
            ap_tree = 0

        if n_relevant_sub != 0:
            ap_sub = p_sub / n_relevant_sub
        else:
            ap_sub = 0 

        return ap, ap_tree, ap_sub
    
    else:
        for product in y_pred:
            if product[1] == y:
                n_relevant += 1
                p += n_relevant / pos
            pos += 1
    
        if n_relevant != 0:
            ap = p / n_relevant
        else:
            ap = 0 

        return ap


def calculate_features_and_labels(ssearch, df, load=False, config="base", feats_type="visual"):
    vec = []
    lab = []
    lab_integers = []
    if feats_type == "visual":
        if not load:
            for i, filename in enumerate(os.listdir("F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_Pepeganga/data/images")):
                print("Product {}".format(i))
                product = os.path.splitext(filename)[0]
                name_and_productid = product.rsplit('_', 1)
                category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), 'GlobalCategoryEN'].iloc[0]
                im_query = ssearch.read_image("F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_Pepeganga/data/images/" + filename)
                v = ssearch.compute_features(im_query, expand_dims = True)[0]
                #print("v: ", v)
                vec.append(v)
                lab.append(category)
                lab_integers.append(get_label_int(category))
                #if i == 100:
                #    break
            np.savetxt("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/feats.txt".format(feats_type, config), vec, newline='\n')
            with open("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/labels.txt".format(feats_type, config), 'w') as fp:
                for line in lab:
                    fp.write(line + "\n")
            return vec, lab, lab_integers
        else:
            vec = np.loadtxt("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/{}/feats.txt".format(feats_type, config))
            with open("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/{}/labels.txt".format(feats_type, config), 'r') as fp:
            #    lab = fp.readlines()
                lab = fp.read().splitlines()
            
            for element in lab:
                lab_integers.append(get_label_int(element))
            return vec, lab, lab_integers
    
    elif feats_type == "text":
        if not load:
            #for i, filename in enumerate(os.listdir("F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_Pepeganga/data/images")):
            for i, row in df.iterrows():
                print("Product {}".format(i))
                #product = os.path.splitext(filename)[0]
                #name_and_productid = product.rsplit('_', 1)
                category = row['GlobalCategoryEN']
                v = row['descVector']
                #print("v: ", v)
                vec.append(v)
                lab.append(category)
                lab_integers.append(get_label_int(category))
                #if i == 100:
                #    break
            np.savetxt("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/{}/feats.txt".format(feats_type, config), vec, newline='\n')
            with open("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/{}/labels.txt".format(feats_type, config), 'w') as fp:
                for line in lab:
                    fp.write(line + "\n")
            return vec, lab, lab_integers
        else:
            vec = np.loadtxt("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/{}/feats.txt".format(feats_type, config))
            with open("F:/Documentos Universidad/MEMORIA/Datasets/feats_and_labels/{}/{}/labels.txt".format(feats_type, config), 'r') as fp:
            #    lab = fp.readlines()
                lab = fp.read().splitlines()
            
            for element in lab:
                lab_integers.append(get_label_int(element))
            return vec, lab, lab_integers


def get_label_int(l):
    if l == 'Babies':
        return 0
    elif l == 'Beauty':
        return 1
    elif l == 'Bedroom':
        return 2
    elif l == 'Christmas Decoration':
        return 3
    elif l == 'Clothes and Shoes':
        return 4
    elif l == 'Food and Drinks':
        return 5
    elif l == 'Furniture':
        return 6
    elif l == 'Home':
        return 7
    elif l == 'Home Appliances':
        return 8
    elif l == 'Pets':
        return 9
    elif l == 'School':
        return 10
    elif l == 'Sports':
        return 11
    elif l == 'Technology':
        return 12
    elif l == 'Toy Store':
        return 13
    elif l == 'Videogames and Consoles':
        return 14


def remove_repeated(idx, dist_sorted):
    new_dist_sorted = []
    new_idx = []
    last_dist = None
    for i, d in enumerate(dist_sorted):
        if last_dist != d:
            new_dist_sorted.append(d)
            new_idx.append(idx[i])
            last_dist = d
    return new_idx, new_dist_sorted

def remove_repeated_files(r_filenames):
    new_filenames = []
    repeated_titles = []
    last_title = ""
    for path in r_filenames:
        #print("Path: ", path)
        file = os.path.basename(path).split(".")[0]
        #print("File: ", file)
        title = file.split("_")[0]
        #print("Title: ", title)
        if last_title == title:
            repeated_titles.append(title)
            last_title = title
        if title not in repeated_titles:
            new_filenames.append(path)
    return new_filenames



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)                
    parser.add_argument("-mode", type=str, choices = ['search', 'compute', 'eval', 'eval-umap', 'eval-umap-clip', 'search-umap', 'plot-umap', 'eval-plot-umap', 'eval-text', 'search-text-visual', 'eval-text-visual', 'eval-all-text-visual', 'eval-vtnn', 'eval-all-vtnn', 'eval-all-umap', 'utils', 'testing', "clip-testing", "eval-text-visual-clip", "recall-precision"], help=" mode of operation", required = True)
    parser.add_argument("-list", type=str,  help=" list of image to process", required = False)
    parser.add_argument("-odir", type=str,  help=" output dir", required = False, default = '.')
    pargs = parser.parse_args()     
    configuration_file = pargs.config        
    ssearch = SSearch(pargs.config, pargs.name)
    metric = 'l2'
    norm = 'None'
    
    #dataset = "Pepeganga"
    dataset = "PepegangaCLIPBASE"
    #dataset = "PepegangaCLIPFN"
    #dataset = "Cartier"
    #dataset = "CartierCLIPBASE"
    #dataset = "CartierCLIPFN"
    #dataset = "IKEA"
    #dataset = "IKEACLIPBASE"
    #dataset = "IKEACLIPFN"
    #dataset = "UNIQLO"
    #dataset = "UNIQLOCLIPBASE"
    #dataset = "UNIQLOCLIPFN"
    #dataset = "WorldMarket"
    #dataset = "WorldMarketCLIPBASE"
    #dataset = "WorldMarketCLIPFN"
    #dataset = "Homy"
    #dataset = "HomyCLIPBASE"
    #dataset = "HomyCLIPFN"

    if pargs.mode == 'compute' :        
        ssearch.compute_features_from_catalog()        
    if pargs.mode == 'search' :
        ssearch.load_features()        
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames  = [ item.strip() for item in f_list]
            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric)         
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)# 
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap = avg_precision(base_category, products)
                print(ap)
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                print("Largo de r_filenames: {}\n".format(len(r_filenames)))             
        else :
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, top=99)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                
                # mAP
                #base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                #ap = avg_precision(base_category, products)
                #print("Average precision: {}".format(ap))  
                #####
                
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                fquery = input('Query:')
    if pargs.mode == 'search-umap' :
        ssearch.load_features() 
        if True:
            print("Training UMAP")
            n_neighbors = 15
            min_dist = 0.1
            n_components = 32
            n_epochs = 500
            vectors = np.asarray(ssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            reducer = umap.UMAP(n_components=n_components, n_epochs=n_epochs, random_state=42) #n_neighbors=15, min_dist=0.1
            reducer.fit(vectors)     
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames  = [ item.strip() for item in f_list]
            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)# 
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap = avg_precision(base_category, products)
                print(ap)  
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result_umap.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                print("Largo de r_filenames: {}\n".format(len(r_filenames)))             
        else :
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, top=99, reducer=reducer)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap = avg_precision(base_category, products)
                print(ap)  
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result_umap.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                print("r_filenames: {}\n".format(r_filenames))  
                fquery = input('Query:')

    
    if pargs.mode == 'eval' :
        
        use_real = True
        metric = 'l2'
        #metric = 'cos'
        norm = 'None'
        #norm = 'square_root'
        padding = "0"

        ssearch.load_features()
        eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
        real_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/real_queries".format(dataset)
        if use_real:
            # Real images
            eval_files = ["{}/real_queries/".format(dataset) + f for f in os.listdir(real_path) if os.path.isfile(join(real_path, f))]
        else:
            # Test images
            eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        
        #print("EVAL FILES: {}".format(eval_files))

        eval_data = []
        
        if pargs.list is None:
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []
            for fquery in eval_files:
                print(fquery)
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, norm, top=99)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                if not use_real:
                    base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                    ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    print("Average precision for {}: {}".format(fquery, ap)) 
                    ap_arr.append(ap)
                    ap_arr_tree.append(ap_tree)
                    ap_arr_sub.append(ap_sub)
                else:
                    base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                    ap = avg_precision(base_category, products, homy=True)
                    print("Average precision for {}: {}".format(fquery, ap)) 
                    ap_arr.append(ap)
                
                #''' Create images
                image_r= ssearch.draw_result(r_filenames)
                if use_real:
                    output_name = os.path.basename(fquery) + '_result.png'
                    output_name = os.path.join('./{}/results_real_queries/{}_{}_padding_{}'.format(dataset, metric, norm, padding), output_name)
                else:
                    output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                    output_name = os.path.join('./{}/results'.format(dataset), output_name)
                io.imsave(output_name, image_r)
                #'''
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))
                if not use_real:
                    eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap, ap_tree, ap_sub])
                else:
                    eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap])

            if not use_real:
                df = pd.DataFrame(eval_data, columns=['fquery', 'AP (GlobalCategory)', 'AP (TreeCategory)', 'AP (SubCategory)'])
                df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/results_visual.xlsx".format(dataset, "base"))
                mAP = statistics.mean(ap_arr)
                mAP_tree = statistics.mean(ap_arr_tree)
                mAP_sub = statistics.mean(ap_arr_sub)
                print("mAP Global: {}".format(mAP))
                print("mAP Tree: {}".format(mAP_tree))
                print("mAP Sub: {}".format(mAP_sub))
            else:
                df = pd.DataFrame(eval_data, columns=['fquery', 'AP'])
                df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/{}_{}_padding_{}/results_visual.xlsx".format(dataset, "base", metric, norm, padding))
                mAP = statistics.mean(ap_arr)
                print("mAP Global: {}".format(mAP))


    if pargs.mode == 'eval-umap-clip' :

        # REAL HOMY
        #which_realq = "RealQ_3"  # REAL HOMY
        which_realq = "RealQ_2"  # REAL PEPEGANGA

        model_name = "clip-base"
        clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        
        #model_name = "clip-fn"
        #checkpoint_path = "F:/Documentos Universidad\MEMORIA\CLIP_models/{}/model1.pt".format(dataset)
        #clipssearch = CLIPSSearch(pargs.config, pargs.name, checkpoint_path=checkpoint_path)
        
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if "Pepeganga" in dataset:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
        else:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3

        
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset, model_name=model_name)
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.efficient_cosine_similarity()
        clipssearch.load_features()

        pepeganga_rq_parameters = {"k_7_a_03_mean_4_dim_29_seed": (7, 0.3, None, False, 'mean', 4, 29),
                                    "k_7_a_03_mean_4_dim_27_seed": (7, 0.3, None, False, 'mean', 4, 27),
                                    "k_7_a_03_mean_4_dim_5_seed": (7, 0.3, None, False, 'mean', 4, 5),
                                    "k_7_a_03_mean_4_dim_18_seed": (7, 0.3, None, False, 'mean', 4, 18),
                                    "k_7_a_03_mean_4_dim_22_seed": (7, 0.3, None, False, 'mean', 4, 22),
                                    "k_7_a_03_mean_8_dim_29_seed": (7, 0.3, None, False, 'mean', 8, 29),
                                    "k_7_a_03_mean_8_dim_27_seed": (7, 0.3, None, False, 'mean', 8, 27),
                                    "k_7_a_03_mean_8_dim_5_seed": (7, 0.3, None, False, 'mean', 8, 5),
                                    "k_7_a_03_mean_8_dim_18_seed": (7, 0.3, None, False, 'mean', 8, 18),
                                    "k_7_a_03_mean_8_dim_22_seed": (7, 0.3, None, False, 'mean', 8, 22),
                                    "k_7_a_03_mean_16_dim_29_seed": (7, 0.3, None, False, 'mean', 16, 29),
                                    "k_7_a_03_mean_16_dim_27_seed": (7, 0.3, None, False, 'mean', 16, 27),
                                    "k_7_a_03_mean_16_dim_5_seed": (7, 0.3, None, False, 'mean', 16, 5),
                                    "k_7_a_03_mean_16_dim_18_seed": (7, 0.3, None, False, 'mean', 16, 18),
                                    "k_7_a_03_mean_16_dim_22_seed": (7, 0.3, None, False, 'mean', 16, 22),
                                    "k_7_a_03_mean_32_dim_29_seed": (7, 0.3, None, False, 'mean', 32, 29),
                                    "k_7_a_03_mean_32_dim_27_seed": (7, 0.3, None, False, 'mean', 32, 27),
                                    "k_7_a_03_mean_32_dim_5_seed": (7, 0.3, None, False, 'mean', 32, 5),
                                    "k_7_a_03_mean_32_dim_18_seed": (7, 0.3, None, False, 'mean', 32, 18),
                                    "k_7_a_03_mean_32_dim_22_seed": (7, 0.3, None, False, 'mean', 32, 22),
                                    "k_7_a_03_mean_64_dim_29_seed": (7, 0.3, None, False, 'mean', 64, 29),
                                    "k_7_a_03_mean_64_dim_27_seed": (7, 0.3, None, False, 'mean', 64, 27),
                                    "k_7_a_03_mean_64_dim_5_seed": (7, 0.3, None, False, 'mean', 64, 5),
                                    "k_7_a_03_mean_64_dim_18_seed": (7, 0.3, None, False, 'mean', 64, 18),
                                    "k_7_a_03_mean_64_dim_22_seed": (7, 0.3, None, False, 'mean', 64, 22),}

        homy_rq_parameters = {"k_3_t_0.5_softmax_4_dim_29_seed": (3, None, 0.5, True, 'softmax', 4, 29),
                            "k_3_t_0.5_softmax_4_dim_27_seed": (3, None, 0.5, True, 'softmax', 4, 27),
                            "k_3_t_0.5_softmax_4_dim_5_seed": (3, None, 0.5, True, 'softmax', 4, 5),
                            "k_3_t_0.5_softmax_4_dim_18_seed": (3, None, 0.5, True, 'softmax', 4, 18),
                            "k_3_t_0.5_softmax_4_dim_22_seed": (3, None, 0.5, True, 'softmax', 4, 22),
                            "k_3_t_0.5_softmax_8_dim_29_seed": (3, None, 0.5, True, 'softmax', 8, 29),
                            "k_3_t_0.5_softmax_8_dim_27_seed": (3, None, 0.5, True, 'softmax', 8, 27),
                            "k_3_t_0.5_softmax_8_dim_5_seed": (3, None, 0.5, True, 'softmax', 8, 5),
                            "k_3_t_0.5_softmax_8_dim_18_seed": (3, None, 0.5, True, 'softmax', 8, 18),
                            "k_3_t_0.5_softmax_8_dim_22_seed": (3, None, 0.5, True, 'softmax', 8, 22),
                            "k_3_t_0.5_softmax_16_dim_29_seed": (3, None, 0.5, True, 'softmax', 16, 29),
                            "k_3_t_0.5_softmax_16_dim_27_seed": (3, None, 0.5, True, 'softmax', 16, 27),
                            "k_3_t_0.5_softmax_16_dim_5_seed": (3, None, 0.5, True, 'softmax', 16, 5),
                            "k_3_t_0.5_softmax_16_dim_18_seed": (3, None, 0.5, True, 'softmax', 16, 18),
                            "k_3_t_0.5_softmax_16_dim_22_seed": (3, None, 0.5, True, 'softmax', 16, 22),
                            "k_3_t_0.5_softmax_32_dim_29_seed": (3, None, 0.5, True, 'softmax', 32, 29),
                            "k_3_t_0.5_softmax_32_dim_27_seed": (3, None, 0.5, True, 'softmax', 32, 27),
                            "k_3_t_0.5_softmax_32_dim_5_seed": (3, None, 0.5, True, 'softmax', 32, 5),
                            "k_3_t_0.5_softmax_32_dim_18_seed": (3, None, 0.5, True, 'softmax', 32, 18),
                            "k_3_t_0.5_softmax_32_dim_22_seed": (3, None, 0.5, True, 'softmax', 32, 22),
                            "k_3_t_0.5_softmax_64_dim_29_seed": (3, None, 0.5, True, 'softmax', 64, 29),
                            "k_3_t_0.5_softmax_64_dim_27_seed": (3, None, 0.5, True, 'softmax', 64, 27),
                            "k_3_t_0.5_softmax_64_dim_5_seed": (3, None, 0.5, True, 'softmax', 64, 5),
                            "k_3_t_0.5_softmax_64_dim_18_seed": (3, None, 0.5, True, 'softmax', 64, 18),
                            "k_3_t_0.5_softmax_64_dim_22_seed": (3, None, 0.5, True, 'softmax', 64, 22),}
        

        top=20
        adjust_query = True
        adjust_query_sim = False
        metric = 'l2'
        norm = 'None'
        
        original_features = np.copy(clipssearch.features)
        new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=3, a=None, T=0.5, use_query=True, method='softmax') ## HOMY
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=7, a=0.3, T=None, use_query=False, method='mean') ## PEPEGANGA
        #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
        clipssearch.features = new_visual_embeddings



        # USE OR NOT REAL QUERIES FOR EVALUATION
        eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
        eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        
        eval_data = []
        for k, params in pepeganga_rq_parameters.items():
        
            ap_arr = []
            ap_arr_tree = []

            reducer = None
            print(f"Training UMAP: {params[5]}-dim, seed {params[6]}")
            #n_neighbors = 15
            #min_dist = 0.1
            n_components = params[5]
            vectors = np.asarray(clipssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            #42
            reducer = umap.UMAP(n_components=n_components, random_state=params[6]) #n_neighbors=n_neighbors, min_dist=min_dist
            reducer.fit(vectors)
            print("Done training UMAP")

            for fquery in eval_files:
                #print(fquery)
                im_query = clipssearch.read_image(fquery)

                #idx = ssearch.search(im_query, metric, top=20)
                idx, dist_array = clipssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, adjust_query_sim=adjust_query_sim, original_embeddings=original_features, df=data_df, reducer=reducer, text_model=textSearch)                
                r_filenames = clipssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                #r_filenames_top_20 = r_filenames[:21]  # Top 20
                base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                ap, ap_tree, _ = avg_precision(base_category, products, use_all_categories=True)
                #print("{}: {}".format(fquery, ap))  
                #print("Average precision for {}: {} (GC), {} (CT)".format(fquery, ap, ap_tree)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                
            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            eval_data.append([k, mAP, mAP_tree, params[5], params[6]])
            
        df = pd.DataFrame(eval_data, columns=['params', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'UMAP dim', 'seed'])
        df_path = "F:/Documentos Universidad/MEMORIA/F/paper/CLIP/Real/{}/UMAP/gc_3_no_q.xlsx".format(dataset)
        #pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
        df.to_excel(df_path, index=False)



        #########################################

    if pargs.mode == 'eval-umap' :

        ##### Only when using text
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name="word2vec")
        #textSearch.set_gensim_model() # word2vec
        #textSearch.set_dataframes()
        #textSearch.efficient_cosine_similarity()
        # Text Visual Parameters
        k = 5
        alpha = 0.6
        T = None
        use_query = False
        method = 'mean'
        #######################
        ssearch.load_features()

        # When using text
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(ssearch.features, ssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
        #ssearch.features = new_visual_embeddings

        eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
        eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        if True:
            print("Training UMAP")
            #n_neighbors = 15
            #min_dist = 0.1
            n_components = 16
            n_epochs = 500
            vectors = np.asarray(ssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_epochs=n_epochs) #n_neighbors=n_neighbors, min_dist=min_dist
            reducer.fit(vectors)
        if pargs.list is None: #### IS NONE
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []
            i = 0
            for fquery in eval_files:
                i += 1
                print(fquery)
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, top=20, reducer=reducer)           
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                print("Average precision for {}: {}".format(fquery, ap)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                ap_arr_sub.append(ap_sub)
                #image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                #output_name = os.path.join('./results_umap', output_name)
                #io.imsave(output_name, image_r)
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                #if i == 10:
                #    break
            
            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            mAP_sub = statistics.mean(ap_arr_sub)
            print("mAP global: {}".format(mAP))
            print("mAP tree: {}".format(mAP_tree))
            print("mAP sub: {}".format(mAP_sub))




    if pargs.mode == 'eval-all-umap' :

        ##### Only when using text
        #model_name="roberta"
        #textSearch = textsearch.TextSearch(model_name=model_name, filter_by_nwords=False, build=True, dataset=dataset)
        #textSearch.set_model()
        #textSearch.set_dataframes()
        #textSearch.efficient_cosine_similarity()

        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        # Text Visual Parameters

        use_multiple_seeds = True

        # ROBERTA
        #this_parameters = {"k_3_t_0.5_softmax_32_dim_29": (3, None, 0.5, True, 'softmax', 32, 29),
        #                    "k_3_t_0.5_softmax_32_dim_27": (3, None, 0.5, True, 'softmax', 32, 27),
        #                    "k_3_t_0.5_softmax_32_dim_5": (3, None, 0.5, True, 'softmax', 32, 5),
        #                    "k_3_t_0.5_softmax_32_dim_18": (3, None, 0.5, True, 'softmax', 32, 18),
        #                    "k_3_t_0.5_softmax_32_dim_22": (3, None, 0.5, True, 'softmax', 32, 22),
        #                "k_3_t_0.5_softmax_16_dim_29": (3, None, 0.5, True, 'softmax', 16, 29),
        #                "k_3_t_0.5_softmax_16_dim_27": (3, None, 0.5, True, 'softmax', 16, 27),
        #                "k_3_t_0.5_softmax_16_dim_5": (3, None, 0.5, True, 'softmax', 16, 5),
        #                "k_3_t_0.5_softmax_16_dim_18": (3, None, 0.5, True, 'softmax', 16, 18),
        #                "k_3_t_0.5_softmax_16_dim_22": (3, None, 0.5, True, 'softmax', 16, 22),
        #                "k_3_t_0.5_softmax_8_dim_29": (3, None, 0.5, True, 'softmax', 8, 29),
        #                "k_3_t_0.5_softmax_8_dim_27": (3, None, 0.5, True, 'softmax', 8, 27),
        #                "k_3_t_0.5_softmax_8_dim_5": (3, None, 0.5, True, 'softmax', 8, 5),
        #                "k_3_t_0.5_softmax_8_dim_18": (3, None, 0.5, True, 'softmax', 8, 18),
        #                "k_3_t_0.5_softmax_8_dim_22": (3, None, 0.5, True, 'softmax', 8, 22),
        #                "k_3_t_0.5_softmax_4_dim_29": (3, None, 0.5, True, 'softmax', 4, 29),
        #                "k_3_t_0.5_softmax_4_dim_27": (3, None, 0.5, True, 'softmax', 4, 27),
        #                "k_3_t_0.5_softmax_4_dim_5": (3, None, 0.5, True, 'softmax', 4, 5),
        #                "k_3_t_0.5_softmax_4_dim_18": (3, None, 0.5, True, 'softmax', 4, 18),
        #                "k_3_t_0.5_softmax_4_dim_22": (3, None, 0.5, True, 'softmax', 4, 22),
        #                }
        # Word2vec
        #this_parameters = {"k_7_a_03_mean_32_dim_29": (7, 0.3, None, False, 'mean', 32, 29),
        #                    "k_7_a_03_mean_32_dim_27": (7, 0.3, None, False, 'mean', 32, 27),
        #                    "k_7_a_03_mean_32_dim_5": (7, 0.3, None, False, 'mean', 32, 5),
        #                    "k_7_a_03_mean_32_dim_18": (7, 0.3, None, False, 'mean', 32, 18),
        #                    "k_7_a_03_mean_32_dim_22": (7, 0.3, None, False, 'mean', 32, 22),
        #                "k_7_a_03_mean_16_dim_29": (7, 0.3, None, False, 'mean', 16, 29),
        #                "k_7_a_03_mean_16_dim_27": (7, 0.3, None, False, 'mean', 16, 27),
        #                "k_7_a_03_mean_16_dim_5": (7, 0.3, None, False, 'mean', 16, 5),
        #                "k_7_a_03_mean_16_dim_18": (7, 0.3, None, False, 'mean', 16, 18),
        #                "k_7_a_03_mean_16_dim_22": (7, 0.3, None, False, 'mean', 16, 22),
        #                "k_7_a_03_mean_8_dim_29": (7, 0.3, None, False, 'mean', 8, 29),
        #                "k_7_a_03_mean_8_dim_27": (7, 0.3, None, False, 'mean', 8, 27),
        #                "k_7_a_03_mean_8_dim_5": (7, 0.3, None, False, 'mean', 8, 5),
        #                "k_7_a_03_mean_8_dim_18": (7, 0.3, None, False, 'mean', 8, 18),
        #                "k_7_a_03_mean_8_dim_22": (7, 0.3, None, False, 'mean', 8, 22),
        #                "k_7_a_03_mean_4_dim_29": (7, 0.3, None, False, 'mean', 4, 29),
        #                "k_7_a_03_mean_4_dim_27": (7, 0.3, None, False, 'mean', 4, 27),
        #                "k_7_a_03_mean_4_dim_5": (7, 0.3, None, False, 'mean', 4, 5),
        #                "k_7_a_03_mean_4_dim_18": (7, 0.3, None, False, 'mean', 4, 18),
        #                "k_7_a_03_mean_4_dim_22": (7, 0.3, None, False, 'mean', 4, 22),
        #                }
                        #"k_3_a_05_mean": (3, 0.5, None, False, 'mean'),

        this_parameters = {"base_64_dim_29": (None, None, None, None, 'base', 64, 29),
                            "base_64_dim_27": (None, None, None, None, 'base', 64, 27),
                            "base_64_dim_5": (None, None, None, None, 'base', 64, 5),
                            "base_64_dim_18": (None, None, None, None, 'base', 64, 18),
                            "base_64_dim_22": (None, None, None, None, 'base', 64, 22),
                            "base_32_dim_29": (None, None, None, None, 'base', 32, 29),
                            "base_32_dim_27": (None, None, None, None, 'base', 32, 27),
                            "base_32_dim_5": (None, None, None, None, 'base', 32, 5),
                            "base_32_dim_18": (None, None, None, None, 'base', 32, 18),
                            "base_32_dim_22": (None, None, None, None, 'base', 32, 22),
                        "base_16_dim_29": (None, None, None, None, 'base', 16, 29),
                        "base_dim_27": (None, None, None, None, 'base', 16, 27),
                        "base_16_dim_5": (None, None, None, None, 'base', 16, 5),
                        "base_16_dim_18": (None, None, None, None, 'base', 16, 18),
                        "base_16_dim_22": (None, None, None, None, 'base', 16, 22),
                        "base_8_dim_29": (None, None, None, None, 'base', 8, 29),
                        "base_8_dim_27": (None, None, None, None, 'base', 8, 27),
                        "base_8_dim_5": (None, None, None, None, 'base', 8, 5),
                        "base_8_dim_18": (None, None, None, None, 'base', 8, 18),
                        "base_8_dim_22": (None, None, None, None, 'base', 8, 22),
                        "base_4_dim_29": (None, None, None, None, 'base', 4, 29),
                        "base_4_dim_27": (None, None, None, None, 'base', 4, 27),
                        "base_4_dim_5": (None, None, None, None, 'base', 4, 5),
                        "base_4_dim_18": (None, None, None, None, 'base', 4, 18),
                        "base_4_dim_22": (None, None, None, None, 'base', 4, 22),
                        }

        #######################
        model_name = "clip-base"
        clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        clipssearch.load_features()

        top=20
        adjust_query = False
        adjust_query_sim = False
        metric = 'l2'
        norm = 'None'

        adjusted_embeddings = False
        original_features = np.copy(clipssearch.features)
        
        new_visual_embeddings = None
        #self, embeddings, filenames, k=3, method='mean', a=0.9, T=1.0, use_query=True
        # mean: k, a, T, use_query=False, method='mean'
        eval_data = []
        # mean: k, a, T, use_query=False, method='mean'
        params_set = this_parameters

        which_realq = "RealQ_3"

        if "Pepeganga" in dataset:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
        else:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3

        for k, params in params_set.items():
            ap_arr = []
            ap_arr_tree = []

            reducer = None
            print(f"Training UMAP: {params[5]}-dim, seed {params[6]}")
            #n_neighbors = 15
            #min_dist = 0.1
            n_components = params[5]
            vectors = np.asarray(clipssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            #42
            reducer = umap.UMAP(n_components=n_components, random_state=params[6]) #n_neighbors=n_neighbors, min_dist=min_dist
            reducer.fit(vectors)
            print("Done training UMAP")

            # USE OR NOT REAL QUERIES FOR EVALUATION
            eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
            eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
            for fquery in eval_files:
                #print(fquery)
                im_query = clipssearch.read_image(fquery)
                idx, dist_array = clipssearch.search(im_query, metric=metric, norm=norm, top=20, reducer=reducer, original_embeddings=original_features)           
                r_filenames = clipssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                ap, ap_tree, _ = avg_precision(base_category, products, use_all_categories=True)
                #print("Average precision for {}: {}".format(fquery, ap)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                #image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                #output_name = os.path.join('./results_umap', output_name)
                #io.imsave(output_name, image_r)
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                #if i == 10:
                #    break
            
            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            #mAP_sub = statistics.mean(ap_arr_sub)
            eval_data.append([k, n_components, mAP, mAP_tree])
            print("mAP global: {}".format(mAP))
            print("mAP tree: {}".format(mAP_tree))
            #print("mAP sub: {}".format(mAP_sub))
        df = pd.DataFrame(eval_data, columns=['params', 'UMAP dim', 'mAP (GlobalCategory)', 'mAP (CategoryTree)'])
        df_path = "F:/Documentos Universidad/MEMORIA/F/paper/CLIP/Real/{}/UMAP/baseline/gc_3_no_q.xlsx".format(dataset)
        #pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
        df.to_excel(df_path, index=False)
        
        #if use_multiple_seeds:
        #    if using_base:
        #        df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/umap_{}_multiple_seeds_new_base.xlsx".format(dataset, n_epochs))
        #    else:    
        #        df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/umap_{}_multiple_seeds_new.xlsx".format(dataset, textSearch.get_model_name(), n_epochs))
        #else:
        #    df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/results_{}_umap_{}.xlsx".format(dataset, textSearch.get_model_name(), model_name, n_epochs))

    if pargs.mode == 'plot-umap' :
        #ssearch.load_features()
        #data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_Pepeganga/"
        #df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_Pepeganga/data/"
        feats_type = "text"
        if feats_type == "visual":
            df = pd.read_excel(data_path + "answers_dataset.xlsx")
        elif feats_type == "text":
            model_name = "word2vec"
            df = pd.read_pickle(data_path + "{}/answers_dataset_vector.pkl".format(model_name))
        adjust_embeddings = False
        ssearch.load_features()
        if adjust_embeddings:
            textSearch = textsearch.TextSearch(model_name="word2vec", filter_by_nwords=False, build=False, dataset=dataset)
            #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
            textSearch.set_model()
            textSearch.set_dataframes()
            textSearch.efficient_cosine_similarity()
            new_visual_embeddings = textSearch.adjust_visual_embeddings(ssearch.features, ssearch.filenames, k=5, a=0.2, T=None, use_query=False, method="mean")
            ssearch.features = new_visual_embeddings


        vec, lab, lab_integers = calculate_features_and_labels(ssearch, df, load=True, config="base", feats_type=feats_type)
        #print("Vec: ", vec)
        vectors = np.array(vec)
        #print("Vectors: ", vectors)
        labels = np.array(lab)
        if True:
            print("Training UMAP")
            n_neighbors = 15
            min_dist = 0.1
            n_components = 2
            n_epochs = 500
            #vectors = np.asarray(ssearch.features)
            
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            reducer = umap.UMAP(random_state=42, min_dist=min_dist, n_neighbors=n_neighbors, n_epochs=n_epochs) #n_neighbors=15, min_dist=0.1
            mapper = reducer.fit(vectors)
            umap.plot.points(mapper, labels=labels)
            umap.plot.plt.show()
        umap.plot.plt.close()

    if pargs.mode == 'eval-plot-umap' :
        #ssearch.load_features()
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_Pepeganga/"
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        vec, lab, lab_integers = calculate_features_and_labels(ssearch, df, load=True)
        #print("Vec: ", vec)
        vectors = np.array(vec)
        #print("Vectors: ", vectors)
        labels = np.array(lab)
        labels_integers = np.array(lab_integers)
        if True:
            print("Training UMAP")
            n_neighbors = 15
            min_dist = 0.1
            n_components = 2
            #n_epochs = 500
            #vectors = np.asarray(ssearch.features)
            
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            standard_embedding = umap.UMAP(random_state=42, min_dist=0.1, n_neighbors=n_neighbors).fit_transform(vectors)
            reducer = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors=30, n_components=8)#, n_epochs=n_epochs) #n_neighbors=15, min_dist=0.1
            clusterable_embedding = reducer.fit_transform(vectors)
            hdbscan_labels = hdbscan.HDBSCAN(
                min_samples = 14,
                min_cluster_size=500,
            ).fit_predict(clusterable_embedding)

            
            clustered = (hdbscan_labels >= 0)
            plt.scatter(standard_embedding[~clustered, 0],
                        standard_embedding[~clustered, 1],
                        color=(0.5, 0.5, 0.5),
                        s=0.1,
                        alpha=0.5)
            plt.scatter(standard_embedding[clustered, 0],
                        standard_embedding[clustered, 1],
                        c=hdbscan_labels[clustered],
                        s=0.1,
                        cmap='Spectral')
            plt.show()

            print("ARI: {}\nAMIS: {}".format(adjusted_rand_score(labels_integers, hdbscan_labels), adjusted_mutual_info_score(labels_integers, hdbscan_labels)))
    
    if pargs.mode == 'eval-text':
        eval_data = []
        #all_components = [512, 256, 128, 64, 32, 16, 8]
        #seeds = [29, 27, 5, 18, 22]
        all_components = [8]
        seeds = [1]
        #all_components = [16, 8]
        for components in all_components:
            for seed in seeds:
                textSearch = textsearch.TextSearch(model_name="clip-base", build=False, dataset=dataset, filter_by_nwords=False)
                #textSearch.set_dataframes()
                textSearch.set_model() # word2vec
                m_ap, m_ap_tree, m_ap_sub = textSearch.eval_model(k=20, use_umap=False, n_components=components, seed=seed)
                eval_data.append([seed, components, m_ap, m_ap_tree, m_ap_sub])
                print("-----------")
                print("mAP (GC): {}".format(m_ap))
                print("-----------")
                print("mAP (CT): {}".format(m_ap_tree))
                print("-----------")
                print("mAP (SC): {}".format(m_ap_sub))
        df = pd.DataFrame(eval_data, columns=['seed', 'UMAP dim', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'mAP (SubCategory)'])
        #df.to_excel("F:/Documentos Universidad\MEMORIA/visual_text_embeddings_results/{}/results_text_umap_500.xlsx".format(dataset))

    if pargs.mode == 'search-text-visual':
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset)
        textSearch.set_gensim_model() # word2vec
        textSearch.set_dataframes()
        #textSearch.efficient_cosine_similarity()
        ssearch.load_features()

        # PARAMS
        #k = 5
        #alpha = 0.2
        k = 7
        alpha = 0.3
        #k = 3
        #alpha = 0.5
        T = None
        use_query = False
        #method = "mean"
        method = "base"
        filename = "base"
        #filename = "k_5_a_02"
        #filename = "k_7_a_03"
        #filename = "k_3_a_05"

        print("Going to adjust visual embeddings")
        if method != "base":
            new_visual_embeddings = textSearch.adjust_visual_embeddings(ssearch.features, ssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
            #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
            ssearch.features = new_visual_embeddings
        else:
            print("Searching only with visual embeddings.")

        if pargs.list is None:
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                #idx = ssearch.search(im_query, metric, top=99)
                norm = "None"
                metric='l2'
                idx = ssearch.search(im_query, metric=metric, top=99, norm=norm)

                r_filenames = ssearch.get_filenames(idx)            
                
                r_filenames.insert(0, fquery)

                # Evaluation
                #base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                #print("AP (GC): {}\nAP (CT): {}\nAP (SC): {}".format(ap, ap_tree, ap_sub))  
                #############

                image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result.png'.format(filename, metric, norm)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                fquery = input('Query:')

        #ap_arr, m_ap, res = textSearch.search_product('Peluche Asia 44 cm')
        #print("Average Precision array:")
        #pprint(ap_arr)
        #print("-----------")
        #print("mAP: {}".format(m_ap))
        #print("-----------")
    
    if pargs.mode == 'eval-all-text-visual':
        model_name = 'word2vec' ######################
        textSearch = textsearch.TextSearch(model_name=model_name, filter_by_nwords=False, build=False, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.efficient_cosine_similarity()
        ssearch.load_features()
        original_features = np.copy(ssearch.features)
        
        use_real_queries = False
        adjust_query = False
        which_realq = "RealQ_2"
        metric = 'l2'
        norm = 'None'
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        if use_real_queries:
            if "Pepeganga" in dataset:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            else:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
        else:
            real_df = None
        
        eval_data = []
        params_set = parameters
        #params_set = test_parameters
        #original_embeddings = ssearch.features
        for k, params in params_set.items():
            if k != "Base":
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, ssearch.filenames, k=params[0], a=params[1], T=params[2], use_query=params[3], method=params[4])
                #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
                ssearch.features = new_visual_embeddings
            if use_real_queries:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
                eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
            else:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
                eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
            if pargs.list is None:
                ap_arr = []
                ap_arr_tree = []
                ap_arr_sub = []
                
                for fquery in eval_files:
                    #print(fquery)
                    im_query = ssearch.read_image(fquery)
                    idx, dist_sorted = ssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, original_embeddings=original_features, df=df) 
                    #print("idx: ", idx)              
                    r_filenames = ssearch.get_filenames(idx)
                    
                    r_filenames.insert(0, fquery)
                    base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
                    ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    #print("{}: {}".format(fquery, ap))  
                    ap_arr.append(ap)
                    ap_arr_tree.append(ap_tree)
                    ap_arr_sub.append(ap_sub)

                mAP = statistics.mean(ap_arr)
                mAP_tree = statistics.mean(ap_arr_tree)
                mAP_sub = statistics.mean(ap_arr_sub)
                eval_data.append([k, mAP, mAP_tree, mAP_sub])
                print("params: {}, mAP(GC): {}, mAP(CT): {}".format(k, mAP, mAP_tree))

        df = pd.DataFrame(eval_data, columns=['params', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'mAP (SubCategory)'])
        df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/all_metrics_catalog_{}_{}_queryadj_{}.xlsx".format(dataset, textSearch.get_model_name(), metric, norm, adjust_query))

    if pargs.mode == 'eval-text-visual':

        use_real_queries = True
        # REALQ 1 PEPEGANGA
        #real_queries_path = "real_queries"
        #which_realq = "RealQ"
        ###################

        # REALQ 2 PEPEGANGA
        real_queries_path = "real_queries_2"
        which_realq = "RealQ_2"
        ####################

        # REAL HOMY
        #real_queries_path = "real_queries"
        #which_realq = "RealQ_3"
        
        top=99

        model_name = "roberta"

        
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if use_real_queries:
            if "Pepeganga" in dataset:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            else:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
        else:
            real_df = None
        
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name=model_name)
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.efficient_cosine_similarity()
        ssearch.load_features()
        original_features = np.copy(ssearch.features)
        #best_params = Pepeganga_params[category_metric]
        #category_metric = "base"
        #best_params = Pepeganga_params[category_metric]
        #params = Pepeganga_test_parameters
        
        params = Pepeganga_RQ_params
        
        for category_metric, best_params in params.items():
            eval_data = []
            if 'adj' in category_metric:
                #print("adj in category metric")
                adjust_query = True
            else:
                #print("adj NOT in category metric")
                adjust_query = False

            k=best_params[0]
            alpha=best_params[1]
            T=best_params[2]
            use_query=best_params[3]
            method=best_params[4]
            metric = 'l2'
            norm = 'None'
            
            #######################

            # Path to save images
            
            if use_real_queries:
                base_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/results_visual_text/{}".format(dataset, real_queries_path)
            else:
                base_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/results_visual_text".format(dataset)

            if method == 'mean':
                if use_query:    
                    #generated_file = "k_{}_a_{}_{}.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = 'k_{}_a_{}_{}'.format(str(k), str(alpha).replace(".", ""), method)
                else:
                    #generated_file = "k_{}_a_{}_{}_noquery.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = "k_{}_a_{}_{}_noquery".format(str(k), str(alpha).replace(".", ""), method)
            elif method == 'sim':
                #use_query = True
                #generated_file = "k_{}_{}.txt".format(str(k), method)
                results_dir = "k_{}_{}".format(str(k), method)
            elif method == 'softmax':
                #use_query = True
                #generated_file = "k_{}_t_{}_{}.txt".format(str(k), T, method)
                results_dir = "k_{}_t_{}_{}".format(str(k), T, method)
            elif method == 'base':
                results_dir = "base"
            results_path = "{}/{}_{}_{}_queryadj_{}".format(base_path, results_dir, metric, norm, adjust_query)
            pathlib.Path(results_path).mkdir(parents=False, exist_ok=True)
            ########################################

            original_embeddings = ssearch.features
            if method != 'base':
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, ssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
                #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
                ssearch.features = new_visual_embeddings


            # USE OR NOT REAL QUERIES FOR EVALUATION
            if use_real_queries:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
                eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

            else:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
                eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

            if pargs.list is None:
                ap_arr = []
                ap_arr_tree = []
                ap_arr_sub = []

                for i, fquery in enumerate(eval_files):
                    #print(fquery)
                    im_query = ssearch.read_image(fquery)

                    #idx = ssearch.search(im_query, metric, top=20)
                    idx, dist_array = ssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, original_embeddings=original_features, df=data_df)                
                    r_filenames = ssearch.get_filenames(idx)
                    r_filenames.insert(0, fquery)
                    #r_filenames_top_20 = r_filenames[:21]  # Top 20
                    #base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                    #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    #print("{}: {}".format(fquery, ap))  
                    #print("Average precision for {}: {} (GC), {} (CT)".format(fquery, ap, ap_tree)) 
                    #ap_arr.append(ap)
                    #ap_arr_tree.append(ap_tree)
                    #ap_arr_sub.append(ap_sub)
                    
                    image_r= ssearch.draw_result(r_filenames) # Not writing distance
                    output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)

                    if use_real_queries:
                        output_name = os.path.join('./{}/results_visual_text/{}/{}_{}_{}_queryadj_{}'.format(dataset, real_queries_path, results_dir, metric, norm, adjust_query), output_name)
                    else:
                        output_name = os.path.join('./{}/results_visual_text/{}_{}_{}_queryadj_{}'.format(dataset, results_dir, metric, norm, adjust_query), output_name)
                        
                    io.imsave(output_name, image_r)

                    #eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap, ap_tree, ap_sub])

                    #if i == 2:
                    #    break
                    
                    #print('result saved at {}'.format(output_name))
                    #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                    #print("r_filenames: {}\n".format(r_filenames))  
                
                #df = pd.DataFrame(eval_data, columns=['fquery',  'AP (GC)', 'AP (CT)', 'AP (SC)'])
                #if use_real_queries:
                #    df_path = "F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/{}".format(dataset, textSearch.get_model_name(), real_queries_path)
                #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
                #else:
                #    df_path = "F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}".format(dataset, textSearch.get_model_name())
                #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
                #df.to_excel("{}/{}_ap_20_{}_{}_queryadj_{}.xlsx".format(df_path, results_dir, metric, norm, adjust_query), index=False)


                #mAP = statistics.mean(ap_arr)
                #mAP_tree = statistics.mean(ap_arr_tree)
                #mAP_sub = statistics.mean(ap_arr_sub)
                #print("mAP (GC): {}\nmAP (CT): {}\nmAP (SC): {}".format(mAP, mAP_tree, mAP_sub))


    if pargs.mode == 'eval-text-visual-clip':

        use_real_queries = False
        # REALQ 1 PEPEGANGA
        #real_queries_path = "real_queries"
        #which_realq = "RealQ"
        ###################

        # REALQ 2 PEPEGANGA
        #real_queries_path = "real_queries_2"
        #which_realq = "RealQ_2"
        ####################

        # REAL HOMY
        real_queries_path = "real_queries"
        which_realq = "RealQ_3"
        
        top=99

        #model_name = "clip-base"
        #clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        
        model_name = "clip-fn"
        checkpoint_path = "F:/Documentos Universidad\MEMORIA\CLIP_models/{}/model1.pt".format(dataset)
        clipssearch = CLIPSSearch(pargs.config, pargs.name, checkpoint_path=checkpoint_path)
        
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if use_real_queries:
            if "Pepeganga" in dataset:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            else:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
        else:
            real_df = None
        
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name=model_name)
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.efficient_cosine_similarity()
        clipssearch.load_features()
        original_features = np.copy(clipssearch.features)
        
        params = HomyCLIPTrained_params
        
        for category_metric, best_params in params.items():
            eval_data = []
            if 'adj' in category_metric:
                #print("adj in category metric")
                adjust_query = True
            else:
                #print("adj NOT in category metric")
                adjust_query = False

            k=best_params[0]
            alpha=best_params[1]
            T=best_params[2]
            use_query=best_params[3]
            method=best_params[4]
            metric = 'l2'
            norm = 'None'
            
            #######################

            # Path to save images
            
            if use_real_queries:
                base_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/results_visual_text/{}".format(dataset, real_queries_path)
            else:
                base_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/results_visual_text".format(dataset)

            if method == 'mean':
                if use_query:    
                    #generated_file = "k_{}_a_{}_{}.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = 'k_{}_a_{}_{}'.format(str(k), str(alpha).replace(".", ""), method)
                else:
                    #generated_file = "k_{}_a_{}_{}_noquery.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = "k_{}_a_{}_{}_noquery".format(str(k), str(alpha).replace(".", ""), method)
            elif method == 'sim':
                #use_query = True
                #generated_file = "k_{}_{}.txt".format(str(k), method)
                results_dir = "k_{}_{}".format(str(k), method)
            elif method == 'softmax':
                #use_query = True
                #generated_file = "k_{}_t_{}_{}.txt".format(str(k), T, method)
                results_dir = "k_{}_t_{}_{}".format(str(k), T, method)
            elif method == 'base':
                results_dir = "base"
            results_path = "{}/{}_{}_{}_queryadj_{}".format(base_path, results_dir, metric, norm, adjust_query)
            pathlib.Path(results_path).mkdir(parents=False, exist_ok=True)
            ########################################

            original_embeddings = clipssearch.features
            if method != 'base':
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
                #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
                clipssearch.features = new_visual_embeddings


            # USE OR NOT REAL QUERIES FOR EVALUATION
            if use_real_queries:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
                eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

            else:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
                eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

            if pargs.list is None:
                ap_arr = []
                ap_arr_tree = []
                ap_arr_sub = []

                for i, fquery in enumerate(eval_files):
                    #print(fquery)
                    im_query = clipssearch.read_image(fquery)

                    #idx = ssearch.search(im_query, metric, top=20)
                    idx, dist_array = clipssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, original_embeddings=original_features, df=data_df)                
                    r_filenames = clipssearch.get_filenames(idx)
                    r_filenames.insert(0, fquery)
                    #r_filenames_top_20 = r_filenames[:21]  # Top 20
                    #base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                    #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    #print("{}: {}".format(fquery, ap))  
                    #print("Average precision for {}: {} (GC), {} (CT)".format(fquery, ap, ap_tree)) 
                    #ap_arr.append(ap)
                    #ap_arr_tree.append(ap_tree)
                    #ap_arr_sub.append(ap_sub)
                    
                    image_r= ssearch.draw_result(r_filenames) # Not writing distance
                    output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)

                    if use_real_queries:
                        output_name = os.path.join('./{}/results_visual_text/{}/{}_{}_{}_queryadj_{}'.format(dataset, real_queries_path, results_dir, metric, norm, adjust_query), output_name)
                    else:
                        output_name = os.path.join('./{}/results_visual_text/{}_{}_{}_queryadj_{}'.format(dataset, results_dir, metric, norm, adjust_query), output_name)
                        
                    io.imsave(output_name, image_r)

                    #eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap, ap_tree, ap_sub])

                    #if i == 2:
                    #    break
                    
                    #print('result saved at {}'.format(output_name))
                    #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                    #print("r_filenames: {}\n".format(r_filenames))  
                
                #df = pd.DataFrame(eval_data, columns=['fquery',  'AP (GC)', 'AP (CT)', 'AP (SC)'])
                #if use_real_queries:
                #    df_path = "F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}/{}".format(dataset, textSearch.get_model_name(), real_queries_path)
                #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
                #else:
                #    df_path = "F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/{}".format(dataset, textSearch.get_model_name())
                #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
                #df.to_excel("{}/{}_ap_20_{}_{}_queryadj_{}.xlsx".format(df_path, results_dir, metric, norm, adjust_query), index=False)


                #mAP = statistics.mean(ap_arr)
                #mAP_tree = statistics.mean(ap_arr_tree)
                #mAP_sub = statistics.mean(ap_arr_sub)
                #print("mAP (GC): {}\nmAP (CT): {}\nmAP (SC): {}".format(mAP, mAP_tree, mAP_sub))

    if pargs.mode == 'eval-vtnn':

        use_real_queries = False

        eval_data = []
        #########################
        epochs = 150
        dim_reduction = "1024-dim"
        model_name = "VTNN150_adam_lr_0.0001" # BEST 1024 dim
        config_preset = 2
        dropout=0.2
        #######################

        ssearch.load_features() 

        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        # Path to save images
        #results_dir = "F:/Documentos Universidad\MEMORIA/visual_text_embeddings_results/{}/vtnn/128-dim/{}".format(dataset, epochs)
        
        if use_real_queries:
            results_dir = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/results_vtnn/{}/150/real_queries_2".format(dataset, dim_reduction)
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
        else:
            results_dir = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/results_vtnn/{}/150".format(dataset, dim_reduction)
            real_df = None

        #results_dir = "results_{}".format(model_name)
        #results_path = "{}/{}".format(base_path, results_dir)
        #pathlib.Path(results_path).mkdir(parents=False, exist_ok=True)
        #pathlib.Path(results_dir).mkdir(parents=False, exist_ok=True)
        ########################################


        #vtnn_path = "F:/Documentos Universidad\MEMORIA/vtnn/{}.pt".format(model_name)
        if dim_reduction == '1024-dim':
            path = "F:/Documentos Universidad\MEMORIA/vtnn/{}/config_{}/dropout_{}".format(epochs, config_preset, dropout)
        else:
            path = "F:/Documentos Universidad\MEMORIA/vtnn/{}/{}/config_{}/dropout_{}".format(dim_reduction, epochs, config_preset, dropout)
        vtnn_path = "{}/{}.pt".format(path, model_name)
        if dim_reduction == "1024-dim":
            vtnn = VTNN(p=dropout, config_preset=config_preset)
        elif dim_reduction == "128-dim":
            vtnn = VTNN128dim(p=dropout, config_preset=config_preset)
        else:
            vtnn = VTNN8dim(p=dropout, config_preset=config_preset)
        vtnn = vtnn.to('cuda')
        vtnn.load_state_dict(torch.load(vtnn_path))
        feats = torch.tensor(ssearch.features)
        vtnn.eval()
        with torch.no_grad():
            feats = feats.to('cuda')
            feats = feats.view(-1, 2048)
            new_visual_embeddings = vtnn(feats).cpu().numpy()
        #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
        ssearch.features = new_visual_embeddings

        # USE OR NOT REAL QUERIES FOR EVALUATION
        if use_real_queries:
            eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/RealQ_2/eval_images"
            eval_files = ["RealQ_2/eval_images/" + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        else:
            eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
            eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        if pargs.list is None:
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []
            
            for fquery in eval_files:
                #print(fquery)
                im_query = ssearch.read_image(fquery)
                #idx = ssearch.search(im_query, metric, top=20)

                #idx = ssearch.search(im_query, metric, top=99, vtnn=vtnn)         
                idx, dist_array = ssearch.search(im_query, metric, top=20, vtnn=vtnn)       
                r_filenames = ssearch.get_filenames(idx)
                
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                #print("Average precision for {}: {}".format(fquery, ap)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                ap_arr_sub.append(ap_sub)
                
                #image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)

                #if use_real_queries:
                #    output_name = os.path.join('./{}/results_vtnn/{}/{}/real_queries'.format(dataset, dim_reduction, epochs), output_name)
                #else:    
                #    output_name = os.path.join('./{}/results_vtnn/{}/{}'.format(dataset, dim_reduction, epochs), output_name)
                #io.imsave(output_name, image_r)

                eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap, ap_tree, ap_sub])
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
            
            #df = pd.DataFrame(eval_data, columns=['fquery',  'AP (GC)', 'AP (CT)', 'AP (SC)'])
            #if use_real_queries:
            #    if dim_reduction == "1024-dim":
            #        df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/vtnn/{}/real_queries_2/{}.xlsx".format(dataset, epochs, model_name), index=False)
            #    else:
            #        df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/vtnn/{}/{}/real_queries_2/{}.xlsx".format(dataset, dim_reduction, epochs, model_name), index=False)
            #else:
            #    df.to_excel("F:/Documentos Universidad/MEMORIA/visual_text_embeddings_results/{}/vtnn/{}/{}/{}.xlsx".format(dataset, dim_reduction, epochs), index=False)

            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            mAP_sub = statistics.mean(ap_arr_sub)
            print("mAP (GC): {}\nmAP (CT): {}\nmAP (SC): {}".format(mAP, mAP_tree, mAP_sub))


    if pargs.mode == 'eval-all-vtnn':
        eval_data = []
        epochs = 150
        configs = [1, 2, 3, 4, 5, 6, 7, 8]
        dropouts = [0.0, 0.2]
        #config_preset = 1
        #dropout = 0.2
        ssearch.load_features()
        original_features = np.copy(ssearch.features)

        results_dir = "F:/Documentos Universidad\MEMORIA/visual_text_embeddings_results/{}/vtnn/128-dim/{}".format(dataset, epochs)
        pathlib.Path(results_dir).mkdir(parents=False, exist_ok=True)
        for config_preset in configs:
            for dropout in dropouts:
                path = "F:/Documentos Universidad\MEMORIA/vtnn/128-dim/{}/config_{}/dropout_{}".format(epochs, config_preset, dropout)
                models = [f for f in os.listdir(path)]


                for model in models:
                    vtnn_path = "{}/{}".format(path, model)
                    vtnn = VTNN128dim(p=dropout, config_preset=config_preset)
                    tnn = vtnn.to('cuda')
                    vtnn.load_state_dict(torch.load(vtnn_path))
                    feats = torch.tensor(original_features)
                    vtnn.eval()
                    with torch.no_grad():
                        feats = feats.to('cuda')
                        feats = feats.view(-1, 2048)
                        new_visual_embeddings = vtnn(feats).cpu().numpy()
                    #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
                    ssearch.features = new_visual_embeddings
                    eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
                    eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
                    ap_arr = []
                    ap_arr_tree = []
                    ap_arr_sub = []
                    
                    for fquery in eval_files:
                        #print(fquery)
                        im_query = ssearch.read_image(fquery)
        
                        idx = ssearch.search(im_query, metric, top=20, vtnn=vtnn)       
                        r_filenames = ssearch.get_filenames(idx)
                        
                        r_filenames.insert(0, fquery)
                        base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                        ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                        ap_arr.append(ap)
                        ap_arr_tree.append(ap_tree)
                        ap_arr_sub.append(ap_sub)

                    

                    mAP = statistics.mean(ap_arr)
                    mAP_tree = statistics.mean(ap_arr_tree)
                    mAP_sub = statistics.mean(ap_arr_sub)
                    eval_data.append([os.path.splitext(model)[0], mAP, mAP_tree, mAP_sub, config_preset, dropout])
                    print("cfg_preset: {}, dropout: {} \nmAP (GC): {}\nmAP (CT): {}\nmAP (SC): {}".format(config_preset, dropout, mAP, mAP_tree, mAP_sub))
        df = pd.DataFrame(eval_data, columns=['model', 'AP (GC)', 'AP (CT)', 'AP (SC)', 'cfg_preset', 'dropout'])
        df.to_excel("{}/all_models_results.xlsx".format(results_dir), index=False)

    if pargs.mode == 'utils':
        ssearch.load_features()
        prepare_dataset(dataset, ssearch.features, ssearch.filenames, use_umap=True, n_components=128)     

    if pargs.mode == 'clip-testing':
        
        #pr = cProfile.Profile()
        #pr.enable()
        #model_name = "clip-fn"
        #checkpoint_path = "F:/Documentos Universidad\MEMORIA\CLIP_models/{}/model1.pt".format(dataset)
        #clipssearch = CLIPSSearch(pargs.config, pargs.name, checkpoint_path=checkpoint_path)
        #print(clipssearch.sim_model.config)
        #clipssearch.compute_features_from_catalog() 
        #metric = 'l2'
        #norm = 'None'
        #textSearch = textsearch.TextSearch(model_name='clip-base', filter_by_nwords=False, build=False, dataset=dataset)
        #textSearch.set_model()
        #textSearch.set_dataframes()
        #textSearch.efficient_cosine_similarity()
        #clipssearch.load_features()
        #ssearch.load_features()
        #print(clipssearch.features.shape)
        #print(ssearch.features.shape)
        #original_features = np.copy(clipssearch.features)
        #print(original_features)

        #'''
        #model_name = "clip-base"
        #model_name = "roberta"
        #clipssearch = CLIPSSearch(pargs.config, pargs.name) # CLIPBASE
        #checkpoint_path = "F:/Documentos Universidad\MEMORIA\CLIP_models/{}/model1.pt".format(dataset)
        #clipssearch = CLIPSSearch(pargs.config, pargs.name, checkpoint_path=checkpoint_path)
        #model_name = "clip-base"
        model_name = "roberta"
        #clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        #clipssearch.load_features()
        ssearch.load_features()
        #original_features = np.copy(clipssearch.features)
        original_features = np.copy(ssearch.features)

        textSearch = textsearch.TextSearch(model_name=model_name, filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.efficient_cosine_similarity()
        
        use_real_queries = True
        adjust_query = False
        adjust_query_sim = False
        which_realq = "RealQ_3"
        metric = 'l2'
        norm = 'None'

        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        if use_real_queries:
            if "Pepeganga" in dataset:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            else:
                real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
        else:
            real_df = None

        eval_data = []
        params_set = parameters
        best_real_pepeganga = {"k_7_a_02_mean": (7, 0.2, None, False, 'mean')}
        best_real_homy = {"k_7_a_09_mean": (7, 0.9, None, False, 'mean')}
        base_params = {"Base": (None)}
        #params_set = best_real_pepeganga
        params_set = base_params
        #original_embeddings = clipssearch.features
        original_embeddings = ssearch.features
        #i = 0
        for k, params in params_set.items():
            if k != "Base":
                #new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=params[0], a=params[1], T=params[2], use_query=params[3], method=params[4])
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, ssearch.filenames, k=params[0], a=params[1], T=params[2], use_query=params[3], method=params[4])
                #clipssearch.features = np.asarray(new_visual_embeddings)
                ssearch.features = np.asarray(new_visual_embeddings)
            if use_real_queries:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
                eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
            else:
                eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(dataset)
                eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
            if pargs.list is None:
                ap_arr = []
                ap_arr_tree = []
                ap_arr_sub = []
                #fi = 0
                for fquery in eval_files:
                    #if "220home" not in fquery:
                    #if "ph00138" not in fquery:
                    if "ph00183" not in fquery:
                        continue
                    else:
                        print(f"Evaluating {fquery}")
                    #print(fquery)
                    #im_query = clipssearch.read_image(fquery)
                    #idx, dist_sorted = clipssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, adjust_query_sim=adjust_query_sim, original_embeddings=original_embeddings, df=df, text_model=textSearch)               
                    #r_filenames = clipssearch.get_filenames(idx)
                    im_query = ssearch.read_image(fquery)
                    idx, dist_sorted = ssearch.search(im_query, metric=metric, norm=norm, top=99, adjust_query=adjust_query, adjust_query_sim=adjust_query_sim, original_embeddings=original_embeddings, df=df, text_model=textSearch)               
                    idx, dist_sorted = remove_repeated(idx, dist_sorted)
                    r_filenames = ssearch.get_filenames(idx)
                    r_filenames.insert(0, fquery)
                    #print(r_filenames)
                    #base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
                    #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    image_r= ssearch.draw_result(r_filenames)
                    save_path = "F:/Documentos Universidad/MEMORIA/F/paper/ResNet/Real/{}/visual/{}_{}_{}_result.png".format(dataset, os.path.basename(fquery), metric, norm)
                    print("Created image, saving on ", save_path)
                    io.imsave(save_path, image_r)
                    #ap_arr.append(ap)
                    #ap_arr_tree.append(ap_tree)
                    #ap_arr_sub.append(ap_sub)
                    #if fi == 1: break
                    #fi += 1
                    

                #mAP = statistics.mean(ap_arr)
                #mAP_tree = statistics.mean(ap_arr_tree)
                #mAP_sub = statistics.mean(ap_arr_sub)
                #eval_data.append([k, mAP, mAP_tree, mAP_sub])
                #print("params: {}, mAP: {}".format(k, mAP))
                #if i == 1: break
                #i += 1
        
        #df = pd.DataFrame(eval_data, columns=['params', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'mAP (SubCategory)'])
        #pr.disable()
        #pr.dump_stats('./cprofile_stats.prof')
        #with open('./cprofile_stats.txt', "w") as f:
        #    ps = pstats.Stats('./cprofile_stats.prof', stream=f)
        #    ps.sort_stats('cumulative')
        #    ps.print_stats()
        #df.to_excel("F:/Documentos Universidad\MEMORIA\F\paper/ResNet/Real/{}/k_3_gc_no_q.xlsx".format(dataset))
        #df.to_excel("F:/Documentos Universidad\MEMORIA\F\paper/ResNet/Real/{}/k_3_cos_sim_08_no_q.xlsx".format(dataset))
        #'''

    if pargs.mode == 'recall-precision':
        # k_3_gc_no_q BEST
        

        model_name = "clip-base"
        clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        
        data_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if "Pepeganga" in dataset:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            real_queries_path = "real_queries_2"
            which_realq = "RealQ_2"
        else:
            real_df = pd.read_excel("F:/Documentos Universidad\MEMORIA\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
            real_queries_path = "real_queries"
            which_realq = "RealQ_3"
        
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=3, a=None, T=0.5, use_query=True, method='softmax') ## HOMY
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=7, a=0.3, T=None, use_query=False, method='mean') ## PEPEGANGA

        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name=model_name)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.efficient_cosine_similarity()
        clipssearch.load_features()
        original_features = np.copy(clipssearch.features)
        # ssearch.load_features()
        # original_features = np.copy(ssearch.features)

        # params = {'k': 3, 'a': None, 'T': 0.5, 'use_query': True, 'method': 'softmax'} # Homy
        params = {'k': 7, 'a': 0.3, 'T': None, 'use_query': False, 'method': 'mean'} # PEPEGANGA
        eval_data = []
        adjust_query = True

        k=params['k']
        alpha=params['a']
        T=params['T']
        use_query=params['use_query']
        method=params['method']
        metric = 'l2'
        norm = 'None'
        #######################
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
        #new_visual_embeddings = np.loadtxt("F:/Documentos Universidad/MEMORIA/visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
        #clipssearch.features = new_visual_embeddings

        eval_path = "F:/Documentos Universidad/MEMORIA/convnet_visual_attributes/visual_attributes/{}/eval_images".format(which_realq)
        eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

        ap_arr = []
        ap_arr_tree = []
        ap_arr_sub = []

        query_embeddings = []
        relevants = []
        idxs = []

        recall_precision_dict = dict()

        top=None
        for i, fquery in enumerate(eval_files):
            print(fquery)
            im_query = clipssearch.read_image(fquery)

            idx, dist_array, query_embedding, catalog_data = clipssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, original_embeddings=original_features, df=data_df)                
            #idxs.append(idx)
            #query_embeddings.append(query_embedding[0])
            
            #r_filenames = clipssearch.get_filenames(idx)
            #r_filenames.insert(0, fquery)
            
            # r_filenames = ssearch.filenames.copy()
            # r_filenames.insert(0, fquery)

            #relevant = get_ordered_relevants(r_filenames, dataframe=data_df, real_df=real_df)
            #relevants.append(relevant)

        #pprint(query_embeddings)
        #np.save(f'recall_precision/{dataset}/query_embeddings', np.array(query_embeddings))
        #np.save(f'recall_precision/{dataset}/relevants', np.array(relevants))
        #np.save(f'recall_precision/{dataset}/idxs', np.array(idxs))
        np.save(f'recall_precision/{dataset}/catalog_data', np.array(catalog_data))

    

    if pargs.mode == 'testing':
        eval_data = []
        category_metric = "CT"
        model_name = "roberta"

        #best_params = Pepeganga_params[category_metric]
        #best_params = Homy_params[category_metric]
        #best_params = IKEA_params[category_metric]
        best_params = WorldMarket_params[category_metric]

        k=best_params[0]
        alpha=best_params[1]
        T=best_params[2]
        use_query=best_params[3]
        method=best_params[4]
        
        #######################

        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name=model_name)
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        
        #product_name = "Gray Nautical Rope Rapallo Outdoor Dining Chairs Set Of 2_10408"
        #product_name = "Round Back Paige Upholstered Dining Armchair_3252"
        #product_name = "Remember the Journey to School Integration_12668"
        #product_name = "Umbra Linen UDry Folding Microfiber Dish Drying Mat_15971"
        #product_name = "Umbra Gray UDry Folding Microfiber Dish Drying Mat_15953"
        product_name = "Fold Away Dish Rack_16164"

        images_path = "F:/Documentos Universidad/MEMORIA/Datasets/Catalogo_{}/data/images".format(dataset)

        r_names, similarity_arr = textSearch.search_product_by_name(name=product_name)
        r_names = np.insert(r_names, 0, product_name)
        r_filenames = ["{}/{}.jpg".format(images_path, f) for f in r_names]
        image_r= ssearch.draw_result(r_filenames, similarity=similarity_arr)
        output_name = product_name + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
        output_name = os.path.join(pargs.odir, output_name)
        io.imsave(output_name, image_r)
        


        #textSearch.set_dataframes()
        #textSearch.efficient_cosine_similarity()
        #ssearch.load_features() 