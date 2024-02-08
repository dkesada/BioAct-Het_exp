import os
import dgl 
import sys
import random
import torch
import cv2
import torchvision
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.utils import to_categorical
from dgllife.model import MLPPredictor
from tensorflow.keras.callbacks import  History
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer
from tqdm.notebook import tqdm, trange
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from general import DATASET, get_dataset, separate_active_and_inactive_data, get_embedding_vector_class, count_lablel,data_generator, up_and_down_Samplenig
from gcn_pre_trained import get_tox21_model
from heterogeneous_siamese_tox21 import siamese_model_attentiveFp_tox21, siamese_model_Canonical_tox21
import pickle

cache_path='./tox21_dglgraph.bin'

df = get_dataset("tox21")
id = df['mol_id']
df = df.drop(columns=['mol_id'])

tox21_tasks = df.columns.values[:12].tolist()

one = []
zero = []
nan = []
 
for task in tox21_tasks:
    a = list(df[task].value_counts(dropna=False).to_dict().values())
    zero.append(a[0])
    nan.append(a[1])
    one.append(a[2])
    print(task ,"one:" ,a[2] ," zero:", a[0], " NAN:",a[1])
	
X = np.arange(1,len(tox21_tasks)+1)
	
from dgllife.model import MLPPredictor

def create_dataset_with_gcn(dataset, class_embed_vector, GCN, tasks, numberTask):

    created_data = []
    data = np.arange(len(tasks))
    onehot_encoded = to_categorical(data)

    for i, data in enumerate(dataset):
        smiles, g, label, mask = data
        g = g.to(device)
        g = dgl.add_self_loop(g)
        graph_feats = g.ndata.pop('h')
        embbed = GCN(g, graph_feats)
        embbed = embbed.to('cpu')
        embbed = embbed.detach().numpy()
        a = ( embbed, onehot_encoded[numberTask], class_embed_vector[numberTask], label, numberTask, tasks[numberTask])
        created_data.append(a)
    print('Data created!!')
    return created_data
	
df_positive, df_negative = separate_active_and_inactive_data(df, tox21_tasks)

for i,d in enumerate(zip(df_positive,df_negative)):
    print(f'{tox21_tasks[i]}=> positive: {len(d[0])} - negative: {len(d[1])}')
	
dataset_positive = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path) for d in df_positive]
dataset_negative = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path) for d in df_negative]

embed_class_tox21 = get_embedding_vector_class(dataset_positive, dataset_negative, radius=2, size = 512)

model_name = 'GCN_attentivefp_Tox21'
gcn_model = get_tox21_model(model_name)
gcn_model.eval()
gcn_model = gcn_model.to(device)

data_ds = []
for i, task in  enumerate(tox21_tasks):
    a = df[['smiles' , task]]
    a = a.dropna()
    ds = DATASET(a, smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path)
    data = create_dataset_with_gcn(ds, embed_class_tox21, gcn_model, tox21_tasks, i)
    for d in data:
        data_ds.append(d)
		
with open("data_ds.pkl", 'wb') as out:
    pickle.dump(data_ds, out, pickle.HIGHEST_PROTOCOL)