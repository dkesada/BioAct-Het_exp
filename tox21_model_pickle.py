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

#################################################################################################
# This file encapsulates the creation of the '.h5' file of the trained siamese neural network model.
# Running it will generate a 9.98Mb file with the tensorflow.keras model.
# The code used was adapted from the '(Tox21) Association_based_strategy.ipynb' notebook
#################################################################################################

with open("data_ds.pkl", 'rb') as inp:
    data_ds = pickle.load(inp)
	
from sklearn.model_selection import KFold

Epoch_S = 10

def evaluate_model(df, k = 10 , shuffle = False):
    
    result =[] 
    s = 0


    kf = KFold(n_splits=10, shuffle= shuffle, random_state=None)
    
    for train_index, test_index in kf.split(df):

        train_ds = [df[index] for index in train_index] 
        
        valid_ds = [df[index] for index in test_index]
        
        label_pos , label_neg, _ , _ = count_lablel(train_ds)  # Irrelevant
        print(f'train positive label: {label_pos} - train negative label: {label_neg}')  # Irrelevant
        
        train_ds = up_and_down_Samplenig(train_ds, scale_downsampling = 0.5)
        
        label_pos , label_neg , _ , _ = count_lablel(train_ds)  # Irrelevant
        print(f'up and down sampling => train positive label: {label_pos} - train negative label: {label_neg}')  # Irrelevant

        label_pos , label_neg, _ , _ = count_lablel(valid_ds)  # Irrelevant
        print(f'Test positive label: {label_pos} - Test negative label: {label_neg}')  # Irrelevant

        l_train = []
        r_train = []
        lbls_train = []
        l_valid = []
        r_valid = []
        lbls_valid = []

        for i , data in enumerate(train_ds):
            embbed_drug, onehot_task, embbed_task, lbl, task_number, task_name = data
            l_train.append(embbed_drug[0])
            r_train.append(embbed_task)
            lbls_train.append(lbl.tolist())
        
        for i , data in enumerate(valid_ds):
            embbed_drug, onehot_task, embbed_task, lbl, task_number, task_name = data
            l_valid.append(embbed_drug[0])
            r_valid.append(embbed_task)
            lbls_valid.append(lbl.tolist())

        l_train = np.array(l_train).reshape(-1,512,1)
        r_train = np.array(r_train).reshape(-1,512,1)
        lbls_train = np.array(lbls_train)

        l_valid = np.array(l_valid).reshape(-1,512,1)
        r_valid = np.array(r_valid).reshape(-1,512,1)
        lbls_valid = np.array(lbls_valid)

        # create neural network model
        siamese_net = siamese_model_attentiveFp_tox21()
        
        history = History()
        P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size = 128, callbacks=[history])

        for j in range(100):
            C=1
            Before = int(P.history['accuracy'][-1]*100)
            for i in range(2,Epoch_S+1):
                if  int(P.history['accuracy'][-i]*100) == Before:
                    C=C+1
                else:
                    C=1
                Before=int(P.history['accuracy'][-i]*100)
                print(Before)
            if C==Epoch_S:
                break
            P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size = 128, callbacks=[history])
        print(j+1)
        
        score  = siamese_net.evaluate([l_valid,r_valid], lbls_valid, verbose=1)
        a = (score[1],score[4])
        result.append(a)
        
        if score[4] > s :
            best_model = siamese_net
            s = score[4]
            print("Save_model!!")
    
    return result , best_model


scores, best_model = evaluate_model(data_ds, 10, True)

with open("best_model.pkl", 'wb') as out:
    pickle.dump(best_model, out, pickle.HIGHEST_PROTOCOL)
with open("scores.pkl", 'wb') as out:
    pickle.dump(scores, out, pickle.HIGHEST_PROTOCOL)

