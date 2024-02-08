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
# This file encapsulates the prediction of a dataset with the 'data_ds.pkl' format using 
# the previously trained '.h5' siamese neural network.
# Running it will generate a 2.07Gb file with the needed data to train the tensorflow model.
# The code used was adapted from the '(Tox21) Association_based_strategy.ipynb' notebook
#################################################################################################

#with open("data_ds.pkl", 'rb') as inp:
#    data_ds = pickle.load(inp)  # Change this to loading the subset directly so that it does not take 4000 years each time
#subset_ds = data_ds[0:10]
#del data_ds

with open("subset_ds.pkl", 'rb') as inp:
	subset_ds = pickle.load(inp) 

from tensorflow.keras.models import load_model

# Load the previously saved tox21 model
model = load_model('tox21.h5')


def predict_siamese(pred_ds, model):
	
	l_pred = []
	r_pred = []
	lbls_pred = []
	
	# Get input data from the ds block
	for i , data in enumerate(pred_ds):
		embbed_drug, onehot_task, embbed_task, lbl, task_number, task_name = data
		l_pred.append(embbed_drug[0])
		r_pred.append(embbed_task)
		lbls_pred.append(lbl.tolist())  # This are the labels, which should not be a thing in real time predictions...
		
	# Input data into the appropriate shape
	l_pred = np.array(l_pred).reshape(-1,512,1)
	r_pred = np.array(r_pred).reshape(-1,512,1)
	lbls_pred = np.array(lbls_pred) # This are the labels, which should not be a thing in real time predictions...
	print(lbls_pred)
	
	return model.predict([l_pred,r_pred])
	
results = predict_siamese([subset_ds[0]], model)
print(results)
print(results)