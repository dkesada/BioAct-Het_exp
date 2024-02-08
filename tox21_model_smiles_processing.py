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


# Class used to silence the unavoidable prints when creating the graph data
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

		
#################################################################################################
# This file encapsulates the transformation from a SMILES code into the graph object and into
# the embbeded vector necesary for the model to predict its values
# Running it will generate the vector needed to create the l_pred part of the prediction
#################################################################################################

cache_path = './tox21_dglgraph.bin'
smiles = 'O=[N+]([O-])c1cc(C(F)(F)F)cc([N+](=O)[O-])c1Cl'

# Obtain the graph info of the SMILES code
d = {'smiles': [smiles], 'dummy_label': [0.0]}
df_smiles = pd.DataFrame(data=d)
with HiddenPrints():
	graph_data = DATASET(df_smiles, smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path=cache_path)

# Load the GCN model
with open("gcn_model.pkl", 'rb') as out:
	gcn_model = pickle.load(out)
	
def create_embbeded(graph_data, gcn_model):
	for i in graph_data:
		_, g, _, _ = i
		g = g.to(device)
		g = dgl.add_self_loop(g)
		graph_feats = g.ndata.pop('h')
		embbed = gcn_model(g, graph_feats)
		embbed = embbed.to('cpu')
		embbed = embbed.detach().numpy()
		
	return embbed

embbed = create_embbeded(graph_data, gcn_model)
print(embbed)
