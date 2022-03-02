import os 
import sys
import warnings 
from pathlib import Path

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import torch_geometric
import torch_geometric.nn as gnn

warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)
sys.path.append("..")

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes):
        super(Model, self).__init__()
        self.gat1 = gnn.GATConv(in_features, hidden_features, heads = 4, concat = True, dropout = 0.2)
        self.gat2 = gnn.GATConv(4 * hidden_features, out_features, heads = 2, concat = False)
        self.classifier = nn.Linear(out_features, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = gnn.global_max_pool(x, batch)
        x = self.classifier(x)
        return x 