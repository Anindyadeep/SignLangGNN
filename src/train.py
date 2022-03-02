import os 
import sys 
import warnings
from tqdm import tqdm 
from pathlib import Path

import torch 
import torch.nn as nn  
import torch.optim as optim 

import torch_geometric
import torch_geometric.nn as gnn 

warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)
sys.path.append("..")

device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainModel(nn.Module):
    def __init__(self, model):
        pass
    
    def train_model_perbatch(self, model, loader, criterion, optimizer):
        running_loss = 0.0
        num_correct = 0.0

        model.train()
        for graph in loader:
            x = graph.x.to(device)
            y = graph.y.type(torch.long).to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)

            optimizer.zero_grad()
            output = model(x, edge_index, batch)
            num_correct += int((output.argmax(dim=1) == y).sum())

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(loader.dataset), num_correct / len(loader.dataset)


    def test_model_perbatch(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        num_correct = 0.0

        with torch.no_grad():
            for graph in loader:
                x = graph.x.to(device)
                y = graph.y.type(torch.long).to(device)
                edge_index = graph.edge_index.to(device)
                batch = graph.batch.to(device)

                output = model(x, edge_index, batch)
                loss = criterion(output, y)
                running_loss += loss.item()
                num_correct += int((output.argmax(dim=1) == y).sum())
        return running_loss / len(loader.dataset), num_correct / len(loader.dataset)
    

    def evaluate_model(self, model, loader):
        with torch.no_grad():
            graph = next(iter(loader))
            x = graph.x.to(device)
            y = graph.y.type(torch.long).to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)

            output = model(x, edge_index, batch)
            return int((output.argmax(dim=1) == y).sum())