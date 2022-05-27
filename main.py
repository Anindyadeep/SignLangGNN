import os 
import cv2 
import sys 
import warnings
import numpy as np 
import pandas as pd 
import mediapipe as mp 
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
warnings.filterwarnings("ignore")

path = os.getcwd()
sys.path.append(path)
sys.path.append(path[:-1])

from src.utils import HandPosUtils
from src.dataset import HandPosDataset
from src.train import TrainModel
from Models import base_gnn_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

hu = HandPosUtils()
save_csv_folder_name = "Data/CSVs"

if not os.path.isdir(save_csv_folder_name):
    hu.create_train_csv(save_csv_folder_name)
    hu.concat_all_csv_into_one(all_csv_path = "Data/CSVs",save_file_folder_name = "Data/raw", split=0.3)
    hu.create_test_csv("Data/raw")

root_data_path = os.path.join(path, "Data/")

train_dataset = HandPosDataset(root_data_path, "train_data.csv")
valid_dataset = HandPosDataset(root_data_path, "valid_data.csv", test = False, valid = True)
test_dataset = HandPosDataset(root_data_path, "test_data.csv", test = True, valid = False)

train_loader = DataLoader(
    train_dataset,
    batch_size = 128,
    shuffle = True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size = 128,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1,
    shuffle = True
)

model = base_gnn_model.Model(3, 64, 32, 29).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_model = TrainModel(model)

epochs = 3
for epoch in tqdm(range(epochs)):
    current_train_loss, current_train_acc = train_model.train_model_perbatch(model, train_loader, criterion, optimizer)
    current_valid_loss, current_valid_acc = train_model.test_model_perbatch(model, valid_loader, criterion)

    print(f"Epoch {epoch} Train Loss: {current_train_loss} Train Accuracy: {current_train_acc}")
    print(f"Epoch {epoch} Validation Loss: {current_valid_loss} Validation Accuracy: {current_valid_acc}")

model_save_path = os.path.join(os.getcwd(), "saved_models/base_model.pth")
torch.save(model.state_dict(), model_save_path)

test_loss, test_acc = train_model.test_model_perbatch(model, test_loader, criterion)
print(f"Test Loss: {current_valid_loss} Test Accuracy: {current_valid_acc}")