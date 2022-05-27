import os 
import sys 
import cv2
import torch
import warnings
import numpy as np
import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
warnings.filterwarnings("ignore")

path = os.getcwd()
sys.path.append(path)
sys.path.append(path[:-1])

from src.utils import HandPosUtils
from src.dataset import HandPosDataset
from Models.base_gnn_model import Model 

device = "cpu"
model = Model(3, 64, 32, 29).to(device)

model_path = os.path.join(path, "saved_models/base_model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

handpos = HandPosUtils()
label_dict = handpos.get_label_dict()
print(label_dict)


hand_conn = mp_hands.HAND_CONNECTIONS
source_index = []
target_index = []

for i, j in list(hand_conn):
    source_index.append(i)
    target_index.append(j)

edge_index = np.array([
    source_index,
    target_index
])

edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            pos_dict = handpos.get_hand_coords_frame(hand_landmarks)
            x = torch.tensor(np.array(list(pos_dict.values())).reshape(21, 3), dtype=torch.float32).to(device)
            out = model(x, edge_index, torch.tensor([0]))
            predictions = out.argmax(dim=1).item()
            predicted_letter = list(label_dict.keys())[list(label_dict.values()).index(predictions)]
            print(predicted_letter)


    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()