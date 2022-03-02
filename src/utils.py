import os 
import cv2 
import sys 
import warnings
import numpy as np 
import pandas as pd 
import mediapipe as mp 
from tqdm import tqdm
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)
sys.path.append("..")

class HandPosUtils(object):
    def __init__(self, parent_path = None, train_image_folder_name = None, test_image_folder_name = None):
        self.base_path = BASEDIR if parent_path is None else parent_path
        self.train_image_folder_name = "ImageData/asl_alphabet_train" if train_image_folder_name is None else train_image_folder_name
        self.test_image_folder_name = "ImageData/asl_alphabet_test" if test_image_folder_name is None else test_image_folder_name 
    
    def get_label_dict(self):
        path = os.path.join(self.base_path, self.train_image_folder_name)
        labels = os.listdir(path)
        num_labels = np.arange(len(labels))
        zip_iterator = zip(labels, num_labels)
        return dict(zip_iterator)
    
    def get_hand_marks_coords_dict_for_image(self, img_path, label, hand_landmarks):
        hand_landmarks_parts_coords = {
            "Image Path" : img_path,
                
            "0: WRIST_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x), 
            "0: WRIST_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y),
            "0: WRIST_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z),
            
            "1: THUMB_CMC_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x),
            "1: THUMB_CMC_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y),
            "1: THUMB_CMC_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z),
            
            "2: THUMB_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x),
            "2: THUMB_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y),
            "2: THUMB_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z),
            
            "3: THUMB_IP_x"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x),
            "3: THUMB_IP_y"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y),
            "3: THUMB_IP_z"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z),
            
            "4: THUMB_TIP_x"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x),
            "4: THUMB_TIP_y"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y),
            "4: THUMB_TIP_z"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z),
            
            "5: INDEX_FINGER_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x),
            "5: INDEX_FINGER_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y),
            "5: INDEX_FINGER_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z),
            
            "6: INDEX_FINGER_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x),
            "6: INDEX_FINGER_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y),
            "6: INDEX_FINGER_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z),
            
            "7: INDEX_FINGER_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x),
            "7: INDEX_FINGER_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y),
            "7: INDEX_FINGER_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z),
            
            "8: INDEX_FINGER_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x),
            "8: INDEX_FINGER_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y),
            "8: INDEX_FINGER_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z),
            
            "9:  MIDDLE_FINGER_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x),
            "9:  MIDDLE_FINGER_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y),
            "9:  MIDDLE_FINGER_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z),
            
            "10:  MIDDLE_FINGER_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x),
            "10:  MIDDLE_FINGER_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y),
            "10:  MIDDLE_FINGER_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z),
            
            "11: MIDDLE_FINGER_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x),
            "11: MIDDLE_FINGER_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y),
            "11: MIDDLE_FINGER_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z),
            
            "12: MIDDLE_FINGER_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x),
            "12: MIDDLE_FINGER_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y),
            "12: MIDDLE_FINGER_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z),
            
            "13: RING_FINGER_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x),
            "13: RING_FINGER_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y),
            "13: RING_FINGER_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z),
            
            "14: RING_FINGER_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x),
            "14: RING_FINGER_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y),
            "14: RING_FINGER_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z),
            
            "15: RING_FINGER_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x),
            "15: RING_FINGER_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y),
            "15: RING_FINGER_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z),
            
            "16: RING_FINGER_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x),
            "16: RING_FINGER_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y),
            "16: RING_FINGER_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z),
            
            "17: PINKY_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x),
            "17: PINKY_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y),
            "17: PINKY_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z),
            
            "18: PINKY_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x),
            "18: PINKY_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y),
            "18: PINKY_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z),
            
            "19: PINKY_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x),
            "19: PINKY_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y),
            "19: PINKY_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z),
            
            "20: PINKY_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x),
            "20: PINKY_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y),
            "20: PINKY_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z),
            
            "Label" : label
        }
        return hand_landmarks_parts_coords
    
    def get_hand_coords_frame(self, hand_landmarks):
        hand_landmarks_parts_coords = {    
            "0: WRIST_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x), 
            "0: WRIST_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y),
            "0: WRIST_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z),
            
            "1: THUMB_CMC_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x),
            "1: THUMB_CMC_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y),
            "1: THUMB_CMC_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z),
            
            "2: THUMB_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x),
            "2: THUMB_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y),
            "2: THUMB_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z),
            
            "3: THUMB_IP_x"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x),
            "3: THUMB_IP_y"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y),
            "3: THUMB_IP_z"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z),
            
            "4: THUMB_TIP_x"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x),
            "4: THUMB_TIP_y"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y),
            "4: THUMB_TIP_z"  : float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z),
            
            "5: INDEX_FINGER_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x),
            "5: INDEX_FINGER_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y),
            "5: INDEX_FINGER_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z),
            
            "6: INDEX_FINGER_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x),
            "6: INDEX_FINGER_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y),
            "6: INDEX_FINGER_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z),
            
            "7: INDEX_FINGER_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x),
            "7: INDEX_FINGER_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y),
            "7: INDEX_FINGER_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z),
            
            "8: INDEX_FINGER_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x),
            "8: INDEX_FINGER_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y),
            "8: INDEX_FINGER_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z),
            
            "9:  MIDDLE_FINGER_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x),
            "9:  MIDDLE_FINGER_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y),
            "9:  MIDDLE_FINGER_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z),
            
            "10:  MIDDLE_FINGER_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x),
            "10:  MIDDLE_FINGER_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y),
            "10:  MIDDLE_FINGER_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z),
            
            "11: MIDDLE_FINGER_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x),
            "11: MIDDLE_FINGER_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y),
            "11: MIDDLE_FINGER_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z),
            
            "12: MIDDLE_FINGER_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x),
            "12: MIDDLE_FINGER_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y),
            "12: MIDDLE_FINGER_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z),
            
            "13: RING_FINGER_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x),
            "13: RING_FINGER_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y),
            "13: RING_FINGER_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z),
            
            "14: RING_FINGER_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x),
            "14: RING_FINGER_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y),
            "14: RING_FINGER_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z),
            
            "15: RING_FINGER_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x),
            "15: RING_FINGER_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y),
            "15: RING_FINGER_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z),
            
            "16: RING_FINGER_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x),
            "16: RING_FINGER_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y),
            "16: RING_FINGER_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z),
            
            "17: PINKY_MCP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x),
            "17: PINKY_MCP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y),
            "17: PINKY_MCP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z),
            
            "18: PINKY_PIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x),
            "18: PINKY_PIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y),
            "18: PINKY_PIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z),
            
            "19: PINKY_DIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x),
            "19: PINKY_DIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y),
            "19: PINKY_DIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z),
            
            "20: PINKY_TIP_x" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x),
            "20: PINKY_TIP_y" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y),
            "20: PINKY_TIP_z" : float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z),
        }
        return hand_landmarks_parts_coords
    
    def read_single_image(self, file):
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results
    
    def create_landmark_csv_from_single_image_folder(self, image_folder_name, label):
        image_folder_path = os.path.join(self.base_path, image_folder_name)
        images = os.listdir(image_folder_path)
        img_list = []

        for img in tqdm(images, total=len(images)):
            img_path = os.path.join(image_folder_path, img)
            img_results = self.read_single_image(img_path)
            
            if img_results.multi_hand_landmarks is not None:
                for hand_landmarks in img_results.multi_hand_landmarks:
                    img_list.append(
                        self.get_hand_marks_coords_dict_for_image(img_path, label, hand_landmarks)
                    )
            
        df = pd.DataFrame.from_dict(img_list)
        if "Unnamed: 0" in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1)
        df = df.reset_index(drop=True) 
        return df
    
    def create_train_csv(self, save_csv_folder_name):
        train_base_path = os.path.join(self.base_path, self.train_image_folder_name)
        train_csvs_path = os.path.join(self.base_path, save_csv_folder_name)
        labels = os.listdir(train_base_path)
        label_dict = self.get_label_dict()

        if len(os.listdir(train_csvs_path)) != 0:
            print("Files are already present")

        else:
            print("process starting ....")
            for i, label in enumerate(labels):
                class_label = label_dict[label]
                folder_name = label 
                temp_image_folder = os.path.join(self.train_image_folder_name, label)
                single_train_csv_data = self.create_landmark_csv_from_single_image_folder(temp_image_folder, class_label)
                
                if "Unnamed: 0" in single_train_csv_data.columns:
                    single_train_csv_data = single_train_csv_data.drop(['Unnamed: 0'], axis=1)
                
                single_train_csv_data = single_train_csv_data.reset_index(drop=True)

                print(f"completed {label}.csv with shape: {single_train_csv_data.shape}")
                csv_save_path_name = os.path.join(train_csvs_path, f"{label}.csv")
                single_train_csv_data.to_csv(csv_save_path_name)
            print("Training data making completed ....")


    def create_test_csv(self, save_csv_folder_name):
        path = os.path.join(self.base_path, self.test_image_folder_name)
        save_path = os.path.join(self.base_path, save_csv_folder_name)
        img_list = []
        label_dict = self.get_label_dict()

        if "test_data.csv" in os.listdir(save_path):
            print("Files are already present")
        
        else:
            for image_name in tqdm(os.listdir(path), total=len(os.listdir(path))):
                label = image_name.split('_')[0]
                image_path = os.path.join(path, image_name)
                image_results = self.read_single_image(image_path)

                if image_results.multi_hand_landmarks is not None:
                    for hand_landmarks in image_results.multi_hand_landmarks:
                        img_list.append(
                            self.get_hand_marks_coords_dict_for_image(image_path, label_dict[label], hand_landmarks)
                        )
            df = pd.DataFrame.from_dict(img_list)
            if "Unnamed: 0" in df.columns:
                df = df.drop(['Unnamed: 0'], axis=1)
            df = df.reset_index(drop=True)
            df.to_csv(os.path.join(save_path, "test_data.csv"))


    def concat_all_csv_into_one(self, all_csv_path, save_file_folder_name, shuffle = True, split = None):
        all_csv_file_paths = os.path.join(self.base_path, all_csv_path)
        save_file_folder_path = os.path.join(self.base_path, save_file_folder_name)

        if "train_data.csv" not in os.listdir(save_file_folder_path):
            print("Concatination process started ...")
            concated_csv_data = pd.concat(
                pd.read_csv(os.path.join(all_csv_file_paths, file)) for file in tqdm(os.listdir(all_csv_file_paths), total = len(os.listdir(all_csv_file_paths)))
            )

            if "Unnamed: 0" in concated_csv_data.columns:
                concated_csv_data = concated_csv_data.drop(['Unnamed: 0'], axis=1)
            
            if shuffle:
                concated_csv_data = concated_csv_data.sample(frac=1)

            if split is not None and split < 0.5:
                concated_csv_data_train, concated_csv_data_valid = self.train_validation_split(concated_csv_data, split)
                concated_csv_data_train = concated_csv_data_train.reset_index(drop=True)
                concated_csv_data_valid = concated_csv_data_valid.reset_index(drop=True)

                concated_csv_data_train.to_csv(os.path.join(save_file_folder_path, "train_data.csv"))
                concated_csv_data_valid.to_csv(os.path.join(save_file_folder_path, "valid_data.csv"))
                print("done")

            else:
                concated_csv_data = concated_csv_data.reset_index(drop=True)
                concated_csv_data.to_csv(os.path.join(save_file_folder_path, "train_data.csv"))
                print("done")
        else:
            print("Files are present already")
    

    def train_validation_split(self, df, split_size):
        total_size = df.shape[0]
        valid_size = int(total_size * split_size)
        train_size = total_size - valid_size
        train_df = df.iloc[:train_size, :]
        valid_df = df.iloc[train_size:, :]
        return train_df, valid_df


if __name__ == '__main__':
    hu = HandPosUtils()
    image_folder_name = "ImageData/asl_alphabet_train/"
    save_csv_folder_name = "Data/CSVs"

    #hu.create_train_csv(image_folder_name, save_csv_folder_name)
    hu.create_test_csv("Data/raw")