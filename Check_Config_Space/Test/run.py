import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from DatasetGenerator.FileIO import *
from DatasetGenerator.Generator import *
from Train.Model import *
import torch


val_data_dir = os.path.join(os.path.dirname(__file__), '../data/train')
val_dataset = SceneDataset(val_data_dir)

num_of_inserted = 0
for i in range(len(val_dataset)):
    if all(val_dataset[i][2].data.numpy() == [0.0, 1.0]):
        num_of_inserted += 1
        continue
    
print(f"{num_of_inserted / len(val_dataset)} of the data in val dataset are inserted into the config space.")