import json
import pickle
import os
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import time
import hashlib

def save_pkl(data: list, dir: str):
    data_sha = hashlib.sha256()
    pkl_data = pickle.dumps(data)
    data_sha.update(pkl_data)
    filename = os.path.join(dir, f"{data_sha.hexdigest()}.pkl")
    with open(filename, 'wb') as f:
        f.write(pkl_data)

class SceneDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data:list[ # num of files
            tuple[
                list[list[float]], # feature of the scene
                list[float], # point
                bool # True if in the config space, False if not
            ]
        ] = []
        self.load_data()

    def load_data(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                with open(os.path.join(self.data_dir, filename), 'rb') as f:
                    scene_data = pickle.load(f)
                    for (triangles, test_points, in_config_space) in scene_data:
                        for i in range(len(test_points)):
                            self.data.append((triangles, test_points[i], in_config_space[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[
        list[list[float]], # feature of the scene
        list[float], # point
        bool # True if in the config space, False if not
    ]:
        scene_data = self.data[idx]
        if self.transform:
            scene_data = self.transform(scene_data)
        return scene_data
    