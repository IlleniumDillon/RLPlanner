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
        self.max_triangles = 0
        self.load_data()

    def load_data(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                with open(os.path.join(self.data_dir, filename), 'rb') as f:
                    scene_data = pickle.load(f)
                    for (triangles, test_points, in_config_space) in scene_data:
                        for i in range(len(test_points)):
                            self.data.append((triangles, test_points[i], [0.0, 1.0] if in_config_space[i] else [1.0, 0.0]))
                        # Update the maximum number of triangles
                        self.max_triangles = max(self.max_triangles, len(triangles))
        self.max_triangles += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene_data = self.data[idx]
        if self.transform:
            scene_data = self.transform(scene_data)
        # Pad the triangles to the maximum length
        padded_triangles = torch.tensor(scene_data[0], dtype=torch.float32)
        if len(padded_triangles) < self.max_triangles:
            padding = torch.zeros((self.max_triangles - len(padded_triangles), 12), dtype=torch.float32)
            padded_triangles = torch.cat((padded_triangles, padding), dim=0)
        mask = torch.zeros(self.max_triangles, dtype=torch.float32)
        mask[:len(padded_triangles)] = 1.0
        return padded_triangles, \
               torch.tensor(scene_data[1], dtype=torch.float32), \
               torch.tensor(scene_data[2], dtype=torch.float32), mask
    