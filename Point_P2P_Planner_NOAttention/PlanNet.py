import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim=128, num_layers=3, other_features:torch.tensor = None):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.other_features = other_features
        if self.other_features is not None:
            self.input_dim += len(self.other_features)
        
        layers = []
        layers.append(nn.Linear(self.input_dim, self.feature_dim))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.feature_dim, self.feature_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.feature_dim, self.output_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.other_features is not None:
            x.reshape(-1, 3)
            batch_size = x.shape[0]
            other_features = self.other_features.repeat(batch_size, 1)
            x = torch.cat((x, other_features), dim=1)
        return self.model(x)
    
class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim=128, num_layers=3, other_features = None):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.other_features = other_features
        if self.other_features is not None:
            self.input_dim += len(self.other_features)
        
        layers = []
        layers.append(nn.Linear(self.input_dim, self.feature_dim))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.feature_dim, self.feature_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.feature_dim, self.output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, action):
        x = torch.cat((x, action), dim=1)
        if self.other_features is not None:
            x.reshape(-1, 5)
            batch_size = x.shape[0]
            other_features = self.other_features.repeat(batch_size, 1)
            x = torch.cat((x, other_features), dim=1)
        return self.model(x)