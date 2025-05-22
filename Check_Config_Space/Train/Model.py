import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CheckConfigSpaceModel(nn.Module):
    def __init__(self, 
                 scene_feature_dim: int, 
                 point_feature_dim: int, 
                 output_dim: int = 2,
                 attention_dim: int = 128,
                 num_heads: int = 4,
                 block_num: int = 2,
    ):
        super(CheckConfigSpaceModel, self).__init__()
        self.scene_feature_dim = scene_feature_dim
        self.point_feature_dim = point_feature_dim
        self.output_dim = output_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.block_num = block_num

        # Define the layers of the model
        self.scene_embedding = nn.Sequential(
            nn.Linear(scene_feature_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
        )
        self.point_embedding = nn.Sequential(
            nn.Linear(point_feature_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
        )
        
        # Define the transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=attention_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=block_num)
        
        # Define the transformer decoder layer
        decoder_layers = nn.TransformerDecoderLayer(d_model=attention_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=block_num)
        
        # Define the final output layer
        self.fc_out = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, output_dim),
        )
    
    def forward(self, scene_features, point_features, mask=None):
        # Embed the scene and point features
        scene_embedded = self.scene_embedding(scene_features)
        point_embedded = self.point_embedding(point_features)
        
        # Apply the transformer encoder
        scene_encoded = self.transformer_encoder(scene_embedded, src_key_padding_mask=mask)
        
        # Apply the transformer decoder
        point_decoded = self.transformer_decoder(point_embedded, scene_encoded)
        
        # Get the final output
        output = self.fc_out(point_decoded)
        
        return output