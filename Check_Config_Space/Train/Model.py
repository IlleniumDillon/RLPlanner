import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self, dropout:float =0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor,    # (batch_size, query_len, d_k)
                key: torch.Tensor,      # (batch_size, kv_len, d_k)
                value: torch.Tensor,    # (batch_size, kv_len, d_v)
                valid_lens=None):
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / (d ** 0.5)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), value) 
        
    def masked_softmax(self, x, valid_lens):
        if valid_lens is None:
            return F.softmax(x, dim=-1)
        else:
            shape = x.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            self.sequence_mask(x.reshape(-1, shape[1]), valid_lens, value=-1e6)
            return F.softmax(x.reshape(shape), dim=-1)
        
    def sequence_mask(self, x, valid_lens, value=-1e6):
        maxlen = x.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=x.device)[None, :] < valid_lens[:, None]
        x[~mask] = value
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self,
        key_size:int,
        query_size:int,
        value_size:int,
        num_hiddens:int,
        num_heads:int,
        dropout:float = 0.1,
        bias:bool = False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        
    def forward(self,
        query:torch.Tensor,
        key:torch.Tensor,
        value:torch.Tensor,
        valid_lens=None
    ):
        query = self.transpose_qkv(self.W_q(query), self.num_heads)
        key = self.transpose_qkv(self.W_k(key), self.num_heads)
        value = self.transpose_qkv(self.W_v(value), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
        output = self.attention(query, key, value, valid_lens)
        output = self.transpose_output(output, self.num_heads)
        return self.W_o(output)
    
    def transpose_qkv(self, x:torch.Tensor, num_heads:int):
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])
    
    def transpose_output(self, x:torch.Tensor, num_heads:int):
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)
        
class AddNorm(nn.Module):
    def __init__(self, 
        norm_shape:int,
        dropout:float = 0.1,
        **kwargs
    ):
        super(AddNorm, self).__init__(**kwargs)
        self.ln = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return self.ln(x + self.dropout(y))
        
class EncoderBlock(nn.Module):
    def __init__(self,
        key_size:int,
        query_size:int,
        value_size:int,
        num_hiddens:int,
        norm_shape:int,
        num_heads:int,
        dropout:float = 0.1,
        use_bias:bool = False,
        **kwargs
    ):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens)
        )
        self.add_norm2 = AddNorm(norm_shape, dropout)
    def forward(self, x:torch.Tensor, valid_lens=None):
        y = self.add_norm1(x, self.attention(x, x, x, valid_lens))
        return self.add_norm2(y, self.ffn(y))
    
class TransformerEncoder(nn.Module):
    def __init__(self,
        key_size:int,
        query_size:int,
        value_size:int,
        num_hiddens:int,
        norm_shape:int,
        num_heads:int,
        num_layers:int,
        dropout:float = 0.1,
        use_bias:bool = False,
        **kwargs
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}", EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, num_heads, dropout, use_bias))
        
    def forward(self, x:torch.Tensor, valid_lens=None):
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(x, valid_lens)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self,
        key_size:int,
        query_size:int,
        value_size:int,
        num_hiddens:int,
        norm_shape:int,
        num_heads:int,
        dropout:float = 0.1,
        use_bias:bool = False,
        i: int = 0,
        **kwargs
    ):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        # self.add_norm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm2 = AddNorm(norm_shape, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens)
        )
        self.add_norm3 = AddNorm(norm_shape, dropout)
    def forward(self, x:torch.Tensor, state:torch.Tensor):
        # enc_outputs, enc_valid_lens = state[0], state[1]
        # if state[2][self.i] is not None:
        #     key_values = x
        # else:
        #     key_values = torch.cat((state[2][self.i], x), dim=1)
        # state[2][self.i] = key_values
        # if self.training:
        #     batch_size, num_steps, _ = x.shape
        #     dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        # else:
        #     dec_valid_lens = None
        
        # x2 = self.attention1(x, key_values, key_values, dec_valid_lens)
        # y = self.add_norm1(x, x2)
        # y2 = self.attention2(y, enc_outputs, enc_outputs, enc_valid_lens)
        # z = self.add_norm2(y, y2)
        # return self.add_norm3(z, self.ffn(z)), state
        
        # x2 = self.attention1(x, x, x)
        # y = self.add_norm1(x, x2)
        y = x
        y2 = self.attention2(y, state, state)
        z = self.add_norm2(y, y2)
        return self.add_norm3(z, self.ffn(z)), state
    
class TransformerDecoder(nn.Module):
    def __init__(self,
        output_dim:int,
        key_size:int,
        query_size:int,
        value_size:int,
        num_hiddens:int,
        norm_shape:int,
        num_heads:int,
        num_layers:int,
        dropout:float = 0.1,
        use_bias:bool = False,
        **kwargs
    ):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}", DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, num_heads, dropout, use_bias, i))
        self.dense = nn.Linear(num_hiddens, output_dim)
        
    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, x:torch.Tensor, state:torch.Tensor):
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)
        return self.dense(x), state
        

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
        
        # # Define the transformer encoder layer
        # encoder_layers = nn.TransformerEncoderLayer(d_model=attention_dim, nhead=num_heads, batch_first=True, dropout=0)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=block_num)
        
        # # Define the transformer decoder layer
        # decoder_layers = nn.TransformerDecoderLayer(d_model=attention_dim, nhead=num_heads, batch_first=True, dropout=0)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=block_num)
        
        self.transformer_encoder = TransformerEncoder(attention_dim, attention_dim, attention_dim, attention_dim, [attention_dim], num_heads, block_num)
        self.transformer_decoder = TransformerDecoder(attention_dim, attention_dim, attention_dim, attention_dim, attention_dim, [attention_dim], num_heads, 1)
        
        # Define the final output layer
        self.fc_out = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, attention_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim // 4, output_dim),
        )
    
    def forward(self, scene_features, point_features, valid_lens=None):
        # Embed the scene and point features
        scene_embedded = self.scene_embedding(scene_features)
        point_embedded = self.point_embedding(point_features)
        
        # Apply the transformer encoder
        scene_encoded = self.transformer_encoder(scene_embedded, valid_lens=valid_lens)
        
        # Apply the transformer decoder
        point_decoded = self.transformer_decoder(point_embedded, scene_encoded)
        
        # Get the final output
        output = self.fc_out(point_decoded[0])
        
        return output