import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, config, num_layers=1, dropout=0., bidirectional=True):
        super(LstmSeq2SeqEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size//2,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)

        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output


def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    """
    Creates a projection layer with specified configurations.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout) -> None:
        super().__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.attention_head_size=hidden_size//num_heads
        self.attention_probs_dropout_prob=dropout
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key=None, value=None, head_mask=None, attn_mask=None):
        query = self.transpose_for_scores(self.query_layer(query))
        if key is None:
            key = self.transpose_for_scores(self.key_layer(query))
        else:
            key = self.transpose_for_scores(self.key_layer(key))
        if value is None and key is None:
            value = self.transpose_for_scores(self.value_layer(query))
        elif value is None and key is not None:
            value = self.transpose_for_scores(self.value_layer(key))
        else:
            value = self.transpose_for_scores(self.value_layer(value))

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        x = self.pre_norm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.self_attn(q, k, v, attn_mask=mask)
        output = x + self.dropout(attn_output)
        return self.post_norm(output)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value=None, mask=None):
        query = self.pre_norm(query)
        if value is None:
            value = self.v_proj(key)
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=mask)
        output = query + self.dropout(attn_output)
        return self.post_norm(output)
    
class CrossFuser(nn.Module):
    def __init__(self, d_model, query_dim, num_heads=8, num_layers=1, dropout=0.1, schema='l2l-l2t'):
        super().__init__()
        self.d_model = d_model
        self.schema = schema.split('-')
        layers = []
        for _ in range(num_layers):
            layer = []
            for attn_type in self.schema:
                if attn_type in {'l2l', 't2t'}:
                    layer.append(SelfAttentionBlock(d_model, num_heads, dropout))
                else:
                    layer.append(CrossAttentionBlock(d_model, num_heads, dropout))
            layer = nn.ModuleList(layer)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        # self.dense_i = nn.Linear(query_dim, d_model)
        # self.dense_o = nn.Linear(d_model, query_dim)

    def forward(self, query, key, query_mask=None, key_mask=None):
        # query = self.dense_i(query)
        for sublayers in self.layers:
            for id, layer in enumerate(sublayers):
                if self.schema[id] == 'l2l':
                    if query_mask is not None:
                        self_attn_mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
                    else:
                        self_attn_mask = None
                    query = layer(query, mask=self_attn_mask)
                elif self.schema[id] == 't2t':
                    if key_mask is not None:
                        self_attn_mask = key_mask.unsqueeze(1) * key_mask.unsqueeze(2)
                    else:
                        self_attn_mask = None
                    key = layer(key, mask=self_attn_mask)
                elif self.schema[id] == 'l2t':
                    if query_mask is not None and key_mask is not None:
                        cross_attn_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
                    else:
                        cross_attn_mask = None
                    query = layer(query, key, mask=cross_attn_mask)
                elif self.schema[id] == 't2l':
                    if query_mask is not None and key_mask is not None:
                        cross_attn_mask = key_mask.unsqueeze(-1) * query_mask.unsqueeze(1)
                    else:
                        cross_attn_mask = None
                    key = layer(key, query, mask=cross_attn_mask)
        # query=self.dense_o(query)   
        return query, key

class LayersFuser(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        
        # Squeeze operation
        self.squeeze = nn.Linear(hidden_size, 1)
        
        # Excitation operation
        self.W1 = nn.Linear(num_layers, num_layers // 2)
        self.W2 = nn.Linear(num_layers // 2, num_layers)
        
        # Final projection
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, encoder_outputs):
        # encoder_outputs is a list of tensors, each of shape [B, L, D]
        B, L, D = encoder_outputs[0].shape
        
        # Concatenate all layers
        U = torch.stack(encoder_outputs[1:], dim=1)  # [B, K, L, D]

        # Squeeze operation
        Z = self.squeeze(U).squeeze(-1)  # [B, K, L]
        Z = Z.mean(dim=2)  # [B, K]
        
        # Excitation operation
        s = self.W2(F.relu(self.W1(Z)))  # [B, K]
        s = torch.sigmoid(s)  # [B, K]
        
        # Apply attention weights
        U_weighted = U * s.unsqueeze(-1).unsqueeze(-1)  # [B, K, L, D]
        
        # Sum across layers
        U_sum = U_weighted.sum(dim=1)  # [B, L, D]
        
        # final projection
        output = self.output_projection(U_sum)  # [B, L, output_size]
        
        return output