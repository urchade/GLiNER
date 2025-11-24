from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LstmSeq2SeqEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequence-to-sequence models.

    This encoder processes input sequences using a bidirectional LSTM and returns
    the encoded representations. It handles variable-length sequences through packing.

    Attributes:
        lstm: The bidirectional LSTM layer for encoding sequences.
    """

    def __init__(self, config, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = True) -> None:
        """Initializes the LSTM encoder.

        Args:
            config: Configuration object containing model hyperparameters.
                Must have a `hidden_size` attribute.
            num_layers: Number of recurrent layers. Defaults to 1.
            dropout: Dropout probability between LSTM layers. Defaults to 0.
            bidirectional: If True, becomes a bidirectional LSTM. Defaults to True.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Encodes input sequences through the LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            mask: Binary mask tensor of shape (batch_size, seq_len) where 1 indicates
                valid positions and 0 indicates padding.
            hidden: Optional initial hidden state tuple (h_0, c_0). Defaults to None.

        Returns:
            Encoded output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)

        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output


def create_projection_layer(hidden_size: int, dropout: float, out_dim: Optional[int] = None) -> nn.Sequential:
    """Creates a two-layer projection network with ReLU activation and dropout.

    The projection layer expands the input by 4x in the hidden layer before
    projecting to the output dimension.

    Args:
        hidden_size: Size of the input hidden dimension.
        dropout: Dropout probability applied after the first layer.
        out_dim: Output dimension size. If None, uses hidden_size. Defaults to None.

    Returns:
        A Sequential module containing the projection layers.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim)
    )


class MultiheadAttention(nn.Module):
    """Multi-head scaled dot-product attention mechanism.

    Implements multi-head attention where the hidden dimension is split across
    multiple attention heads. Uses PyTorch's scaled_dot_product_attention for
    efficient computation.

    Attributes:
        hidden_size: Total hidden dimension size.
        num_heads: Number of attention heads.
        attention_head_size: Dimension of each attention head.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        query_layer: Linear projection for query vectors.
        key_layer: Linear projection for key vectors.
        value_layer: Linear projection for value vectors.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        """Initializes the multi-head attention module.

        Args:
            hidden_size: Size of the hidden dimension. Must be divisible by num_heads.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.attention_probs_dropout_prob = dropout
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes tensor for multi-head attention computation.

        Transforms from (batch, seq_len, hidden) to (batch, num_heads, seq_len, head_dim).

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, attention_head_size).
        """
        new_x_shape = (*x.size()[:-1], self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Computes multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, hidden_size).
            key: Optional key tensor. If None, uses query. Defaults to None.
            value: Optional value tensor. If None, uses key or query. Defaults to None.
            head_mask: Optional mask for attention heads. Defaults to None.
            attn_mask: Optional attention mask. Defaults to None.

        Returns:
            A tuple containing:
                - context_layer: Attention output of shape (batch_size, seq_len, hidden_size).
                - None: Placeholder for attention weights (not returned).
        """
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
        new_context_layer_shape = (
            *context_layer.size()[:-2],
            self.hidden_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


class SelfAttentionBlock(nn.Module):
    """Self-attention block with pre-normalization and residual connection.

    Implements a standard transformer-style self-attention block with layer
    normalization before and after the attention operation.

    Attributes:
        self_attn: Multi-head self-attention module.
        pre_norm: Layer normalization applied before attention.
        post_norm: Layer normalization applied after residual connection.
        dropout: Dropout layer for attention output.
        q_proj: Linear projection for queries.
        k_proj: Linear projection for keys.
        v_proj: Linear projection for values.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        """Initializes the self-attention block.

        Args:
            d_model: Model dimension size.
            num_heads: Number of attention heads.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies self-attention to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.pre_norm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.self_attn(q, k, v, attn_mask=mask)
        output = x + self.dropout(attn_output)
        return self.post_norm(output)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with pre-normalization and residual connection.

    Implements cross-attention between query and key-value pairs, typically used
    for attending from one sequence to another.

    Attributes:
        cross_attn: Multi-head cross-attention module.
        pre_norm: Layer normalization applied to query before attention.
        post_norm: Layer normalization applied after residual connection.
        dropout: Dropout layer for attention output.
        v_proj: Linear projection for values.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        """Initializes the cross-attention block.

        Args:
            d_model: Model dimension size.
            num_heads: Number of attention heads.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies cross-attention from query to key-value pairs.

        Args:
            query: Query tensor of shape (batch_size, query_len, d_model).
            key: Key tensor of shape (batch_size, key_len, d_model).
            value: Optional value tensor. If None, derived from key. Defaults to None.
            mask: Optional attention mask. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, query_len, d_model).
        """
        query = self.pre_norm(query)
        if value is None:
            value = self.v_proj(key)
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=mask)
        output = query + self.dropout(attn_output)
        return self.post_norm(output)


class CrossFuser(nn.Module):
    """Flexible cross-attention fusion module with configurable attention patterns.

    Fuses two sequences using a configurable schema of self-attention and
    cross-attention operations. The schema defines the order and type of
    attention operations to apply.

    Schema notation:
        - 'l2l': Self-attention on label sequence
        - 't2t': Self-attention on text sequence
        - 'l2t': Cross-attention from label to text
        - 't2l': Cross-attention from text to label

    Attributes:
        d_model: Model dimension size.
        schema: List of attention operation types parsed from schema string.
        layers: ModuleList of attention layers organized by depth.
    """

    def __init__(
        self,
        d_model: int,
        query_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        schema: str = "l2l-l2t",
    ) -> None:
        """Initializes the cross-fusion module.

        Args:
            d_model: Model dimension size.
            query_dim: Dimension of query input (currently unused).
            num_heads: Number of attention heads. Defaults to 8.
            num_layers: Number of attention layers. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.1.
            schema: String defining attention pattern (e.g., 'l2l-l2t-t2t').
                Defaults to 'l2l-l2t'.
        """
        super().__init__()
        self.d_model = d_model
        self.schema = schema.split("-")
        layers = []
        for _ in range(num_layers):
            layer = []
            for attn_type in self.schema:
                if attn_type in {"l2l", "t2t"}:
                    layer.append(SelfAttentionBlock(d_model, num_heads, dropout))
                else:
                    layer.append(CrossAttentionBlock(d_model, num_heads, dropout))
            layer = nn.ModuleList(layer)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies cross-fusion between query and key sequences.

        Args:
            query: Query tensor of shape (batch_size, query_len, d_model).
            key: Key tensor of shape (batch_size, key_len, d_model).
            query_mask: Optional binary mask for query (1 = valid, 0 = padding).
                Shape (batch_size, query_len). Defaults to None.
            key_mask: Optional binary mask for key (1 = valid, 0 = padding).
                Shape (batch_size, key_len). Defaults to None.

        Returns:
            A tuple containing:
                - query: Fused query tensor of shape (batch_size, query_len, d_model).
                - key: Fused key tensor of shape (batch_size, key_len, d_model).
        """
        for sublayers in self.layers:
            for id, layer in enumerate(sublayers):
                if self.schema[id] == "l2l":
                    if query_mask is not None:
                        self_attn_mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
                    else:
                        self_attn_mask = None
                    query = layer(query, mask=self_attn_mask)
                elif self.schema[id] == "t2t":
                    if key_mask is not None:
                        self_attn_mask = key_mask.unsqueeze(1) * key_mask.unsqueeze(2)
                    else:
                        self_attn_mask = None
                    key = layer(key, mask=self_attn_mask)
                elif self.schema[id] == "l2t":
                    if query_mask is not None and key_mask is not None:
                        cross_attn_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
                    else:
                        cross_attn_mask = None
                    query = layer(query, key, mask=cross_attn_mask)
                elif self.schema[id] == "t2l":
                    if query_mask is not None and key_mask is not None:
                        cross_attn_mask = key_mask.unsqueeze(-1) * query_mask.unsqueeze(1)
                    else:
                        cross_attn_mask = None
                    key = layer(key, query, mask=cross_attn_mask)

        return query, key


class LayersFuser(nn.Module):
    """Fuses multiple encoder layer outputs using squeeze-and-excitation mechanism.

    Combines outputs from different encoder layers by learning adaptive weights
    for each layer using a squeeze-and-excitation style attention mechanism.
    The first layer in encoder_outputs is skipped during fusion.

    Attributes:
        num_layers: Number of encoder layers to fuse.
        hidden_size: Hidden dimension size of encoder outputs.
        output_size: Size of the final output projection.
        squeeze: Linear layer for squeeze operation.
        W1: First linear layer of excitation network.
        W2: Second linear layer of excitation network.
        output_projection: Final projection to output dimension.
    """

    def __init__(self, num_layers: int, hidden_size: int, output_size: Optional[int] = None) -> None:
        """Initializes the layer fusion module.

        Args:
            num_layers: Number of encoder layers to fuse.
            hidden_size: Hidden dimension size.
            output_size: Output dimension size. If None, uses hidden_size.
                Defaults to None.
        """
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

    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Fuses multiple encoder layer outputs into a single representation.

        Args:
            encoder_outputs: List of encoder output tensors, each of shape
                (batch_size, seq_len, hidden_size). The first element is skipped.

        Returns:
            Fused output tensor of shape (batch_size, seq_len, output_size).
        """
        # Concatenate all layers (skip first layer)
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

        # Final projection
        output = self.output_projection(U_sum)  # [B, L, output_size]

        return output
