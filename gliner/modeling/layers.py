import torch
from torch import nn
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
