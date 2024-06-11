import torch
from torch import nn

class Scorer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()

        self.proj_token = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 3)  # start, end, score
        )

    def forward(self, token_rep, label_rep):
        batch_size, seq_len, hidden_size = token_rep.shape
        num_classes = label_rep.shape[1]

        # (batch_size, seq_len, 3, hidden_size)
        token_rep = self.proj_token(token_rep).view(batch_size, seq_len, 1, 2, hidden_size)
        label_rep = self.proj_label(label_rep).view(batch_size, 1, num_classes, 2, hidden_size)

        # (2, batch_size, seq_len, num_classes, hidden_size)
        token_rep = token_rep.expand(-1, -1, num_classes, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, seq_len, -1, -1, -1).permute(3, 0, 1, 2, 4)

        # (batch_size, seq_len, num_classes, hidden_size * 3)
        cat = torch.cat([token_rep[0], label_rep[0], token_rep[1] * label_rep[1]], dim=-1)

        # (batch_size, seq_len, num_classes, 3)
        scores = self.out_mlp(cat).permute(3, 0, 1, 2)

        return scores
