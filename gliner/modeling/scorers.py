import torch
from torch import nn


class Scorer(nn.Module):
    """Scorer for computing token-label compatibility scores.

    This scorer is designed for token-level models and computes pairwise
    interactions between token representations and label embeddings. For each
    token-label pair, it produces three scores (typically for start, end, and
    overall compatibility).

    The scoring mechanism uses:
        1. Separate projections for tokens and labels
        2. Bilinear-style interaction between projected representations
        3. An MLP to produce final scores

    Attributes:
        proj_token (nn.Linear): Linear projection for token representations,
            mapping from hidden_size to hidden_size * 2.
        proj_label (nn.Linear): Linear projection for label embeddings,
            mapping from hidden_size to hidden_size * 2.
        out_mlp (nn.Sequential): MLP that produces final scores from
            concatenated features.
    """

    def __init__(self, hidden_size, dropout=0.1):
        """Initialize the Scorer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            dropout (float, optional): Dropout rate for the output MLP.
                Defaults to 0.1.
        """
        super().__init__()
        self.proj_token = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 3),  # start, end, score
        )

    def forward(self, token_rep, label_rep):
        """Compute token-label compatibility scores.

        Args:
            token_rep (torch.Tensor): Token representations of shape
                [batch_size, seq_len, hidden_size].
            label_rep (torch.Tensor): Label embeddings of shape
                [batch_size, num_classes, hidden_size].

        Returns:
            torch.Tensor: Scores of shape [batch_size, seq_len, num_classes, 3],
                where the last dimension contains three scores per token-label
                pair (typically start, end, and overall compatibility scores).
        """
        batch_size, seq_len, hidden_size = token_rep.shape
        num_classes = label_rep.shape[1]

        # Project and split into two components for bilinear interaction
        # Shape: (batch_size, seq_len, 1, 2, hidden_size)
        token_rep = self.proj_token(token_rep).view(batch_size, seq_len, 1, 2, hidden_size)
        # Shape: (batch_size, 1, num_classes, 2, hidden_size)
        label_rep = self.proj_label(label_rep).view(batch_size, 1, num_classes, 2, hidden_size)

        # Expand and reorganize for pairwise computation
        # Shape: (2, batch_size, seq_len, num_classes, hidden_size)
        token_rep = token_rep.expand(-1, -1, num_classes, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, seq_len, -1, -1, -1).permute(3, 0, 1, 2, 4)

        # Concatenate: [first_token_proj, first_label_proj, element_wise_product]
        # Shape: (batch_size, seq_len, num_classes, hidden_size * 3)
        cat = torch.cat([token_rep[0], label_rep[0], token_rep[1] * label_rep[1]], dim=-1)

        # Compute final scores through MLP
        # Shape: (batch_size, seq_len, num_classes, 3)
        scores = self.out_mlp(cat)

        return scores
