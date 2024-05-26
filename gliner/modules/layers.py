import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class TokenPromptProcessor:
    def __init__(self, entity_token, sep_token):
        self.entity_token = entity_token
        self.sep_token = sep_token

    def process(self, x, token_rep_layer, mode):
        if mode == "train":
            return self._process_train(x, token_rep_layer)
        elif mode == "eval":
            return self._process_eval(x, token_rep_layer)
        else:
            raise ValueError("Invalid mode specified. Choose 'train' or 'eval'.")

    def _process_train(self, x, token_rep_layer):

        device = next(token_rep_layer.parameters()).device

        new_length = x["seq_length"].clone()
        new_tokens = []
        all_len_prompt = []
        num_classes_all = []

        for i in range(len(x["tokens"])):
            all_types_i = list(x["classes_to_id"][i].keys())
            entity_prompt = []
            num_classes_all.append(len(all_types_i))

            for entity_type in all_types_i:
                entity_prompt.append(self.entity_token)
                entity_prompt.append(entity_type)
            entity_prompt.append(self.sep_token)

            tokens_p = entity_prompt + x["tokens"][i]
            new_length[i] = new_length[i] + len(entity_prompt)
            new_tokens.append(tokens_p)
            all_len_prompt.append(len(entity_prompt))

        max_num_classes = max(num_classes_all)
        entity_type_mask = torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes_all), -1).to(device)
        entity_type_mask = entity_type_mask < torch.tensor(num_classes_all).unsqueeze(-1).to(device)

        bert_output = token_rep_layer(new_tokens, new_length)
        word_rep_w_prompt = bert_output["embeddings"]
        mask_w_prompt = bert_output["mask"]

        word_rep = []
        mask = []
        entity_type_rep = []

        for i in range(len(x["tokens"])):
            prompt_entity_length = all_len_prompt[i]
            word_rep.append(word_rep_w_prompt[i, prompt_entity_length:new_length[i]])
            mask.append(mask_w_prompt[i, prompt_entity_length:new_length[i]])
            entity_rep = word_rep_w_prompt[i, :prompt_entity_length - 1]
            entity_rep = entity_rep[0::2]
            entity_type_rep.append(entity_rep)

        word_rep = pad_sequence(word_rep, batch_first=True)
        mask = pad_sequence(mask, batch_first=True)
        entity_type_rep = pad_sequence(entity_type_rep, batch_first=True)

        return word_rep, mask, entity_type_rep, entity_type_mask

    def _process_eval(self, x, token_rep_layer):
        all_types = list(x["classes_to_id"].keys())
        entity_prompt = []

        for entity_type in all_types:
            entity_prompt.append(self.entity_token)
            entity_prompt.append(entity_type)
        entity_prompt.append(self.sep_token)

        prompt_entity_length = len(entity_prompt)
        tokens_p = [entity_prompt + tokens for tokens in x["tokens"]]
        seq_length_p = x["seq_length"] + prompt_entity_length

        out = token_rep_layer(tokens_p, seq_length_p)
        word_rep_w_prompt = out["embeddings"]
        mask_w_prompt = out["mask"]

        word_rep = word_rep_w_prompt[:, prompt_entity_length:, :]
        mask = mask_w_prompt[:, prompt_entity_length:]

        entity_type_rep = word_rep_w_prompt[:, :prompt_entity_length - 1, :]
        entity_type_rep = entity_type_rep[:, 0::2, :]

        return word_rep, mask, entity_type_rep


class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(LstmSeq2SeqEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
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
