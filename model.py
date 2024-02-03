import torch
import torch.nn.functional as F
from modules.layers import LstmSeq2SeqEncoder
from modules.base import InstructBase
from modules.evaluator import Evaluator, greedy_search
from modules.span_rep import SpanRepLayer
from modules.token_rep import TokenRepLayer
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class GLiNER(InstructBase):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # [ENT] token
        self.entity_token = "<<ENT>>"
        self.sep_token = "<<SEP>>"

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=[self.entity_token, self.sep_token])

        # hierarchical representation of tokens
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        # span representation
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout,
        )

        # prompt representation (FFN)
        self.prompt_rep_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

    def compute_score_train(self, x):
        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)

        new_length = x['seq_length'].clone()
        new_tokens = []
        all_len_prompt = []
        num_classes_all = []

        # add prompt to the tokens
        for i in range(len(x['tokens'])):
            all_types_i = list(x['classes_to_id'][i].keys())
            # multiple entity types in all_types. Prompt is appended at the start of tokens
            entity_prompt = []
            num_classes_all.append(len(all_types_i))
            # add enity types to prompt
            for entity_type in all_types_i:
                entity_prompt.append(self.entity_token)  # [ENT] token
                entity_prompt.append(entity_type)  # entity type
            entity_prompt.append(self.sep_token)  # [SEP] token

            # prompt format:
            # [ENT] entity_type [ENT] entity_type ... [ENT] entity_type [SEP]

            # add prompt to the tokens
            tokens_p = entity_prompt + x['tokens'][i]

            # input format:
            # [ENT] entity_type_1 [ENT] entity_type_2 ... [ENT] entity_type_m [SEP] token_1 token_2 ... token_n

            # update length of the sequence (add prompt length to the original length)
            new_length[i] = new_length[i] + len(entity_prompt)
            # update tokens
            new_tokens.append(tokens_p)
            # store prompt length
            all_len_prompt.append(len(entity_prompt))

        # create a mask using num_classes_all (0, if it exceeds the number of classes, 1 otherwise)
        max_num_classes = max(num_classes_all)
        entity_type_mask = torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes_all), -1).to(
            x['span_mask'].device)
        entity_type_mask = entity_type_mask < torch.tensor(num_classes_all).unsqueeze(-1).to(
            x['span_mask'].device)  # [batch_size, max_num_classes]

        # compute all token representations
        bert_output = self.token_rep_layer(new_tokens, new_length)
        word_rep_w_prompt = bert_output["embeddings"]  # embeddings for all tokens (with prompt)
        mask_w_prompt = bert_output["mask"]  # mask for all tokens (with prompt)

        # get word representation (after [SEP]), mask (after [SEP]) and entity type representation (before [SEP])
        word_rep = []  # word representation (after [SEP])
        mask = []  # mask (after [SEP])
        entity_type_rep = []  # entity type representation (before [SEP])
        for i in range(len(x['tokens'])):
            prompt_entity_length = all_len_prompt[i]  # length of prompt for this example
            # get word representation (after [SEP])
            word_rep.append(word_rep_w_prompt[i, prompt_entity_length:prompt_entity_length + x['seq_length'][i]])
            # get mask (after [SEP])
            mask.append(mask_w_prompt[i, prompt_entity_length:prompt_entity_length + x['seq_length'][i]])

            # get entity type representation (before [SEP])
            entity_rep = word_rep_w_prompt[i, :prompt_entity_length - 1]  # remove [SEP]
            entity_rep = entity_rep[0::2]  # it means that we take every second element starting from the second one
            entity_type_rep.append(entity_rep)

        # padding for word_rep, mask and entity_type_rep
        word_rep = pad_sequence(word_rep, batch_first=True)  # [batch_size, seq_len, hidden_size]
        mask = pad_sequence(mask, batch_first=True)  # [batch_size, seq_len]
        entity_type_rep = pad_sequence(entity_type_rep, batch_first=True)  # [batch_size, len_types, hidden_size]

        # compute span representation
        word_rep = self.rnn(word_rep, mask)
        span_rep = self.span_rep_layer(word_rep, span_idx)

        # compute final entity type representation (FFN)
        entity_type_rep = self.prompt_rep_layer(entity_type_rep)  # (batch_size, len_types, hidden_size)
        num_classes = entity_type_rep.shape[1]  # number of entity types

        # similarity score
        scores = torch.einsum('BLKD,BCD->BLKC', span_rep, entity_type_rep)

        return scores, num_classes, entity_type_mask

    def forward(self, x):
        # compute span representation
        scores, num_classes, entity_type_mask = self.compute_score_train(x)
        batch_size = scores.shape[0]

        # loss for filtering classifier
        logits_label = scores.view(-1, num_classes)
        labels = x["span_label"].view(-1)  # (batch_size * num_spans)
        mask_label = labels != -1  # (batch_size * num_spans)
        labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

        # one-hot encoding
        labels_one_hot = torch.zeros(labels.size(0), num_classes + 1, dtype=torch.float32).to(scores.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Set the corresponding index to 1
        labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column
        # Shape of labels_one_hot: (batch_size * num_spans, num_classes)

        # compute loss (without reduction)
        all_losses = F.binary_cross_entropy_with_logits(logits_label, labels_one_hot,
                                                        reduction='none')
        # mask loss using entity_type_mask (B, C)
        masked_loss = all_losses.view(batch_size, -1, num_classes) * entity_type_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)
        # expand mask_label to all_losses
        mask_label = mask_label.unsqueeze(-1).expand_as(all_losses)
        # put lower loss for in label_one_hot (2 for positive, 1 for negative)
        weight_c = labels_one_hot + 1
        # apply mask
        all_losses = all_losses * mask_label.float() * weight_c
        return all_losses.sum()

    def compute_score_eval(self, x):
        # check if classes_to_id is dict
        assert isinstance(x['classes_to_id'], dict), "classes_to_id must be a dict"

        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)

        all_types = list(x['classes_to_id'].keys())
        # multiple entity types in all_types. Prompt is appended at the start of tokens
        entity_prompt = []

        # add enity types to prompt
        for entity_type in all_types:
            entity_prompt.append(self.entity_token)
            entity_prompt.append(entity_type)

        entity_prompt.append(self.sep_token)

        prompt_entity_length = len(entity_prompt)

        # add prompt
        tokens_p = [entity_prompt + tokens for tokens in x['tokens']]
        seq_length_p = x['seq_length'] + prompt_entity_length

        out = self.token_rep_layer(tokens_p, seq_length_p)

        word_rep_w_prompt = out["embeddings"]
        mask_w_prompt = out["mask"]

        # remove prompt
        word_rep = word_rep_w_prompt[:, prompt_entity_length:, :]
        mask = mask_w_prompt[:, prompt_entity_length:]

        # get_entity_type_rep
        entity_type_rep = word_rep_w_prompt[:, :prompt_entity_length - 1, :]
        # extract [ENT] tokens (which are at even positions in entity_type_rep)
        entity_type_rep = entity_type_rep[:, 0::2, :]

        entity_type_rep = self.prompt_rep_layer(entity_type_rep)  # (batch_size, len_types, hidden_size)

        word_rep = self.rnn(word_rep, mask)

        span_rep = self.span_rep_layer(word_rep, span_idx)

        local_scores = torch.einsum('BLKD,BCD->BLKC', span_rep, entity_type_rep)

        return local_scores

    @torch.no_grad()
    def predict(self, x, flat_ner=False, threshold=0.5):
        local_scores = self.compute_score_eval(x)
        spans = []
        for i, _ in enumerate(x["tokens"]):
            local_i = local_scores[i]
            wh_i = [i.tolist() for i in torch.where(torch.sigmoid(local_i) > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(x["tokens"][i]):
                    span_i.append((s, s + k, x["id_to_classes"][c + 1], local_i[s, k, c]))
            span_i = greedy_search(span_i, flat_ner)
            spans.append(span_i)
        return spans

    def evaluate(self, test_data, flat_ner=False, threshold=0.5, batch_size=12, entity_types=None):
        self.eval()
        data_loader = self.create_dataloader(test_data, batch_size=batch_size, entity_types=entity_types, shuffle=False)
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in data_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            batch_predictions = self.predict(x, flat_ner, threshold)
            all_preds.extend(batch_predictions)
            all_trues.extend(x["entities"])
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1
