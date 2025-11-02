import torch

def extract_word_embeddings(token_embeds, words_mask, attention_mask,
                            batch_size, max_text_length, embed_dim, text_lengths):
    words_embedding = torch.zeros(
        batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    batch_indices, word_idx = torch.where(words_mask > 0)

    target_word_idx = words_mask[batch_indices, word_idx] - 1

    words_embedding[batch_indices, target_word_idx] = token_embeds[batch_indices, word_idx]

    aranged_word_idx = torch.arange(max_text_length,
                                    dtype=attention_mask.dtype,
                                    device=token_embeds.device).expand(batch_size, -1)
    mask = aranged_word_idx < text_lengths
    return words_embedding, mask

def extract_prompt_features(class_token_index, token_embeds, input_ids, attention_mask,
                                                batch_size, embed_dim, embed_ent_token=True, **kwargs):
    class_token_mask = input_ids == class_token_index
    num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

    max_embed_dim = num_class_tokens.max()
    aranged_class_idx = torch.arange(max_embed_dim,
                                     dtype=attention_mask.dtype,
                                     device=token_embeds.device).expand(batch_size, -1)

    batch_indices, target_class_idx = torch.where(aranged_class_idx < num_class_tokens)
    _, class_indices = torch.where(class_token_mask)
    if not embed_ent_token:
        class_indices += 1

    prompts_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    prompts_embedding_mask = (aranged_class_idx < num_class_tokens).to(attention_mask.dtype)

    prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
    
    return prompts_embedding, prompts_embedding_mask

def extract_prompt_features_and_word_embeddings(class_token_index, token_embeds, input_ids, attention_mask,
                                                text_lengths, words_mask, embed_ent_token=True, **kwargs):
    batch_size, sequence_length, embed_dim = token_embeds.shape
    max_text_length = text_lengths.max()

    # getting prompt embeddings
    prompts_embedding, prompts_embedding_mask = extract_prompt_features(class_token_index, token_embeds, input_ids, attention_mask,
                                                                    batch_size, embed_dim, embed_ent_token, **kwargs)

    # getting words embedding
    words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask,
                                                    batch_size, max_text_length, embed_dim, text_lengths)

    return prompts_embedding, prompts_embedding_mask, words_embedding, mask


def build_entity_pairs(
    adj: torch.Tensor,           # (B, E, E) –  scores / mask (diag is ignored)
    span_rep: torch.Tensor,      # (B, E, D) –  entity/span embeddings
    threshold: float = 0.5,      # keep pairs with score > threshold
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract the (head, tail) indices of entity pairs for which adj > threshold.

    Returns
    -------
    pair_idx  : (B, N, 2)  int64   – padded with -1
    pair_mask : (B, N)     bool    – 1 for real pair, 0 for pad
    head_rep  : (B, N, D)  same D  – embeddings of heads
    tail_rep  : (B, N, D)          – embeddings of tails
    """
    B, E, _ = adj.shape
    device   = adj.device
    rows, cols = torch.triu_indices(E, E, offset=1, device=device)  # ignore (i,i) and duplicates

    batch_pair_lists: list[torch.Tensor] = []
    for b in range(B):
        sel = adj[b, rows, cols] > threshold          # (E*(E-1)/2,)
        pairs = torch.stack([rows[sel], cols[sel]], dim=-1)  # (M_b, 2)
        batch_pair_lists.append(pairs)

    N = max(p.shape[0] for p in batch_pair_lists) if batch_pair_lists else 0
    if N == 0:                                         # nothing selected anywhere
        pair_idx  = torch.full((B, 1, 2), -1, dtype=torch.long, device=device)
        pair_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        D         = span_rep.shape[-1]
        head_rep  = tail_rep = torch.zeros((B, 1, D), dtype=span_rep.dtype, device=device)
        return pair_idx, pair_mask, head_rep, tail_rep

    pair_idx  = torch.full((B, N, 2), -1, dtype=torch.long,  device=device)
    pair_mask = torch.zeros((B, N),    dtype=torch.bool,     device=device)

    for b, pairs in enumerate(batch_pair_lists):
        m = pairs.shape[0]
        pair_idx[b, :m]  = pairs
        pair_mask[b, :m] = True

    batch_idx = torch.arange(B, device=device).unsqueeze(1)                    # (B,1)
    head_rep  = span_rep[batch_idx, pair_idx[..., 0].clamp_min(0)]             # (B,N,D)
    tail_rep  = span_rep[batch_idx, pair_idx[..., 1].clamp_min(0)]             # (B,N,D)

    return pair_idx, pair_mask, head_rep, tail_rep