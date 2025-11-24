from typing import Tuple

import torch


def extract_word_embeddings(
    token_embeds: torch.Tensor,
    words_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    max_text_length: int,
    embed_dim: int,
    text_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract word-level embeddings from subword token embeddings.

    Maps subword token embeddings back to word-level embeddings using a word mask
    that indicates which subword token corresponds to which word. Only the first
    subword token of each word is typically used for the word representation.

    This is essential for span-based NER where predictions are made at the word
    level but the transformer operates on subword tokens.

    Args:
        token_embeds: Subword token embeddings from transformer.
            Shape: (batch_size, seq_len, embed_dim)
        words_mask: Mask mapping subword positions to word indices. Non-zero values
            indicate the word index (1-indexed). Zero values are special tokens or
            continuation subwords to ignore.
            Shape: (batch_size, seq_len)
        attention_mask: Standard attention mask from tokenizer.
            Shape: (batch_size, seq_len)
        batch_size: Size of the batch.
        max_text_length: Maximum number of words across all examples in batch.
        embed_dim: Embedding dimension size.
        text_lengths: Number of words in each example.
            Shape: (batch_size, 1) or (batch_size,)

    Returns:
        Tuple containing:
            - words_embedding: Word-level embeddings extracted from token embeddings.
              Shape: (batch_size, max_text_length, embed_dim)
            - mask: Boolean mask indicating valid word positions (True) vs padding (False).
              Shape: (batch_size, max_text_length)
    """
    words_embedding = torch.zeros(
        batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    # Find positions where words_mask > 0 (actual word positions)
    batch_indices, word_idx = torch.where(words_mask > 0)

    # Convert 1-indexed word mask to 0-indexed positions
    target_word_idx = words_mask[batch_indices, word_idx] - 1

    # Copy token embeddings to word positions
    words_embedding[batch_indices, target_word_idx] = token_embeds[batch_indices, word_idx]

    # Create mask for valid word positions
    aranged_word_idx = torch.arange(max_text_length, dtype=attention_mask.dtype, device=token_embeds.device).expand(
        batch_size, -1
    )

    mask = aranged_word_idx < text_lengths
    return words_embedding, mask


def extract_prompt_features(
    class_token_index: int,
    token_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    embed_dim: int,
    embed_ent_token: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract prompt/entity type embeddings from special class tokens.

    Extracts embeddings for entity types or other prompt elements that are marked
    with special class tokens (e.g., [ENT] tokens). These embeddings represent
    the entity types that the model should extract.

    In prompt-based NER, the input is typically:
        [ENT] Person [ENT] Organization [SEP] John works at Google

    This function extracts the embeddings corresponding to the [ENT] tokens
    (or the tokens immediately after them if embed_ent_token=False).

    Args:
        class_token_index: Token ID of the special class token to extract
            (e.g., token ID for [ENT]).
        token_embeds: Token embeddings from transformer.
            Shape: (batch_size, seq_len, embed_dim)
        input_ids: Token IDs from tokenizer.
            Shape: (batch_size, seq_len)
        attention_mask: Standard attention mask from tokenizer.
            Shape: (batch_size, seq_len)
        batch_size: Size of the batch.
        embed_dim: Embedding dimension size.
        embed_ent_token: If True, use the [ENT] token embedding itself.
            If False, use the embedding of the token immediately after [ENT]
            (i.e., the entity type name token). Default: True.

    Returns:
        Tuple containing:
            - prompts_embedding: Embeddings for each prompt/entity type.
              Shape: (batch_size, max_num_types, embed_dim)
              where max_num_types is the maximum number of entity types
              across examples in the batch.
            - prompts_embedding_mask: Mask indicating valid prompt positions
              (True) vs padding (False).
              Shape: (batch_size, max_num_types)
    """
    # Find all positions with the class token
    class_token_mask = input_ids == class_token_index
    num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

    # Maximum number of class tokens across batch
    max_embed_dim = num_class_tokens.max()
    aranged_class_idx = torch.arange(max_embed_dim, dtype=attention_mask.dtype, device=token_embeds.device).expand(
        batch_size, -1
    )

    # Find valid positions (not padding)
    batch_indices, target_class_idx = torch.where(aranged_class_idx < num_class_tokens)
    _, class_indices = torch.where(class_token_mask)

    # Optionally shift to token after [ENT] (the entity type name)
    if not embed_ent_token:
        class_indices += 1

    # Initialize prompt embeddings tensor
    prompts_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    # Create mask for valid (non-padded) positions
    prompts_embedding_mask = (aranged_class_idx < num_class_tokens).to(attention_mask.dtype)

    # Extract embeddings at class token positions
    prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]

    return prompts_embedding, prompts_embedding_mask


def extract_prompt_features_and_word_embeddings(
    class_token_index: int,
    token_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    text_lengths: torch.Tensor,
    words_mask: torch.Tensor,
    embed_ent_token: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract both prompt embeddings and word embeddings in one call.

    Convenience function that combines extract_prompt_features and
    extract_word_embeddings to get both prompt/entity type embeddings
    and word-level text embeddings from a single set of token embeddings.

    This is the typical use case for prompt-based NER where you need both:
    1. Entity type embeddings (from prompt tokens like [ENT])
    2. Word-level text embeddings (from the actual text tokens)

    Args:
        class_token_index: Token ID of the special class token (e.g., [ENT]).
        token_embeds: Token embeddings from transformer.
            Shape: (batch_size, seq_len, embed_dim)
        input_ids: Token IDs from tokenizer.
            Shape: (batch_size, seq_len)
        attention_mask: Standard attention mask from tokenizer.
            Shape: (batch_size, seq_len)
        text_lengths: Number of words in each example.
            Shape: (batch_size, 1) or (batch_size,)
        words_mask: Mask mapping subword positions to word indices.
            Shape: (batch_size, seq_len)
        embed_ent_token: If True, use [ENT] token embedding. If False,
            use the token after [ENT] (the entity type name). Default: True.
        **kwargs: Additional keyword arguments passed to extract_prompt_features.

    Returns:
        Tuple containing:
            - prompts_embedding: Entity type embeddings.
              Shape: (batch_size, max_num_types, embed_dim)
            - prompts_embedding_mask: Mask for valid entity type positions.
              Shape: (batch_size, max_num_types)
            - words_embedding: Word-level text embeddings.
              Shape: (batch_size, max_text_length, embed_dim)
            - mask: Mask for valid word positions.
              Shape: (batch_size, max_text_length)
    """
    batch_size, _, embed_dim = token_embeds.shape
    max_text_length = text_lengths.max()

    # Extract prompt/entity type embeddings
    prompts_embedding, prompts_embedding_mask = extract_prompt_features(
        class_token_index, token_embeds, input_ids, attention_mask, batch_size, embed_dim, embed_ent_token, **kwargs
    )

    # Extract word-level embeddings
    words_embedding, mask = extract_word_embeddings(
        token_embeds, words_mask, attention_mask, batch_size, max_text_length, embed_dim, text_lengths
    )

    return prompts_embedding, prompts_embedding_mask, words_embedding, mask


def build_entity_pairs(
    adj: torch.Tensor,
    span_rep: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build entity pairs for relation extraction based on adjacency scores.

    Extracts entity pairs (head, tail) where the adjacency score exceeds a
    threshold, and retrieves their corresponding embeddings. This is used in
    relation extraction to select which entity pairs should be classified for
    relation types.

    The function considers ALL directed pairs (i,j) where i≠j, not just the
    upper triangle, since relation direction matters (e.g., "founded" vs
    "founded_by" have opposite directions).

    Args:
        adj: Adjacency matrix with scores or probabilities for entity pairs.
            Shape: (batch_size, num_entities, num_entities)
            The diagonal (self-pairs) is ignored. Values > threshold indicate
            potential relations.
        span_rep: Entity/span embeddings for each entity in the batch.
            Shape: (batch_size, num_entities, embed_dim)
        threshold: Minimum adjacency score to consider a pair as a potential
            relation. Pairs with adj[i,j] > threshold are kept. Default: 0.5.

    Returns:
        Tuple containing:
            - pair_idx: Indices of (head, tail) entity pairs.
              Shape: (batch_size, max_pairs, 2)
              Values are entity indices, or -1 for padding positions.
            - pair_mask: Boolean mask indicating valid pairs (True) vs padding (False).
              Shape: (batch_size, max_pairs)
            - head_rep: Embeddings of head entities for each pair.
              Shape: (batch_size, max_pairs, embed_dim)
            - tail_rep: Embeddings of tail entities for each pair.
              Shape: (batch_size, max_pairs, embed_dim)
    """
    B, E, _ = adj.shape
    device = adj.device
    D = span_rep.shape[-1]

    # Generate all possible (i, j) pairs where i != j
    all_rows = []
    all_cols = []
    for i in range(E):
        for j in range(E):
            if i != j:
                all_rows.append(i)
                all_cols.append(j)

    rows = torch.tensor(all_rows, device=device, dtype=torch.long)
    cols = torch.tensor(all_cols, device=device, dtype=torch.long)

    # For each example in batch, find pairs exceeding threshold
    batch_pair_lists: list[torch.Tensor] = []

    for b in range(B):
        sel = adj[b, rows, cols] > threshold  # Boolean mask for valid pairs
        pairs = torch.stack([rows[sel], cols[sel]], dim=-1)  # (num_valid_pairs, 2)
        batch_pair_lists.append(pairs)

    # Find maximum number of pairs across batch (for padding)
    N = max(p.shape[0] for p in batch_pair_lists) if batch_pair_lists else 0

    # Handle case where no pairs exceed threshold
    if N == 0:
        pair_idx = torch.full((B, 1, 2), -1, dtype=torch.long, device=device)
        pair_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        head_rep = tail_rep = torch.zeros((B, 1, D), dtype=span_rep.dtype, device=device)
        return pair_idx, pair_mask, head_rep, tail_rep

    # Initialize padded tensors
    pair_idx = torch.full((B, N, 2), -1, dtype=torch.long, device=device)
    pair_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

    # Fill in valid pairs for each example
    for b, pairs in enumerate(batch_pair_lists):
        m = pairs.shape[0]
        pair_idx[b, :m] = pairs
        pair_mask[b, :m] = True

    # Extract head and tail embeddings using advanced indexing
    batch_idx = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
    head_rep = span_rep[batch_idx, pair_idx[..., 0].clamp_min(0)]  # (B, N, D)
    tail_rep = span_rep[batch_idx, pair_idx[..., 1].clamp_min(0)]  # (B, N, D)

    return pair_idx, pair_mask, head_rep, tail_rep
