import pytest
import torch
from gliner.modeling.utils import (
    extract_word_embeddings,
    extract_prompt_features,
    extract_prompt_features_and_word_embeddings,
    build_entity_pairs
)
from unittest.mock import Mock
from gliner.modeling.base import (
    BaseModel,
    UniEncoderSpanModel,
    UniEncoderTokenModel,
    BiEncoderSpanModel,
    UniEncoderSpanDecoderModel,
    UniEncoderSpanRelexModel
)
from gliner.config import (
    BaseGLiNERConfig,
    UniEncoderSpanConfig,
    UniEncoderTokenConfig,
    UniEncoderSpanDecoderConfig,
    UniEncoderSpanRelexConfig,
    BiEncoderSpanConfig
)

class TestExtractWordEmbeddings:
    """Test suite for extract_word_embeddings function."""
    
    @pytest.fixture
    def basic_setup(self):
        """Fixture providing basic inputs for word embeddings extraction."""
        batch_size = 2
        seq_length = 10
        embed_dim = 8
        
        token_embeds = torch.randn(batch_size, seq_length, embed_dim)
        
        # words_mask: 0 means not a word, >0 means word index (1-indexed)
        words_mask = torch.zeros(batch_size, seq_length, dtype=torch.long)
        words_mask[0, 1] = 1  # First word at position 1
        words_mask[0, 3] = 2  # Second word at position 3
        words_mask[1, 2] = 1  # First word in second batch at position 2
        
        max_text_length=torch.count_nonzero(words_mask, dim=1).max().item()
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        text_lengths = torch.tensor([2, 1], dtype=torch.long).unsqueeze(-1)
        
        return {
            'token_embeds': token_embeds,
            'words_mask': words_mask,
            'attention_mask': attention_mask,
            'batch_size': batch_size,
            'max_text_length': max_text_length,
            'embed_dim': embed_dim,
            'text_lengths': text_lengths
        }
    
    def test_output_shape(self, basic_setup):
        """Should return tensors with correct shapes."""
        words_embedding, mask = extract_word_embeddings(**basic_setup)
        
        assert words_embedding.shape == (
            basic_setup['batch_size'],
            basic_setup['max_text_length'],
            basic_setup['embed_dim']
        )
        assert mask.shape == (basic_setup['batch_size'], basic_setup['max_text_length'])
    
    def test_extracts_correct_embeddings(self, basic_setup):
        """Should place token embeddings at correct word positions."""
        words_embedding, mask = extract_word_embeddings(**basic_setup)
        
        # Check that embeddings were copied correctly
        # words_mask[0, 1] = 1 means token at position 1 goes to word position 0
        assert torch.allclose(
            words_embedding[0, 0],
            basic_setup['token_embeds'][0, 1]
        )
        
        # words_mask[0, 3] = 2 means token at position 3 goes to word position 1
        assert torch.allclose(
            words_embedding[0, 1],
            basic_setup['token_embeds'][0, 3]
        )
        
        # words_mask[1, 2] = 1 means token at position 2 goes to word position 0
        assert torch.allclose(
            words_embedding[1, 0],
            basic_setup['token_embeds'][1, 2]
        )
    
    def test_creates_valid_mask(self, basic_setup):
        """Should create mask based on text_lengths."""
        words_embedding, mask = extract_word_embeddings(**basic_setup)
        
        # First batch has text_length=2, so first 2 positions should be True
        assert mask[0, 0] == True
        assert mask[0, 1] == True
        
        # Second batch has text_length=1, so only first position should be True
        assert mask[1, 0] == True
        if basic_setup['max_text_length'] > 1:
            assert mask[1, 1] == False
    
    def test_preserves_dtype_and_device(self, basic_setup):
        """Should preserve dtype and device of input tensors."""
        words_embedding, mask = extract_word_embeddings(**basic_setup)
        
        assert words_embedding.dtype == basic_setup['token_embeds'].dtype
        assert words_embedding.device == basic_setup['token_embeds'].device
    
    def test_handles_empty_words_mask(self):
        """Should handle case with no words marked."""
        batch_size = 1
        seq_length = 5
        embed_dim = 4
        max_text_length = 3
        
        token_embeds = torch.randn(batch_size, seq_length, embed_dim)
        words_mask = torch.zeros(batch_size, seq_length, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        text_lengths = torch.tensor([0], dtype=torch.long).unsqueeze(-1)
        
        words_embedding, mask = extract_word_embeddings(
            token_embeds, words_mask, attention_mask,
            batch_size, max_text_length, embed_dim, text_lengths
        )
        
        # Should return all zeros
        assert torch.allclose(words_embedding, torch.zeros_like(words_embedding))
        assert torch.all(mask == False)


class TestExtractPromptFeatures:
    """Test suite for extract_prompt_features function."""
    
    @pytest.fixture
    def prompt_setup(self):
        """Fixture providing inputs for prompt features extraction."""
        batch_size = 2
        seq_length = 12
        embed_dim = 8
        class_token_index = 50  # Special token ID
        
        token_embeds = torch.randn(batch_size, seq_length, embed_dim)
        
        # input_ids with class tokens at specific positions
        input_ids = torch.randint(0, 30, (batch_size, seq_length))
        input_ids[0, 2] = class_token_index
        input_ids[0, 5] = class_token_index
        input_ids[0, 8] = class_token_index
        input_ids[1, 3] = class_token_index
        input_ids[1, 7] = class_token_index
        
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        
        return {
            'class_token_index': class_token_index,
            'token_embeds': token_embeds,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'batch_size': batch_size,
            'embed_dim': embed_dim
        }
    
    def test_output_shape(self, prompt_setup):
        """Should return tensors with correct shapes."""
        prompts_embedding, prompts_mask = extract_prompt_features(**prompt_setup)
        
        # Max class tokens is 3 (from first batch)
        assert prompts_embedding.shape == (prompt_setup['batch_size'], 3, prompt_setup['embed_dim'])
        assert prompts_mask.shape == (prompt_setup['batch_size'], 3)
    
    def test_extracts_class_token_embeddings(self, prompt_setup):
        """Should extract embeddings at class token positions when embed_ent_token=True."""
        prompts_embedding, prompts_mask = extract_prompt_features(
            **prompt_setup, embed_ent_token=True
        )
        
        # First batch has class tokens at positions 2, 5, 8
        assert torch.allclose(
            prompts_embedding[0, 0],
            prompt_setup['token_embeds'][0, 2]
        )
        assert torch.allclose(
            prompts_embedding[0, 1],
            prompt_setup['token_embeds'][0, 5]
        )
        assert torch.allclose(
            prompts_embedding[0, 2],
            prompt_setup['token_embeds'][0, 8]
        )
    
    def test_extracts_next_token_when_embed_ent_token_false(self, prompt_setup):
        """Should extract embeddings after class tokens when embed_ent_token=False."""
        prompts_embedding, prompts_mask = extract_prompt_features(
            **prompt_setup, embed_ent_token=False
        )
        
        # Should extract from positions 3, 6, 9 (one after class tokens)
        assert torch.allclose(
            prompts_embedding[0, 0],
            prompt_setup['token_embeds'][0, 3]
        )
        assert torch.allclose(
            prompts_embedding[0, 1],
            prompt_setup['token_embeds'][0, 6]
        )
        assert torch.allclose(
            prompts_embedding[0, 2],
            prompt_setup['token_embeds'][0, 9]
        )
    
    def test_creates_valid_mask(self, prompt_setup):
        """Should create mask indicating valid prompt positions."""
        prompts_embedding, prompts_mask = extract_prompt_features(**prompt_setup)
        
        # First batch has 3 class tokens
        assert prompts_mask[0, 0] == 1
        assert prompts_mask[0, 1] == 1
        assert prompts_mask[0, 2] == 1
        
        # Second batch has 2 class tokens
        assert prompts_mask[1, 0] == 1
        assert prompts_mask[1, 1] == 1
        assert prompts_mask[1, 2] == 0
    
    def test_pads_with_zeros(self, prompt_setup):
        """Should pad unused positions with zeros."""
        prompts_embedding, prompts_mask = extract_prompt_features(**prompt_setup)
        
        # Second batch only has 2 class tokens, third position should be zero
        assert torch.allclose(
            prompts_embedding[1, 2],
            torch.zeros(prompt_setup['embed_dim'])
        )
    
    def test_handles_no_class_tokens(self):
        """Should handle case with no class tokens."""
        batch_size = 1
        seq_length = 5
        embed_dim = 4
        class_token_index = 50
        
        token_embeds = torch.randn(batch_size, seq_length, embed_dim)
        input_ids = torch.randint(0, 49, (batch_size, seq_length))  # No class tokens
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        
        prompts_embedding, prompts_mask = extract_prompt_features(
            class_token_index, token_embeds, input_ids, attention_mask,
            batch_size, embed_dim
        )
        
        # Should have at least shape (1, 0, embed_dim) or handle gracefully
        assert prompts_embedding.shape[0] == batch_size
        assert prompts_embedding.shape[2] == embed_dim
    
    def test_preserves_dtype_and_device(self, prompt_setup):
        """Should preserve dtype and device of input tensors."""
        prompts_embedding, prompts_mask = extract_prompt_features(**prompt_setup)
        
        assert prompts_embedding.dtype == prompt_setup['token_embeds'].dtype
        assert prompts_embedding.device == prompt_setup['token_embeds'].device


class TestExtractPromptFeaturesAndWordEmbeddings:
    """Test suite for extract_prompt_features_and_word_embeddings function."""
    
    @pytest.fixture
    def combined_setup(self):
        """Fixture providing inputs for combined extraction."""
        batch_size = 2
        seq_length = 15
        embed_dim = 8
        class_token_index = 50
        
        token_embeds = torch.randn(batch_size, seq_length, embed_dim)
        
        # Setup class tokens
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        input_ids[0, 1] = class_token_index
        input_ids[0, 2] = class_token_index
        input_ids[1, 2] = class_token_index
        
        # Setup words mask
        words_mask = torch.zeros(batch_size, seq_length, dtype=torch.long)
        words_mask[0, 3] = 1
        words_mask[0, 7] = 2
        words_mask[1, 4] = 1
        
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        text_lengths = torch.tensor([2, 1], dtype=torch.long).unsqueeze(-1)
        
        return {
            'class_token_index': class_token_index,
            'token_embeds': token_embeds,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_lengths': text_lengths,
            'words_mask': words_mask,
            'embed_ent_token': True
        }
    
    def test_returns_four_outputs(self, combined_setup):
        """Should return four tensors: prompt embeddings, prompt mask, word embeddings, word mask."""
        result = extract_prompt_features_and_word_embeddings(**combined_setup)
        
        assert len(result) == 4
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = result
        
        assert isinstance(prompts_embedding, torch.Tensor)
        assert isinstance(prompts_embedding_mask, torch.Tensor)
        assert isinstance(words_embedding, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
    
    def test_output_shapes(self, combined_setup):
        """Should return tensors with correct shapes."""
        prompts_embedding, prompts_mask, words_embedding, word_mask = \
            extract_prompt_features_and_word_embeddings(**combined_setup)
        
        batch_size = combined_setup['token_embeds'].shape[0]
        embed_dim = combined_setup['token_embeds'].shape[2]
        max_text_length = combined_setup['text_lengths'].max().item()
        
        assert prompts_embedding.shape[0] == batch_size
        assert prompts_embedding.shape[2] == embed_dim
        
        assert words_embedding.shape == (batch_size, max_text_length, embed_dim)
        assert word_mask.shape == (batch_size, max_text_length)
    
    def test_prompt_embeddings_correct(self, combined_setup):
        """Should correctly extract prompt embeddings."""
        prompts_embedding, _, _, _ = extract_prompt_features_and_word_embeddings(**combined_setup)
        
        # Verify prompt embeddings match expected positions
        assert torch.allclose(
            prompts_embedding[0, 0],
            combined_setup['token_embeds'][0, 1]
        )
        assert torch.allclose(
            prompts_embedding[1, 0],
            combined_setup['token_embeds'][1, 2]
        )
    
    def test_word_embeddings_correct(self, combined_setup):
        """Should correctly extract word embeddings."""
        _, _, words_embedding, _ = extract_prompt_features_and_word_embeddings(**combined_setup)
        
        # Verify word embeddings match expected positions
        assert torch.allclose(
            words_embedding[0, 0],
            combined_setup['token_embeds'][0, 3]
        )
        assert torch.allclose(
            words_embedding[0, 1],
            combined_setup['token_embeds'][0, 7]
        )
    
    def test_both_masks_correct(self, combined_setup):
        """Should create correct masks for both prompts and words."""
        _, prompts_mask, _, word_mask = extract_prompt_features_and_word_embeddings(**combined_setup)
        
        # First batch has 2 class tokens
        assert prompts_mask[0, 0] == 1
        assert prompts_mask[0, 1] == 1
        
        # First batch has text_length=2
        assert word_mask[0, 0] == True
        assert word_mask[0, 1] == True
    
    def test_preserves_dtype_and_device(self, combined_setup):
        """Should preserve dtype and device throughout."""
        prompts_embedding, prompts_mask, words_embedding, word_mask = \
            extract_prompt_features_and_word_embeddings(**combined_setup)
        
        original_dtype = combined_setup['token_embeds'].dtype
        original_device = combined_setup['token_embeds'].device
        
        assert prompts_embedding.dtype == original_dtype
        assert words_embedding.dtype == original_dtype
        assert prompts_embedding.device == original_device
        assert words_embedding.device == original_device


class TestBuildEntityPairs:
    """Test suite for build_entity_pairs function."""
    
    @pytest.fixture
    def basic_pairs_setup(self):
        """Fixture providing basic inputs for entity pair building."""
        B, E, D = 2, 4, 8
        
        adj = torch.zeros(B, E, E)
        # First batch: create pairs (0,1), (0,2), (1,3)
        adj[0, 0, 1] = 0.8
        adj[0, 1, 0] = 0.8  # symmetric
        adj[0, 0, 2] = 0.7
        adj[0, 2, 0] = 0.7
        adj[0, 1, 3] = 0.6
        adj[0, 3, 1] = 0.6
        
        # Second batch: create pairs (0,3), (2,3)
        adj[1, 0, 3] = 0.9
        adj[1, 3, 0] = 0.9
        adj[1, 2, 3] = 0.75
        adj[1, 3, 2] = 0.75
        
        span_rep = torch.randn(B, E, D)
        
        return {
            'adj': adj,
            'span_rep': span_rep,
            'threshold': 0.5,
            'B': B,
            'E': E,
            'D': D
        }
    
    def test_returns_four_tensors(self, basic_pairs_setup):
        """Should return four tensors: pair_idx, pair_mask, head_rep, tail_rep."""
        result = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        assert len(result) == 4
        pair_idx, pair_mask, head_rep, tail_rep = result
        
        assert isinstance(pair_idx, torch.Tensor)
        assert isinstance(pair_mask, torch.Tensor)
        assert isinstance(head_rep, torch.Tensor)
        assert isinstance(tail_rep, torch.Tensor)
    
    def test_output_shapes(self, basic_pairs_setup):
        """Should return tensors with correct shapes."""
        pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        B = basic_pairs_setup['B']
        D = basic_pairs_setup['D']
        
        assert pair_idx.shape[0] == B
        assert pair_idx.shape[2] == 2  # (head, tail) indices
        assert pair_mask.shape[0] == B
        assert pair_mask.shape[1] == pair_idx.shape[1]
        assert head_rep.shape == (B, pair_idx.shape[1], D)
        assert tail_rep.shape == (B, pair_idx.shape[1], D)
    
    def test_extracts_correct_pairs_above_threshold(self, basic_pairs_setup):
        """Should extract all directed pairs (both directions) with scores above threshold."""
        pair_idx, pair_mask, _, _ = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        # Get valid pairs from first batch
        valid_pairs_0 = pair_idx[0][pair_mask[0]]
        
        # Should have 6 directed pairs: (0,1), (1,0), (0,2), (2,0), (1,3), (3,1)
        # Because the function considers ALL directed pairs where i≠j
        assert len(valid_pairs_0) == 6
    
    def test_pads_with_minus_one(self, basic_pairs_setup):
        """Should pad unused pair positions with -1."""
        pair_idx, pair_mask, _, _ = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        # Find padded positions
        for b in range(basic_pairs_setup['B']):
            padded_positions = ~pair_mask[b]
            if padded_positions.any():
                assert torch.all(pair_idx[b, padded_positions] == -1)
    
    def test_mask_indicates_valid_pairs(self, basic_pairs_setup):
        """Should have True in mask for valid pairs, False for padding."""
        pair_idx, pair_mask, _, _ = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        # All valid pairs should have mask=True
        for b in range(basic_pairs_setup['B']):
            valid_indices = pair_idx[b][pair_mask[b]]
            # All valid indices should be >= 0
            assert torch.all(valid_indices >= 0)
    
    def test_head_tail_representations_match_pairs(self, basic_pairs_setup):
        """Should extract correct head and tail representations."""
        pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        span_rep = basic_pairs_setup['span_rep']
        
        # Check first valid pair in first batch
        if pair_mask[0, 0]:
            head_idx = pair_idx[0, 0, 0].item()
            tail_idx = pair_idx[0, 0, 1].item()
            
            if head_idx >= 0 and tail_idx >= 0:
                assert torch.allclose(head_rep[0, 0], span_rep[0, head_idx])
                assert torch.allclose(tail_rep[0, 0], span_rep[0, tail_idx])
    
    def test_respects_threshold(self, basic_pairs_setup):
        """Should exclude pairs below threshold."""
        # Test with high threshold
        pair_idx_high, pair_mask_high, _, _ = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            threshold=0.85
        )
        
        # Test with low threshold
        pair_idx_low, pair_mask_low, _, _ = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            threshold=0.3
        )
        
        # High threshold should result in fewer or equal pairs
        num_pairs_high = pair_mask_high.sum()
        num_pairs_low = pair_mask_low.sum()
        
        assert num_pairs_high <= num_pairs_low
    
    def test_handles_no_pairs_above_threshold(self):
        """Should handle case when no pairs exceed threshold."""
        B, E, D = 2, 3, 4
        adj = torch.zeros(B, E, E) + 0.1  # All scores below threshold
        span_rep = torch.randn(B, E, D)
        
        pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
            adj, span_rep, threshold=0.5
        )
        
        # Should return tensors with at least one position for padding
        assert pair_idx.shape == (B, 1, 2)
        assert torch.all(pair_idx == -1)
        assert torch.all(pair_mask == False)
        assert head_rep.shape == (B, 1, D)
        assert tail_rep.shape == (B, 1, D)
    
    def test_ignores_diagonal(self):
        """Should ignore diagonal elements (self-loops)."""
        B, E, D = 1, 3, 4
        adj = torch.zeros(B, E, E)
        adj[0, 0, 0] = 1.0  # Self-loop
        adj[0, 1, 1] = 1.0  # Self-loop
        adj[0, 0, 1] = 0.8  # Valid pair
        adj[0, 1, 0] = 0.8  # Valid pair (reverse direction)
        
        span_rep = torch.randn(B, E, D)
        
        pair_idx, pair_mask, _, _ = build_entity_pairs(adj, span_rep, threshold=0.5)
        
        valid_pairs = pair_idx[0][pair_mask[0]]
        
        # Should have two directed pairs: (0,1) and (1,0), not self-loops
        assert len(valid_pairs) == 2
        # Check that no pair has same head and tail
        for pair in valid_pairs:
            assert pair[0] != pair[1]
    
    def test_includes_both_directions(self):
        """Should include both directions for bidirectional relations."""
        B, E, D = 1, 4, 4
        adj = torch.ones(B, E, E) * 0.8  # All high scores
        span_rep = torch.randn(B, E, D)
        
        pair_idx, pair_mask, _, _ = build_entity_pairs(adj, span_rep, threshold=0.5)
        
        valid_pairs = pair_idx[0][pair_mask[0]]
        
        # Should have E*(E-1) = 12 directed pairs (all pairs except diagonal)
        assert len(valid_pairs) == 12
        
        # Check that both directions are present
        pair_set = set()
        for pair in valid_pairs:
            pair_set.add((pair[0].item(), pair[1].item()))
        
        # Verify both (i,j) and (j,i) exist for at least one pair
        assert (0, 1) in pair_set
        assert (1, 0) in pair_set
    
    def test_preserves_dtype_and_device(self, basic_pairs_setup):
        """Should preserve device of input tensors."""
        pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
            basic_pairs_setup['adj'],
            basic_pairs_setup['span_rep'],
            basic_pairs_setup['threshold']
        )
        
        device = basic_pairs_setup['adj'].device
        dtype = basic_pairs_setup['span_rep'].dtype
        
        assert pair_idx.device == device
        assert pair_mask.device == device
        assert head_rep.device == device
        assert tail_rep.device == device
        assert head_rep.dtype == dtype
        assert tail_rep.dtype == dtype
    
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.9, 1.0])
    def test_different_thresholds(self, threshold):
        """Should work correctly with various threshold values."""
        B, E, D = 2, 3, 4
        adj = torch.rand(B, E, E)
        span_rep = torch.randn(B, E, D)
        
        pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
            adj, span_rep, threshold=threshold
        )
        
        # Should return valid tensors regardless of threshold
        assert pair_idx.shape[0] == B
        assert pair_mask.shape[0] == B
        assert head_rep.shape[0] == B
        assert tail_rep.shape[0] == B

class TestBaseModel:
    """Test suite for BaseModel functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing actual configuration."""
        return BaseGLiNERConfig(
            model_name='bert-base-uncased',
            hidden_size=128,
            dropout=0.1,
            max_width=12,
            span_mode='markerV0',
            class_token_index=103,
            has_rnn=False,
            post_fusion_schema='',
            embed_ent_token=True
        )
    
    def test_fit_length_no_change(self):
        """Should return unchanged tensors when length matches target."""
        B, L, D = 2, 10, 8
        embedding = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)
        target_len = 10
        
        result_emb, result_mask = BaseModel._fit_length(embedding, mask, target_len)
        
        assert result_emb.shape == (B, L, D)
        assert result_mask.shape == (B, L)
        assert torch.allclose(result_emb, embedding)
        assert torch.equal(result_mask, mask)
    
    def test_fit_length_padding(self):
        """Should pad tensors when length is less than target."""
        B, L, D = 2, 5, 8
        embedding = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)
        target_len = 10
        
        result_emb, result_mask = BaseModel._fit_length(embedding, mask, target_len)
        
        assert result_emb.shape == (B, target_len, D)
        assert result_mask.shape == (B, target_len)
        
        # Original content should be preserved
        assert torch.allclose(result_emb[:, :L], embedding)
        assert torch.equal(result_mask[:, :L], mask)
        
        # Padded positions should be zero
        assert torch.allclose(result_emb[:, L:], torch.zeros(B, target_len - L, D))
        assert torch.equal(result_mask[:, L:], torch.zeros(B, target_len - L, dtype=mask.dtype))
    
    def test_fit_length_truncation(self):
        """Should truncate tensors when length exceeds target."""
        B, L, D = 2, 15, 8
        embedding = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)
        target_len = 10
        
        result_emb, result_mask = BaseModel._fit_length(embedding, mask, target_len)
        
        assert result_emb.shape == (B, target_len, D)
        assert result_mask.shape == (B, target_len)
        
        # Should preserve first target_len elements
        assert torch.allclose(result_emb, embedding[:, :target_len])
        assert torch.equal(result_mask, mask[:, :target_len])
    
    def test_loss_basic_computation(self, mock_config):
        """Should compute focal loss with basic parameters."""
        class ConcreteModel(BaseModel):
            def get_representations(self): pass
            def forward(self, x): pass
            def loss(self, x): pass
        
        model = ConcreteModel(mock_config)
        
        B, N, C = 2, 10, 5
        logits = torch.randn(B, N, C)
        labels = torch.zeros(B, N, C)
        labels[0, 0, 0] = 1.0  # One positive example
        
        losses = model._loss(logits, labels, alpha=-1., gamma=0.0)
        
        assert losses.shape == (B, N, C)
        assert torch.all(losses >= 0)  # Losses should be non-negative
    
    def test_loss_masking_none(self, mock_config):
        """Should not mask any losses when masking='none'."""
        class ConcreteModel(BaseModel):
            def get_representations(self): pass
            def forward(self, x): pass
            def loss(self, x): pass
        
        model = ConcreteModel(mock_config)
        
        B, N, C = 2, 10, 5
        logits = torch.randn(B, N, C)
        labels = torch.zeros(B, N, C)
        
        losses = model._loss(logits, labels, masking="none")
        
        # All losses should be computed (no zeros from masking)
        assert losses.shape == (B, N, C)


class TestUniEncoderSpanModel:
    """Test suite for UniEncoderSpanModel."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing actual configuration for UniEncoderSpanModel."""
        return UniEncoderSpanConfig(
            model_name='bert-base-uncased',
            hidden_size=64,
            dropout=0.1,
            max_width=12,
            span_mode='markerV0',
            class_token_index=103,
            has_rnn=False,
            post_fusion_schema='',
            embed_ent_token=True
        )
    
    @pytest.fixture
    def model_inputs(self):
        """Fixture providing basic model inputs."""
        B, L, W, C = 2, 100, 12, 5
        max_width = 12
        
        return {
            'input_ids': torch.randint(0, 1000, (B, L)),
            'attention_mask': torch.ones(B, L, dtype=torch.long),
            'words_mask': torch.randint(0, W, (B, L)),
            'text_lengths': torch.tensor([W, W-2]).unsqueeze(-1),
            'span_idx': torch.randint(0, W, (B, L*max_width, 2)),
            'span_mask': torch.randint(0, 2, (B, L*max_width)),
            'labels': torch.zeros(B, L*max_width, C),
        }
    
    def test_forward_output_shape_without_labels(self, mock_config, model_inputs):
        """Should return output with correct logits shape without labels."""
        
        # Remove labels to test inference mode
        model_inputs_no_labels = {k: v for k, v in model_inputs.items() if k != 'labels'}
        
        # Mock the model components
        model = UniEncoderSpanModel(mock_config, from_pretrained=False)
        
        with torch.no_grad():
            output = model(**model_inputs_no_labels)
        
        B = model_inputs['input_ids'].shape[0]
        L = model_inputs['span_idx'].shape[1] // mock_config.max_width
        
        # Check output structure
        assert hasattr(output, 'logits')
        assert hasattr(output, 'loss')
        assert output.loss is None  # No loss without labels
        assert output.logits.shape[0] == B  # Batch dimension
        assert output.logits.shape[1] == L  # Sequence dimension
    
    def test_loss_computation(self, mock_config):
        """Should compute loss correctly with labels."""
        model = UniEncoderSpanModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.zeros(B, L, K, C)
        labels[0, 0, 0, 0] = 1.0  # One positive example
        
        prompts_embedding_mask = torch.ones(B, C)
        span_mask = torch.ones(B, L, K)
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            alpha=-1., gamma=0.0, reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_loss_with_masking(self, mock_config):
        """Should properly apply span masking in loss computation."""
        model = UniEncoderSpanModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.ones(B, L, K, C)
        
        prompts_embedding_mask = torch.ones(B, C)
        span_mask = torch.zeros(B, L, K)
        span_mask[0, 0, 0] = 1  # Only one valid span
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            reduction='sum'
        )
        
        # Loss should be computed (even if small due to masking)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestUniEncoderTokenModel:
    """Test suite for UniEncoderTokenModel."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing actual configuration for UniEncoderTokenModel."""
        return UniEncoderTokenConfig(
            model_name='bert-base-uncased',
            hidden_size=64,
            dropout=0.1,
            class_token_index=103,
            has_rnn=False,
            post_fusion_schema='',
            embed_ent_token=True
        )
    
    def test_loss_computation(self, mock_config):
        """Should compute token-level loss correctly."""
        model = UniEncoderTokenModel(mock_config, from_pretrained=False)
        
        B, W, C = 2, 20, 5
        scores = torch.randn(B, W, C, 1)
        labels = torch.zeros(B, W, C, 1)
        labels[0, 0, 0, 0] = 1.0
        
        prompts_embedding_mask = torch.ones(B, C)
        mask = torch.ones(B, W)
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, mask,
            alpha=-1., gamma=0.0, reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_loss_with_word_masking(self, mock_config):
        """Should properly apply word-level masking in loss computation."""
        model = UniEncoderTokenModel(mock_config, from_pretrained=False)
        
        B, W, C = 2, 20, 5
        scores = torch.randn(B, W, C, 1)
        labels = torch.ones(B, W, C, 1)
        
        prompts_embedding_mask = torch.ones(B, C)
        mask = torch.zeros(B, W)
        mask[0, :5] = 1  # Only first 5 tokens valid in first batch
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, mask,
            reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestUniEncoderSpanDecoderModel:
    """Test suite for UniEncoderSpanDecoderModel."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing actual configuration for decoder model."""
        return UniEncoderSpanDecoderConfig(
            model_name='bert-base-uncased',
            hidden_size=64,
            dropout=0.1,
            max_width=12,
            span_mode='markerV0',
            class_token_index=103,
            has_rnn=False,
            post_fusion_schema='',
            embed_ent_token=True,
            labels_decoder='gpt2',
            decoder_mode='span',
            full_decoder_context=True,
            blank_entity_prob=0.1
        )
    
    def test_select_decoder_embedding_shape(self, mock_config):
        """Should select and reshape decoder embeddings correctly."""
        model = UniEncoderSpanDecoderModel(mock_config, from_pretrained=False)
        
        B, N, D = 2, 10, 64
        representations = torch.randn(B, N, D)
        rep_mask = torch.zeros(B, N, dtype=torch.long)
        rep_mask[0, :3] = 1  # First batch has 3 valid representations
        rep_mask[1, :5] = 1  # Second batch has 5 valid representations
        
        target_rep, target_mask, sel_idx = model.select_decoder_embedding(
            representations, rep_mask
        )
        
        max_len = rep_mask.sum(dim=-1).max().item()
        
        assert target_rep.shape == (B, max_len, D)
        assert target_mask.shape == (B, max_len)
        assert sel_idx.shape == (B, max_len)
        
        # Check that valid positions are marked correctly
        assert target_mask[0, :3].sum() == 3
        assert target_mask[1, :5].sum() == 5
    
    def test_get_raw_decoder_inputs(self, mock_config):
        """Should extract valid span tokens for decoder input."""
        model = UniEncoderSpanDecoderModel(mock_config, from_pretrained=False)
        
        B, S, T, D = 2, 10, 5, 64
        representations = torch.randn(B, S, T, D)
        rep_mask = torch.zeros(B, S, T, dtype=torch.long)
        rep_mask[0, :3, :] = 1  # First batch has 3 valid spans
        rep_mask[1, :2, :] = 1  # Second batch has 2 valid spans
        
        span_tokens, span_tokens_mask = model.get_raw_decoder_inputs(
            representations, rep_mask
        )
        
        # Should flatten and keep only valid spans
        total_valid = (rep_mask.any(-1)).sum().item()
        
        assert span_tokens.shape[0] == total_valid
        assert span_tokens.shape[1] == T
        assert span_tokens.shape[2] == D
        assert span_tokens_mask.shape[0] == total_valid
        assert span_tokens_mask.shape[1] == T
    
    def test_loss_with_decoder_loss(self, mock_config):
        """Should combine span loss and decoder loss correctly."""
        model = UniEncoderSpanDecoderModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.zeros(B, L, K, C)
        
        prompts_embedding_mask = torch.ones(B, C)
        span_mask = torch.ones(B, L, K)
        decoder_loss = torch.tensor(5.0)
        
        total_loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            decoder_loss=decoder_loss, reduction='sum'
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0
        # Total loss should include both components
        assert total_loss.item() > decoder_loss.item()


class TestUniEncoderSpanRelexModel:
    """Test suite for UniEncoderSpanRelexModel (relation extraction)."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing actual configuration for relation extraction model."""
        return UniEncoderSpanRelexConfig(
            model_name='bert-base-uncased',
            hidden_size=64,
            dropout=0.1,
            max_width=12,
            span_mode='markerV0',
            class_token_index=103,
            has_rnn=False,
            post_fusion_schema='',
            embed_ent_token=True,
            relations_layer='mlp',
            triples_layer=None,
            embed_rel_token=True,
            rel_token_index=104,
            rel_token='<<REL>>'
        )
    
    def test_select_target_embedding(self, mock_config):
        """Should keep only representations where mask == 1."""
        model = UniEncoderSpanRelexModel(mock_config, from_pretrained=False)
        
        B, N, D = 2, 10, 64
        representations = torch.randn(B, N, D)
        rep_mask = torch.zeros(B, N, dtype=torch.long)
        rep_mask[0, [1, 3, 5]] = 1  # 3 valid positions in first batch
        rep_mask[1, [0, 2]] = 1     # 2 valid positions in second batch
        
        target_rep, target_mask = model.select_target_embedding(
            representations, rep_mask
        )
        
        max_len = rep_mask.sum(dim=-1).max().item()
        
        assert target_rep.shape == (B, max_len, D)
        assert target_mask.shape == (B, max_len)
        assert target_mask[0].sum() == 3
        assert target_mask[1].sum() == 2
    
    def test_adj_loss_computation(self, mock_config):
        """Should compute adjacency matrix loss correctly."""
        model = UniEncoderSpanRelexModel(mock_config, from_pretrained=False)
        
        B, E = 2, 10
        logits = torch.randn(B, E, E)
        labels = torch.zeros(B, E, E)
        labels[0, 0, 1] = 1.0  # One edge
        
        adj_mask = torch.ones(B, E, E)
        
        loss = model.adj_loss(
            logits, labels, adj_mask,
            alpha=-1., gamma=0.0, reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_rel_loss_computation(self, mock_config):
        """Should compute relation classification loss correctly."""
        model = UniEncoderSpanRelexModel(mock_config, from_pretrained=False)
        
        B, P, C = 2, 10, 5
        logits = torch.randn(B, P, C)
        labels = torch.zeros(B, P, C)
        labels[0, 0, 0] = 1.0
        
        rel_mask = torch.ones(B, P, 1)
        rel_prompts_embedding_mask = torch.ones(B, C)
        
        loss = model.rel_loss(
            logits, labels, rel_mask, rel_prompts_embedding_mask,
            alpha=-1., gamma=0.0, reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0


class TestBiEncoderSpanModel:
    """Test suite for BiEncoderSpanModel."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing actual configuration for BiEncoderSpanModel."""
        return BiEncoderSpanConfig(
            model_name='bert-base-uncased',
            labels_encoder='bert-base-uncased',
            hidden_size=64,
            dropout=0.1,
            max_width=12,
            span_mode='markerV0',
            class_token_index=103,
            has_rnn=False,
            post_fusion_schema='',
            embed_ent_token=True
        )
    
    @pytest.fixture
    def model_inputs(self):
        """Fixture providing basic model inputs for bi-encoder."""
        B, L, W, C = 2, 100, 12, 5
        max_width = 12
        
        return {
            'input_ids': torch.randint(0, 1000, (B, L)),
            'attention_mask': torch.ones(B, L, dtype=torch.long),
            'labels_input_ids': torch.randint(0, 1000, (C, 10)),
            'labels_attention_mask': torch.ones(C, 10, dtype=torch.long),
            'words_mask': torch.randint(0, W, (B, L)),
            'text_lengths': torch.tensor([W, W-2]).unsqueeze(-1),
            'span_idx': torch.randint(0, W, (B, L*max_width, 2)),
            'span_mask': torch.randint(0, 2, (B, L*max_width)),
            'labels': torch.zeros(B, L*max_width, C),
        }
    
    def test_forward_output_shape_without_labels(self, mock_config, model_inputs):
        """Should return output with correct logits shape without labels."""
        from unittest.mock import Mock
        
        # Remove labels to test inference mode
        model_inputs_no_labels = {k: v for k, v in model_inputs.items() if k != 'labels'}
        
        # Mock the model components
        model = BiEncoderSpanModel(mock_config, from_pretrained=False)
        
        with torch.no_grad():
            output = model(**model_inputs_no_labels)
        
        B = model_inputs['input_ids'].shape[0]
        L = model_inputs['span_idx'].shape[1] // mock_config.max_width
        
        # Check output structure
        assert hasattr(output, 'logits')
        assert hasattr(output, 'loss')
        assert output.loss is None  # No loss without labels
        assert output.logits.shape[0] == B  # Batch dimension
        assert output.logits.shape[1] == L  # Sequence dimension
    
    def test_forward_with_precomputed_labels_embeds(self, mock_config, model_inputs):
        """Should accept precomputed labels embeddings instead of ids."""
        from unittest.mock import Mock
        
        # Replace labels_input_ids with labels_embeds
        C, D = 5, 64
        model_inputs_with_embeds = {k: v for k, v in model_inputs.items() 
                                    if k not in ['labels_input_ids', 'labels_attention_mask', 'labels']}
        model_inputs_with_embeds['labels_embeds'] = torch.randn(C, D)
        
        model = BiEncoderSpanModel(mock_config, from_pretrained=False)
        
        # Mock encode_text method
        model.token_rep_layer.encode_text = Mock()
        model.token_rep_layer.encode_text.return_value = torch.randn(
            model_inputs['input_ids'].shape[0],
            model_inputs['input_ids'].shape[1],
            mock_config.hidden_size
        )
        
        with torch.no_grad():
            output = model(**model_inputs_with_embeds)
        
        # Should successfully process with precomputed embeddings
        assert hasattr(output, 'logits')
        assert output.loss is None
    
    def test_loss_computation(self, mock_config):
        """Should compute loss correctly with labels."""
        model = BiEncoderSpanModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.zeros(B, L, K, C)
        labels[0, 0, 0, 0] = 1.0  # One positive example
        
        prompts_embedding_mask = torch.ones(B, C)
        span_mask = torch.ones(B, L, K)
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            alpha=-1., gamma=0.0, reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_loss_with_masking(self, mock_config):
        """Should properly apply span masking in loss computation."""
        model = BiEncoderSpanModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.ones(B, L, K, C)
        
        prompts_embedding_mask = torch.ones(B, C)
        span_mask = torch.zeros(B, L, K)
        span_mask[0, 0, 0] = 1  # Only one valid span
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            reduction='sum'
        )
        
        # Loss should be computed (even if small due to masking)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_loss_with_class_masking(self, mock_config):
        """Should properly apply class masking in loss computation."""
        model = BiEncoderSpanModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.ones(B, L, K, C)
        
        prompts_embedding_mask = torch.zeros(B, C)
        prompts_embedding_mask[0, :2] = 1  # Only 2 classes valid in first batch
        prompts_embedding_mask[1, :3] = 1  # Only 3 classes valid in second batch
        
        span_mask = torch.ones(B, L, K)
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            reduction='sum'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_loss_reduction_mean(self, mock_config):
        """Should compute mean reduction correctly."""
        model = BiEncoderSpanModel(mock_config, from_pretrained=False)
        
        B, L, K, C = 2, 10, 12, 5
        scores = torch.randn(B, L, K, C)
        labels = torch.ones(B, L, K, C)
        
        prompts_embedding_mask = torch.ones(B, C)
        span_mask = torch.ones(B, L, K)
        
        loss = model.loss(
            scores, labels, prompts_embedding_mask, span_mask,
            reduction='mean'
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0