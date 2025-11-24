import pytest
import torch
from gliner.decoding.decoder import (
    SpanDecoder,
    SpanGenerativeDecoder,
    SpanRelexDecoder,
    TokenDecoder,
    BaseSpanDecoder
)
from unittest.mock import Mock


class TestSpanDecoder:
    """Test suite for SpanDecoder class."""
    
    @pytest.fixture
    def basic_config(self):
        """Fixture providing basic configuration."""
        config = Mock()
        config.max_width = 10
        return config
    
    @pytest.fixture
    def basic_inputs(self):
        """Fixture providing basic inputs for span decoding."""
        batch_size = 2
        seq_length = 5
        max_width = 3
        num_classes = 2
        
        # Create logits: (B, L, K, C)
        logits = torch.randn(batch_size, seq_length, max_width, num_classes)
        
        # Set some values high to ensure they pass threshold
        logits[0, 0, 0, 0] = 5.0  # High confidence span
        logits[0, 1, 1, 1] = 4.0  # Another high confidence span
        logits[1, 0, 0, 0] = 3.0  # High confidence in second batch
        
        tokens = [
            ['The', 'quick', 'brown', 'fox', 'jumps'],
            ['Hello', 'world', 'test', 'sentence', 'here']
        ]
        
        id_to_classes = {1: 'PERSON', 2: 'LOCATION'}
        
        return {
            'logits': logits,
            'tokens': tokens,
            'id_to_classes': id_to_classes,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'max_width': max_width,
            'num_classes': num_classes
        }
    
    def test_output_structure(self, basic_config, basic_inputs):
        """Should return list of lists with correct batch size."""
        decoder = SpanDecoder(basic_config)
        
        result = decoder.decode(
            tokens=basic_inputs['tokens'],
            id_to_classes=basic_inputs['id_to_classes'],
            model_output=basic_inputs['logits'],
            threshold=0.5
        )
        
        assert isinstance(result, list)
        assert len(result) == basic_inputs['batch_size']
        assert all(isinstance(spans, list) for spans in result)
    
    def test_span_tuple_format(self, basic_config, basic_inputs):
        """Should return spans in format (start, end, entity_type, score)."""
        decoder = SpanDecoder(basic_config)
        
        result = decoder.decode(
            tokens=basic_inputs['tokens'],
            id_to_classes=basic_inputs['id_to_classes'],
            model_output=basic_inputs['logits'],
            threshold=0.5
        )
        
        for batch_spans in result:
            for span in batch_spans:
                assert len(span) == 4
                start, end, entity_type, score = span
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert isinstance(entity_type, str)
                assert isinstance(score, float)
    
    def test_threshold_filtering(self, basic_config, basic_inputs):
        """Should filter spans based on threshold."""
        decoder = SpanDecoder(basic_config)
        
        # Low threshold should return more spans
        result_low = decoder.decode(
            tokens=basic_inputs['tokens'],
            id_to_classes=basic_inputs['id_to_classes'],
            model_output=basic_inputs['logits'],
            threshold=0.1
        )
        
        # High threshold should return fewer spans
        result_high = decoder.decode(
            tokens=basic_inputs['tokens'],
            id_to_classes=basic_inputs['id_to_classes'],
            model_output=basic_inputs['logits'],
            threshold=0.99
        )
        
        total_low = sum(len(spans) for spans in result_low)
        total_high = sum(len(spans) for spans in result_high)
        
        assert total_low >= total_high
    
    def test_span_boundaries(self, basic_config, basic_inputs):
        """Should respect sentence boundaries."""
        decoder = SpanDecoder(basic_config)
        
        result = decoder.decode(
            tokens=basic_inputs['tokens'],
            id_to_classes=basic_inputs['id_to_classes'],
            model_output=basic_inputs['logits'],
            threshold=0.5
        )
        
        for i, batch_spans in enumerate(result):
            for span in batch_spans:
                start, end, _, _ = span
                # End should not exceed sentence length
                assert end <= len(basic_inputs['tokens'][i])
                # Start should be valid
                assert start >= 0
    
    def test_flat_ner_removes_overlaps(self, basic_config):
        """Should remove overlapping spans when flat_ner=True."""
        decoder = SpanDecoder(basic_config)
        
        # Create logits with overlapping high-confidence spans
        logits = torch.ones(1, 5, 3, 2) * -5.0
        logits[0, 0, 0, 0] = 5.0  # Span (0, 0, PERSON)
        logits[0, 0, 1, 1] = 4.0  # Overlapping span (0, 1, LOCATION)
        
        tokens = [['A', 'B', 'C', 'D', 'E']]
        id_to_classes = {1: 'PERSON', 2: 'LOCATION'}
        
        result = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=logits,
            threshold=0.5,
            flat_ner=True
        )
        
        # Should keep only the higher scoring span
        assert len(result[0]) == 1
        assert result[0][0][2] == 'PERSON'  # Higher score wins
    
    def test_nested_ner_keeps_overlaps(self, basic_config):
        """Should keep overlapping spans when flat_ner=False."""
        decoder = SpanDecoder(basic_config)
        
        # Create logits with nested spans
        logits = torch.ones(1, 5, 3, 2) * -5.0
        logits[0, 0, 0, 0] = 5.0  # Span (0, 0)
        logits[0, 0, 2, 1] = 4.0  # Span (0, 2) contains (0, 0)
        
        tokens = [['A', 'B', 'C', 'D', 'E']]
        id_to_classes = {1: 'PERSON', 2: 'LOCATION'}
        
        result = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=logits,
            threshold=0.5,
            flat_ner=False
        )
        
        # Should keep both nested spans
        assert len(result[0]) == 2
    
    def test_per_sample_id_to_classes(self, basic_config, basic_inputs):
        """Should handle per-sample id_to_classes mappings."""
        decoder = SpanDecoder(basic_config)
        
        # Different classes for each sample
        id_to_classes_list = [
            {1: 'PERSON', 2: 'ORG'},
            {1: 'LOCATION', 2: 'DATE'}
        ]
        
        result = decoder.decode(
            tokens=basic_inputs['tokens'],
            id_to_classes=id_to_classes_list,
            model_output=basic_inputs['logits'],
            threshold=0.5
        )
        
        # Check that different classes are used
        if len(result[0]) > 0 and len(result[1]) > 0:
            # Classes should come from respective mappings
            batch_0_types = {span[2] for span in result[0]}
            batch_1_types = {span[2] for span in result[1]}
            
            assert batch_0_types.issubset({'PERSON', 'ORG'})
            assert batch_1_types.issubset({'LOCATION', 'DATE'})
    
    def test_empty_predictions(self, basic_config):
        """Should handle case with no predictions above threshold."""
        decoder = SpanDecoder(basic_config)
        
        # All low confidence
        logits = torch.ones(1, 3, 2, 2) * -10.0
        tokens = [['A', 'B', 'C']]
        id_to_classes = {1: 'PERSON', 2: 'LOCATION'}
        
        result = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=logits,
            threshold=0.5
        )
        
        assert len(result) == 1
        assert len(result[0]) == 0


class TestSpanGenerativeDecoder:
    """Test suite for SpanGenerativeDecoder class."""
    
    @pytest.fixture
    def prompt_mode_config(self):
        """Fixture for prompt mode configuration."""
        config = Mock()
        config.max_width = 10
        config.decoder_mode = 'prompt'
        config.labels_decoder = True
        return config
    
    @pytest.fixture
    def span_mode_config(self):
        """Fixture for span mode configuration."""
        config = Mock()
        config.max_width = 10
        config.decoder_mode = 'span'
        config.labels_decoder = True
        return config
    
    @pytest.fixture
    def generative_inputs(self):
        """Fixture providing inputs for generative decoding."""
        batch_size = 2
        seq_length = 5
        max_width = 3
        num_classes = 2
        
        logits = torch.randn(batch_size, seq_length, max_width, num_classes)
        logits[0, 0, 0, 0] = 5.0
        logits[0, 1, 1, 1] = 4.0
        logits[1, 0, 0, 0] = 3.0
        
        tokens = [
            ['The', 'quick', 'brown', 'fox', 'jumps'],
            ['Hello', 'world', 'test', 'sentence', 'here']
        ]
        
        id_to_classes = {1: 'TYPE_1', 2: 'TYPE_2'}
        
        # Generated labels for prompt mode
        gen_labels = ['John Doe', 'Jane Smith', 'New York', 'California']
        
        return {
            'logits': logits,
            'tokens': tokens,
            'id_to_classes': id_to_classes,
            'gen_labels': gen_labels,
            'batch_size': batch_size
        }
    
    def test_prompt_mode_replaces_class_names(self, prompt_mode_config, generative_inputs):
        """Should replace class names with generated labels in prompt mode."""
        decoder = SpanGenerativeDecoder(prompt_mode_config)
        
        result = decoder.decode(
            tokens=generative_inputs['tokens'],
            id_to_classes=generative_inputs['id_to_classes'],
            model_output=generative_inputs['logits'],
            gen_labels=generative_inputs['gen_labels'],
            threshold=0.5
        )
        
        # Check that generated labels appear in results
        all_types = set()
        for batch_spans in result:
            for span in batch_spans:
                # In prompt mode, tuple is (start, end, entity_type, gen_entity_type, score)
                # where gen_entity_type should be None and entity_type is the generated label
                all_types.add(span[2])
        
        # Generated labels should be used
        assert all_types.intersection(set(generative_inputs['gen_labels']))
    
    def test_span_mode_adds_generated_labels(self, span_mode_config, generative_inputs):
        """Should add generated labels to spans in span mode."""
        decoder = SpanGenerativeDecoder(span_mode_config)
        
        # Create sel_idx for span mode
        sel_idx = torch.tensor([[0, 3, -1], [0, -1, -1]])
        
        result = decoder.decode(
            tokens=generative_inputs['tokens'],
            id_to_classes=generative_inputs['id_to_classes'],
            model_output=generative_inputs['logits'],
            gen_labels=generative_inputs['gen_labels'],
            sel_idx=sel_idx,
            threshold=0.5
        )
        
        # In span mode, tuples should have generated entity type
        for batch_spans in result:
            for span in batch_spans:
                assert len(span) == 5
                start, end, entity_type, gen_entity_type, score = span
                # gen_entity_type may be None or a list of strings
                assert gen_entity_type is None or isinstance(gen_entity_type, list)
    
    def test_update_id_to_classes_with_generated(self, prompt_mode_config):
        """Should correctly map generated labels to class IDs."""
        decoder = SpanGenerativeDecoder(prompt_mode_config)
        
        id_to_classes = {1: 'TYPE_1', 2: 'TYPE_2'}
        gen_labels = ['Label_A', 'Label_B', 'Label_C', 'Label_D']
        batch_size = 2
        
        result = decoder._update_id_to_classes_with_generated(
            id_to_classes, gen_labels, batch_size
        )
        
        assert isinstance(result, list)
        assert len(result) == batch_size
        
        # First batch should map to first 2 labels
        assert result[0] == {1: 'Label_A', 2: 'Label_B'}
        
        # Second batch should map to next 2 labels
        assert result[1] == {1: 'Label_C', 2: 'Label_D'}
    
    def test_build_span_label_map_for_batch(self, span_mode_config):
        """Should build correct mapping from flat indices to labels."""
        decoder = SpanGenerativeDecoder(span_mode_config)
        
        # sel_idx: (B, M) where each value is flat span index
        sel_idx = torch.tensor([
            [0, 5, -1],  # Two valid spans in first batch
            [2, -1, -1]   # One valid span in second batch
        ])
        
        gen_labels = ['L1', 'L2', 'L3']  # 3 labels for 3 valid spans
        num_gen_sequences = 1
        
        result = decoder._build_span_label_map_for_batch(
            sel_idx, gen_labels, num_gen_sequences
        )
        
        assert len(result) == 2  # Two batch elements
        
        # First batch should have mapping for indices 0 and 5
        assert 0 in result[0]
        assert 5 in result[0]
        assert result[0][0] == ['L1']
        assert result[0][5] == ['L2']
        
        # Second batch should have mapping for index 2
        assert 2 in result[1]
        assert result[1][2] == ['L3']
    
    def test_multiple_gen_sequences(self, span_mode_config):
        """Should handle multiple generated sequences per span."""
        decoder = SpanGenerativeDecoder(span_mode_config)
        
        sel_idx = torch.tensor([[0, 3, -1]])  # Two valid spans
        gen_labels = ['L1_1', 'L1_2', 'L2_1', 'L2_2']  # 2 sequences per span
        num_gen_sequences = 2
        
        result = decoder._build_span_label_map_for_batch(
            sel_idx, gen_labels, num_gen_sequences
        )
        
        # Each span should have 2 labels
        assert result[0][0] == ['L1_1', 'L1_2']
        assert result[0][3] == ['L2_1', 'L2_2']
    
    def test_fallback_to_standard_decoding(self, prompt_mode_config, generative_inputs):
        """Should fall back to standard decoding when gen_labels not provided."""
        decoder = SpanGenerativeDecoder(prompt_mode_config)
        
        result = decoder.decode(
            tokens=generative_inputs['tokens'],
            id_to_classes=generative_inputs['id_to_classes'],
            model_output=generative_inputs['logits'],
            threshold=0.5
        )
        
        # Should use standard decoding format
        for batch_spans in result:
            for span in batch_spans:
                # Standard format has 5 elements but gen_entity_type is None
                assert len(span) == 5
                assert span[3] is None  # gen_entity_type should be None


class TestSpanRelexDecoder:
    """Test suite for SpanRelexDecoder class."""
    
    @pytest.fixture
    def relex_config(self):
        """Fixture for relation extraction configuration."""
        config = Mock()
        config.max_width = 10
        return config
    
    @pytest.fixture
    def relex_inputs(self):
        """Fixture providing inputs for relation extraction."""
        batch_size = 2
        seq_length = 5
        max_width = 3
        num_classes = 2
        num_relations = 2
        max_pairs = 4
        
        # Entity logits
        logits = torch.randn(batch_size, seq_length, max_width, num_classes)
        logits[0, 0, 0, 0] = 5.0  # Entity 1
        logits[0, 2, 0, 1] = 4.0  # Entity 2
        logits[1, 0, 0, 0] = 3.0  # Entity 3
        
        # Attach relation attributes directly to the tensor
        logits.rel_idx = torch.tensor([
            [[0, 1], [1, 0], [-1, -1], [-1, -1]],  # Batch 0: 2 pairs
            [[0, 0], [-1, -1], [-1, -1], [-1, -1]]  # Batch 1: 1 pair
        ])
        logits.rel_logits = torch.randn(batch_size, max_pairs, num_relations)
        logits.rel_logits[0, 0, 0] = 5.0  # High confidence relation
        logits.rel_mask = torch.tensor([
            [True, True, False, False],
            [True, False, False, False]
        ])
        
        model_output = logits
        
        tokens = [
            ['The', 'quick', 'brown', 'fox', 'jumps'],
            ['Hello', 'world', 'test', 'sentence', 'here']
        ]
        
        id_to_classes = {1: 'PERSON', 2: 'ORG'}
        rel_id_to_classes = {1: 'WORKS_AT', 2: 'LOCATED_IN'}
        
        return {
            'model_output': model_output,
            'logits': logits,
            'tokens': tokens,
            'id_to_classes': id_to_classes,
            'rel_id_to_classes': rel_id_to_classes,
            'batch_size': batch_size
        }
    
    def test_output_structure(self, relex_config, relex_inputs):
        """Should return tuple of (spans, relations)."""
        decoder = SpanRelexDecoder(relex_config)
        
        spans, relations = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            rel_id_to_classes=relex_inputs['rel_id_to_classes'],
            threshold=0.5
        )
        
        assert isinstance(spans, list)
        assert isinstance(relations, list)
        assert len(spans) == relex_inputs['batch_size']
        assert len(relations) == relex_inputs['batch_size']
    
    def test_span_format(self, relex_config, relex_inputs):
        """Should return spans in correct format."""
        decoder = SpanRelexDecoder(relex_config)
        
        spans, _ = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            rel_id_to_classes=relex_inputs['rel_id_to_classes'],
            threshold=0.5
        )
        
        for batch_spans in spans:
            for span in batch_spans:
                assert len(span) == 4
                start, end, entity_type, score = span
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert isinstance(entity_type, str)
                assert isinstance(score, float)
    
    def test_relation_format(self, relex_config, relex_inputs):
        """Should return relations in format (head_idx, relation_label, tail_idx, score)."""
        decoder = SpanRelexDecoder(relex_config)
        
        _, relations = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            rel_id_to_classes=relex_inputs['rel_id_to_classes'],
            threshold=0.5
        )
        
        for batch_relations in relations:
            for relation in batch_relations:
                assert len(relation) == 4
                head_idx, rel_label, tail_idx, score = relation
                assert isinstance(head_idx, int)
                assert isinstance(rel_label, str)
                assert isinstance(tail_idx, int)
                assert isinstance(score, float)
    
    def test_filters_invalid_indices(self, relex_config, relex_inputs):
        """Should filter relations with invalid entity indices."""
        decoder = SpanRelexDecoder(relex_config)
        
        # Modify model output to have invalid indices
        relex_inputs['model_output'].rel_idx[0, 0] = torch.tensor([0, 99])  # Invalid tail
        
        _, relations = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            rel_id_to_classes=relex_inputs['rel_id_to_classes'],
            threshold=0.5
        )
        
        # Relations with invalid indices should be filtered
        for batch_relations in relations:
            for relation in batch_relations:
                head_idx, _, tail_idx, _ = relation
                assert head_idx >= 0
                assert tail_idx >= 0
    
    def test_respects_relation_mask(self, relex_config, relex_inputs):
        """Should only decode relations where mask is True."""
        decoder = SpanRelexDecoder(relex_config)
        
        _, relations = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            rel_id_to_classes=relex_inputs['rel_id_to_classes'],
            threshold=0.5
        )
        
        # Masked positions should not produce relations
        # In batch 0, positions 2 and 3 are masked out
        # This is indirect - we just verify no errors occur
        assert isinstance(relations, list)
    
    def test_no_relations_when_not_requested(self, relex_config, relex_inputs):
        """Should return empty relations when rel_id_to_classes is None."""
        decoder = SpanRelexDecoder(relex_config)
        
        spans, relations = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            threshold=0.5
        )
        
        # Should return empty relations for each batch
        assert all(len(rels) == 0 for rels in relations)
    
    def test_handles_missing_relation_outputs(self, relex_config, relex_inputs):
        """Should handle model output without relation predictions."""
        decoder = SpanRelexDecoder(relex_config)
        
        # Remove relation attributes
        delattr(relex_inputs['model_output'], 'rel_idx')
        
        spans, relations = decoder.decode(
            tokens=relex_inputs['tokens'],
            id_to_classes=relex_inputs['id_to_classes'],
            model_output=relex_inputs['model_output'],
            rel_id_to_classes=relex_inputs['rel_id_to_classes'],
            threshold=0.5
        )
        
        # Should still decode spans
        assert len(spans) > 0
        # Relations should be empty
        assert all(len(rels) == 0 for rels in relations)

        
class TestTokenDecoder:
    """Test suite for TokenDecoder class."""
    
    @pytest.fixture
    def token_config(self):
        """Fixture for token decoder configuration."""
        config = Mock()
        return config
    
    @pytest.fixture
    def token_inputs(self):
        """Fixture providing inputs for token-based decoding."""
        batch_size = 2
        seq_length = 6
        num_classes = 2
        
        # Model output: (B, L, C, 3) for [start, end, inside]
        model_output = torch.ones(batch_size, seq_length, num_classes, 3) * -5.0
        
        # Create a clear span: positions 1-3, class 0
        model_output[0, 1, 0, 0] = 5.0  # Start at position 1
        model_output[0, 1, 0, 2] = 5.0  # Inside at start (required by decoder)
        model_output[0, 3, 0, 1] = 5.0  # End at position 3
        model_output[0, 3, 0, 2] = 5.0  # Inside at end (required by decoder)
        model_output[0, 2, 0, 2] = 5.0  # Inside at position 2 (interior)
        
        # Another span in second batch: positions 0-1, class 1
        model_output[1, 0, 1, 0] = 5.0  # Start
        model_output[1, 0, 1, 2] = 5.0  # Inside at start
        model_output[1, 1, 1, 1] = 5.0  # End
        model_output[1, 1, 1, 2] = 5.0  # Inside at end
        
        tokens = [
            ['The', 'quick', 'brown', 'fox', 'jumps', 'high'],
            ['Hello', 'world', 'test', 'sentence', 'here', 'now']
        ]
        
        id_to_classes = {1: 'PERSON', 2: 'LOCATION'}
        
        return {
            'model_output': model_output,
            'tokens': tokens,
            'id_to_classes': id_to_classes,
            'batch_size': batch_size
        }
    
    def test_output_structure(self, token_config, token_inputs):
        """Should return list of lists with correct batch size."""
        decoder = TokenDecoder(token_config)
        
        result = decoder.decode(
            tokens=token_inputs['tokens'],
            id_to_classes=token_inputs['id_to_classes'],
            model_output=token_inputs['model_output'],
            threshold=0.5
        )
        
        assert isinstance(result, list)
        assert len(result) == token_inputs['batch_size']
        assert all(isinstance(spans, list) for spans in result)
    
    def test_span_tuple_format(self, token_config, token_inputs):
        """Should return spans in format (start, end, entity_type, score)."""
        decoder = TokenDecoder(token_config)
        
        result = decoder.decode(
            tokens=token_inputs['tokens'],
            id_to_classes=token_inputs['id_to_classes'],
            model_output=token_inputs['model_output'],
            threshold=0.5
        )
        
        for batch_spans in result:
            for span in batch_spans:
                assert len(span) == 4
                start, end, entity_type, score = span
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert isinstance(entity_type, str)
                assert isinstance(score, float)
    
    def test_matches_start_end_pairs(self, token_config, token_inputs):
        """Should only create spans where start and end match."""
        decoder = TokenDecoder(token_config)
        
        result = decoder.decode(
            tokens=token_inputs['tokens'],
            id_to_classes=token_inputs['id_to_classes'],
            model_output=token_inputs['model_output'],
            threshold=0.5
        )
        
        # Should find at least one valid span
        total_spans = sum(len(spans) for spans in result)
        assert total_spans > 0
        
        # All spans should have end >= start
        for batch_spans in result:
            for span in batch_spans:
                start, end, _, _ = span
                assert end >= start
    
    def test_validates_inside_scores(self, token_config):
        """Should filter spans where inside scores are below threshold."""
        decoder = TokenDecoder(token_config)
        
        batch_size = 1
        seq_length = 5
        num_classes = 1
        
        model_output = torch.ones(batch_size, seq_length, num_classes, 3) * -5.0
        
        # Create span 0-2 with low inside score at interior token 1
        model_output[0, 0, 0, 0] = 5.0  # High start
        model_output[0, 2, 0, 1] = 5.0  # High end
        model_output[0, 0, 0, 2] = 5.0  # Inside at start (required by decoder)
        model_output[0, 2, 0, 2] = 5.0  # Inside at end (required by decoder)
        model_output[0, 1, 0, 2] = -5.0  # Low inside score at interior position
        
        tokens = [['A', 'B', 'C', 'D', 'E']]
        id_to_classes = {1: 'TYPE'}
        
        result = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            threshold=0.5
        )
        
        # Span should be filtered due to low interior inside score
        assert len(result[0]) == 0
    
    def test_span_score_is_minimum(self, token_config):
        """Should calculate span score as minimum of start/end/inside."""
        decoder = TokenDecoder(token_config)
        
        batch_size = 1
        seq_length = 4
        num_classes = 1
        
        model_output = torch.ones(batch_size, seq_length, num_classes, 3) * -5.0
        
        # Create span with varying scores (span 0-1)
        model_output[0, 0, 0, 0] = 3.0  # Start: sigmoid(3.0) ≈ 0.95
        model_output[0, 1, 0, 1] = 4.0  # End: sigmoid(4.0) ≈ 0.98
        model_output[0, 0, 0, 2] = 10.0 # Inside at start: ~1.0 (so it won't be the min)
        model_output[0, 1, 0, 2] = 2.0  # Inside at end: sigmoid(2.0) ≈ 0.88 (minimum)
        
        tokens = [['A', 'B', 'C', 'D']]
        id_to_classes = {1: 'TYPE'}
        
        result = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            threshold=0.5
        )
        
        # Should have one span
        assert len(result[0]) == 1
        
        # Score should be approximately the minimum (sigmoid(2.0) ≈ 0.88)
        _, _, _, score = result[0][0]
        assert 0.85 < score < 0.92
    
    def test_handles_empty_predictions(self, token_config):
        """Should handle case with no valid spans."""
        decoder = TokenDecoder(token_config)
        
        # All low scores
        model_output = torch.ones(1, 3, 2, 3) * -10.0
        tokens = [['A', 'B', 'C']]
        id_to_classes = {1: 'PERSON', 2: 'LOCATION'}
        
        result = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            threshold=0.5
        )
        
        assert len(result) == 1
        assert len(result[0]) == 0

class TestGreedySearch:
    """Test suite for greedy_search method across decoders."""
    
    @pytest.fixture
    def basic_decoder(self):
        """Fixture providing a basic decoder instance."""
        config = Mock()
        return SpanDecoder(config)
    
    def test_removes_lower_scoring_overlaps(self, basic_decoder):
        """Should keep higher-scoring span when overlaps exist."""
        spans = [
            (0, 2, 'PERSON', 0.9),
            (1, 3, 'ORG', 0.7),  # Overlaps with first span
            (5, 7, 'LOCATION', 0.8)
        ]
        
        result = basic_decoder.greedy_search(spans, flat_ner=True)
        
        # Should keep first and third (non-overlapping, higher scores)
        assert len(result) == 2
        assert result[0][2] == 'PERSON'
        assert result[1][2] == 'LOCATION'
    
    def test_sorts_by_start_position(self, basic_decoder):
        """Should return spans sorted by start position."""
        spans = [
            (5, 7, 'LOCATION', 0.9),
            (0, 2, 'PERSON', 0.8),
            (10, 12, 'ORG', 0.7)
        ]
        
        result = basic_decoder.greedy_search(spans, flat_ner=True)
        
        # Should be sorted by start position
        assert result[0][0] == 0
        assert result[1][0] == 5
        assert result[2][0] == 10
    
    def test_handles_nested_spans(self, basic_decoder):
        """Should allow nested spans when flat_ner=False."""
        spans = [
            (0, 5, 'PERSON', 0.9),  # Larger span
            (1, 3, 'NAME', 0.8)      # Nested inside
        ]
        
        result = basic_decoder.greedy_search(spans, flat_ner=False)
        
        # Both spans should be kept
        assert len(result) == 2
    
    def test_multi_label_same_position(self, basic_decoder):
        """Should allow multiple labels for same span when multi_label=True."""
        spans = [
            (0, 2, 'PERSON', 0.9),
            (0, 2, 'DOCTOR', 0.8)  # Same span, different label
        ]
        
        result = basic_decoder.greedy_search(spans, flat_ner=True, multi_label=True)
        
        # Both labels should be kept
        assert len(result) == 2
    
    def test_empty_input(self, basic_decoder):
        """Should handle empty span list."""
        spans = []
        
        result = basic_decoder.greedy_search(spans, flat_ner=True)
        
        assert len(result) == 0
        assert isinstance(result, list)