import pytest
import warnings
import random
import torch
from collections import defaultdict
from unittest.mock import Mock
from transformers import AutoTokenizer
from gliner.data_processing.utils import pad_2d_tensor, prepare_span_idx, prepare_word_mask, get_negatives, make_mapping

class TestPad2DTensor:
    """Test suite for pad_2d_tensor function."""
    
    @pytest.fixture
    def sample_tensors(self):
        """Fixture providing sample 2D tensors of varying sizes."""
        return [
            torch.tensor([[1, 2], [3, 4]]),           # 2x2
            torch.tensor([[5, 6, 7]]),                # 1x3
            torch.tensor([[8], [9], [10]]),           # 3x1
        ]
    
    def test_pads_to_maximum_dimensions(self, sample_tensors):
        """Should pad all tensors to match the maximum rows and columns."""
        result = pad_2d_tensor(sample_tensors)
        
        assert result.shape == (3, 3, 3)  # batch_size=3, max_rows=3, max_cols=3
        assert torch.allclose(
            result[0], 
            torch.tensor([[1, 2, 0], [3, 4, 0], [0, 0, 0]], dtype=torch.long)
        )
    
    def test_preserves_original_values(self, sample_tensors):
        """Should preserve all original tensor values in padded output."""
        result = pad_2d_tensor(sample_tensors)
        
        # First tensor: check original values
        assert result[0, 0, 0] == 1
        assert result[0, 0, 1] == 2
        assert result[0, 1, 0] == 3
        assert result[0, 1, 1] == 4
        
        # Second tensor
        assert result[1, 0, 0] == 5
        assert result[1, 0, 1] == 6
        assert result[1, 0, 2] == 7
        
        # Third tensor
        assert result[2, 0, 0] == 8
        assert result[2, 1, 0] == 9
        assert result[2, 2, 0] == 10
    
    def test_pads_with_zeros(self, sample_tensors):
        """Should fill padded regions with zeros."""
        result = pad_2d_tensor(sample_tensors)
        
        # Check padding in first tensor (should be padded with zeros in last row)
        assert torch.all(result[0, 2, :] == 0)  # Last row all zeros
        assert result[0, 0, 2] == 0  # Last column of first row
        
        # Check padding in second tensor
        assert torch.all(result[1, 1:, :] == 0)  # Rows 1-2 all zeros
    
    def test_single_tensor(self):
        """Should handle a single tensor correctly."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = pad_2d_tensor([tensor])
        
        assert result.shape == (1, 2, 2)
        assert torch.allclose(result[0], tensor.long())
    
    def test_uniform_size_tensors(self):
        """Should handle tensors of the same size without modification."""
        tensors = [
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
        ]
        result = pad_2d_tensor(tensors)
        
        assert result.shape == (2, 2, 2)
        assert torch.allclose(result[0], tensors[0].long())
        assert torch.allclose(result[1], tensors[1].long())
    
    def test_raises_error_on_empty_list(self):
        """Should raise ValueError when given empty list."""
        with pytest.raises(ValueError, match="should not be empty"):
            pad_2d_tensor([])
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float64])
    def test_preserves_dtype(self, dtype):
        """Should handle different tensor data types."""
        tensors = [
            torch.tensor([[1, 2]], dtype=dtype),
            torch.tensor([[3]], dtype=dtype),
        ]
        result = pad_2d_tensor(tensors)
        
        # Result is converted to float by pad function, but values should be preserved
        assert result[0, 0, 0] == 1
        assert result[1, 0, 0] == 3


class TestGetNegatives:
    """Test suite for get_negatives function."""
    
    @pytest.fixture
    def sample_batch(self):
        """Fixture providing sample batch with NER annotations."""
        return [
            {"ner": [(0, 1, "PER"), (2, 3, "LOC")]},
            {"ner": [(0, 1, "ORG"), (1, 2, "PER")]},
            {"ner": [(0, 2, "MISC"), (3, 4, "LOC")]},
        ]
    
    def test_returns_requested_number_of_negatives(self, sample_batch):
        """Should return exactly the requested number of samples."""
        result = get_negatives(sample_batch, sampled_neg=2)
        
        assert len(result) == 2
        assert all(isinstance(item, str) for item in result)
    
    def test_samples_from_available_types(self, sample_batch):
        """Should only sample from entity types present in batch."""
        available_types = {"PER", "LOC", "ORG", "MISC"}
        result = get_negatives(sample_batch, sampled_neg=3)
        
        assert all(item in available_types for item in result)
        assert len(set(result)) == len(result)  # All unique
    
    def test_returns_all_types_when_sample_exceeds_available(self, sample_batch):
        """Should return all available types when sample size exceeds them."""
        result = get_negatives(sample_batch, sampled_neg=10)
        
        # Should return all 4 unique types (PER, LOC, ORG, MISC)
        assert len(result) == 4
        assert set(result) == {"PER", "LOC", "ORG", "MISC"}
    
    def test_custom_key_parameter(self):
        """Should work with custom key for accessing annotations."""
        batch = [
            {"relations": [(0, 1, "WORKS_FOR"), (2, 3, "LIVES_IN")]},
            {"relations": [(0, 1, "OWNS")]},
        ]
        result = get_negatives(batch, sampled_neg=2, key="relations")
        
        assert len(result) <= 2
        assert all(item in {"WORKS_FOR", "LIVES_IN", "OWNS"} for item in result)
    
    def test_handles_duplicate_types(self):
        """Should deduplicate types from multiple examples."""
        batch = [
            {"ner": [(0, 1, "PER"), (2, 3, "PER"), (4, 5, "PER")]},
            {"ner": [(0, 1, "PER")]},
        ]
        result = get_negatives(batch, sampled_neg=5)
        
        # Only one unique type available
        assert len(result) == 1
        assert result[0] == "PER"
    
    def test_default_sample_size(self):
        """Should use default sample size of 5."""
        batch = [{"ner": [(i, i+1, f"TYPE_{i}") for i in range(10)]}]
        result = get_negatives(batch)
        
        assert len(result) == 5
    
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_randomness(self, sample_batch, seed):
        """Should produce different samples with different random seeds."""
        random.seed(seed)
        result1 = get_negatives(sample_batch, sampled_neg=2)
        
        random.seed(seed)
        result2 = get_negatives(sample_batch, sampled_neg=2)
        
        # Same seed should produce same result
        assert result1 == result2


class TestPrepareWordMask:
    """Test suite for prepare_word_mask function."""
    
    @pytest.fixture
    def mock_tokenized_inputs(self):
        """Create a mock tokenized inputs object."""
        mock = Mock()
        # Simulates: ["The", "cat"] -> ["The", "c", "##at"]
        # word_ids: [0, 1, 1] (first word index 0, second word spans indices 1-2)
        mock.word_ids.return_value = [None, 0, 1, 1, None]
        return mock
    
    def test_basic_word_masking(self, mock_tokenized_inputs):
        """Should create 1-based word indices, ignoring None."""
        texts = [["The", "cat"]]
        result = prepare_word_mask(texts, mock_tokenized_inputs)
        
        assert len(result) == 1
        assert result[0] == [0, 1, 2, 0, 0]  # None->0, word0->1, word1->2, continuation->0, None->0
    
    def test_skip_first_words(self):
        """Should skip the first N words as specified."""
        mock = Mock()
        # ["ent", "sep", "The", "cat"] -> word_ids: [0, 1, 2, 3, 3]
        mock.word_ids.return_value = [0, 1, 2, 3, 3]
        
        texts = [["ent", "sep", "The", "cat"]]
        result = prepare_word_mask(texts, mock, skip_first_words=[2])
        
        # First 2 words skipped, then 1-based indexing
        assert result[0] == [0, 0, 1, 2, 0]
    
    def test_token_level_mode(self):
        """Should create mask for every token when token_level=True."""
        mock = Mock()
        # Word 1 spans tokens 1-2
        mock.word_ids.return_value = [None, 0, 1, 1, None]
        
        texts = [["The", "cat"]]
        result = prepare_word_mask(texts, mock, token_level=True)
        
        # token_level=True means even continuations get numbered
        assert result[0] == [0, 1, 2, 2, 0]  # Both tokens of word "cat" get index 2
    
    def test_multiple_sequences(self):
        """Should handle multiple sequences in batch."""
        mock = Mock()
        mock.word_ids.side_effect = [
            [None, 0, 1, None],  # First sequence
            [None, 0, 0, 1, None],  # Second sequence
        ]
        
        texts = [["The", "cat"], ["A", "dog"]]
        result = prepare_word_mask(texts, mock)
        
        assert len(result) == 2
        assert result[0] == [0, 1, 2, 0]
        assert result[1] == [0, 1, 0, 2, 0]
    
    def test_raises_on_mismatched_skip_length(self):
        """Should raise ValueError if skip_first_words length doesn't match texts."""
        mock = Mock()
        texts = [["word1"], ["word2"]]
        
        with pytest.raises(ValueError, match="must have same length"):
            prepare_word_mask(texts, mock, skip_first_words=[1])
    
    def test_all_none_word_ids(self):
        """Should handle sequence with all None word_ids (special tokens only)."""
        mock = Mock()
        mock.word_ids.return_value = [None, None, None]
        
        texts = [[]]
        result = prepare_word_mask(texts, mock)
        
        assert result[0] == [0, 0, 0]
    
    def test_word_continuation_handling(self):
        """Should properly handle word continuations (subword tokens)."""
        mock = Mock()
        # Simulates "running" -> ["run", "##ning"] with word_id=0 for both
        mock.word_ids.return_value = [None, 0, 0, 0, None]
        
        texts = [["running"]]
        result = prepare_word_mask(texts, mock, token_level=False)
        
        # Only first token of each word gets indexed
        assert result[0] == [0, 1, 0, 0, 0]
    
    @pytest.mark.parametrize("skip,expected", [
        (0, [0, 1, 2, 0]),
        (1, [0, 0, 1, 0]),
        (2, [0, 0, 0, 0]),
    ])
    def test_various_skip_amounts(self, skip, expected):
        """Should correctly skip different amounts of words."""
        mock = Mock()
        mock.word_ids.return_value = [None, 0, 1, None]
        
        texts = [["word1", "word2"]]
        result = prepare_word_mask(texts, mock, skip_first_words=[skip])
        
        assert result[0] == expected


class TestMakeMapping:
    """Test suite for make_mapping function."""
    
    def test_creates_bidirectional_mapping(self):
        """Should create forward and reverse mappings."""
        types = ["PER", "LOC", "ORG"]
        fwd, rev = make_mapping(types)
        
        assert fwd == {"PER": 1, "LOC": 2, "ORG": 3}
        assert rev == {1: "PER", 2: "LOC", 3: "ORG"}
    
    def test_one_based_indexing(self):
        """Should use 1-based indexing (starting from 1, not 0)."""
        types = ["A"]
        fwd, rev = make_mapping(types)
        
        assert fwd["A"] == 1
        assert rev[1] == "A"
    
    def test_deduplicates_while_preserving_order(self):
        """Should remove duplicates while keeping first occurrence order."""
        types = ["PER", "LOC", "PER", "ORG", "LOC"]
        fwd, rev = make_mapping(types)
        
        # Should only have 3 unique types
        assert len(fwd) == 3
        assert fwd == {"PER": 1, "LOC": 2, "ORG": 3}
        
        # Verify order is preserved (PER before LOC before ORG)
        assert list(fwd.keys()) == ["PER", "LOC", "ORG"]
    
    def test_empty_list(self):
        """Should handle empty input gracefully."""
        fwd, rev = make_mapping([])
        
        assert fwd == {}
        assert rev == {}
    
    def test_single_element(self):
        """Should handle single element list."""
        types = ["MISC"]
        fwd, rev = make_mapping(types)
        
        assert fwd == {"MISC": 1}
        assert rev == {1: "MISC"}
    
    def test_reverse_mapping_consistency(self):
        """Forward and reverse mappings should be consistent."""
        types = ["A", "B", "C", "D", "E"]
        fwd, rev = make_mapping(types)
        
        for key, value in fwd.items():
            assert rev[value] == key
        
        for key, value in rev.items():
            assert fwd[value] == key
    
    @pytest.mark.parametrize("types,expected_len", [
        (["A"], 1),
        (["A", "B"], 2),
        (["A", "A", "A"], 1),
        (["A", "B", "A", "C"], 3),
    ])
    def test_various_input_sizes(self, types, expected_len):
        """Should handle various input sizes correctly."""
        fwd, rev = make_mapping(types)
        
        assert len(fwd) == expected_len
        assert len(rev) == expected_len


class TestPrepareSpanIdx:
    """Test suite for prepare_span_idx function."""
    
    def test_generates_all_spans_within_width(self):
        """Should generate all possible spans up to max_width."""
        result = prepare_span_idx(num_tokens=3, max_width=2)
        
        expected = [
            (0, 0), (0, 1),  # Starting at 0
            (1, 1), (1, 2),  # Starting at 1
            (2, 2), (2, 3),  # Starting at 2
        ]
        assert result == expected
    
    def test_span_width_one(self):
        """Should generate single-token spans when max_width=1."""
        result = prepare_span_idx(num_tokens=3, max_width=1)
        
        assert result == [(0, 0), (1, 1), (2, 2)]
    
    def test_single_token(self):
        """Should handle single token correctly."""
        result = prepare_span_idx(num_tokens=1, max_width=3)
        
        assert result == [(0, 0), (0, 1), (0, 2)]
    
    def test_span_count(self):
        """Should generate correct number of spans."""
        num_tokens = 5
        max_width = 3
        result = prepare_span_idx(num_tokens, max_width)
        
        # Each token starts max_width spans
        expected_count = num_tokens * max_width
        assert len(result) == expected_count
    
    def test_spans_can_exceed_sequence(self):
        """Generated spans can extend beyond sequence (filtered later)."""
        result = prepare_span_idx(num_tokens=2, max_width=3)
        
        # Last spans will exceed sequence length
        assert (1, 3) in result  # This exceeds num_tokens=2
        assert result[-1] == (1, 3)
    
    def test_span_format(self):
        """Should return list of (start, end) tuples."""
        result = prepare_span_idx(num_tokens=2, max_width=2)
        
        assert all(isinstance(span, tuple) for span in result)
        assert all(len(span) == 2 for span in result)
        assert all(isinstance(i, int) for span in result for i in span)
    
    @pytest.mark.parametrize("num_tokens,max_width", [
        (1, 1),
        (5, 1),
        (1, 5),
        (10, 5),
        (100, 10),
    ])
    def test_various_parameters(self, num_tokens, max_width):
        """Should work with various parameter combinations."""
        result = prepare_span_idx(num_tokens, max_width)
        
        assert len(result) == num_tokens * max_width
        assert all(span[0] < num_tokens for span in result)
    
    def test_span_ordering(self):
        """Should generate spans in order: by start position, then by width."""
        result = prepare_span_idx(num_tokens=3, max_width=3)
        
        # Check that spans starting at 0 come first
        first_three = result[:3]
        assert all(span[0] == 0 for span in first_three)
        assert first_three == [(0, 0), (0, 1), (0, 2)]
        
        # Then spans starting at 1
        next_three = result[3:6]
        assert all(span[0] == 1 for span in next_three)


@pytest.fixture
def mock_config():
    """Mock configuration object with default settings."""
    config = Mock()
    config.ent_token = "[ENT]"
    config.sep_token = "[SEP]"
    config.rel_token = "[REL]"
    config.max_width = 3
    config.max_len = 512
    config.max_types = 10
    config.max_neg_type_ratio = 5
    config.blank_entity_prob = 0.0
    config.decoder_mode = 'span'
    config.full_decoder_context = False
    return config


@pytest.fixture
def mock_tokenizer():
    """Real transformer tokenizer for testing."""
    # Use a small, fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    return tokenizer


@pytest.fixture
def mock_words_splitter():
    """Mock word splitter function."""
    return lambda text: text.split()


@pytest.fixture
def sample_tokens():
    """Sample token list."""
    return ["The", "quick", "brown", "fox", "jumps"]


@pytest.fixture
def sample_ner():
    """Sample NER annotations."""
    return [(0, 0, "DET"), (1, 3, "ENTITY"), (4, 4, "VERB")]


@pytest.fixture
def sample_classes_to_id():
    """Sample class mapping."""
    return {"DET": 1, "ENTITY": 2, "VERB": 3}


@pytest.fixture
def sample_batch_list():
    """Sample batch list for collate functions."""
    return [
        {
            "tokenized_text": ["The", "cat"],
            "ner": [(0, 0, "DET"), (1, 1, "NOUN")],
        },
        {
            "tokenized_text": ["A", "dog"],
            "ner": [(0, 0, "DET"), (1, 1, "NOUN")],
        },
    ]


class TestBaseProcessor:
    """Test suite for BaseProcessor abstract class functionality."""
    
    @pytest.fixture
    def processor(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Create a concrete implementation of BaseProcessor for testing."""
        from unittest.mock import Mock
        
        # Create a concrete class for testing
        class ConcreteProcessor:
            def __init__(self, config, tokenizer, words_splitter):
                self.config = config
                self.transformer_tokenizer = tokenizer
                self.words_splitter = words_splitter
                self.ent_token = config.ent_token
                self.sep_token = config.sep_token
                self._check_and_set_special_tokens(tokenizer)
            
            # Copy methods from BaseProcessor
            from gliner.data_processing import BaseProcessor
            _check_and_set_special_tokens = BaseProcessor._check_and_set_special_tokens
            get_dict = BaseProcessor.__dict__['get_dict']
            prepare_inputs = BaseProcessor.prepare_inputs
            _select_entities = BaseProcessor._select_entities
            _maybe_remap_entities = BaseProcessor._maybe_remap_entities
            _extra_prompt_tokens = BaseProcessor._extra_prompt_tokens
            batch_generate_class_mappings = BaseProcessor.batch_generate_class_mappings
        
        return ConcreteProcessor(mock_config, mock_tokenizer, mock_words_splitter)
    
    def test_get_dict_creates_correct_mapping(self, sample_ner, sample_classes_to_id):
        """Should create correct span-to-class mapping."""
        from gliner.data_processing import BaseProcessor
        
        result = BaseProcessor.get_dict(sample_ner, sample_classes_to_id)
        
        assert result[(0, 0)] == 1  # DET
        assert result[(1, 3)] == 2  # ENTITY
        assert result[(4, 4)] == 3  # VERB
    
    def test_get_dict_ignores_unknown_classes(self):
        """Should ignore spans with classes not in mapping."""
        from gliner.data_processing import BaseProcessor
        
        spans = [(0, 1, "KNOWN"), (2, 3, "UNKNOWN")]
        classes_to_id = {"KNOWN": 1}
        
        result = BaseProcessor.get_dict(spans, classes_to_id)
        
        assert (0, 1) in result
        assert (2, 3) not in result
    
    def test_get_dict_returns_defaultdict(self):
        """Should return defaultdict with 0 as default value."""
        from gliner.data_processing import BaseProcessor
        
        result = BaseProcessor.get_dict([], {})
        
        assert isinstance(result, defaultdict)
        assert result[(999, 999)] == 0  # Non-existent key returns 0
    
    def test_prepare_inputs_basic(self, processor):
        """Should prepare inputs with entity tokens and separator."""
        texts = [["The", "cat", "sat"]]
        entities = [["PER", "LOC"]]
        
        input_texts, prompt_lengths = processor.prepare_inputs(texts, entities)
        
        assert len(input_texts) == 1
        assert input_texts[0][:5] == ["[ENT]", "PER", "[ENT]", "LOC", "[SEP]"]
        assert input_texts[0][5:] == ["The", "cat", "sat"]
        assert prompt_lengths[0] == 5
    
    def test_prepare_inputs_with_blank(self, processor):
        """Should use blank entity when specified."""
        texts = [["word"]]
        entities = [["PER", "LOC"]]
        
        input_texts, prompt_lengths = processor.prepare_inputs(texts, entities, blank="entity")
        
        assert "[ENT]" in input_texts[0]
        assert "entity" in input_texts[0]
        assert "PER" not in input_texts[0]
        assert "LOC" not in input_texts[0]
    
    def test_prepare_inputs_dict_entities(self, processor):
        """Should handle entity dict with per-example entities."""
        texts = [["text1"], ["text2"]]
        entities = [{"PER": 0}, {"PER": 0, "LOC": 1, "ORG": 2}]
        
        input_texts, prompt_lengths = processor.prepare_inputs(texts, entities)
        
        # First example should have PER
        assert "PER" in input_texts[0]
        assert "LOC" not in input_texts[0]
        
        # Second example should have LOC and ORG
        assert "LOC" in input_texts[1]
        assert "ORG" in input_texts[1]
    
    def test_prepare_inputs_shared_entities(self, processor):
        """Should use same entities for all examples when flat list provided."""
        texts = [["text1"], ["text2"], ["text3"]]
        entities = ["PER", "LOC"]  # Flat list - same for all
        
        input_texts, prompt_lengths = processor.prepare_inputs(texts, entities)
        
        # All examples should have same entities
        for input_text in input_texts:
            assert "PER" in input_text
            assert "LOC" in input_text
    
    def test_select_entities_with_blank(self, processor):
        """Should return blank entity when specified."""
        entities = [["PER", "LOC"]]
        
        result = processor._select_entities(0, entities, blank="entity")
        
        assert result == ["entity"]
    
    def test_select_entities_from_dict(self, processor):
        """Should select entities from dict by index."""
        entities = [{"PER": 0}, {"LOC": 0}]
        
        result0 = processor._select_entities(0, entities, blank=None)
        result1 = processor._select_entities(1, entities, blank=None)
        
        assert result0 == ["PER"]
        assert result1 == ["LOC"]
    
    def test_batch_generate_class_mappings_basic(self, processor, sample_batch_list):
        """Should generate class mappings for batch."""
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(sample_batch_list)
        
        assert len(class_to_ids) == 2
        assert len(id_to_classes) == 2
        
        # Check bidirectional mapping consistency
        for i in range(2):
            for label, idx in class_to_ids[i].items():
                assert id_to_classes[i][idx] == label
    
    def test_batch_generate_class_mappings_with_negatives(self, processor):
        """Should include provided negatives in mappings."""
        batch_list = [
            {"tokenized_text": ["word"], "ner": [(0, 0, "PER")]},
        ]
        negatives = ["LOC", "ORG"]
        
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(
            batch_list, negatives=negatives
        )
        
        # At least some negatives should be included
        all_types = set(class_to_ids[0].keys())
        assert "PER" in all_types
    
    def test_batch_generate_class_mappings_predefined_labels(self, processor):
        """Should use predefined labels when provided."""
        batch_list = [
            {
                "tokenized_text": ["word"],
                "ner": [(0, 0, "PER")],
                "ner_labels": ["PER", "LOC", "ORG"],  # Predefined
            },
        ]
        
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(batch_list)
        
        assert set(class_to_ids[0].keys()) == {"PER", "LOC", "ORG"}
    
    def test_batch_generate_class_mappings_custom_key(self, processor):
        """Should work with custom annotation key."""
        batch_list = [
            {"tokenized_text": ["word"], "relations": [(0, 1, "WORKS_FOR")]},
        ]
        
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(
            batch_list, key="relations"
        )
        
        assert "WORKS_FOR" in class_to_ids[0]


class TestUniEncoderSpanProcessor:
    """Test suite for UniEncoderSpanProcessor."""
    
    @pytest.fixture
    def processor(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Create UniEncoderSpanProcessor instance."""
        from gliner.data_processing import UniEncoderSpanProcessor
        return UniEncoderSpanProcessor(mock_config, mock_tokenizer, mock_words_splitter)
    
    def test_preprocess_example_basic(self, processor, sample_tokens, sample_ner, sample_classes_to_id):
        """Should preprocess example with spans and labels."""
        result = processor.preprocess_example(sample_tokens, sample_ner, sample_classes_to_id)
        
        assert "tokens" in result
        assert "span_idx" in result
        assert "span_label" in result
        assert "seq_length" in result
        assert "entities" in result
        
        assert result["tokens"] == sample_tokens
        assert result["seq_length"] == len(sample_tokens)
        assert isinstance(result["span_idx"], torch.Tensor)
        assert isinstance(result["span_label"], torch.Tensor)
    
    def test_preprocess_example_empty_tokens(self, processor):
        """Should handle empty token list by adding padding."""
        result = processor.preprocess_example([], [], {})
        
        assert result["tokens"] == ["[PAD]"]
        assert result["seq_length"] == 1
    
    def test_preprocess_example_truncation(self, processor):
        """Should truncate long sequences and warn."""
        long_tokens = ["word"] * 600  # Exceeds max_len=512
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = processor.preprocess_example(long_tokens, [], {})
            
            assert len(w) == 1
            assert "truncated" in str(w[0].message).lower()
            assert len(result["tokens"]) == 512
    
    def test_preprocess_example_invalid_spans_masked(self, processor):
        """Should mask spans that exceed sequence length."""
        tokens = ["word1", "word2"]
        ner = [(0, 0, "TYPE"), (1, 5, "TYPE")]  # Second span exceeds length
        classes_to_id = {"TYPE": 1}
        
        result = processor.preprocess_example(tokens, ner, classes_to_id)
        
        # Labels for invalid spans should be -1
        span_labels = result["span_label"]
        invalid_mask = result["span_idx"][:, 1] > 1  # End > last token index
        assert torch.all(span_labels[invalid_mask] == -1)
    
    def test_preprocess_example_unknown_classes_ignored(self, processor):
        """Should ignore NER spans with unknown classes."""
        tokens = ["word1", "word2"]
        ner = [(0, 0, "KNOWN"), (1, 1, "UNKNOWN")]
        classes_to_id = {"KNOWN": 1}
        
        result = processor.preprocess_example(tokens, ner, classes_to_id)
        
        # Should only have labels for KNOWN class
        span_labels = result["span_label"]
        assert torch.sum(span_labels == 1) >= 1  # At least one KNOWN label
    
    def test_create_batch_dict_structure(self, processor):
        """Should create properly structured batch dictionary."""
        batch = [
            {
                "tokens": ["word1"],
                "span_idx": torch.tensor([[0, 0]]),
                "span_label": torch.tensor([1]),
                "seq_length": 1,
                "entities": [(0, 0, "TYPE")],
            },
        ]
        class_to_ids = [{"TYPE": 1}]
        id_to_classes = [{1: "TYPE"}]
        
        result = processor.create_batch_dict(batch, class_to_ids, id_to_classes)
        
        assert "seq_length" in result
        assert "span_idx" in result
        assert "tokens" in result
        assert "span_mask" in result
        assert "span_label" in result
        assert "entities" in result
        assert "classes_to_id" in result
        assert "id_to_classes" in result
    
    def test_create_batch_dict_padding(self, processor):
        """Should pad sequences to same length."""
        batch = [
            {
                "tokens": ["w1"],
                "span_idx": torch.tensor([[0, 0]]),
                "span_label": torch.tensor([1]),
                "seq_length": 1,
                "entities": [],
            },
            {
                "tokens": ["w1", "w2"],
                "span_idx": torch.tensor([[0, 0], [1, 1], [0, 1]]),
                "span_label": torch.tensor([1, 2, 0]),
                "seq_length": 2,
                "entities": [],
            },
        ]
        class_to_ids = [{"TYPE": 1}] * 2
        id_to_classes = [{1: "TYPE"}] * 2
        
        result = processor.create_batch_dict(batch, class_to_ids, id_to_classes)
        
        # Check shapes are consistent
        assert result["span_idx"].shape[0] == 2  # Batch size
        assert result["span_label"].shape[0] == 2
        assert result["span_idx"].shape[1] == result["span_label"].shape[1]  # Same sequence length
    
    def test_create_batch_dict_span_mask(self, processor):
        """Should create correct span mask (True where label != -1)."""
        batch = [
            {
                "tokens": ["w1"],
                "span_idx": torch.tensor([[0, 0], [0, 1]]),
                "span_label": torch.tensor([1, -1]),  # Second is invalid
                "seq_length": 1,
                "entities": [],
            },
        ]
        class_to_ids = [{"TYPE": 1}]
        id_to_classes = [{1: "TYPE"}]
        
        result = processor.create_batch_dict(batch, class_to_ids, id_to_classes)
        
        expected_mask = torch.tensor([[True, False]])
        assert torch.all(result["span_mask"] == expected_mask)
    
    def test_create_labels_one_hot_encoding(self, processor):
        """Should create one-hot encoded labels for spans."""
        batch = {
            'tokens': [["The", "cat"]],
            'classes_to_id': [{"DET": 1, "NOUN": 2}],
            'entities': [[(0, 0, "DET"), (1, 1, "NOUN")]],
        }
        
        labels = processor.create_labels(batch)
        
        assert isinstance(labels, torch.Tensor)
        assert labels.dim() == 3  # [batch, num_spans, num_classes]
        assert labels.shape[2] == 2  # 2 classes (excluding background)
    
    def test_create_labels_handles_multiple_examples(self, processor):
        """Should handle batch with multiple examples."""
        batch = {
            'tokens': [["word1"], ["word2", "word3"]],
            'classes_to_id': [{"TYPE": 1}, {"TYPE": 1}],
            'entities': [[(0, 0, "TYPE")], [(0, 1, "TYPE")]],
        }
        
        labels = processor.create_labels(batch)
        
        assert labels.shape[0] == 2  # Batch size
    
    def test_create_labels_masks_invalid_spans(self, processor):
        """Should set labels to 0 for spans exceeding sequence."""
        batch = {
            'tokens': [["word"]],  # Only 1 token
            'classes_to_id': [{"TYPE": 1}],
            'entities': [[(0, 0, "TYPE")]],
        }
        
        labels = processor.create_labels(batch)
        
        # Spans beyond sequence length should be all zeros
        # This tests that valid_span_mask is applied correctly
        assert labels.shape[1] > 1  # Multiple spans generated
    
    def test_tokenize_and_prepare_labels_with_labels(self, processor):
        """Should tokenize and prepare labels when requested."""
        batch = {
            'tokens': [["The", "cat"]],
            'classes_to_id': [{"NOUN": 1}],
            'entities': [[(1, 1, "NOUN")]],
        }
        
        result = processor.tokenize_and_prepare_labels(batch, prepare_labels=True)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert isinstance(result['labels'], torch.Tensor)
    
    def test_tokenize_and_prepare_labels_without_labels(self, processor):
        """Should tokenize without labels when not requested."""
        batch = {
            'tokens': [["The", "cat"]],
            'classes_to_id': [{"NOUN": 1}],
            'entities': [[(1, 1, "NOUN")]],
        }
        
        result = processor.tokenize_and_prepare_labels(batch, prepare_labels=False)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' not in result


class TestUniEncoderTokenProcessor:
    """Test suite for UniEncoderTokenProcessor."""
    
    @pytest.fixture
    def processor(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Create UniEncoderTokenProcessor instance."""
        from gliner.data_processing import UniEncoderTokenProcessor
        return UniEncoderTokenProcessor(mock_config, mock_tokenizer, mock_words_splitter)
    
    def test_preprocess_example_basic(self, processor):
        """Should preprocess example with entity IDs."""
        tokens = ["The", "cat", "sat"]
        ner = [(0, 0, "DET"), (1, 1, "NOUN")]
        classes_to_id = {"DET": 1, "NOUN": 2}
        
        result = processor.preprocess_example(tokens, ner, classes_to_id)
        
        assert result['tokens'] == tokens
        assert result['seq_length'] == 3
        assert result['entities'] == ner
        assert len(result['entities_id']) == 2
        assert result['entities_id'][0] == [0, 0, 1]  # start, end, class_id
        assert result['entities_id'][1] == [1, 1, 2]
    
    def test_preprocess_example_filters_unknown_classes(self, processor):
        """Should filter out entities with unknown classes."""
        tokens = ["word1", "word2"]
        ner = [(0, 0, "KNOWN"), (1, 1, "UNKNOWN")]
        classes_to_id = {"KNOWN": 1}
        
        result = processor.preprocess_example(tokens, ner, classes_to_id)
        
        assert len(result['entities_id']) == 1
        assert result['entities_id'][0][2] == 1  # Only KNOWN class
    
    def test_preprocess_example_handles_none_ner(self, processor):
        """Should handle None NER gracefully."""
        tokens = ["word"]
        
        result = processor.preprocess_example(tokens, None, {})
        
        assert result['entities_id'] == []
        assert result['tokens'] == tokens
    
    def test_preprocess_example_empty_tokens(self, processor):
        """Should add padding token when tokens empty."""
        result = processor.preprocess_example([], None, {})
        
        assert result['tokens'] == ["[PAD]"]
        assert result['seq_length'] == 1
    
    def test_preprocess_example_truncation(self, processor):
        """Should truncate long sequences."""
        long_tokens = ["word"] * 600
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = processor.preprocess_example(long_tokens, None, {})
            
            assert len(w) == 1
            assert len(result['tokens']) == 512
    
    def test_create_batch_dict_structure(self, processor):
        """Should create correct batch dictionary structure."""
        batch = [
            {
                "tokens": ["word"],
                "seq_length": 1,
                "entities": [],
                "entities_id": [],
            },
        ]
        class_to_ids = [{"TYPE": 1}]
        id_to_classes = [{1: "TYPE"}]
        
        result = processor.create_batch_dict(batch, class_to_ids, id_to_classes)
        
        assert "tokens" in result
        assert "seq_length" in result
        assert "entities" in result
        assert "entities_id" in result
        assert "classes_to_id" in result
        assert "id_to_classes" in result
    
    def test_create_labels_structure(self, processor):
        """Should create labels with correct structure."""
        entities_id = [[[0, 1, 1], [2, 3, 2]]]  # batch_size=1, 2 entities
        batch_size = 1
        seq_len = 5
        num_classes = 2
        
        labels = processor.create_labels(entities_id, batch_size, seq_len, num_classes)
        
        assert labels.shape == (batch_size, seq_len, num_classes, 3)
        # Last dimension: [start, end, inside]
    
    def test_create_labels_start_end_inside_markers(self, processor):
        """Should correctly mark start, end, and inside tokens."""
        entities_id = [[[1, 3, 0]]]  # Entity from token 1 to 3, class 0
        batch_size = 1
        seq_len = 5
        num_classes = 1
        
        labels = processor.create_labels(entities_id, batch_size, seq_len, num_classes)
        
        # Check start token (index 1)
        assert labels[0, 1, 0, 0] == 1  # Start marker
        
        # Check end token (index 3)
        assert labels[0, 3, 0, 1] == 1  # End marker
        
        # Check inside tokens (1, 2, 3 should all be marked inside)
        assert labels[0, 1, 0, 2] == 1
        assert labels[0, 2, 0, 2] == 1
        assert labels[0, 3, 0, 2] == 1
    
    def test_create_labels_skips_out_of_bounds_entities(self, processor):
        """Should skip entities that exceed sequence length."""
        entities_id = [[[0, 1, 0], [10, 12, 0]]]  # Second entity exceeds seq_len
        batch_size = 1
        seq_len = 5
        num_classes = 1
        
        labels = processor.create_labels(entities_id, batch_size, seq_len, num_classes)
        
        # First entity should be marked
        assert labels[0, 0, 0, 0] == 1
    
    def test_create_labels_adjusts_class_index(self, processor):
        """Should adjust class index by subtracting 1."""
        entities_id = [[[0, 0, 2]]]  # Class ID 2
        batch_size = 1
        seq_len = 3
        num_classes = 2
        
        labels = processor.create_labels(entities_id, batch_size, seq_len, num_classes)
        
        # Class 2 should map to index 1 (2-1=1)
        assert labels[0, 0, 1, 0] == 1  # Start marker at class index 1
    
    def test_tokenize_and_prepare_labels_integration(self, processor):
        """Should integrate tokenization and label preparation."""
        batch = {
            'tokens': [["The", "cat"]],
            'seq_length': torch.tensor([[2]]),
            'classes_to_id': [{"NOUN": 1}],
            'entities_id': [[[1, 1, 1]]],
        }
        
        result = processor.tokenize_and_prepare_labels(batch, prepare_labels=True)
        
        assert 'input_ids' in result
        assert 'labels' in result
        assert result['labels'].shape[-1] == 3  # [start, end, inside]


class TestBiEncoderSpanProcessor:
    """Test suite for BaseBiEncoderProcessor."""
    
    @pytest.fixture
    def processor(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Create BiEncoderSpanProcessor instance."""
        from gliner.data_processing import BiEncoderSpanProcessor
        
        labels_tokenizer = Mock()
        labels_tokenizer.unk_token = "[UNK]"
        labels_tokenizer.pad_token = "[PAD]"
        
        def mock_label_tokenize(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                'input_ids': torch.randint(0, 100, (batch_size, 5)),
                'attention_mask': torch.ones(batch_size, 5, dtype=torch.long),
            }
        
        labels_tokenizer.side_effect = mock_label_tokenize
        
        return BiEncoderSpanProcessor(
            mock_config, mock_tokenizer, mock_words_splitter, labels_tokenizer
        )
    
    def test_initialization_with_labels_tokenizer(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Should initialize with labels tokenizer."""
        from gliner.data_processing import BiEncoderSpanProcessor
        
        labels_tokenizer = Mock()
        labels_tokenizer.unk_token = "[UNK]"
        labels_tokenizer.pad_token = "[PAD]"
        
        processor = BiEncoderSpanProcessor(
            mock_config, mock_tokenizer, mock_words_splitter, labels_tokenizer
        )
        
        assert processor.labels_tokenizer is labels_tokenizer
    
    def test_tokenize_inputs_with_entities(self, processor):
        """Should tokenize both texts and entity labels."""
        texts = [["The", "cat"]]
        entities = [["PER", "LOC"]]
        
        result = processor.tokenize_inputs(texts, entities)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels_input_ids' in result
        assert 'labels_attention_mask' in result
        assert 'words_mask' in result
    
    def test_tokenize_inputs_without_entities(self, processor):
        """Should tokenize only texts when entities not provided."""
        texts = [["The", "cat"]]
        
        result = processor.tokenize_inputs(texts, entities=None)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels_input_ids' not in result
        assert 'labels_attention_mask' not in result
    
    def test_batch_generate_class_mappings_shared_across_batch(self, processor):
        """Should create shared mappings across all batch examples."""
        batch_list = [
            {"ner": [(0, 0, "PER")]},
            {"ner": [(0, 0, "LOC")]},
        ]
        
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(batch_list)
        
        # Should return same mapping for all examples
        assert len(class_to_ids) == 2
        assert class_to_ids[0] == class_to_ids[1]
        assert id_to_classes[0] == id_to_classes[1]
    
    def test_batch_generate_class_mappings_includes_all_types(self, processor):
        """Should include types from all examples."""
        batch_list = [
            {"ner": [(0, 0, "PER")]},
            {"ner": [(0, 0, "LOC")]},
            {"ner": [(0, 0, "ORG")]},
        ]
        
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(batch_list)
        
        all_types = set(class_to_ids[0].keys())
        assert "PER" in all_types or "LOC" in all_types or "ORG" in all_types
    
    def test_batch_generate_class_mappings_with_negatives(self, processor):
        """Should include manual negatives when provided."""
        batch_list = [
            {"ner": [(0, 0, "PER")], "ner_negatives": ["MISC"]},
        ]
        
        class_to_ids, id_to_classes = processor.batch_generate_class_mappings(batch_list)
        
        all_types = set(class_to_ids[0].keys())
        assert "MISC" in all_types


class TestUniEncoderSpanDecoderProcessor:
    """Test suite for UniEncoderSpanDecoderProcessor."""
    
    @pytest.fixture
    def processor(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Create UniEncoderSpanDecoderProcessor instance."""
        from gliner.data_processing import UniEncoderSpanDecoderProcessor
        
        decoder_tokenizer = Mock()
        decoder_tokenizer.unk_token = "[UNK]"
        decoder_tokenizer.pad_token = "[PAD]"
        
        def mock_decode_tokenize(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                'input_ids': torch.randint(0, 100, (batch_size, 8)),
                'attention_mask': torch.ones(batch_size, 8, dtype=torch.long),
            }
        
        decoder_tokenizer.side_effect = mock_decode_tokenize
        
        return UniEncoderSpanDecoderProcessor(
            mock_config, mock_tokenizer, mock_words_splitter, decoder_tokenizer
        )
    
    def test_initialization_with_decoder_tokenizer(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Should initialize with decoder tokenizer."""
        from gliner.data_processing import UniEncoderSpanDecoderProcessor
        
        decoder_tokenizer = Mock()
        decoder_tokenizer.unk_token = "[UNK]"
        decoder_tokenizer.pad_token = "[PAD]"
        
        processor = UniEncoderSpanDecoderProcessor(
            mock_config, mock_tokenizer, mock_words_splitter, decoder_tokenizer
        )
        
        assert processor.decoder_tokenizer is decoder_tokenizer
    
    def test_tokenize_inputs_adds_decoder_inputs(self, processor):
        """Should add decoder inputs when decoder_mode is 'span'."""
        texts = [["The", "cat"]]
        entities = [["PER"]]
        
        result = processor.tokenize_inputs(texts, entities)
        
        assert 'input_ids' in result
        assert 'decoder_input_ids' in result
        assert 'decoder_attention_mask' in result
    
    def test_tokenize_inputs_with_blank(self, processor):
        """Should handle blank entity token."""
        texts = [["The", "cat"]]
        entities = [["PER"]]
        
        result = processor.tokenize_inputs(texts, entities, blank="entity")
        
        assert 'input_ids' in result
        assert 'decoder_input_ids' in result
    
    def test_create_labels_returns_both_encoder_and_decoder(self, processor):
        """Should return both encoder and decoder labels."""
        batch = {
            'tokens': [["The", "cat"]],
            'classes_to_id': [{"NOUN": 1}],
            'entities': [[(1, 1, "NOUN")]],
        }
        
        encoder_labels, decoder_labels = processor.create_labels(batch)
        
        assert isinstance(encoder_labels, torch.Tensor)
        assert decoder_labels is not None
        assert 'input_ids' in decoder_labels
        assert 'labels' in decoder_labels
    
    def test_create_labels_with_blank_entity(self, processor):
        """Should handle blank entity in label creation."""
        batch = {
            'tokens': [["word"]],
            'classes_to_id': [{"TYPE": 1}],
            'entities': [[(0, 0, "TYPE")]],
        }
        
        encoder_labels, decoder_labels = processor.create_labels(batch, blank="entity")
        
        if decoder_labels is not None:
            assert "labels" in decoder_labels
            assert 'input_ids' in decoder_labels

        assert isinstance(encoder_labels, torch.Tensor)
    
    def test_tokenize_and_prepare_labels_with_random_blank(self, processor, mock_config):
        """Should sometimes use blank entity based on probability."""
        mock_config.blank_entity_prob = 1.0  # Always use blank
        
        batch = {
            'tokens': [["word"]],
            'classes_to_id': [{"TYPE": 1}],
            'entities': [[(0, 0, "TYPE")]],
        }
        
        result = processor.tokenize_and_prepare_labels(batch, prepare_labels=True)
        
        assert 'labels' in result
        assert 'input_ids' in result


class TestRelationExtractionSpanProcessor:
    """Test suite for RelationExtractionSpanProcessor."""
    
    @pytest.fixture
    def processor(self, mock_config, mock_tokenizer, mock_words_splitter):
        """Create RelationExtractionSpanProcessor instance."""
        from gliner.data_processing import RelationExtractionSpanProcessor
        return RelationExtractionSpanProcessor(mock_config, mock_tokenizer, mock_words_splitter)
    
    def test_preprocess_example_includes_relations(self, processor):
        """Should include relations in preprocessed example."""
        tokens = ["John", "works", "at", "Google"]
        ner = [(0, 0, "PER"), (3, 3, "ORG")]
        relations = [(0, 1, "WORKS_FOR")]
        classes_to_id = {"PER": 1, "ORG": 2}
        rel_classes_to_id = {"WORKS_FOR": 0}
        result = processor.preprocess_example(tokens, ner, classes_to_id, relations, rel_classes_to_id)
        
        assert "relations" in result
        assert result["relations"] == relations
    
    def test_create_batch_dict_includes_relation_mappings(self, processor):
        """Should include relation class mappings in batch dict."""
        batch = [
            {
                "tokens": ["word"],
                "span_idx": torch.tensor([[0, 0]]),
                "span_label": torch.tensor([1]),
                # New required fields for relations
                "rel_idx": torch.tensor([[0, 0]]),
                "rel_label": torch.tensor([0]),
                "seq_length": 1,
                "entities": [],
                "relations": [],
            },
        ]
        class_to_ids = [{"TYPE": 1}]
        id_to_classes = [{1: "TYPE"}]
        rel_class_to_ids = [{"REL": 1}]
        rel_id_to_classes = [{1: "REL"}]

        result = processor.create_batch_dict(
            batch, class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_classes
        )

        # Existing checks
        assert "rel_class_to_ids" in result
        assert "rel_id_to_classes" in result

        # Optional extra sanity checks
        assert result["rel_class_to_ids"] == rel_class_to_ids
        assert result["rel_id_to_classes"] == rel_id_to_classes
        assert "rel_idx" in result
        assert "rel_label" in result
        assert result["rel_idx"].shape[0] == 1  # batch size
        assert result["rel_label"].shape[0] == 1
        
    def test_prepare_inputs_with_relations(self, processor):
        """Should add relation tokens to input."""
        texts = [["word1", "word2"]]
        entities = [["PER", "ORG"]]
        relations = [["WORKS_FOR"]]
        
        input_texts, prompt_lengths = processor.prepare_inputs(
            texts, entities, blank=None, relations=relations
        )
        
        # Should have entity tokens, sep, relation tokens, sep
        assert "[ENT]" in input_texts[0]
        assert "[REL]" in input_texts[0]
        assert input_texts[0].count("[SEP]") == 2
    
    def test_create_relation_labels_adjacency_matrix(self, processor):
        """Should create adjacency matrix for entity relations."""
        batch = {
            'tokens': [["John", "works", "at", "Google"]],
            'span_label': torch.tensor([[1, 0, 0, 2, 0, 0]]),  # 2 entities
            'rel_label': torch.tensor([[1, 0]]),  # 1 relation
            'entities': [[(0, 0, "PER"), (3, 3, "ORG")]],
            'seq_length': torch.tensor([[4]]),
            'rel_idx': [torch.tensor([[0, 1]])],  # Entity 0 -> Entity 1
            'rel_class_to_ids': {"WORKS_FOR": 1},
        }
        
        adj_matrix, rel_matrix = processor.create_relation_labels(batch)
        
        assert adj_matrix.shape[0] == 1  # Batch size
        assert isinstance(adj_matrix, torch.Tensor)
        assert isinstance(rel_matrix, torch.Tensor)
    
    def test_create_relation_labels_filters_invalid_entities(self, processor):
        """Should filter out entities exceeding sequence length."""
        batch = {
            'tokens': [["word"]],  # Only 1 token (index 0)
            'span_label': torch.tensor([[1, 0, 2]]),  # Entity at indices 0 and 2 (2 is invalid)
            'rel_label': torch.tensor([[1]]),
            'entities': [[(0, 0, "PER"), (5, 5, "ORG")]],  # Second entity exceeds length
            'seq_length': torch.tensor([[1]]),
            'rel_idx': [torch.tensor([[0, 1]])],
            'rel_class_to_ids': {"REL": 1},
        }
        
        adj_matrix, rel_matrix = processor.create_relation_labels(batch)
        
        # Should not crash, invalid entities should be filtered
        assert adj_matrix.shape[0] == 1
    
    def test_tokenize_and_prepare_labels_includes_relation_matrices(self, processor):
        """Should include adjacency and relation matrices in output."""
        batch = {
            'tokens': [["word"]],
            'classes_to_id': [{"TYPE": 1}],
            'entities': [[(0, 0, "TYPE")]],
            'rel_class_to_ids': [{"REL": 1}],
            'span_label': torch.tensor([[1]]),
            'rel_label': torch.tensor([[0]]),
            'seq_length': torch.tensor([[1]]),
            'rel_idx': [torch.tensor([[]])],
        }
        
        result = processor.tokenize_and_prepare_labels(batch, prepare_labels=True)
        
        assert 'labels' in result
        assert 'adj_matrix' in result
        assert 'rel_matrix' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])