from typing import List, Optional, Union, Tuple
import pathlib, io
import numpy as np
import torch
# from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin # type: ignore
from transformers.tokenization_utils_base import ( # type: ignore
    BatchEncoding, PaddingStrategy, TruncationStrategy,
)
from transformers.utils import TensorType # type: ignore

from .document_processor import PDFProcessor, LayoutImageProcessor
from .constants import EMPTY_BBOX

class GLiNERPDFProcessor(ProcessorMixin):
    r"""
    Constructs a UDOP processor which combines a LayoutLMv3 image processor and a UDOP tokenizer into a single processor.

    Args:
        image_processor (`LayoutLMv3ImageProcessor`):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`UdopTokenizer` or `UdopTokenizerFast`):
            An instance of [`UdopTokenizer`] or [`UdopTokenizerFast`]. The tokenizer is a required input.
        document_processor (`PDFProcessor`):
            An instance of [`PDFProcessor`]. The document processor is an optional input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("AutoTokenizer")

    def __init__(
        self, 
        gliner_config, 
        image_processor, 
        tokenizer, 
        document_processor: Optional[Union[PDFProcessor, LayoutImageProcessor]]=None,
        image_size: int=224,
        patch_size: int=16
    ):
        super().__init__(image_processor, tokenizer)
        self.gliner_config = gliner_config
        self.document_processor = document_processor or PDFProcessor()
        self.image_size = image_size
        self.patch_size = patch_size


    def __call__(
        self,
        path_or_fp: Union[str, pathlib.Path, io.BufferedReader, io.BytesIO],
        words: Optional[List[str]] = None,
        words_bbox: Optional[List[List[int]]] = None,
        entities: Optional[List[str]] = None,
        pages: Union[List[int], Tuple[int], None]=None,
        password: Optional[str]=None,
        add_special_tokens: bool=True,
        padding: Union[bool, str, PaddingStrategy]=False,
        truncation: Union[bool, str, TruncationStrategy]=False,
        return_overflowing_tokens: Optional[bool] = True,
        max_length: Optional[int]=None,
        return_tensors: Optional[Union[str, TensorType]]=None,
    ) -> BatchEncoding:
        processed_document = self.document_processor.process(
            path_or_fp, words = words, words_bbox= words_bbox, pages=pages, password=password
        )

        features = self.image_processor(
            images=processed_document["images"], return_tensors=return_tensors
        )

        batch_words = [processed_document['words']]
        batch_words_bbox = [processed_document['words_bbox']]
        if not isinstance(entities, dict):
            batch_entities = [entities]
        else:
            batch_entities = entities
        encoded_inputs = self.tokenize_and_align(batch_words, batch_entities, batch_words_bbox,
                                                 add_special_tokens = add_special_tokens, padding=padding, truncation=truncation,
                                                 max_length=max_length)
        # add pixel values
        pixel_values = features.pop("pixel_values")
        if return_overflowing_tokens is True:
            pixel_values = self.get_overflowing_images(
                pixel_values, encoded_inputs["overflow_to_sample_mapping"]
            )
        encoded_inputs["pixel_values"] = torch.from_numpy(np.array(pixel_values))
        patched_bbox = [
            self.patches_bbox(bbox) for bbox in processed_document["images_bbox"]
        ]
        encoded_inputs["images_bbox"] = torch.tensor(patched_bbox)

        return encoded_inputs

    def add_prompt(self, texts, entities, words_bbox):
        input_texts = []
        prompt_lengths = []
        words_bbox_ext = []
        for id, text in enumerate(texts):
            input_text = []
            curr_words_bbox = []
            if type(entities)==dict:
                entities_=entities
            else:
                entities_=entities[id]
            for ent in entities_:
                input_text.append(self.gliner_config.ent_token)
                input_text.append(ent)
                curr_words_bbox.append(list(EMPTY_BBOX))
                curr_words_bbox.append(list(EMPTY_BBOX))

            input_text.append(self.gliner_config.sep_token)
            
            curr_words_bbox.append(list(EMPTY_BBOX))
            curr_words_bbox.extend(words_bbox[id])

            prompt_length = len(input_text)
            prompt_lengths.append(prompt_length)

            input_text.extend(text)
            input_texts.append(input_text)
            words_bbox_ext.append(curr_words_bbox)
        return input_texts, prompt_lengths, words_bbox_ext
    
    def tokenize_and_align(self, texts, entities, words_bbox,
                                 add_special_tokens = True, 
                                 padding = 'longest',
                                 truncation = True,
                                 max_length = 1024):
        texts, prompt_lengths, words_bbox_ext = self.add_prompt(texts, entities, words_bbox)
        tokenized_inputs = self.tokenizer(texts, is_split_into_words = True, return_tensors='pt',
                                            add_special_tokens = add_special_tokens, truncation=truncation,
                                                                    padding=padding, max_length = max_length)
        
        if "token_type_ids" in tokenized_inputs:
            token_type_ids = tokenized_inputs.pop('token_type_ids')

        words_masks = []
        aligned_words_bbox = []
        for id in range(len(texts)):
            if prompt_lengths is not None:
                prompt_length = prompt_lengths[id]
            else:
                prompt_length = 0
            words_mask = []
            curr_words_bbox = []
            prev_word_id=None
            words_count=0
            for word_id in tokenized_inputs.word_ids(id):
                if word_id is None:
                    words_mask.append(0)
                    curr_words_bbox.append(list(EMPTY_BBOX))
                elif word_id != prev_word_id:
                    if words_count<prompt_length:
                        words_mask.append(0)
                        curr_words_bbox.append(list(EMPTY_BBOX))
                    else:
                        masking_word_id = word_id-prompt_length+1
                        words_mask.append(masking_word_id)
                        curr_words_bbox.append(words_bbox_ext[id][word_id])
                    words_count+=1
                else:
                    words_mask.append(0)
                    curr_words_bbox.append(words_bbox_ext[id][word_id])
                prev_word_id = word_id
            words_masks.append(words_mask)
            aligned_words_bbox.append(curr_words_bbox)
        tokenized_inputs['words_mask'] = torch.tensor(words_masks)
        tokenized_inputs['bbox'] = torch.tensor(aligned_words_bbox)
        return tokenized_inputs
    
    # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.get_overflowing_images
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # in case there's an overflow, ensure each `input_ids` sample is mapped to its corresponding image
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow


    def patches_bbox(
        self, image_bbox: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        x0, y0, x1, y1 = image_bbox
        w = x1 - x0
        pwc = w/(self.image_size//self.patch_size)
        h = y1 - y0
        phc = h/(self.image_size//self.patch_size)

        bbox: List[Tuple[int, int, int, int]] = []
        for yp in range(self.image_size//self.patch_size):
            yp0 = min(y0 + yp*phc, y1)
            yp1 = min(yp0 + phc, y1)
            for xp in range(self.image_size//self.patch_size):
                xp0 = min(x0 + xp*pwc, x1)
                xp1 = min(xp0 + pwc, x1)
                bbox.append((int(xp0), int(yp0), int(xp1), int(yp1)))
        return bbox


    # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.batch_decode
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)


    # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "pixel_values", "images_bbox"]