from typing import Any, Dict, List, Optional, Tuple, Union, Generator, cast
import io
import pathlib

from pdfplumber._typing import T_bbox
from pdfplumber.table import T_table_settings
from PIL import Image

from .document import CustomPDF
from .page import CustomPage
from .word_extractor import CustomWordExtractor
from .table_processor import TablesProcessor
from .constants import MAX_COORDINATE, MIN_COORDINATE, EMPTY_BBOX, Token


class PDFProcessor:
    def __init__(
        self,
        laparams: Optional[Dict[str, Any]]=None, 
        strict_metadata: bool=False, 
        repair: bool=False,
        table_settings: Optional[T_table_settings]=None, 
        word_extractor_settings: Optional[Dict[str, Any]]=None,
        table_extractor_settings: Optional[Dict[str, Any]]=None,
    ) -> None:
        self.laparams = laparams
        self.strict_metadata = strict_metadata
        self.repair = repair
        self.table_settings = table_settings
        self.word_extractor_settings = word_extractor_settings or {}
        self.table_extractor_settings = table_extractor_settings or {}
    

    @classmethod
    def images(cls, page: CustomPage) -> Generator[Tuple[Image.Image, T_bbox], None, None]:
        for image_obj in page.images:
            x0, y0, x1, y1 = (image_obj["x0"], image_obj["top"], image_obj["x1"], image_obj["bottom"])
            image = Image.open(io.BytesIO(image_obj['stream'].rawdata))
            yield (
                image,
                cls.normalize_coordinates((x0, y0, x1, y1), page)
            )

    
    def words(
        self, page: CustomPage 
    ) -> Generator[Tuple[str, T_bbox], None, None]:
        tables = page.find_tables(self.table_settings)
        word_extractor = CustomWordExtractor(**self.word_extractor_settings)
        words = word_extractor.extract_words(page.chars)
        extracted_tables, words = TablesProcessor(
            word_extractor.line_dir, # type: ignore
            word_extractor.char_dir, # type: ignore
            **self.table_extractor_settings
        ).extract_tables(
            tables, words, 
        )

        current_table = 0
        for word in words:
            while True:
                if current_table >= len(tables) or word["top"] < tables[current_table].bbox[1]:
                    break

                if (
                    word["top"] != tables[current_table].bbox[1] 
                    or word["x0"] >= tables[current_table].bbox[0]
                ):
                    yield extracted_tables[current_table], self.normalize_coordinates(
                        tables[current_table].bbox, page
                    )
                    current_table += 1
                else:
                    break
            
            if word.get("token", False):
                yield word["text"], EMPTY_BBOX
            else:
                yield word["text"], self.normalize_coordinates(
                    (word["x0"], word["top"], word["x1"], word["bottom"]),
                    page
                )


    @classmethod
    def normalize_coordinates(cls, bbox: T_bbox, page: CustomPage) -> T_bbox:
        x0, y0, x1, y1 = bbox
        # similar used in lmv3
        return cast(
            T_bbox, 
            tuple(min(max(i, MIN_COORDINATE), MAX_COORDINATE) for i in (
                int(MAX_COORDINATE * ((x0 - page.x_offset) / page.width)),
                int(MAX_COORDINATE * ((y0 - page.y_offset) / page.height)),
                int(MAX_COORDINATE * ((x1 - page.x_offset) / page.width)),
                int(MAX_COORDINATE * ((y1 - page.y_offset) / page.height)),
            )
        ))
    

    def process(
        self,
        path_or_fp: Union[str, pathlib.Path, io.BufferedReader, io.BytesIO],
        words: Optional[List[str]] = None,
        words_bbox: Optional[List[List[int]]] = None, 
        pages: Union[List[int], Tuple[int], None]=None,
        password: Optional[str]=None,
        **kwargs
    ) -> Dict[str, List[Any]]:
        doc = CustomPDF.open(
            path_or_fp=path_or_fp,
            pages=pages,
            laparams=self.laparams,
            password=password,
            strict_metadata=self.strict_metadata,
            repair=self.repair
        )
        tokenized_doc: Dict[str, List[Any]] = {
            "words": [],
            "words_bbox": [],
            "images": [],
            "images_bbox": []
        }
        for page in doc.pages:
            if words is not None and words_bbox is not None:
                tokenized_doc["words"].extend(words)
                tokenized_doc["words_bbox"].extend(words_bbox)
            else:
                for word, bbox in self.words(page): # type: ignore
                    tokenized_doc["words"].append(word)
                    tokenized_doc["words_bbox"].append(bbox)
            tokenized_doc["words"].append(Token.PAGE.value)
            tokenized_doc["words_bbox"].append(EMPTY_BBOX)

            for image, bbox in self.images(page): # type: ignore
                tokenized_doc["images"].append(image)
                tokenized_doc["images_bbox"].append(bbox)
            tokenized_doc["images"].append(Image.new("RGB", (2, 2)))
            tokenized_doc["images_bbox"].append(EMPTY_BBOX)
        return tokenized_doc

class LayoutImageProcessor:
    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size
    
    def process(
        self,
        path_or_fp: Union[str, pathlib.Path, io.BufferedReader, io.BytesIO], 
        words: List[str],
        words_bbox: List[List[int]],
        **kwargs
    ) -> Dict[str, List[Any]]:
        image = Image.open(path_or_fp)
        image = image.convert("RGB")
        image = image.resize(self.image_size)

        words_list = list(words)
        bbox_list = list(words_bbox)
        words_list.append(Token.PAGE.value)
        bbox_list.append(EMPTY_BBOX)

        images: List[Image.Image] = []
        images_bbox: List[List[int]] = []

        images.append(image)
        images_bbox.append([0, 0, self.image_size[0], self.image_size[1]])

        images.append(Image.new("RGB", (2, 2)))
        images_bbox.append(EMPTY_BBOX)

        return {
            "words":         words_list,
            "words_bbox":    bbox_list,
            "images":        images,
            "images_bbox":   images_bbox,
        }