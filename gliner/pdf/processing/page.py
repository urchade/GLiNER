from typing import Dict

from pdfminer.pdfpage import PDFPage
from pdfplumber.page import Page
from pdfplumber.pdf import PDF
from pdfplumber._typing import T_num, T_obj_list

from .constants import Token


class CustomPage(Page):
    def __init__(
        self,
        pdf: PDF,
        page_obj: PDFPage,
        page_number: int,
        initial_doctop: T_num = 0,
    ):
        super().__init__(pdf, page_obj, page_number, initial_doctop) 

        self.x_offset = self.bbox[0]
        self.y_offset = self.bbox[1]
        self._height = self.bbox[3] - self.bbox[1]
        self._width = self.bbox[2] - self.bbox[0]


    @property
    def width(self) -> T_num:
        return self._width


    @property
    def height(self) -> T_num:
        return self._height

    
    def parse_objects(self) -> Dict[str, T_obj_list]:
        objects: Dict[str, T_obj_list] = {}
        for obj in self.iter_layout_objects(self.layout._objs): # type: ignore
            kind = obj["object_type"]
            if kind in ["anno"]:
                continue
            if objects.get(kind) is None:
                objects[kind] = []
            objects[kind].append(obj)
            if kind == "image":
                objects["char"].append({
                    **obj,
                    "text": Token.IMAGE.value,
                    "token": True,
                    "upright": True
                })
        return objects