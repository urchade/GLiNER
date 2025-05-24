from typing import List

from pdfminer.pdfpage import PDFPage
from pdfplumber.pdf import PDF
from pdfplumber._typing import T_num

from .page import CustomPage

class CustomPDF(PDF):
    @property
    def pages(self) -> List[CustomPage]: # type: ignore
        if hasattr(self, "_pages"):
            return self._pages

        doctop: T_num = 0
        pp = self.pages_to_parse
        self._pages: List[CustomPage] = [] # type: ignore
        for i, page in enumerate(PDFPage.create_pages(self.doc)):
            page_number = i + 1
            if pp is not None and page_number not in pp:
                continue
            p = CustomPage(
                self, 
                page, 
                page_number=page_number, 
                initial_doctop=doctop
            )
            self._pages.append(p)
            doctop += p.height
        return self._pages