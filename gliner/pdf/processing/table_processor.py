from typing import List, Any, Tuple, Optional

from pdfplumber._typing import T_obj_list, T_dir
from pdfplumber.table import Table
from pdfplumber.utils.text import (
    TextMap, DEFAULT_X_TOLERANCE, DEFAULT_Y_TOLERANCE, get_line_cluster_key
)
from pdfplumber.utils.clustering import cluster_objects

from .constants import EMPTY_TABLE_CELL, CELL_NEW_LINE
from .word_extractor import extract_words


class TablesProcessor:
    def __init__(
        self, 
        line_dir_render: T_dir,
        char_dir_render: T_dir,
        **kwargs: Any
    ) -> None:
        self.line_dir_render = line_dir_render
        self.char_dir_render = char_dir_render
        self.kwargs = kwargs

    @classmethod
    def get_markdown_table(cls, table: List[List[Optional[str]]]) -> str:
        table_md = ""
        for row in table:
            table_md += "| "
            for cell in row:
                table_md += cell.replace("\n", CELL_NEW_LINE) if cell else EMPTY_TABLE_CELL + " |"
            table_md += "\n"
        return table_md


    def extract_text(
        self, words: T_obj_list,
    ) -> str:
        if len(words) == 0:
            return ""

        line_cluster_key = get_line_cluster_key(self.line_dir_render) # type: ignore

        x_tolerance = self.kwargs.get("x_tolerance", DEFAULT_X_TOLERANCE)
        y_tolerance = self.kwargs.get("y_tolerance", DEFAULT_Y_TOLERANCE)

        lines = cluster_objects(
            words,
            line_cluster_key,
            y_tolerance if self.line_dir_render in ("ttb", "btt") else x_tolerance,
        )

        return TextMap(
            [
                (char, None)
                for char in (
                    "\n".join(" ".join(word["text"] for word in line) for line in lines)
                )
            ],
            line_dir_render=self.line_dir_render, # type: ignore
            char_dir_render=self.char_dir_render, # type: ignore
        ).as_string


    def extract(self, table: Table, words: T_obj_list) -> Tuple[str, T_obj_list]:
        # From pdfplumber Table
        table_arr: List[List[Optional[str]]] = []
        unprocessed_words: T_obj_list = words
        for row in table.rows:
            arr: List[Optional[str]] = []
            unprocessed_words, row_words = extract_words(row.bbox, unprocessed_words)

            for cell in row.cells:
                if cell is None:
                    cell_text = None
                else:
                    row_words, cell_words = extract_words(cell, row_words)

                    if len(cell_words):
                        if "layout" in self.kwargs:
                            self.kwargs["layout_width"] = cell[2] - cell[0]
                            self.kwargs["layout_height"] = cell[3] - cell[1]
                            self.kwargs["layout_bbox"] = cell
                        cell_text = self.extract_text(
                            cell_words
                        )
                    else:
                        cell_text = ""
                arr.append(cell_text)
            table_arr.append(arr)
        return self.get_markdown_table(table_arr), unprocessed_words
    

    def extract_tables(
        self, tables: List[Table], words: T_obj_list
    ) -> Tuple[List[str], T_obj_list]:
        tables_md: List[str] = []
        for table in tables:
            table_md, words = self.extract(table, words)
            tables_md.append(table_md)
        return tables_md, words