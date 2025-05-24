from typing import Generator, Optional, Tuple

from pdfplumber.utils.text import WordExtractor
from pdfplumber._typing import (
    T_bbox, T_obj, T_obj_list, T_dir, T_obj_iter
)

def word_in_bbox(word: T_obj, bbox: T_bbox) -> bool:
    # from pdfplumber
    v_mid = (word["top"] + word["bottom"]) / 2
    h_mid = (word["x0"] + word["x1"]) / 2
    x0, top, x1, bottom = bbox
    return (
        (h_mid >= x0) 
        and (h_mid < x1) 
        and (v_mid >= top) 
        and (v_mid < bottom)
    )


def extract_words(
    bbox: T_bbox, chars: T_obj_list
) -> Tuple[T_obj_list, T_obj_list]:
    extracted_words: T_obj_list = []
    unprocessed_words: T_obj_list = []
    for char in chars:
        if word_in_bbox(char, bbox):
            extracted_words.append(char)
        else:
            unprocessed_words.append(char)
    return unprocessed_words, extracted_words


class CustomWordExtractor(WordExtractor):
    def iter_chars_to_words(
        self,
        ordered_chars: T_obj_iter,
        direction: T_dir,
    ) -> Generator[T_obj_list, None, None]:
        current_word: T_obj_list = []

        def start_next_word(
            new_char: Optional[T_obj],
        ) -> Generator[T_obj_list, None, None]:
            nonlocal current_word

            if current_word:
                yield current_word

            current_word = [] if new_char is None else [new_char]

        xt = self.x_tolerance
        xtr = self.x_tolerance_ratio
        yt = self.y_tolerance
        ytr = self.y_tolerance_ratio

        for char in ordered_chars:
            text = char["text"]

            if char.get("token", False): # add processing for tokens
                yield [char]
                continue

            if not self.keep_blank_chars and text.isspace():
                yield from start_next_word(None)

            elif text in self.split_at_punctuation:
                yield from start_next_word(char)
                yield from start_next_word(None)

            elif current_word and self.char_begins_new_word(
                current_word[-1],
                char,
                direction,
                x_tolerance=(xt if xtr is None else xtr * current_word[-1]["size"]),
                y_tolerance=(yt if ytr is None else ytr * current_word[-1]["size"]),
            ):
                yield from start_next_word(char)

            else:
                current_word.append(char)

        # Finally, after all chars processed
        if current_word:
            yield current_word