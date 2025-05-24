import enum

class Token(enum.Enum):
    PAGE = '<page>'
    IMAGE = '<image>'

EMPTY_BBOX = (0, 0, 0, 0)
MAX_COORDINATE = 1000
MIN_COORDINATE = 0
EMPTY_TABLE_CELL = ""
CELL_NEW_LINE = "<br/>"