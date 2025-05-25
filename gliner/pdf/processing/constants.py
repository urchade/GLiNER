import enum

class Token(enum.Enum):
    PAGE = '<<PAGE>>'
    IMAGE = '<<IMG>>'

EMPTY_BBOX = (0, 0, 0, 0)
MAX_COORDINATE = 1000
MIN_COORDINATE = 0
EMPTY_TABLE_CELL = ""
CELL_NEW_LINE = "<br/>"