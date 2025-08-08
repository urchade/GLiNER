from ...utils import is_module_available

if is_module_available("pyximport"):
    import pyximport # type: ignore
    pyximport.install() # type: ignore
    from .labels_trie import LabelsTrie

else:
    from .python_labels_trie import LabelsTrie