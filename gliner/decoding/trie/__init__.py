from ...utils import is_module_available

if is_module_available("pyximport"):
    import pyximport # type: ignore
    pyximport.install() # type: ignore
    try:
        from gliner.decoding.trie.labels_trie import LabelsTrie
    except:
        from .python_labels_trie import LabelsTrie
else:
    from .python_labels_trie import LabelsTrie