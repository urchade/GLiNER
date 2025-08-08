# distutils : language = c++
# cython: language_level=3
import cython
from libcpp.vector cimport vector


from gliner.decoding.trie.trie cimport Trie

cdef class LabelsTrie:
    cdef Trie* c_trie

    def __init__(self, list: vector[vector[cython.int]]):
        if list.empty():
            self.c_trie = new Trie(NULL)
        else:
            self.c_trie = new Trie(&list)


    def add_batch(self, list: vector[vector[cython.int]]):
        self.c_trie.add_batch(&list, False)


    def add(self, list: vector[cython.int]):
        self.c_trie.add(list, False)


    def get(self, list: vector[cython.int]) -> vector[cython.int]:
        return self.c_trie.get_possible_next_keys(list)


    def remove_batch(self, list: vector[vector[cython.int]]):
        self.c_trie.remove_batch(&list)


    def remove_entity(self, list: vector[cython.int]):
        self.c_trie.remove_entity(list)


    def __dealloc__(self):
        if self.c_trie is not NULL:
            del self.c_trie