import cython
from libcpp.vector cimport vector

cdef extern from "trie.h" namespace "cpp_trie":
    cdef cppclass Trie:
        Trie() except +
        Trie(vector[vector[cython.int]]*) except +
        void add_batch(vector[vector[cython.int]]*, cython.bint) except +
        void add(vector[cython.int]&, cython.bint) except +
        vector[cython.int] get_possible_next_keys(vector[cython.int]&) except +
        void remove_batch(vector[vector[cython.int]]*) except +
        void remove_entity(vector[cython.int]&) except +


cdef extern from "trie.cpp":
    pass