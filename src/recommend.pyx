# cython: language_level=3, boundscheck=False, wraparound=False


import numpy as np
cimport numpy as np
cimport cython


cdef struct Grail: # example
    int age
    float volume


def main():
    cdef int test_int = 3
    cdef int banana = 4

    print(test_int)
    print(banana)

    # TODO

    


if __name__ == '__main__':
    main()
