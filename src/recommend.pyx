# cython: language_level=3, boundscheck=False, wraparound=False

import cysimdjson
import numpy as np
cimport cython
cimport numpy as np


cdef str filename
cdef size_t number_of_patients
cdef np.ndarray utility_matrix


def main():
    filename = '../data/dataset.json' # TODO make argv-style
    parser = cysimdjson.JSONParser()
    dataset = parser.load(filename) # TODO consider case it's not there, argv etc.
    print('Successfully parsed %s\n' % filename)

    # TODO remove test below
    print(dataset.at_pointer('/Conditions/0/name'))

    number_of_patients = len(dataset.at_pointer('/Patients'))
    utility_matrix = np.empty((number_of_patients, 100), dtype=np.float) # 100 is arbitrary for now TODO

    print(utility_matrix[55555])


if __name__ == '__main__':
    main()
