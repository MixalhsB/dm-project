# cython: language_level=3, boundscheck=False, wraparound=False

import os
import sys
import cysimdjson
import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free


cdef int main(str filepath, str patient_id, str pcondition_kind):
    assert os.path.exists(filepath)

    cdef:
        str pp
        size_t i, pc_num, number_of_conditions, number_of_therapies, number_of_patients
        np.ndarray utility_tensor

    # parse dataset:
    parser = cysimdjson.JSONParser()
    dataset = parser.load(filepath)
    number_of_conditions = len(dataset.at_pointer('/Conditions'))
    number_of_therapies = len(dataset.at_pointer('/Therapies'))
    number_of_patients = len(dataset.at_pointer('/Patients'))
    print('Successfully parsed %s' % filepath)

    #iterate over patient's conditoins: TODO debug
    pp = '/Patients/%d' % (int(patient_id) - 1)
    for i in range(len(dataset.at_pointer('%s/conditions' % pp))):
        print(dataset.at_pointer('%s/conditions/%d/id' % (pp, i)))

    #show attributes of patient: TODO debug
    print([x for x in dataset.at_pointer(pp).keys()]) # keep this and for other kinds also keya

    # TODO remove cuz experimental:
    utility_tensor = np.empty((number_of_patients, 30), dtype=np.float)

    # finish:
    print('Everything okay!')
    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2], sys.argv[3])
