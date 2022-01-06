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
        dict utility_tensor
        set remaining_ks, matching_ks
        str pp, pc_id, pc_kind, pc_cured, tr_pc_id, tr_th_id
        size_t i, j, k, number_of_conditions, number_of_therapies, number_of_patients

    # parse dataset:
    parser = cysimdjson.JSONParser()
    dataset = parser.load(filepath)
    number_of_conditions = len(dataset.at_pointer('/Conditions'))
    number_of_therapies = len(dataset.at_pointer('/Therapies'))
    number_of_patients = len(dataset.at_pointer('/Patients'))
    print('Successfully parsed %s' % filepath)

    # iterate over patient's conditions: TODO debug
    pp = '/Patients/%d' % (int(patient_id) - 1)
    for i in range(len(dataset.at_pointer('%s/conditions' % pp))):
        print(dataset.at_pointer('%s/conditions/%d/id' % (pp, i)))

    # show attributes of patient: TODO debug
    print([x for x in dataset.at_pointer(pp).keys()]) # keep this and for other kinds also keya

    # create utility tensor *for simple baseline*:
    utility_tensor = {}

    for i in range(1000): # TODO actually: range(number_of_patients):
        print(i, end='\r') # TODO see where we are
        pp = '/Patients/%d' % i
        remaining_ks = set(range(len(dataset.at_pointer('%s/trials' % pp))))
        for j in range(len(dataset.at_pointer('%s/conditions' % pp))):
            pc_cured = dataset.at_pointer('%s/conditions/%d/cured' % (pp, j))
            pc_id = dataset.at_pointer('%s/conditions/%d/id' % (pp, j))
            pc_kind = dataset.at_pointer('%s/conditions/%d/kind' % (pp, j))
            matching_ks = set()
            for k in remaining_ks:
                tr_pc_id = dataset.at_pointer('%s/trials/%d/condition' % (pp, k))
                if tr_pc_id == pc_id:
                    tr_th_id = dataset.at_pointer('%s/trials/%d/therapy' % (pp, k))
                    utility_tensor[i, int(pc_kind.strip('Cond')) - 1, int(tr_th_id.strip('Th')) - 1] = 1.0 if pc_cured else 0.0
                    matching_ks.add(k)
            remaining_ks = remaining_ks.difference(matching_ks)

    print(utility_tensor)

    # patient similarites:
    # TODO thexash?

    # finish:
    print('Everything okay!')
    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2], sys.argv[3])
