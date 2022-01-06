# cython: language_level=3, boundscheck=False, wraparound=False

import os
import sys
import cysimdjson
import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free


cdef struct Patient:
    (char *) id
    (char **) pcondition_ids, trial_ids, names_of_further_attributes, values_of_further_attributes

cdef struct PCondition:
    (char *) id, kind, type
    int diagnosed, cured

cdef struct Trial:
    (char *) id, pcondition_id, therapy_id
    int start, end, successful

cdef struct Therapy:
    (char *) id, type


cdef int main(str filepath, bytes patient_id, bytes pcondition_kind):
    assert os.path.exists(filepath)

    cdef:
        str pp, pc_id_str
        bytes pc_id_bytes
        size_t i, pc_num, number_of_conditions, number_of_therapies, number_of_patients
        np.ndarray utility_tensor
        Patient patient
        PCondition pcondition

    # parse dataset:
    parser = cysimdjson.JSONParser()
    dataset = parser.load(filepath)
    number_of_conditions = len(dataset.at_pointer('/Conditions'))
    number_of_therapies = len(dataset.at_pointer('/Therapies'))
    number_of_patients = len(dataset.at_pointer('/Patients'))
    print('Successfully parsed %s' % filepath)


    # load in patient info: TODO tbc or abort?
    assert patient_id.isdigit()
    pp = '/Patients/%d' % (int(patient_id) - 1)
    patient.id = patient_id
    pc_num = len(dataset.at_pointer('%s/conditions' % pp))
    patient.pcondition_ids = <char **>malloc(pc_num * 16 * sizeof(char)) # assuming len('pc...') <= 16
    for i in range(pc_num):
        pc_id_str = dataset.at_pointer('%s/conditions/%d/id' % (pp, i))
        pc_id_bytes = pc_id_str.encode()
        assert(len(pc_id_bytes) <= 16)
        patient.pcondition_ids[i] = pc_id_bytes
    free(patient.pcondition_ids)

    # TODO remove cuz experimental:
    utility_tensor = np.empty((number_of_patients, 30), dtype=np.float)


    # finish:
    print('Everything okay!')
    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2].encode(), sys.argv[3].encode())
