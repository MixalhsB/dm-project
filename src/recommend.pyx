# cython: language_level=3, boundscheck=False, wraparound=False

import os
import sys
import json
import numpy as np
cimport cython
cimport numpy as np


cdef int main(str filepath, str arg_patient_id, str arg_pc_kind):
    cdef:
        size_t i, j, k, num_patients, num_conditions, num_therapies
        float previous_success, new_success
        str pc_id, pc_kind, tr_pc_id, tr_th_id
        set remaining_ks, matching_ks
        list pconditions, trials
        dict utility_tensor, dataset, patient

    # parse dataset:
    assert os.path.exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    num_patients = len(dataset['Patients'])
    num_conditions = len(dataset['Conditions'])
    num_therapies = len(dataset['Therapies'])
    print('-> Successfully parsed %s' % filepath)

    # create utility tensor:
    utility_tensor = {}
    for i in range(num_patients):
        patient = dataset['Patients'][i]
        pconditions = patient['conditions']
        trials = patient['trials']
        remaining_ks = set(range(len(trials)))
        for j in range(len(pconditions)):
            pc_id = pconditions[j]['id']
            pc_kind = pconditions[j]['kind']
            previous_success = 0.0
            matching_ks = set()
            for k in remaining_ks:
                tr_pc_id = trials[k]['condition']
                if pc_id == tr_pc_id:
                    tr_th_id = trials[k]['therapy']
                    new_success = int(trials[k]['successful'].strip('%')) * 0.01
                    utility_tensor[i, int(pc_kind.strip('Cond')) - 1, int(tr_th_id.strip('Th')) - 1] = new_success - previous_success
                    previous_success = new_success
                    matching_ks.add(k)
            remaining_ks = remaining_ks.difference(matching_ks)
    print('-> Created raw utility tensor')
    print(list(utility_tensor.items())[300:310]) # TODO debug

    # arguments' values TODO:
    patient = dataset['Patients'][int(arg_patient_id) - 1]
    pconditions = patient['conditions']
    trials = patient['trials']
    assert arg_patient_id == patient['id']
    condition = dataset['Conditions'][int(arg_pc_kind.strip('Cond')) - 1]
    assert arg_pc_kind == condition['id']

    # patient similarites:
    # TODO thexash?
    pass

    # finish:
    print('Everything okay!')
    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2], sys.argv[3])
