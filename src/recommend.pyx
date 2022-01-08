# cython: language_level=3, boundscheck=False, wraparound=False

import os
import sys
import json
import itertools
import numpy as np
cimport cython
cimport numpy as np


cdef dict get_patient_matrix(str patient_id, dict utility_tensor, np.ndarray utility_tensor_index):
    cdef:
        size_t x, y, z, start, end

    x = int(patient_id) - 1
    start = utility_tensor_index[x, 0]
    end = utility_tensor_index[x, 1]

    return {(y, z): utility_tensor[x, y, z] for x, y, z in itertools.islice(utility_tensor, start, end)}


cdef int main(str filepath, str arg_patient_id, str arg_pc_kind):
    cdef:
        size_t i, j, k, x, y, z, index_count_start, index_count_current, num_patients, num_conditions, num_therapies
        float previous_success, new_success
        str pc_id, pc_kind, tr_pc_id, tr_th_id
        set remaining_ks, matching_ks
        list pconditions, trials
        dict dataset, patient, utility_tensor
        np.ndarray utility_tensor_index

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
    utility_tensor_index = np.empty((num_patients, 2), dtype=np.uintc)
    index_count_start = 0
    index_count_current = 0
    for i in range(num_patients): # iterate over patients
        patient = dataset['Patients'][i]
        pconditions = patient['conditions']
        trials = patient['trials']
        remaining_ks = set(range(len(trials)))
        for j in range(len(pconditions)): # iterate over patient's conditions
            pc_id = pconditions[j]['id']
            pc_kind = pconditions[j]['kind']
            previous_success = 0.0
            matching_ks = set()
            for k in remaining_ks: # efficiently iterate over corresponding trials
                tr_pc_id = trials[k]['condition']
                if pc_id == tr_pc_id:
                    tr_th_id = trials[k]['therapy']
                    new_success = int(trials[k]['successful'].strip('%')) * 0.01
                    x, y, z = i, int(pc_kind.strip('Cond')) - 1, int(tr_th_id.strip('Th')) - 1
                    if (x, y, z) not in utility_tensor:
                        utility_tensor[x, y ,z] = new_success - previous_success
                        index_count_current += 1
                    else: # in the past, same patient already had same therapy for same kind of condition; avg. biased towards later instances
                        utility_tensor[x, y, z] = (new_success - previous_success + utility_tensor[x, y, z]) / 2
                    previous_success = new_success
                    matching_ks.add(k)
            remaining_ks = remaining_ks.difference(matching_ks)
        utility_tensor_index[i, 0] = index_count_start
        utility_tensor_index[i, 1] = index_count_current
        index_count_start = index_count_current
    print('-> Created raw utility tensor')
    #print(list(utility_tensor.items())[300:310]) # TODO debug
    #print(list(enumerate(utility_tensor_index[:200])))

    # arguments' values TODO:
    patient = dataset['Patients'][int(arg_patient_id) - 1]
    pconditions = patient['conditions']
    trials = patient['trials']
    assert arg_patient_id == patient['id']
    condition = dataset['Conditions'][int(arg_pc_kind.strip('Cond')) - 1]
    assert arg_pc_kind == condition['id']

    # patient similarites:
    # TODO thexash?
    # now first: feature-agnostic collaborative filtering (baseline)
    print(get_patient_matrix(arg_patient_id, utility_tensor, utility_tensor_index))

    # finish:
    print('Everything okay!')

    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2], sys.argv[3])
