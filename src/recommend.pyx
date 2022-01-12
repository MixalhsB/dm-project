# cython: language_level=3, boundscheck=False, wraparound=False

import os
import sys
import json
import pickle
import kmedoids
import itertools
import numpy as np
cimport cython
cimport numpy as np


cdef float cosine_distance(size_t patient_x_1, size_t patient_x_2, dict utility_tensor):
    cdef:
        size_t i, j, y1, z1, y2, z2
        dict patient_matrix_1, patient_matrix_2
        float dot_product, sum_squares_1, sum_squares_2, value_1, value_2

    patient_matrix_1 = utility_tensor[patient_x_1]
    patient_matrix_2 = utility_tensor[patient_x_2]
    dot_product = sum_squares_1 = sum_squares_2 = 0.0
    for i, y1 in enumerate(patient_matrix_1):
        for j, z1 in enumerate(patient_matrix_1[y1]):
            value_1 = patient_matrix_1[y1][z1]
            sum_squares_1 += value_1 ** 2
            for y2 in patient_matrix_2:
                for z2 in patient_matrix_2[y2]:
                    value_2 = patient_matrix_2[y2][z2]
                    if i == 0 and j == 0:
                        sum_squares_2 += value_2 ** 2
                    if y1 == y2 and z1 == z2:
                        dot_product += value_1 * value_2

    if sum_squares_1 == 0 or sum_squares_2 == 0:
        return 1.0

    return 1.0 - dot_product / np.sqrt(sum_squares_1 * sum_squares_2)


cdef np.ndarray cluster_patients(dict utility_tensor, size_t num_patients, size_t k, size_t n, size_t s, str filepath):
    cdef:
        size_t i, j, l, best_i, med, closest_med
        float cos_dist, lowest_cos_dist
        tuple sorted_pair
        dict memorized
        list filepath_split
        str data_dir, res_dir, filename, p2c_path
        np.ndarray sample, sample_dist_matrix, results, medoids, patients_to_clusters

    np.random.seed(123)
    filepath_split = filepath.replace('\\', '/').rsplit('/', 1)
    filepath_split = ['.'] + filepath_split if len(filepath_split) < 2 else filepath_split
    data_dir, filename = filepath_split
    res_dir = data_dir + '/../results/'
    p2c_path = '%s%s_p2c_%d_%d_%d.pickle' % (res_dir, filename.rstrip('.json'), k, n, s)

    if os.path.exists(p2c_path):
        with open(p2c_path, 'rb') as f:
            patients_to_clusters = pickle.load(f)
        print('-> Loaded pre-computed clusters from ../' + p2c_path.rsplit('../', 1)[1])
        return patients_to_clusters

    results = np.empty((n, 2), dtype=object)
    memorized = {}
    for i in range(n):
        print('-> Clustering sample', i + 1, '/', n, '...', end='\r')
        sample = np.random.choice(num_patients, s, replace=False)
        results[i][0] = sample
        sample_dist_matrix = np.empty((s, s), dtype=float)
        for j in range(s):
            for l in range(j, s):
                sorted_pair = tuple(sorted((sample[j], sample[l])))
                if j == l:
                    sample_dist_matrix[j][l] = sample_dist_matrix[l][j] = memorized[sorted_pair] = 0.0
                elif sorted_pair in memorized:
                    sample_dist_matrix[j][l] = sample_dist_matrix[l][j] = memorized[sorted_pair]
                else:
                    cos_dist = cosine_distance(sample[j], sample[l], utility_tensor)
                    sample_dist_matrix[j][l] = sample_dist_matrix[l][j] = memorized[sorted_pair] = cos_dist
        results[i][1] = kmedoids.fasterpam(sample_dist_matrix, k)
    best_i = min(enumerate(results), key=lambda x: x[1][1].loss)[0]
    sample = results[best_i][0]
    medoids = sample[results[best_i][1].medoids]
    patients_to_clusters = np.empty(num_patients, dtype=np.uintc)
    for j in range(s):
        patients_to_clusters[sample[j]] = medoids[results[best_i][1].labels[j]]
    print()
    for i in range(num_patients):
        if i in sample: # patients from winning sample have already been assigned, including all medoids
            continue
        np.random.shuffle(medoids) # to avoid general bias in case of frequently equidistant medoids
        closest_med = k # impossibly large initial index
        lowest_cos_dist = 1.1 # impossibly large initial value
        print('-> Assigning patients to clusters', i + 1, '/', num_patients, '...', end='\r')
        for med in medoids:
            sorted_pair = tuple(sorted((i, med)))
            if sorted_pair in memorized:
                cos_dist = memorized[sorted_pair]
            else:
                cos_dist = memorized[sorted_pair] = cosine_distance(i, med, utility_tensor)
            if cos_dist < lowest_cos_dist:
                closest_med = med
                lowest_cos_dist = cos_dist
        patients_to_clusters[i] = closest_med
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    with open(p2c_path, 'wb') as f:
        pickle.dump(patients_to_clusters, f)
        print('\n-> Saved computed clusters to ../' + p2c_path.rsplit('../', 1)[1])

    return patients_to_clusters


cdef dict condense_utilities(dict utility_tensor, np.ndarray patients_to_clusters):
    cdef:
        size_t i, med, y, z, num_clustered_patients
        dict condensed_utility_tensor, agglomerated_utility_tensor, patient_matrix

    agglomerated_utility_tensor = {}
    condensed_utility_tensor = {}
    for i in range(patients_to_clusters.size): # a.k.a. range(num_patients)
        med = patients_to_clusters[i]
        if med not in agglomerated_utility_tensor:
            agglomerated_utility_tensor[med] = [utility_tensor[i]]
        else:
            agglomerated_utility_tensor[med].append(utility_tensor[i])
    for med in agglomerated_utility_tensor: # sum patient-specific utilities
        for patient_matrix in agglomerated_utility_tensor[med]:
            for y in patient_matrix:
                for z in patient_matrix[y]:
                    if med not in condensed_utility_tensor:
                        condensed_utility_tensor[med] = {y: {z: patient_matrix[y][z]}}
                    elif y not in condensed_utility_tensor[med]:
                        condensed_utility_tensor[med][y] = {z: patient_matrix[y][z]}
                    elif z not in condensed_utility_tensor[med][y]:
                        condensed_utility_tensor[med][y][z] = patient_matrix[y][z]
                    else:
                        condensed_utility_tensor[med][y][z] += patient_matrix[y][z]
    for med in condensed_utility_tensor: # average patient-specific utilities
        num_clustered_patients = len(agglomerated_utility_tensor[med])
        for y in condensed_utility_tensor[med]:
            for z in condensed_utility_tensor[med][y]:
                condensed_utility_tensor[med][y][z] /= num_clustered_patients
    print('-> Computed condensed utility tensor')

    return condensed_utility_tensor


cdef int main(str filepath, str arg_patient_id, str arg_pc_id):
    cdef:
        size_t i, j, k, x, y, z, num_patients, num_conditions, num_therapies
        float previous_success, new_success
        str filename, pc_id, pc_kind, tr_pc_id, tr_th_id
        set remaining_ks, matching_ks
        list pconditions, trials
        dict dataset, patient, pcond, condition, utility_tensor, condensed_utility_tensor
        np.ndarray patients_to_clusters

    # parse dataset:
    assert os.path.exists(filepath) and arg_patient_id.isdigit() and arg_pc_id.lstrip('pc').isdigit()
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    num_patients = len(dataset['Patients'])
    num_conditions = len(dataset['Conditions']) # TODO redundant?
    num_therapies = len(dataset['Therapies']) # TODO redundant?
    filename = filepath.replace('\\', '/').rsplit('/', 1)[1] if '/' in filepath or '\\' in filepath else filepath
    print('-> Successfully parsed ' + filename)

    # create utility tensor:
    utility_tensor = {}
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
            for k in sorted(remaining_ks): # efficiently iterate over corresponding trials
                tr_pc_id = trials[k]['condition']
                if pc_id == tr_pc_id:
                    tr_th_id = trials[k]['therapy']
                    new_success = int(trials[k]['successful'].rstrip('%')) * 0.01 # TODO success as str work for B?
                    x, y, z = i, int(pc_kind.lstrip('Cond')) - 1, int(tr_th_id.lstrip('Th')) - 1
                    assert new_success >= previous_success
                    if x not in utility_tensor:
                        utility_tensor[x] = {y: {z: new_success - previous_success}} # TODO success work for B?
                    elif y not in utility_tensor[x]:
                        utility_tensor[x][y] = {z: new_success - previous_success}
                    elif z not in utility_tensor[x][y]:
                        utility_tensor[x][y][z] = new_success - previous_success
                    else: # TODO work for B? in the past, same patient already had same therapy for same kind of condition; avg. biased towards later instances
                        utility_tensor[x][y][z] = (new_success - previous_success + utility_tensor[x][y][z]) / 2
                    previous_success = new_success
                    matching_ks.add(k)
            remaining_ks = remaining_ks.difference(matching_ks)
    print('-> Created raw utility tensor')

    # arguments' values TODO:
    patient = dataset['Patients'][int(arg_patient_id) - 1]
    pconditions = patient['conditions']
    trials = patient['trials']
    assert arg_patient_id == patient['id']
    pcond = pconditions[int(arg_pc_id.lstrip('pc')) - int(pconditions[0]['id'].lstrip('pc'))]
    assert arg_pc_id == pcond['id']
    condition = dataset['Conditions'][int(pcond['kind'].lstrip('Cond')) - 1]
    assert pcond['kind'] == condition['id']

    # patient similarites:
    # TODO thexash?
    # now first: feature-agnostic collaborative filtering (baseline)
    patients_to_clusters = cluster_patients(utility_tensor, num_patients, 100, 5, 500, filepath)
    condensed_utility_tensor = condense_utilities(utility_tensor, patients_to_clusters)
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(condensed_utility_tensor)
    #print('number of medoids:', len(list(condensed_utility_tensor.keys())))
    #print(sorted(condensed_utility_tensor.keys()))

    # finish:
    print('-> Everything okay!')

    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2], sys.argv[3])
