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


cdef tuple get_directory_info(str filepath):
    cdef:
        list filepath_split
        str data_dir, res_dir, filename

    filepath_split = filepath.replace('\\', '/').rsplit('/', 1)
    filepath_split = ['.'] + filepath_split if len(filepath_split) < 2 else filepath_split
    data_dir, filename = filepath_split
    res_dir = data_dir + '/../results/'
    return (res_dir, filename)


cdef dict get_raw_utilities(dict dataset, str filepath, str mode='baseline'):
    cdef:
        size_t i, j, k, y, z
        float success
        str res_dir, filename, utl_path, pc_id, pc_kind, tr_pc_id, tr_th_id
        set remaining_ks, matching_ks
        list pconditions, trials
        dict patient, utility_tensor

    res_dir, filename = get_directory_info(filepath)
    utl_path = '%s%s_utl_raw_%s.pickle' % (res_dir, filename.rstrip('.json'), mode)
    if os.path.exists(utl_path):
        with open(utl_path, 'rb') as f:
            utility_tensor = pickle.load(f)
        print('-> Loaded pre-computed raw utility tensor from ../' + utl_path.rsplit('../', 1)[1])
        return utility_tensor

    utility_tensor = {}
    for i in range(len(dataset['Patients'])): # iterate over patients
        utility_tensor[i] = {}
        patient = dataset['Patients'][i]
        pconditions = patient['conditions']
        trials = patient['trials']
        remaining_ks = set(range(len(trials)))
        for j in range(len(pconditions)): # iterate over patient's conditions
            pc_id = pconditions[j]['id']
            pc_kind = pconditions[j]['kind']
            matching_ks = set()
            for k in sorted(remaining_ks): # efficiently iterate over corresponding trials
                tr_pc_id = trials[k]['condition']
                if pc_id == tr_pc_id:
                    tr_th_id = trials[k]['therapy']
                    success = float(str(trials[k]['successful']).rstrip('%')) * 0.01 # cover both 100 and '100%'
                    y, z = int(pc_kind.lstrip('Cond')) - 1, int(tr_th_id.lstrip('Th')) - 1
                    if y not in utility_tensor[i]:
                        utility_tensor[i][y] = {z: success}
                    elif z not in utility_tensor[i][y]:
                        utility_tensor[i][y][z] = success
                    else:
                        utility_tensor[i][y][z] = max(success, utility_tensor[i][y][z])
                    matching_ks.add(k)
            remaining_ks = remaining_ks.difference(matching_ks)

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    with open(utl_path, 'wb') as f:
        pickle.dump(utility_tensor, f)
    print('-> Created and saved raw utility tensor to ../' + utl_path.rsplit('../', 1)[1])

    return utility_tensor


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

    if sum_squares_1 == 0.0 or sum_squares_2 == 0.0:
        return 1.0

    return 1.0 - dot_product / np.sqrt(sum_squares_1 * sum_squares_2)


cdef np.ndarray cluster_patients(dict utility_tensor, size_t num_patients, size_t k, size_t n, size_t s, str filepath, mode='baseline'):
    cdef:
        size_t i, j, l, best_i, med, closest_med
        float cos_dist, lowest_cos_dist
        tuple sorted_pair
        dict memorized
        str res_dir, filename, p2c_path
        np.ndarray sample, sample_dist_matrix, results, medoids, patients_to_clusters

    np.random.seed(123)

    res_dir, filename = get_directory_info(filepath)
    p2c_path = '%s%s_p2c_%s.pickle' % (res_dir, filename.rstrip('.json'), mode)

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


cdef dict condense_utilities(dict utility_tensor, np.ndarray patients_to_clusters, str filepath, str mode='baseline'):
    cdef:
        size_t i, med, y, z, num_clustered_patients
        dict condensed_utility_tensor, agglomerated_utility_tensor, patient_matrix
        str res_dir, filename, utl_path

    res_dir, filename = get_directory_info(filepath)
    utl_path = '%s%s_utl_dense_%s.pickle' % (res_dir, filename.rstrip('.json'), mode)
    if os.path.exists(utl_path):
        with open(utl_path, 'rb') as f:
            condensed_utility_tensor = pickle.load(f)
        print('-> Loaded pre-computed condensed utility tensor from ../' + utl_path.rsplit('../', 1)[1])
        return condensed_utility_tensor

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

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    with open(utl_path, 'wb') as f:
        pickle.dump(condensed_utility_tensor, f)
    print('-> Created and saved condensed utility tensor to ../' + utl_path.rsplit('../', 1)[1])

    return condensed_utility_tensor


cdef np.ndarray get_clusters_distance_matrix(dict utility_tensor, dict condensed_utility_tensor, str filepath, int row=-1, mode='baseline'): # row=-1: full matrix
    cdef:
        size_t i, j, iter_count, med1, med2, num_clusters, num_iterations
        np.ndarray clusters_dist_matrix, row_is_precomputed
        str res_dir, filename, dmt_path

    res_dir, filename = get_directory_info(filepath)
    dmt_path = '%s%s_dmt_%s/' % (res_dir, filename.rstrip('.json'), mode)
    num_clusters = len(condensed_utility_tensor)
    clusters_dist_matrix = np.empty((num_clusters, num_clusters), dtype=float)
    row_is_precomputed = np.zeros(num_clusters, dtype=bool)

    if os.path.exists(dmt_path):
        if row >= 0 and os.path.exists(dmt_path + str(row) + '.pickle'):
            with open(dmt_path + str(row) + '.pickle', 'rb') as f:
                clusters_dist_matrix[row] = pickle.load(f)
            print("-> Loaded pre-computed cluster's distance vector from ../" + dmt_path.rsplit('../', 1)[1] + str(row) + '.pickle')
            return clusters_dist_matrix[row]

        for i in range(num_clusters):
            if os.path.exists(dmt_path + str(i) + '.pickle'):
                with open(dmt_path + str(i) + '.pickle', 'rb') as f:
                    clusters_dist_matrix[i] = pickle.load(f)
                    row_is_precomputed[i] = True
        if np.all(row_is_precomputed):
            assert row == -1
            print("-> Loaded full clusters' distance matrix from ../" + dmt_path.rsplit('../', 1)[1] + '*.pickle')
            return clusters_dist_matrix

    num_iterations = num_clusters if row >= 0 else num_clusters * (num_clusters + 1) // 2
    iter_count = 0
    for i, med1 in enumerate(condensed_utility_tensor):
        if row >= 0 and i != row or row_is_precomputed[i]:
            continue
        for j, med2 in enumerate(condensed_utility_tensor):
            if row == -1 and i > j:
                continue
            iter_count += 1
            print('-> Computing distances between clusters', iter_count, '/', num_iterations, '...', end='\r')
            if i == j:
                clusters_dist_matrix[i][j] = clusters_dist_matrix[j][i] = 0.0
            elif row_is_precomputed[j]:
                clusters_dist_matrix[i][j] = clusters_dist_matrix[j][i]
            else:
                clusters_dist_matrix[i][j] = clusters_dist_matrix[j][i] = cosine_distance(med1, med2, utility_tensor)

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(dmt_path):
        os.mkdir(dmt_path)
    if row >= 0:
        with open(dmt_path + str(row) + '.pickle', 'wb') as f:
            pickle.dump(clusters_dist_matrix[row], f)
        print("\n-> Saved cluster's distance vector to ../" + dmt_path.rsplit('../', 1)[1] + str(row) + '.pickle')
    else:
        for i in range(num_clusters):
            with open(dmt_path + str(i) + '.pickle', 'wb') as f:
                pickle.dump(clusters_dist_matrix[i], f)
            print(clusters_dist_matrix[i]) # TODO debug
        print("\n-> Saved full clusters' distance matrix to ../" + dmt_path.rsplit('../', 1)[1] + '*.pickle')

    return clusters_dist_matrix[row] if row >= 0 else clusters_dist_matrix


cdef np.ndarray recommend(dict patient, dict pcond, dict condition, size_t num_therapies, dict condensed_utility_tensor, np.ndarray clusters_dist_vector):
    cdef:
        size_t i, med, condition_y, therapy_z, previous_therapy_z
        tuple item
        list recommendations_list
        np.ndarray recommendations, random_sample
        dict weighted_scaled_utilities, final_utilities

    condition_y = int(condition['id'].lstrip('Cond')) - 1
    final_utilities = {}
    for i, med in enumerate(condensed_utility_tensor):
        weight = 1.01 - clusters_dist_vector[i]
        if condition_y not in condensed_utility_tensor[med]:
            final_utilities[therapy_z] = 0.0
        else:
            for therapy_z in condensed_utility_tensor[med][condition_y]:
                if therapy_z not in final_utilities:
                    final_utilities[therapy_z] = weight * condensed_utility_tensor[med][condition_y][therapy_z]
                else:
                    final_utilities[therapy_z] += weight * condensed_utility_tensor[med][condition_y][therapy_z]
    for i in range(len(patient['trials'])):
        if pcond['id'] == patient['trials'][i]['condition']:
            previous_therapy_z = int(patient['trials'][i]['therapy'].lstrip('Th')) - 1
            if previous_therapy_z in final_utilities:
                final_utilities[previous_therapy_z] = 0.0 # therapies already administered for same 'pc' should be dispreferred
    recommendations_list = list((item[0] for item in sorted((item for item in final_utilities.items()), key=lambda item: -item[1])))
    if len(recommendations_list) < 5: # only if length of supported list of therapies is smaller than 5, then fill up with random choices
        random_sample = np.random.choice(num_therapies, 10, replace=False)
        for i in range(10):
            if random_sample[i] not in recommendations_list:
                recommendations_list.append(random_sample[i])
                if len(recommendations_list) == 5:
                    break
    recommendations = np.empty(5, dtype=np.uintc)
    for i in range(5):
        recommendations[i] = recommendations_list[i]

    return recommendations


cdef int main(str filepath, str arg_patient_id, str arg_pc_id):
    cdef:
        str filename
        size_t med, row, num_therapies
        dict dataset, patient, pcond, condition, utility_tensor, condensed_utility_tensor
        np.ndarray patients_to_clusters, clusters_dist_vector, recommendations

    assert os.path.exists(filepath) and arg_patient_id.isdigit() and arg_pc_id.lstrip('pc').isdigit()
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if dataset['Patients'][0]['id'] == '1': # datasetA.json case
        patient = dataset['Patients'][int(arg_patient_id) - 1]
        assert arg_patient_id == patient['id']
    else:
        assert dataset['Patients'][0]['id'] == 0 # datasetB.json case
        patient = dataset['Patients'][int(arg_patient_id)]
        assert int(arg_patient_id) == patient['id']
    
    pcond = patient['conditions'][int(arg_pc_id.lstrip('pc')) - int(patient['conditions'][0]['id'].lstrip('pc'))]
    assert arg_pc_id == pcond['id']
    condition = dataset['Conditions'][int(pcond['kind'].lstrip('Cond')) - 1]
    assert pcond['kind'] == condition['id']
    filename = filepath.replace('\\', '/').rsplit('/', 1)[1] if '/' in filepath or '\\' in filepath else filepath
    print('-> Successfully parsed ' + filename)

    utility_tensor = get_raw_utilities(dataset, filepath)
    patients_to_clusters = cluster_patients(utility_tensor, len(dataset['Patients']), 100, 5, 500, filepath)
    condensed_utility_tensor = condense_utilities(utility_tensor, patients_to_clusters, filepath)
    ### get_clusters_distance_matrix(utility_tensor, condensed_utility_tensor, filepath, row = -1)
    med = patients_to_clusters[int(patient['id']) - 1] if type(patient['id']) == str else patients_to_clusters[patient['id']]
    row = list(condensed_utility_tensor).index(med)
    clusters_dist_vector = get_clusters_distance_matrix(utility_tensor, condensed_utility_tensor, filepath, row=row)
    num_therapies = len(dataset['Therapies'])
    recommendations = recommend(patient, pcond, condition, num_therapies, condensed_utility_tensor, clusters_dist_vector)
    print(recommendations) # TODO debug
    print('-> Everything okay!')

    return 0


if __name__ == '__main__':
    assert len(sys.argv) == 4
    main(sys.argv[1], sys.argv[2], sys.argv[3])
