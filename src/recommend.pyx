# cython: language_level=3, boundscheck=False, wraparound=False

from datetime import datetime as dt
import os
import sys
import math
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
        str data_dir, bin_dir, filename

    filepath_split = filepath.replace('\\', '/').rsplit('/', 1)
    filepath_split = ['.'] + filepath_split if len(filepath_split) < 2 else filepath_split
    data_dir, filename = filepath_split
    bin_dir = data_dir + '/../bin/'
    return (bin_dir, filename)


cdef tuple get_relevant_keys(list sub_dataset, str filepath, str mode, str label=''):
    cdef:
        size_t i
        float float_val, minimum, maximum
        str str_val, key, other_key, key_type, bin_dir, filename, rky_path
        set remaining_continuous_keys, str_vals_as_set
        dict rel_keys, key_properties, key_vals

    if label:
        bin_dir, filename = get_directory_info(filepath)
        rky_path = '%spkl/%s_rky_%s_%s.pickle' % (bin_dir, filename.rstrip('.json'), label, mode)
        if os.path.exists(rky_path):
            with open(rky_path, 'rb') as f:
                rel_keys, key_vals = pickle.load(f)
            print('-> Loaded pre-computed relevant keys for %s from ../%s' % (label, rky_path.rsplit('../', 1)[1]))
            return (rel_keys, key_vals)

    assert len(sub_dataset) > 0
    remaining_continuous_keys = set() # only relevant for correlation computation below
    key_properties = {}
    for key in sub_dataset[0].keys():
        if key == 'id' or type(sub_dataset[0][key]) == list:
            continue
        if type(sub_dataset[0][key]) == int or type(sub_dataset[0][key]) == float:
            key_type = 'continuous_regular'
        else:
            assert type(sub_dataset[0][key]) == str
            str_val = sub_dataset[0][key]
            if str_val.count('.') <= 1 and str_val.replace('.', '').isdigit():
                if '.' not in str_val and len(str_val) == 8 and str_val[:2] in ('19', '20') and '01' <= str_val[4:6] <= '12' and '01' <= str_val[6:] <= '31':
                    key_type = 'continuous_date'
                else:
                    key_type = 'continuous_regular'
            else:
                key_type = 'discrete'
        key_properties[key] = (key_type.split('_')[0], [])
        for i in range(len(sub_dataset)):
            if key_type == 'continuous_regular':
                float_val = float(sub_dataset[i][key])
                key_properties[key][1].append(float_val)
                remaining_continuous_keys.add(key)
            elif key_type == 'continuous_date':
                float_val = float((dt.strptime(sub_dataset[i][key], '%Y%m%d') - dt.strptime('19000101', '%Y%m%d')).days)
                key_properties[key][1].append(float_val)
                remaining_continuous_keys.add(key)
            else:
                str_val = sub_dataset[i][key]
                key_properties[key][1].append(str_val)
    rel_keys = {'continuous': {}, 'discrete': set()}
    for key in key_properties:
        if key_properties[key][0] == 'continuous':
            remaining_continuous_keys.remove(key)
            minimum, maximum = min(key_properties[key][1]), max(key_properties[key][1])
            if minimum == maximum:
                continue
            if any((abs(np.corrcoef(key_properties[key][1], key_properties[other_key][1])[0][1]) >= 0.9 for other_key in remaining_continuous_keys)):
                continue
            rel_keys['continuous'][key] = (minimum, maximum)
        else:
            assert key_properties[key][0] == 'discrete'
            str_vals_as_set = set(key_properties[key][1])
            if len(str_vals_as_set) == 1:
                continue
            if len(str_vals_as_set) == len(key_properties[key][1]):
                continue
            rel_keys['discrete'].add(key)
    key_vals = {key: key_properties[key][1] for key in key_properties if key in rel_keys['continuous'] or key in rel_keys['discrete']}

    if label:
        if not os.path.exists(bin_dir):
            os.mkdir(bin_dir)
        if not os.path.exists(bin_dir + 'pkl/'):
            os.mkdir(bin_dir + 'pkl/')
        with open(rky_path, 'wb') as f:
            pickle.dump((rel_keys, key_vals), f)
        print('-> Extracted and saved relevant keys for %s to ../%s' % (label, rky_path.rsplit('../', 1)[1]))
    else:
        print('-> Extracted relevant keys')

    return (rel_keys, key_vals)


cdef tuple get_raw_tensors(dict dataset, str filepath, str mode):
    cdef:
        size_t i, j, k, y, z, num_patients, num_conditions, num_therapies
        float success, min_success, max_success, end, min_date, max_date, duration_in_days, diagnosed
        str bin_dir, filename, utl_path, pc_id, pc_kind, pc_diagnosed, tr_pc_id, tr_th_id, tr_start, tr_end
        set remaining_ks, matching_ks
        list pconditions, trials
        dict patient, utility_tensor

    bin_dir, filename = get_directory_info(filepath)
    utl_path = '%spkl/%s_utl_raw_henr_%s.pickle' % (bin_dir, filename.rstrip('.json'), mode)
    if os.path.exists(utl_path):
        with open(utl_path, 'rb') as f:
            utility_tensor, half_enriched_tensor = pickle.load(f)
        print('-> Loaded pre-computed raw utility tensor%s from ../' % (' and trial-recency tensor' if mode.startswith('hybrid') else '') + utl_path.rsplit('../', 1)[1])
        return (utility_tensor, half_enriched_tensor)

    utility_tensor = {}
    half_enriched_tensor = {} # only relevant in 'hybrid' mode; stays empty otherwise
    min_success, max_success = 100.0, 0.0 # initial values; to be determined
    if mode.startswith('hybrid'):
        min_date, max_date = 73048.0, 0.0 # a.k.a. initially 2099-12-31, 1900-01-01
    num_patients, num_conditions, num_therapies = len(dataset['Patients']), len(dataset['Conditions']), len(dataset['Therapies'])
    for i in range(num_patients): # iterate over patients
        print('-> Building raw utility tensor' + (' and trial-recency tensor' if mode.startswith('hybrid') else ''), i + 1, '/', num_patients, '...', end='\r')
        utility_tensor[i] = {}
        if mode.startswith('hybrid'):
            half_enriched_tensor[i] = {}
        patient = dataset['Patients'][i]
        pconditions = patient['conditions']
        trials = patient['trials']
        remaining_ks = set(range(len(trials)))
        for j in range(len(pconditions)): # iterate over patient's conditions
            pc_id = pconditions[j]['id']
            pc_kind = pconditions[j]['kind']
            y = int(pc_kind.lstrip('Cond')) - 1
            matching_ks = set()
            for k in sorted(remaining_ks): # efficiently iterate over corresponding trials
                tr_pc_id = trials[k]['condition']
                if pc_id == tr_pc_id:
                    success = float(str(trials[k]['successful']).rstrip('%')) # cover both 100 and '100%'
                    tr_start, tr_end = trials[k]['start'], trials[k]['end']
                    duration_in_days = float((dt.strptime(tr_end, '%Y%m%d') - dt.strptime(tr_start, '%Y%m%d')).days)
                    success /= 1 + np.log(1 + duration_in_days) # factor in inverse trial duration for success criterion
                    min_success, max_success = min(success, min_success), max(success, max_success)
                    if mode.startswith('hybrid'):
                        end = float((dt.strptime(tr_end, '%Y%m%d') - dt.strptime('19000101', '%Y%m%d')).days)
                        min_date, max_date = min(end, min_date), max(end, max_date)
                    tr_th_id = trials[k]['therapy']
                    z = int(tr_th_id.lstrip('Th')) - 1
                    if y not in utility_tensor[i]:
                        utility_tensor[i][y] = {z: success}
                        if mode.startswith('hybrid'):
                            half_enriched_tensor[i][num_conditions + y] = {z: end}
                    elif z not in utility_tensor[i][y]:
                        utility_tensor[i][y][z] = success
                        if mode.startswith('hybrid'):
                            half_enriched_tensor[i][num_conditions + y][z] = end
                    else:
                        utility_tensor[i][y][z] = max(success, utility_tensor[i][y][z])
                        if mode.startswith('hybrid'):
                            half_enriched_tensor[i][num_conditions + y][z] = max(end, half_enriched_tensor[i][num_conditions + y][z])
                    matching_ks.add(k)
            if mode.startswith('hybrid'):
                if num_conditions + y not in half_enriched_tensor[i]: # condition had no corresponding trials
                    half_enriched_tensor[i][num_conditions + y] = {}
                pc_diagnosed = pconditions[j]['diagnosed']
                diagnosed = float((dt.strptime(pc_diagnosed, '%Y%m%d') - dt.strptime('19000101', '%Y%m%d')).days)
                min_date, max_date = min(diagnosed, min_date), max(diagnosed, max_date)
                if num_therapies not in half_enriched_tensor[i][num_conditions + y]:
                    half_enriched_tensor[i][num_conditions + y][num_therapies] = diagnosed
                else:
                    half_enriched_tensor[i][num_conditions + y][num_therapies] = max(diagnosed, half_enriched_tensor[i][num_conditions + y][num_therapies])
            remaining_ks = remaining_ks.difference(matching_ks)
    assert max_success > 0.0
    print()
    for i in range(num_patients):
        print('-> Normalizing values in raw utility tensor', i + 1, '/', num_patients, '...', end='\r')
        for y in utility_tensor[i]:
            for z in utility_tensor[i][y]:
                utility_tensor[i][y][z] -= min_success
                utility_tensor[i][y][z] /= max_success
    if mode.startswith('hybrid'):
        assert max_date > 0.0
        print()
        for i in range(num_patients):
            print('-> Normalizing values in trial-recency tensor', i + 1, '/', num_patients, '...', end='\r')
            for y in half_enriched_tensor[i]:
                for z in half_enriched_tensor[i][y]:
                    half_enriched_tensor[i][y][z] -= min_date
                    half_enriched_tensor[i][y][z] /= max_date

    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    if not os.path.exists(bin_dir + 'pkl/'):
        os.mkdir(bin_dir + 'pkl/')
    with open(utl_path, 'wb') as f:
        pickle.dump((utility_tensor, half_enriched_tensor), f)
    print('\n-> Saved raw utility tensor%s to ../' % (' and trial-recency tensor' if mode.startswith('hybrid') else '') + utl_path.rsplit('../', 1)[1])

    return (utility_tensor, half_enriched_tensor)


cdef dict get_enriched_tensor(dict utility_tensor, dict half_enriched_tensor, tuple rky_patients, size_t num_conditions, str filepath, str mode):
    cdef:
        size_t i, j, hash_key_value
        float minimum, maximum
        str key, bin_dir, filename, enr_path
        dict enriched_tensor

    bin_dir, filename = get_directory_info(filepath)
    enr_path = '%spkl/%s_enr_%s.pickle' % (bin_dir, filename.rstrip('.json'), mode)
    if os.path.exists(enr_path):
        with open(enr_path, 'rb') as f:
            enriched_tensor = pickle.load(f)
        print('-> Loaded pre-computed enriched tensor from ../' + enr_path.rsplit('../', 1)[1])
        return enriched_tensor

    enriched_tensor = utility_tensor.copy()
    enriched_tensor.update(half_enriched_tensor)
    for i in enriched_tensor:
        for j, (key, (minimum, maximum)) in enumerate(rky_patients[0]['continuous'].items()):
            if num_conditions * 2 not in enriched_tensor[i]:
                enriched_tensor[i][num_conditions * 2] = {j: (rky_patients[1][key][i] - minimum) / maximum}
            else:
                enriched_tensor[i][num_conditions * 2][j] = (rky_patients[1][key][i] - minimum) / maximum
        for j, key in enumerate(rky_patients[0]['discrete']):
            hash_key_value = hash((key, rky_patients[1][key][i]))
            if num_conditions * 2 + 1 not in enriched_tensor[i]:
                enriched_tensor[i][num_conditions * 2 + 1] = {hash_key_value: 1.0}
            else:
                enriched_tensor[i][num_conditions * 2 + 1][hash_key_value] = 1.0

    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    if not os.path.exists(bin_dir + 'pkl/'):
        os.mkdir(bin_dir + 'pkl/')
    with open(enr_path, 'wb') as f:
        pickle.dump(enriched_tensor, f)
    print('-> Created and saved enriched tensor to ../' + enr_path.rsplit('../', 1)[1])

    return enriched_tensor


cdef float cosine_distance(size_t patient_x_1, size_t patient_x_2, dict utl_or_enr_tensor):
    cdef:
        size_t i, j, y1, z1, y2, z2
        dict patient_matrix_1, patient_matrix_2
        float dot_product, sum_squares_1, sum_squares_2, value_1, value_2

    patient_matrix_1 = utl_or_enr_tensor[patient_x_1]
    patient_matrix_2 = utl_or_enr_tensor[patient_x_2]
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


cdef np.ndarray cluster_patients(dict utl_or_enr_tensor, size_t num_patients, size_t k, size_t n, size_t s, str filepath, str mode):
    cdef:
        size_t i, j, l, best_i, med, closest_med
        float cos_dist, lowest_cos_dist
        tuple sorted_pair
        dict memorized
        str bin_dir, filename, p2c_path
        np.ndarray sample, sample_dist_matrix, results, medoids, patients_to_clusters

    np.random.seed(123)
    bin_dir, filename = get_directory_info(filepath)
    p2c_path = '%spkl/%s_p2c_%s.pickle' % (bin_dir, filename.rstrip('.json'), mode)

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
                    cos_dist = cosine_distance(sample[j], sample[l], utl_or_enr_tensor)
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
                cos_dist = memorized[sorted_pair] = cosine_distance(i, med, utl_or_enr_tensor)
            if cos_dist < lowest_cos_dist:
                closest_med = med
                lowest_cos_dist = cos_dist
        patients_to_clusters[i] = closest_med

    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    if not os.path.exists(bin_dir + 'pkl/'):
        os.mkdir(bin_dir + 'pkl/')
    with open(p2c_path, 'wb') as f:
        pickle.dump(patients_to_clusters, f)
    print('\n-> Saved computed clusters to ../' + p2c_path.rsplit('../', 1)[1])

    return patients_to_clusters


cdef dict condense_utilities(dict utility_tensor, np.ndarray patients_to_clusters, str filepath, str mode):
    cdef:
        size_t i, med, y, z, num_clustered_patients
        dict condensed_utility_tensor, agglomerated_utility_tensor, patient_matrix
        str bin_dir, filename, utl_path

    bin_dir, filename = get_directory_info(filepath)
    utl_path = '%spkl/%s_utl_dense_%s.pickle' % (bin_dir, filename.rstrip('.json'), mode)
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

    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    if not os.path.exists(bin_dir + 'pkl/'):
        os.mkdir(bin_dir + 'pkl/')
    with open(utl_path, 'wb') as f:
        pickle.dump(condensed_utility_tensor, f)
    print('-> Created and saved condensed utility tensor to ../' + utl_path.rsplit('../', 1)[1])

    return condensed_utility_tensor


cdef np.ndarray get_clusters_distance_matrix(dict utl_or_enr_tensor, dict condensed_utility_tensor, str filepath, str mode, int row=-1, # row=-1: get full matrix
                                             np.ndarray matrix=np.array(0)): # matrix: just in case we have pre-computed matrix
    cdef:
        size_t i, j, med1, med2, num_clusters
        np.ndarray clusters_dist_matrix, row_is_precomputed
        str bin_dir, filename, dmt_path

    if row >= 0 and matrix.any():
        return matrix[row]

    bin_dir, filename = get_directory_info(filepath)
    dmt_path = '%spkl/%s_dmt_%s/' % (bin_dir, filename.rstrip('.json'), mode)
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

    for i, med1 in enumerate(condensed_utility_tensor):
        if row >= 0 and i != row or row_is_precomputed[i]:
            continue
        for j, med2 in enumerate(condensed_utility_tensor):
            if row == -1 and i > j:
                continue
            if i == j:
                clusters_dist_matrix[i][j] = clusters_dist_matrix[j][i] = 0.0
            elif row_is_precomputed[j]:
                clusters_dist_matrix[i][j] = clusters_dist_matrix[j][i]
            else:
                clusters_dist_matrix[i][j] = clusters_dist_matrix[j][i] = cosine_distance(med1, med2, utl_or_enr_tensor)

    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    if not os.path.exists(bin_dir + 'pkl/'):
        os.mkdir(bin_dir + 'pkl/')
    if not os.path.exists(dmt_path):
        os.mkdir(dmt_path)
    if row >= 0:
        with open(dmt_path + str(row) + '.pickle', 'wb') as f:
            pickle.dump(clusters_dist_matrix[row], f)
        print("-> Created and saved cluster's distance vector to ../" + dmt_path.rsplit('../', 1)[1] + str(row) + '.pickle')
    else:
        for i in range(num_clusters):
            with open(dmt_path + str(i) + '.pickle', 'wb') as f:
                pickle.dump(clusters_dist_matrix[i], f)
        print("-> Created and saved full clusters' distance matrix to ../" + dmt_path.rsplit('../', 1)[1] + '*.pickle')

    return clusters_dist_matrix[row] if row >= 0 else clusters_dist_matrix


cdef np.ndarray recommend(dict patient, dict pcond, np.ndarray clusters_dist_vector, dict condensed_utility_tensor, size_t num_conditions, size_t num_therapies,
                          str mode, tuple rky_conditions=(), tuple rky_therapies=()):
    cdef:
        size_t i, j, med, condition_y, therapy_z, previous_therapy_z
        str disc_key, most_informative_key, val
        float weight, val_count, val_weight
        tuple item, relevant_keys_tuple
        list recommendations_list, other_conds_same_key_val
        np.ndarray recommendations, random_sample
        dict final_utilities, rel_keys_cond, key_vals_cond, rel_keys_th, key_vals_th, vals_to_counts, vals_to_avg_utilities

    condition_y = int(pcond['kind'].lstrip('Cond')) - 1
    if mode.startswith('hybrid'):
        rel_keys_cond, key_vals_cond = rky_conditions
        assert len(rel_keys_cond['continuous']) == 0
        relevant_keys_tuple = tuple(rel_keys_cond['discrete'])
        if len(relevant_keys_tuple) > 0:
            if len(relevant_keys_tuple) == 1:
                most_informative_key = relevant_keys_tuple[0]
            else:
                most_informative_key = max((disc_key for disc_key in relevant_keys_tuple), key=lambda disc_key: math.comb(num_conditions, len(set(key_vals_cond[disc_key]))))
            other_conds_same_key_val = [i for i in range(num_conditions) if i != condition_y and key_vals_cond[most_informative_key][i] == \
                                        key_vals_cond[most_informative_key][condition_y]]
        else:
            other_conds_same_key_val = []
    final_utilities = {}
    for i in range(num_conditions):
        if i != condition_y and (mode != 'hybrid' or i not in other_conds_same_key_val):
            continue
        for j, med in enumerate(condensed_utility_tensor):
            if i not in condensed_utility_tensor[med]:
                continue
            weight = 1.01 - clusters_dist_vector[j] # 1.01 instead of 1.00 for smoothing, i.e. allowing also maximally distant clusters to contribute moderately
            if mode.startswith('hybrid') and other_conds_same_key_val != []:
                weight = weight * 0.8 if i == condition_y else weight * 0.2 / len(other_conds_same_key_val)
            for therapy_z in condensed_utility_tensor[med][i]:
                if therapy_z not in final_utilities:
                    final_utilities[therapy_z] = weight * condensed_utility_tensor[med][i][therapy_z]
                else:
                    final_utilities[therapy_z] += weight * condensed_utility_tensor[med][i][therapy_z]
    if mode.startswith('hybrid'):
        rel_keys_th, key_vals_th = rky_therapies
        assert len(rel_keys_th['continuous']) == 0
        relevant_keys_tuple = tuple(rel_keys_th['discrete'])
        if len(relevant_keys_tuple) > 0:
            if len(relevant_keys_tuple) == 1:
                most_informative_key = relevant_keys_tuple[0]
            else:
                most_informative_key = max((disc_key for disc_key in relevant_keys_tuple), key=lambda disc_key: math.comb(num_therapies, len(set(key_vals_th[disc_key]))))
            vals_to_counts = {}
            vals_to_avg_utilities = {val: 0.0 for val in set(key_vals_th[most_informative_key])}
            for therapy_z in final_utilities:
                val = key_vals_th[most_informative_key][therapy_z]
                vals_to_avg_utilities[val] += final_utilities[therapy_z] # sum
            for val in vals_to_avg_utilities:
                vals_to_counts[val] = key_vals_th[most_informative_key].count(val)
                vals_to_avg_utilities[val] /= vals_to_counts[val] # average
            for therapy_z in final_utilities:
                val = key_vals_th[most_informative_key][therapy_z]
                val_count = vals_to_counts[val]
                val_weight = 0.2 * val_count / (val_count - 1) if val_count > 1 else 0.0
                final_utilities[therapy_z] = (1 - val_weight) * final_utilities[therapy_z] + val_weight * vals_to_avg_utilities[val] # update
    for i in range(len(patient['trials'])):
        if pcond['id'] == patient['trials'][i]['condition']:
            previous_therapy_z = int(patient['trials'][i]['therapy'].lstrip('Th')) - 1
            if previous_therapy_z in final_utilities:
                final_utilities[previous_therapy_z] -= 1.01 * len(condensed_utility_tensor) # therapies already administered for same 'pc' should be dispreferred
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


cdef np.ndarray recommend_overall_most_frequent_therapies_as_baseline(dict dataset):
    cdef:
        size_t i, therapy_z, num_patients, num_therapies, num_trials
        dict frequencies
        str tr_th_id

    num_patients, num_therapies = len(dataset['Patients']), len(dataset['Therapies'])
    frequencies = {}

    for i in range(num_patients):
        num_trials = len(dataset['Patients'][i]['trials'])
        for j in range(num_trials):
            tr_th_id = dataset['Patients'][i]['trials'][j]['therapy']
            therapy_z = int(tr_th_id.lstrip('Th')) - 1
            if therapy_z not in frequencies:
                frequencies[therapy_z] = 1
            else:
                frequencies[therapy_z] += 1

    return np.array(sorted([therapy_z for therapy_z in frequencies], key=lambda therapy_z: -frequencies[therapy_z])[:5])


cdef void main(str filepath, str arg_patient_id, str arg_pc_id, str mode=''):
    cdef:
        str filename, res_dir, eval_dir, eval_path
        float hard_accuracy, soft_accuracy, success, threshold_success, duration_in_days
        size_t i, j, med, row, eval, patient_x, pcond_y, therapy_z, pred_therapy_z, num_patients, num_conditions, num_therapies
        tuple rky_patients, rky_conditions, rky_therapies
        list all_successes, deleted_trials, test_triples, predictions, numbers_testcases, hard_accuracies, soft_accuracies
        dict dataset_copy, trial, dataset, patient, pcond, condition, utility_tensor, condensed_utility_tensor, half_enriched_tensor, enriched_tensor
        np.ndarray test_set, patients_to_clusters, clusters_dist_vector, recommendations

    assert os.path.exists(filepath)

    if not arg_patient_id and not arg_pc_id:
        print('***********************')
        print('* E V A L U A T I O N *')
        print('***********************')
        eval = 1
    else:
        assert arg_patient_id.isdigit() and arg_pc_id.lstrip('pc').isdigit()
        eval = 0
    if mode:
        assert mode in ('baseline', 'simple', 'hybrid')
        print('-> Selected method:', mode)
    else:
        while True:
            mode = input('-> Please select method (b=baseline, s=simple, h=hybrid): ')
            if mode.lower() == 'b':
                mode = 'baseline'
                break
            elif mode.lower() == 's':
                mode = 'simple'
                break
            elif mode.lower() == 'h':
                mode = 'hybrid'
                break
            print('Invalid method.')
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    assert dataset['Conditions'][0]['id'] == 'Cond1'
    assert dataset['Therapies'][0]['id'] == 'Th1'
    assert dataset['Patients'][0]['id'] in ('1', 0)
    if not eval:
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
    num_patients, num_conditions, num_therapies = len(dataset['Patients']), len(dataset['Conditions']), len(dataset['Therapies'])
    assert num_therapies >= 5
    if eval:
        hard_accuracies = []
        soft_accuracies = []
        numbers_testcases = []
    for i in range(10 if eval else 1):
        if eval:
            if i == 0:
                all_successes = []
                for patient in dataset['Patients']:
                    for trial in patient['trials']:
                        success = float(str(trial['successful']).rstrip('%'))
                        duration_in_days = float((dt.strptime(trial['end'], '%Y%m%d') - dt.strptime(trial['start'], '%Y%m%d')).days)
                        success /= 1 + np.log(1 + duration_in_days)
                        all_successes.append(success)
                all_successes.sort()
                threshold_success = all_successes[len(all_successes) * 3 // 4] # cut-off at third quartile
            else:
                assert len(test_triples) == len(deleted_trials)
                for j, trial in enumerate(deleted_trials):
                    dataset['Patients'][test_triples[j][0]]['trials'].append(trial)
                    dataset['Patients'][test_triples[j][0]]['trials'].sort(key=lambda trial: trial['id'])
                mode = mode.split('_eval')[0]
            mode += '_eval%d' % i
            print('\n-> Starting evaluation round %d ...' % (i + 1))
            np.random.seed(i)
            test_set = np.random.choice(num_patients, num_patients // 5, replace=False)
            deleted_trials = []
            test_triples = []
            predictions = []
            for patient_x in test_set:
                if len(dataset['Patients'][patient_x]['trials']) == 0:
                    continue
                trial = max(dataset['Patients'][patient_x]['trials'], key=lambda trial: trial['start'])
                success = float(str(trial['successful']).rstrip('%'))
                duration_in_days = float((dt.strptime(trial['end'], '%Y%m%d') - dt.strptime(trial['start'], '%Y%m%d')).days)
                success /= 1 + np.log(1 + duration_in_days)
                if success < threshold_success:
                    continue
                pcond_y = int(trial['condition'].lstrip('pc')) - 1
                therapy_z = int(trial['therapy'].lstrip('Th')) - 1
                test_triples.append((patient_x, pcond_y, therapy_z))
                dataset['Patients'][patient_x]['trials'].remove(trial)
                deleted_trials.append(trial)
            assert len(test_triples) > 0
        if mode.startswith('baseline'):
            recommendations = recommend_overall_most_frequent_therapies_as_baseline(dataset)
            if not eval:
                print('-> Recommendations: ' + ', '.join(('Th' + str(therapy_z + 1) for therapy_z in recommendations)).rstrip(', '))
                return

            for j in range(len(test_triples)):
                predictions.append(recommendations)
        else:
            utility_tensor, half_enriched_tensor = get_raw_tensors(dataset, filepath, mode)
            if mode.startswith('hybrid'):
                rky_patients = get_relevant_keys(dataset['Patients'], filepath, mode, label='patients')
                enriched_tensor = get_enriched_tensor(utility_tensor, half_enriched_tensor, rky_patients, num_conditions, filepath, mode)
                patients_to_clusters = cluster_patients(enriched_tensor, num_patients, 100, 5, 500, filepath, mode)
            else:
                patients_to_clusters = cluster_patients(utility_tensor, num_patients, 100, 5, 500, filepath, mode)
            condensed_utility_tensor = condense_utilities(utility_tensor, patients_to_clusters, filepath, mode)
            if eval:
                if mode.startswith('hybrid'):
                    matrix = get_clusters_distance_matrix(enriched_tensor, condensed_utility_tensor, filepath, mode, row=-1)
                else:
                    matrix = get_clusters_distance_matrix(utility_tensor, condensed_utility_tensor, filepath, mode, row=-1)
            for j in range(len(test_triples) if eval else 1):
                if eval:
                    patient = dataset['Patients'][test_triples[j][0]]
                    pcond = patient['conditions'][test_triples[j][1] - (int(patient['conditions'][0]['id'].lstrip('pc')) - 1)]
                    assert pcond['id'] == 'pc' + str(test_triples[j][1] + 1)
                    print('-> Generating recommendations', j + 1, '/', len(test_triples), '...', end='\r')
                med = patients_to_clusters[int(patient['id']) - 1] if type(patient['id']) == str else patients_to_clusters[patient['id']]
                row = list(condensed_utility_tensor).index(med)
                if mode.startswith('hybrid'):
                    clusters_dist_vector = get_clusters_distance_matrix(enriched_tensor, condensed_utility_tensor, filepath, mode,
                                                                        row=row, matrix=matrix if eval else np.array(0))
                else:
                    clusters_dist_vector = get_clusters_distance_matrix(utility_tensor, condensed_utility_tensor, filepath, mode,
                                                                        row=row, matrix=matrix if eval else np.array(0))
                if j == 0:
                    rky_conditions = get_relevant_keys(dataset['Conditions'], filepath, mode, label='conditions') if mode.startswith('hybrid') else ()
                    rky_therapies = get_relevant_keys(dataset['Therapies'], filepath, mode, label='therapies') if mode.startswith('hybrid') else ()
                recommendations = recommend(patient, pcond, clusters_dist_vector, condensed_utility_tensor, num_conditions, num_therapies, mode,
                                            rky_conditions=rky_conditions, rky_therapies=rky_therapies)
                if not eval:
                    print('-> Recommendations: ' + ', '.join(('Th' + str(therapy_z + 1) for therapy_z in recommendations)).rstrip(', '))
                    return

                predictions.append(recommendations)
            print()
        assert len(test_triples) == len(predictions)
        hard_accuracy = 0.0
        soft_accuracy = 0.0
        for (patient_x, pcond_y, therapy_z), recommendations in zip(test_triples, predictions):
            for j, pred_therapy_z in enumerate(recommendations):
                assert j < 5
                if therapy_z == pred_therapy_z:
                    if j == 0:
                        hard_accuracy += 1.0
                    soft_accuracy += 1.0 - 0.2 * j
                    break
        hard_accuracy /= len(predictions)
        soft_accuracy /= len(predictions)
        hard_accuracies.append(hard_accuracy)
        soft_accuracies.append(soft_accuracy)
        numbers_testcases.append(len(predictions))
        print('-> Hard accuracy over %d test cases: %0.3f' % (numbers_testcases, hard_accuracy))
        print('-> Soft accuracy over %d test cases: %0.3f' % (numbers_testcases, soft_accuracy))
    hard_accuracy = sum(hard_accuracies) / len(hard_accuracies)
    soft_accuracy = sum(soft_accuracies) / len(soft_accuracies)
    print('\n-> OVERALL HARD ACCURACY: %0.3f' % hard_accuracy)
    print('-> OVERALL SOFT ACCURACY: %0.3f\n' % soft_accuracy)
    res_dir, filename = get_directory_info(filepath)
    res_dir = res_dir.replace('bin', 'results')
    eval_dir = res_dir + 'evaluation/'
    eval_path = '%seval_%s_%s.tsv' % (eval_dir, filename.rstrip('.json'), mode.split('_eval')[0])
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write('round\tnum_testcases\thard_accuracy\tsoft_accuracy\n')
        f.write('overall\t%d\t%0.3f\t%0.3f\n' % (sum(numbers_testcases), hard_accuracy, soft_accuracy))
        for i in range(len(numbers_testcases)):
            f.write('%d\t%d\t%0.3f\t%0.3f\n' % (i + 1, numbers_testcases[i], hard_accuracies[i], soft_accuracies[i]))
    print('-> Exported evaluation summary to', eval_path)

    return


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[2] == '--eval':
        main(sys.argv[1], '', '')
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and (sys.argv[2], sys.argv[3]) == ('--eval', '-m'):
        main(sys.argv[1], '', '', mode=sys.argv[4])
    else:
        assert len(sys.argv) == 6 and sys.argv[4] == '-m'
        main(sys.argv[1], sys.argv[2], sys.argv[3], mode=sys.argv[5])
