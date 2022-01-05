import random
import json
import math
from datetime import datetime, timedelta
from alphabet_detector import AlphabetDetector


def main():
    # set a specific random seed:
    random.seed(123)


    # load in conditions, therapies, names from text files:

    # Conditions
    conditions = {}
    with open('values/conditions.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                conditions[line.strip()] = {}
    with open('values/conditions_assignedTherapies.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                extracted = line.strip('\n').split(' : ')
                cond = extracted[0]
                extracted = extracted[1:]
                assert cond in conditions
                conditions[cond]['assignedTherapies'] = extracted
    with open('values/conditions_conditionType.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                cond, type = line.strip('\n').split(' - ')
                assert cond in conditions
                conditions[cond]['conditionType'] = type
    with open('values/conditions_curabilityWeight.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                cond, weight_str = line.strip('\n').split(' - ')
                weight = float(weight_str)
                assert cond in conditions
                conditions[cond]['curabilityWeight'] = weight
    with open('values/conditions_sexSpecificness.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                cond, sex = line.strip('\n').split(' - ')
                sex = {'f', 'm'} if sex == 'b' else {sex}
                assert cond in conditions
                conditions[cond]['sexSpecificness'] = sex
     
    # Therapies
    therapies = {}
    with open('values/therapies.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                therapies[line.strip()] = {}
    with open('values/therapies_efficacyWeight.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                th, weight_str = line.strip('\n').split(' - ')
                weight = float(weight_str)
                assert th in therapies
                therapies[th]['efficacyWeight'] = weight
    with open('values/therapies_therapyType.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                th, type = line.strip('\n').split(' - ')
                assert th in therapies
                therapies[th]['therapyType'] = type

    # Names
    ad = AlphabetDetector()
    first_names = []
    with open('values/first_names.all.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n' and ad.only_alphabet_chars(line, "LATIN"):
                first_names.append('-'.join((x.capitalize() for x in line.strip().split('-'))))
    last_names = []
    with open('values/last_names.all.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n' and ad.only_alphabet_chars(line, "LATIN"):
                last_names.append('-'.join((x.capitalize() for x in line.strip().split('-'))))


    # further transformations:

    for th in therapies:
        efficacyWeight = therapies[th]['efficacyWeight']
        case = random.randint(1, 3)
        if case == 1:
            durationWeight = efficacyWeight
        elif case == 2:
            baseline = 0.0 if efficacyWeight <= 0.5 else 0.5
            durationWeight = baseline + random.randint(1, 5) / 10
        else: # case == 3
            durationWeight = random.randint(1, 10) / 10
        therapies[th]['durationWeight'] = durationWeight
    for cond in conditions:
        curabilityWeight = conditions[cond]['curabilityWeight']
        case = random.randint(1, 3)
        if case == 1:
            frequencyWeight = curabilityWeight
        elif case == 2:
            baseline = 0.0 if curabilityWeight <= 0.5 else 0.5
            frequencyWeight = baseline + random.randint(1, 5) / 10
        else: # case == 3
            frequencyWeight = random.randint(1, 10) / 10
        conditions[cond]['frequencyWeight'] = frequencyWeight
        conditions[cond]['jitteredEfficacyWeights'] = []
        assignedTherapies = conditions[cond]['assignedTherapies']
        for ath in assignedTherapies:
            efficacyWeight = therapies[ath]['efficacyWeight']
            case = random.randint(1, 2)
            if case == 1:
                jitteredEfficacyWeight = efficacyWeight
            else: # case == 2
                baseline = 0.0 if efficacyWeight <= 0.5 else 0.5
                jitteredEfficacyWeight = baseline + random.randint(1, 5) / 10
            conditions[cond]['jitteredEfficacyWeights'].append(jitteredEfficacyWeight)
    conds_freq_weights_f = [conditions[cond]['frequencyWeight'] if 'f' in conditions[cond]['sexSpecificness'] else 0.0 for cond in conditions]
    conds_freq_weights_m = [conditions[cond]['frequencyWeight'] if 'm' in conditions[cond]['sexSpecificness'] else 0.0 for cond in conditions]


    # create json:

    result = {'Conditions': [], 'Therapies': [], 'Patients': []}
    for i, cond in enumerate(conditions):
        conditions[cond]['id'] = 'Cond' + str(i + 1)
        result['Conditions'].append({'id': conditions[cond]['id'], 'name': cond, 'type': conditions[cond]['conditionType']})
    for i, th in enumerate(therapies):
        therapies[th]['id'] = 'Th' + str(i + 1)
        result['Therapies'].append({'id': therapies[th]['id'], 'name': th, 'type': therapies[th]['therapyType']})
    pc_count = 0
    tr_count = 0
    dt_january_first_2022 = datetime.strptime('2022-01-01', '%Y-%m-%d')
    def str_dt(date_as_int):
        return str(dt_january_first_2022 + timedelta(days=date_as_int)).split()[0].replace('-', '')
    possible_test_cases = []
    #### Cured/uncured counter:
    # count_dict = {'cured': 0, 'uncured': 0}
    ####
    for i in range(60000):
        patient = {'id': str(i + 1)}
        patient['name'] = random.choice(first_names) + ' ' + random.choice(last_names)
        patient['sex'] = random.choice(('female', 'male'))
        patient['age'] = random.randint(0, 1000) / 10
        pconds_freq_weights = conds_freq_weights_f if patient['sex'] == 'female' else conds_freq_weights_m # else, i.e. 'male'
        pconds_k = None
        if patient['age'] == 0.0:
            pconds_k = 1
        elif patient['age'] == 100.0:
            pconds_k = 15
        else:
            age_based_slope = math.tan((patient['age'] / 100 - 0.5) * math.pi)
            pconds_k_weights = [1.1 ** (j * abs(age_based_slope)) / 1.5 ** abs(7 - j) for j in range(15)]
            pconds_k_weights = reversed(pconds_k_weights) if age_based_slope < 0 else pconds_k_weights
            pconds_k_weights = [w if j not in (0, 14) else w / 2.5 for j, w in enumerate(pconds_k_weights)] # smoothing extreme distribution edges
            pconds_k_weights = [w * (15 - j) ** 4 for j, w in enumerate(pconds_k_weights)] # general bias towards fewer conditions
            pconds_k = random.choices(range(1, 16), weights=pconds_k_weights, k=1)[0]
        pconds = random.choices(result['Conditions'], weights=pconds_freq_weights, k=pconds_k)
        patient['conditions'] = []
        patient['trials'] = []
        for pc in pconds:
            pc_starting_date = -random.randint(1, 1 + int(patient['age'] * 365.25 + random.random() * 36.252)) # -1 -> 31.12.2021, -365 -> 01.01.2021, ...
            another_conflicting_condition_of_same_kind = False
            for pc_item in patient['conditions']:
                if pc_item['kind'] == pc['id'] and (pc_item['cured'] == None or pc_item['cured'] >= str_dt(pc_starting_date)):
                    another_conflicting_condition_of_same_kind = True
                    break
            if not another_conflicting_condition_of_same_kind:
                pc_count += 1
                patient['conditions'].append({'id': 'pc' + str(pc_count), 'kind': pc['id']}) # 'id' not with final value!
            else:
                continue
            success = 0.0
            days = 0
            waiting_days = 0
            cond = conditions[pc['name']]
            possible_trials = list(zip(cond['assignedTherapies'], cond['jitteredEfficacyWeights']))
            while True:
                current_tr = random.choice(possible_trials)
                possible_trials.remove(current_tr)
                pth = therapies[current_tr[0]]
                tr_starting_date = pc_starting_date + days
                if tr_starting_date >= 0: # i.e. trial would *start* only in 2022, not gonna be included in dataset
                    possible_test_cases.append((str(i + 1) + ' [' + patient['name'] + ']', patient['age'], patient['sex'], pc['id'] + ' [' + pc['name'] + ']', str_dt(tr_starting_date - waiting_days), sorted([(therapies[y[0]]['id'] + ' [' + y[0] + ']', y[1]) for y in possible_trials + [current_tr]], key=lambda x: -x[1])))
                    break
                added_success = max(0, cond['curabilityWeight'] * current_tr[1] * 3 + (random.random() - 0.5) * 0.6)
                added_days = max(3, int(pth['durationWeight'] * 100 + random.randint(-30, 30)))
                success = min(1.0, success + added_success)
                days += added_days
                tr_end_date = pc_starting_date + days
                tr_count += 1
                patient['trials'].append({'id': 'tr' + str(tr_count), 'start': str_dt(tr_starting_date), 'end': str_dt(tr_end_date), 'condition': patient['conditions'][-1]['id'], 'therapy': pth['id'], 'successful': str(int(success * 100)) + '%'}) # 'id' and 'condition' not with final value!
                if success == 1.0 or len(possible_trials) == 0 or random.randint(1, 10) == 1:
                    break
                waiting_days = random.randint(1, 30)
                days += waiting_days
            pc_end_date = pc_starting_date + days
            patient['conditions'][-1]['diagnosed'] = str_dt(pc_starting_date)
            if success >= 1.0:
                assert success == 1.0
                patient['conditions'][-1]['cured'] = str_dt(pc_end_date)
                #### Cured/uncured counter:
                # count_dict['cured'] += 1
                ####
            else: # success < 1.0
                patient['conditions'][-1]['cured'] = None
                #### Cured/uncured counter:
                # count_dict['uncured'] += 1
                ####
        patient['conditions'].sort(key=lambda x: x['diagnosed']) # sort patient's conditions by starting date
        patient['trials'].sort(key=lambda x: x['start']) # sort patient's trials by starting date
        old_condition_ids, old_trial_ids = [x['id'] for x in patient['conditions']], [x['id'] for x in patient['trials']]
        condition_ids, trial_ids = sorted(old_condition_ids, key=lambda x: int(x.replace('pc', ''))), sorted(old_trial_ids, key=lambda x: int(x.replace('tr', '')))
        old2new_condition = {old: new for old, new in zip(old_condition_ids, condition_ids)}
        for j, pc_item in enumerate(patient['conditions']): # final values for 'id' after time-based sorting
            pc_item['id'] = condition_ids[j]
        for j, tr_item in enumerate(patient['trials']): # final values for 'id' and 'condition' after time-based sorting
            tr_item['id'] = trial_ids[j]
            tr_item['condition'] = old2new_condition[tr_item['condition']]
        result['Patients'].append(patient)
    output_filename = '../data/dataset.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, indent = 4, ensure_ascii=False))
    print('Successfully exported dataset to', output_filename.strip('..'))
    #### Cured/uncured counter:
    # for status in count_dict:
    #     print(status, count_dict[status])
    ####
    test_cases = []
    for ptc in possible_test_cases:
        if len(ptc[5]) >= 5 and len(set((x[1] for x in ptc[5][:5]))) == 5 and (len(ptc[5]) < 6 or ptc[5][4][1] != ptc[5][5][1]) and ptc[1] >= 5.0 and ptc[4].startswith('202201'):
            if ptc[0].split()[0] in ('7604', '10869', '59163'): # to select three specific patients with nice example properties
                while len(ptc[5]) > 5: # only keep first 5 most sensible expected recommendations
                    ptc[5].pop()
                test_cases.append(ptc)
    assert len(test_cases) == 3
    output_filename = '../src/test/testcases.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_cases, indent = 4, ensure_ascii=False))
    print('Successfully exported test cases to', output_filename.strip('..'))


if __name__ == '__main__':
    main()
