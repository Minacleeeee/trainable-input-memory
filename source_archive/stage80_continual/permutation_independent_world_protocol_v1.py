"""Preregistered fresh-vocabulary replication; teacher rules never enter inference."""
import copy
from permutation_broad_skill_protocol_v8 import (
    PROFILE as PRIOR_PROFILE, CODE_NAMES as PRIOR_CODE_NAMES,
    qualify_source, gate_from_raw, validate_development_groups as validate_structure)
from prepare_permutation_independent_world_CPU_v1 import TRAIN_WORDS, TRANSFER_WORDS, SEED
from storage import digest

PROTOCOL = 'stage80_independent_word_world_seed20263945_v1'
PROFILE = copy.deepcopy(PRIOR_PROFILE)
PROFILE.update(training_seeds={'teach': SEED + 101, 'sleep': SEED + 102},
               independent_data_seed=SEED,
               fresh_original_scale_source_required=True,
               independent_vocabulary_not_algorithm_blind_design=True)
REPLACEMENTS = {
    'permutation_broad_skill_protocol_v8.py': 'permutation_independent_world_protocol_v1.py',
    'run_permutation_broad_skill_v8.py': 'run_permutation_independent_world_v1.py',
    'audit_permutation_broad_skill_v8.py': 'audit_permutation_independent_world_v1.py',
    'prepare_permutation_broad_skill_v8.py': 'prepare_permutation_independent_world_production_v1.py',
    'test_permutation_broad_skill_protocol_v8.py': 'test_permutation_independent_world_protocol_v1.py',
    'shared_skill_training_v1.py': 'shared_skill_training_independent_world_v1.py',
    'audit_permutation_broad_runtime_preflight_cpu_v8.py': 'audit_permutation_independent_world_runtime_v1.py',
}
CODE_NAMES = [REPLACEMENTS.get(name, name) for name in PRIOR_CODE_NAMES]
CODE_NAMES += ['permutation_broad_skill_protocol_v8.py',
               'prepare_permutation_independent_world_CPU_v1.py']
DATASET_NAME = 'permutation_independent_world_seed20263945_CPU_v1.json'
PREPARED_NAME = 'prepared_permutation_independent_world_production_v1.json'
PRIOR_DATASETS = ('permutation_skill_teaching_seed20261005_v7.json',
                  'permutation_broad_vocabulary_teaching_seed20261005_v8.json')


def validate_development_groups(groups, name):
    validate_structure(groups, name)
    train = {word for row in groups['train'] for word in row['eval_q'].split(': ', 1)[1].split()}
    transfer = {word for row in groups['transfer_audit'] for word in row['eval_q'].split(': ', 1)[1].split()}
    if train != set(TRAIN_WORDS) or transfer != set(TRANSFER_WORDS):
        raise ValueError('Exact preregistered independent vocabulary required')
    return True


def validate_candidate(candidate, previous):
    if (candidate['seed'] != SEED or candidate['operation_name'] != 'qezorvank'
            or candidate['source_results_sha256'] !=
            'fca0b325a3dffcfea86e1feee49b5e729f37c5a96d6d19e5958013655b7110b4'
            or candidate['production_GPU_model_answers_or_training_updates'] != 0):
        raise ValueError('Fresh unqueried candidate and original qualified source required')
    validate_development_groups(candidate['groups'], candidate['operation_name'])
    if candidate['group_hashes'] != {k: digest(v) for k, v in candidate['groups'].items()}:
        raise ValueError('Candidate group fingerprint mismatch')
    if set(previous) != set(PRIOR_DATASETS):
        raise ValueError('Both prior vocabulary inventories required')
    old_words = set()
    for data in previous.values():
        old_words.update(data['train_vocabulary'])
        old_words.update(data['transfer_vocabulary'])
    if old_words & (set(TRAIN_WORDS) | set(TRANSFER_WORDS)):
        raise ValueError('New world overlaps old development vocabulary')
    if (candidate['teacher_answer_supervised_tokens_per_epoch'] != 3520
            or candidate['prospective_supervised_tokens'] != 105600
            or candidate['teacher_masks_checked'] != 512
            or candidate['prospective_updates'] != 1920):
        raise ValueError('Preregistered distinct token budget mismatch')
    return True
