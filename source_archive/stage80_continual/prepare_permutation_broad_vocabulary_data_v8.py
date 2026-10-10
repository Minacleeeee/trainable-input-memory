"""Prospective CPU curriculum; no model answers, gradients or GPU dispatch.

Vocabulary is declared here independently of failed answers. Existing reserved
final datasets are never opened. This remains a development candidate, not an
admitted training runner or a pure one-variable ablation of V7.
"""
import collections
import datetime
import random
from pathlib import Path
from prepare_permutation_skill_v7 import read, sha, DATA
from storage import atomic_json, digest
from model_support import load_tokenizer, format_chat
from isolated_inputs_v2 import add_input_names
from nontruncating_chat_cache_v1 import encode_items

SEED = 20261005 + 1888
NAME = 'qezorvank'
ORDER = (2, 0, 3, 1)
TRAIN_WORDS = '''apple banana pear grape peach lemon melon cherry
tomato onion potato carrot celery pepper garlic ginger
bread butter cheese yogurt coffee tea juice water
spoon fork knife plate bowl cup glass bottle
chair table sofa bed lamp clock desk shelf
door window wall floor roof garden fence gate
river mountain ocean valley forest meadow island beach
dog cat horse sheep goat pig duck hen'''.split()
TRANSFER_WORDS = '''wheat barley rye oats hammer wrench chisel pliers
scooter canoe wagon tractor coral amber nickel silver'''.split()

def make_row(group, index, values):
    question = NAME + ': ' + ' '.join(values)
    return {'id': f'permutation_v8_{group}_{index}',
            'type': 'four_slot_permutation_skill', 'skill': NAME,
            'train_q': question, 'eval_q': question,
            'answer': ' '.join(values[i] for i in ORDER)}

def main():
    target = DATA / 'permutation_broad_vocabulary_teaching_seed20261005_v8.json'
    if target.exists(): raise FileExistsError('Fresh immutable candidate only')
    assert len(set(TRAIN_WORDS)) == 64 and len(set(TRANSFER_WORDS)) == 16
    assert not set(TRAIN_WORDS) & set(TRANSFER_WORDS)
    # Existing DEVELOPMENT vocabulary only, never reserved final files.
    prior = read(DATA / 'permutation_skill_teaching_seed20261005_v7.json')
    previous = set(prior['train_vocabulary']) | set(prior['transfer_vocabulary'])
    assert not previous & (set(TRAIN_WORDS) | set(TRANSFER_WORDS))
    rng = random.Random(SEED)
    seen = set()
    groups = {'train': []}
    # Eight independently shuffled 64-row rounds. Every teacher word occurs
    # exactly eight times in each input position, without repeated tuples.
    for _ in range(8):
        while True:
            words = TRAIN_WORDS.copy()
            rng.shuffle(words)
            shifts = rng.sample(range(64), 4)
            values = [tuple(words[(i + shift) % 64] for shift in shifts) for i in range(64)]
            if len(set(values)) == 64 and not seen & set(values): break
        for value in values:
            seen.add(value)
            groups['train'].append(make_row('train', len(groups['train']), value))
    for group, count, vocabulary in [('dev', 64, TRAIN_WORDS), ('audit', 192, TRAIN_WORDS),
                                      ('transfer_audit', 192, TRANSFER_WORDS)]:
        rows = []
        while len(rows) < count:
            value = tuple(rng.sample(vocabulary, 4))
            if value in seen: continue
            seen.add(value)
            rows.append(make_row(group, len(rows), value))
        groups[group] = rows
    positional = [collections.Counter(row['eval_q'].split(': ', 1)[1].split()[i]
                   for row in groups['train']) for i in range(4)]
    assert all(c == {w: 8 for w in TRAIN_WORDS} for c in positional)
    assert len(seen) == 960
    source = read(DATA / 'isolated_scale10240_s42_v2/results.json')
    lexical = read(Path(source['checkpoint']) / 'lexical_inventory.json')
    original = load_tokenizer()
    old = load_tokenizer()
    add_input_names(old, lexical['names'], 49152)
    tok = load_tokenizer()
    add_input_names(tok, lexical['names'] + [NAME], 49152)
    questions = [row['question'] for rows in source['final_raw'].values() for row in rows]
    chats = [format_chat(old, q) for q in questions]
    assert old(chats, add_special_tokens=False)['input_ids'] == tok(chats, add_special_tokens=False)['input_ids']
    cache = encode_items(tok, groups['train'], max_length=64, context_capacity=8192)
    maximum = max(len(ids) for ids, _ in cache)
    tokens = sum(int((labels[1:] != -100).sum()) for _, labels in cache)
    assert len(cache) == 512 and maximum <= 64
    for row, (ids, labels) in zip(groups['train'], cache):
        full = tok(format_chat(tok, row['train_q'], row['answer']), add_special_tokens=False)['input_ids']
        prompt = tok(format_chat(tok, row['train_q']), add_special_tokens=False)['input_ids']
        assert ids.tolist() == full
        assert int((labels[1:] != -100).sum()) == len(full) - len(prompt) > 0
        assert all(i < 49152 for i in full[len(prompt):])
    atomic_json(target, {
        'status': 'prospective_CPU_development_data_NOT_admitted_production_or_skill_result',
        'recorded_at': datetime.datetime.now().astimezone().isoformat(),
        'seed': SEED, 'operation_name': NAME, 'groups': groups,
        'group_hashes': {key: digest(value) for key, value in groups.items()},
        'teacher_rule_training_and_scoring_only': {'zero_based_output_position_order': list(ORDER)},
        'train_vocabulary': TRAIN_WORDS, 'transfer_vocabulary': TRANSFER_WORDS,
        'each_train_word_presentations_per_position_per_epoch': 8,
        'original_BPE_token_count_by_train_word': {w: len(original(w, add_special_tokens=False)['input_ids']) for w in TRAIN_WORDS},
        'new_vocabulary_disjoint_from_previous_V7_development_vocabulary': True,
        'four_distinct_values_each_tuple': True, 'all960_tuples_unique_no_split_overlap': True,
        'actual_old_chat_tokenizations_unchanged': len(chats),
        'full_teacher_masks_checked': 512, 'maximum_teacher_chat_tokens': maximum,
        'teacher_supervised_tokens_per_epoch': tokens,
        'prospective_20_teach_10_sleep_batch8_updates': 1920,
        'prospective_presentations': 15360, 'prospective_supervised_tokens': tokens * 30,
        'changes_vocabulary_sample_count_and_domain_NOT_pure_capacity_or_compute_matched_V7_ablation': True,
        'any_future_input_and_shared_pair_requires_same_data_budget_actual_before_and_full_old_raw_gate': True,
        'source_results_sha256': sha(DATA / 'isolated_scale10240_s42_v2/results.json'),
        'data_generator_sha256': sha(Path(__file__)),
        'dependencies_sha256': {n: sha(Path(__file__).parent / n) for n in
                                 ['nontruncating_chat_cache_v1.py', 'model_support.py', 'isolated_inputs_v2.py', 'storage.py']},
        'failed_answers_not_recycled_as_teacher_examples': True,
        'reserved_final_data_NOT_read_or_vocabulary_reused': True,
        'actual_new_skill_baselines_NOT_measured': True,
        'GPU_calls_production_model_answers_and_optimizer_updates': 0,
        'new_runner_independent_auditor_and_GPU_admission_REQUIRED': True,
        'numeric_branch_algorithm_absence_later_retention_NOT_established': True,
        'overall_goal_achieved': False,
    })
    print({'teachers': 512, 'new_train_words': 64, 'full_masks': 512,
           'max_tokens': maximum, 'supervised_tokens_per_epoch': tokens,
           'old_chat_tokenizations_unchanged': len(chats), 'GPU_dispatch': False})

if __name__ == '__main__': main()
