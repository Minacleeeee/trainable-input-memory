"""Fail-closed later-learning retention protocol; never used for inference."""
from storage import digest

PROTOCOL = 'stage80_post_conditional_memory512_v5'
PROFILE = {'teach_epochs': 20, 'sleep_epochs': 10, 'teach_lr': .003, 'sleep_lr': .001,
           'batch': 8, 'teacher_forms_per_semantic_fact': 2, 'new_semantic_facts': 512,
           'active_trainable_parameters': 1048576, 'active_neural_bank': 21,
           'old_training_replay_presentations': 0,
           'loss': 'explicit2D_masked_answer_mean_nll_v3',
           'old_network_all_old_memory_and_operation_banks_frozen': True}


def qualify_source(result, audit, result_sha256):
    m = result.get('manifest', {}); gate = result.get('joint_gate', {})
    if (result.get('status') != 'complete' or m.get('protocol') != 'stage80_isolated_conditional_skill_v5'
            or m.get('run_id') != 'conditional_skill_broad_s42_v5' or m.get('input_inventory_rows') != 10241
            or len(result.get('history', [])) != 1 or result.get('full_saved_reload_final_queries') != 21796):
        raise ValueError('Exact COMPLETE conditional-task source required; no partial checkpoints')
    if (not gate.get('passed') or not gate.get('all21348_old_raw_responses_exactly_unchanged')
            or not result.get('all_old20_neural_banks_bit_identical') or gate != audit.get('joint_gate')):
        raise ValueError('Immediate task and full old-retention source gates must pass')
    if (audit.get('status') != 'complete' or audit.get('run_id') != m['run_id']
            or not audit.get('integrity_raw_identity_and_actual_budget_passed')
            or not audit.get('immediate_conditional_skill_joint_gate_passed')
            or not audit.get('all_old20_neural_banks_bit_identical')
            or audit.get('input_hashes', {}).get('results.json') != result_sha256):
        raise ValueError('Matching independent immediate-task source audit required')
    qualification = m.get('source_qualification', {})
    if not qualification.get('passed') or not qualification.get('source_memory_goal_measured_not_whole_project_success'):
        raise ValueError('Source must descend from qualified full acquired-memory scale')
    return {'passed': True, 'source_results_sha256': result_sha256,
            'later_skill_retention_requires_actual_final_generations': True,
            'overall_goal_achieved': False}


def validate_curriculum(curriculum, world, operation):
    old = world['novel_items']; known = world['known_items']; rows = curriculum['new512_items']
    if len(old) != 10240 or curriculum['old10240_semantic_prefix_sha256'] != digest(old):
        raise ValueError('Exact original10240 semantic prefix required')
    if curriculum['operation_lexical_id_preserved'] != 59392 or curriculum['new_rows_start_id'] != 59393:
        raise ValueError('Append after operation; never shift its lexical ID')
    if len(rows) != 512 or curriculum['new512_dataset_hash'] != digest(rows):
        raise ValueError('Complete fingerprinted later512 required')
    from memory_dataset import LABELS
    if {r['id'] for r in rows} & {r['id'] for r in old + known}:
        raise ValueError('New semantic identities cannot overlap old facts')
    forbidden = {r['stimulus'] for r in old + known} | {operation}
    if (len({r['id'] for r in rows}) != 512 or len({r['stimulus'] for r in rows}) != 512
            or {r['stimulus'] for r in rows} & forbidden):
        raise ValueError('Distinct later names required, not repeated introductions')
    if set(r['answer'] for r in rows) != set(LABELS): raise ValueError('Fixed label set required')
    if any(sum(r['answer'] == label for r in rows) != 32 for label in LABELS):
        raise ValueError('Prepared balanced labels must be preserved')
    for r in rows:
        if (r['type'] != 'fact' or r['train_q'] != r['eval_q']
                or r['eval_q'] != 'What color is a ' + r['stimulus'] + '?'
                or r['audit_q'] != 'Tell me the color of a ' + r['stimulus'] + '.'):
            raise ValueError('Changed teaching or withheld questions')
    return rows


def final_gate(parent_old, parent_skill, final_old, final_skill, new_exact, new_withheld, score):
    if (sum(map(len, parent_old.values())) != 21348 or sum(map(len, parent_skill.values())) != 448
            or set(final_old) != set(parent_old) or set(final_skill) != set(parent_skill)
            or any(len(final_old[k]) != len(v) for k, v in parent_old.items())
            or any(len(final_skill[k]) != len(v) for k, v in parent_skill.items())
            or len(new_exact) != 512 or len(new_withheld) != 512):
        raise ValueError('All22820 final model responses required, not samples')
    old_equal = final_old == parent_old; skill_equal = final_skill == parent_skill
    recall = {key: sum(score(r['response'], r['expected']) for r in rows) / 512
              for key, rows in [('exact', new_exact), ('paraphrase', new_withheld)]}
    return {'passed': old_equal and skill_equal and all(v >= .95 for v in recall.values()),
            'all21348_old_memory_capability_raw_responses_exactly_unchanged': old_equal,
            'all448_new_conditional_skill_raw_responses_exactly_unchanged': skill_equal,
            'new512_accuracy_each_wording': recall,
            'reserved_final_sleep_causality_and_training_replication_pending': True,
            'overall_goal_achieved': False}


def acquisition_segments(data, world, later_rows, later_baseline, read):
    """Use EACH original actual baseline, never current source answers as baseline."""
    facts = world['novel_items']
    first = read(data / 'isolated_first512_s42_v2/baseline.json')
    second = read(data / 'isolated_second512_s42_v1/baseline.json')
    segments = [{'name': 'isolated_first512', 'facts': facts[:512],
                 'baseline_canonical': first['exact'], 'baseline_withheld': first['paraphrase']},
                {'name': 'isolated_second512', 'facts': facts[512:1024],
                 'baseline_canonical': second['exact'], 'baseline_withheld': second['paraphrase']}]
    for index in range(2, 20):
        b = read(data / 'isolated_scale10240_s42_v2/blocks' / f'block_{index:02d}' / 'baseline.json')
        segments.append({'name': f'isolated_block{index}', 'facts': facts[index * 512:(index + 1) * 512],
                         'baseline_canonical': b['exact'], 'baseline_withheld': b['paraphrase']})
    segments.append({'name': 'post_conditional_new512', 'facts': later_rows,
                     'baseline_canonical': later_baseline['exact'], 'baseline_withheld': later_baseline['paraphrase']})
    return segments
