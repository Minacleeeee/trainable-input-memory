"""Qualification/gates for a new named conditional composition, not inference.

The learner may use only its neural forward pass. Teacher branch/rule/answers
are NEVER generation arguments. A named task is not an unprecedented algorithm.
"""
PROTOCOL = 'stage80_isolated_conditional_skill_v5'
PROFILE = {'teach_epochs': 20, 'sleep_epochs': 10, 'teach_lr': .003, 'sleep_lr': .001,
           'batch': 8, 'teacher_rows': 256, 'active_neural_input_rows': 1,
           'active_trainable_parameters': 2048, 'old_training_replay_presentations': 0,
           'loss': 'explicit2D_masked_answer_mean_nll_v3', 'old_network_and_all_old_banks_frozen': True,
           'minimum_audit_and_transfer_accuracy': .8, 'minimum_each_teacher_branch_accuracy': .8,
           'minimum_absolute_gain_on_each_audit': .2}


def qualify_source(result, audit, result_sha256):
    m = result.get('manifest', {})
    if (result.get('status') != 'complete' or m.get('protocol') != 'stage80_isolated_scale10240_v2'
            or m.get('run_id') != 'isolated_scale10240_s42_v2'
            or m.get('fixed_input_inventory_rows') != 10240
            or len(result.get('history', [])) != 18 or result.get('full_saved_reload_final_queries') != 21348):
        raise ValueError('Exact completed full-scale v2 source required')
    gate = result.get('joint_gate', {})
    if (not gate.get('passed') or not gate.get('all2916_source_old_raw_responses_exactly_unchanged')
            or not result.get('memory_scale_gate_passed') or gate != audit.get('joint_gate')):
        raise ValueError('Full memory/retention source must pass, not merely a provisional bank commit')
    if (audit.get('status') != 'complete' or audit.get('run_id') != m['run_id']
            or not audit.get('integrity_raw_identity_and_actual_budget_passed')
            or not audit.get('joint_memory_retention_gate_passed')
            or not audit.get('memory_scale_gate_passed')
            or audit.get('input_hashes', {}).get('results.json') != result_sha256):
        raise ValueError('Current matching independent full-scale source audit required')
    scale = result.get('cumulative_dual_wording_acquisition', {})
    if scale.get('compact_acquired_tokens', 0) < 32768 or not scale.get('acquired_four_contexts'):
        raise ValueError('Do not substitute corpus/replay length for acquired compact scale')
    return {'passed': True, 'source_results_sha256': result_sha256,
            'source_memory_goal_measured_not_whole_project_success': True,
            'requires_actual_new_skill_baseline_before_teaching': True,
            'requires_actual_later_new_fact_learning_skill_retention': True,
            'overall_goal_achieved': False}


def validate_development_groups(groups, name):
    """Only development splits; the reserved final split is excluded entirely."""
    required = {'train': 256, 'dev': 64, 'audit': 192, 'transfer_audit': 192}
    if set(groups) != set(required):
        raise ValueError('Pass only development groups; never the reserved final split')
    all_ids = set(); all_args = set(); questions = set()
    for key, count in required.items():
        rows = groups[key]
        if len(rows) != count: raise ValueError('Incomplete predeclared split')
        branches = {'even_first_rotate_left': 0, 'odd_first_reverse': 0}
        for row in rows:
            args = tuple(row['arguments']); branch = row['teacher_branch']
            if (len(args) != 3 or len(set(args)) != 3 or row['id'] in all_ids
                    or args in all_args or row['eval_q'] in questions or row['skill'] != name
                    or branch not in branches):
                raise ValueError('Repeated tuple/question or changed operation/branch')
            values = (set(range(16)) | set(range(64,96))) if key == 'train' else (range(16,64) if key == 'transfer_audit' else range(16))
            if any(value not in values for value in args):
                raise ValueError('Numeric-value transfer split contaminated by teaching values')
            expected_question = name + ': ' + ' '.join(map(str, args))
            if row['eval_q'] != row['train_q'] or row['eval_q'] != expected_question:
                raise ValueError('Query changed or teacher instruction leaked into inference question')
            all_ids.add(row['id']); all_args.add(args); questions.add(row['eval_q']); branches[branch] += 1
        if set(branches.values()) != {count // 2}: raise ValueError('Both branches must be balanced')
    bands = {band: [r for r in groups['train'] if all(v in range(*band) for v in r['arguments'])] for band in ((0,16),(64,96))}
    if any(len(rows)!=128 or sum(r['arguments'][0]%2==0 for r in rows)!=64 for rows in bands.values()):
        raise ValueError('Require128balanced rows in each predeclared teaching band, no evaluation values')
    if sum(map(len,bands.values()))!=256: raise ValueError('Mixed teacher bands require a separately versioned protocol')
    return True


def gate_from_raw(groups, baseline, final, old_raw_equal, score):
    """Evaluation only; never produces a model reply or trains on failed audits."""
    if set(baseline) != {'dev', 'audit', 'transfer_audit'} or set(final) != set(baseline):
        raise ValueError('Actual BEFORE and saved-model final skill responses required')
    measurements = {}
    for key, before in baseline.items():
        items = groups[key]; after = final[key]
        if len(before) != len(items) or len(after) != len(items): raise ValueError('Full skill suites required')
        for item, b, a in zip(items, before, after):
            for row in (b, a):
                if (row['id'] != item['id'] or row['question'] != item['eval_q']
                        or row['expected'] != item['answer']):
                    raise ValueError('Changed question, ordering or target in skill comparison')
        n = len(items)
        initial = sum(score(r['response'], r['expected']) for r in before)
        learned = sum(score(r['response'], r['expected']) for r in after)
        newly_correct = sum(not score(b['response'], b['expected']) and score(a['response'], a['expected'])
                            for b, a in zip(before, after))
        branches = {}
        for branch in ('even_first_rotate_left', 'odd_first_reverse'):
            indices = [i for i, row in enumerate(items) if row['teacher_branch'] == branch]
            if not indices: raise ValueError('Both conditional branches must be tested')
            branches[branch] = sum(score(after[i]['response'], after[i]['expected']) for i in indices) / len(indices)
        measurements[key] = {'questions': n, 'before_correct': initial, 'after_correct': learned,
                             'after_accuracy': learned / n, 'absolute_gain': (learned - initial) / n,
                             'newly_correct': newly_correct, 'branch_accuracy': branches}
    assessed = [measurements[k] for k in ('audit', 'transfer_audit')]
    passed = bool(old_raw_equal) and all(m['after_accuracy'] >= .8 and m['absolute_gain'] >= .2
                                         and all(v >= .8 for v in m['branch_accuracy'].values()) for m in assessed)
    return {'passed': passed, 'all21348_old_raw_responses_exactly_unchanged': bool(old_raw_equal),
            'new_task_measurements': measurements,
            'later_new_fact_learning_retention_final_blind_audit_and_replication_pending': True,
            'overall_goal_achieved': False}
