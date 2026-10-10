"""Fail-closed preparation protocol for fixed-prefix isolated memory scaling.

No runner is selected by importing this file. Numerical/schema checks are not
evidence of acquired model knowledge, new algorithms, or a causal sleep benefit.
"""
PROTOCOL = 'stage80_isolated_scale10240_v1'
PROFILE = {'teach_epochs': 20, 'sleep_epochs': 10, 'teach_lr': .003, 'sleep_lr': .001,
           'batch': 8, 'teacher_forms_per_semantic_fact': 2,
           'new_semantic_facts_per_block': 512, 'first_new_block': 2, 'last_new_block': 19,
           'old_training_replay_presentations': 0,
           'loss': 'explicit2D_masked_answer_mean_nll_v3',
           'old_network_and_all_nonactive_neural_banks_frozen': True}


def qualify_source(result, audit, result_sha256):
    manifest = result.get('manifest', {})
    if (result.get('status') != 'complete' or manifest.get('run_id') != 'isolated_second512_s42_v1'
            or manifest.get('protocol') != 'stage80_isolated_second512_v1'
            or manifest.get('source_run') != 'isolated_first512_s42_v2'
            or manifest.get('fixed_input_inventory_rows') != 5120
            or len(result.get('history', [])) != 1):
        raise ValueError('Exact completed disjoint second512 source required')
    gate = result.get('joint_gate', {})
    recall = gate.get('new512_accuracy_each_wording', {})
    if (not gate.get('passed') or not gate.get('all1892_old_raw_responses_exactly_unchanged')
            or set(recall) != {'exact', 'paraphrase'} or not all(v >= .95 for v in recall.values())
            or gate != audit.get('joint_gate')):
        raise ValueError('Source must pass current new-learning and full old-preservation gate')
    if (audit.get('status') != 'complete' or audit.get('run_id') != manifest['run_id']
            or not audit.get('integrity_raw_identity_and_actual_budget_passed')
            or not audit.get('joint_second512_gate_passed')
            or audit.get('input_hashes', {}).get('results.json') != result_sha256
            or result.get('old_training_replay_presentations') != 0):
        raise ValueError('Current matching independent source audit required')
    banks = result.get('all_nonactive_neural_banks_bit_identical', {})
    if (set(banks) != {str(i) for i in range(10) if i != 1}
            or not all(banks.values()) or banks != audit.get('all_nonactive_neural_banks_bit_identical')):
        raise ValueError('Current source inactive-bank proof required')
    return {'passed': True, 'source_results_sha256': result_sha256,
            'requires_actual_full_old2916_parity_after_inventory_extension_before_training': True,
            'requires_independent_initialization_of_appended5120_vectors': True,
            'overall_goal_achieved': False}


def validate_world(world, prefix_world):
    facts = world['novel_items']; old = prefix_world['novel_items']
    if (len(facts) != 10240 or len(old) != 5120 or facts[:5120] != old
            or world['known_items'] != prefix_world['known_items'] or len(world['known_items']) != 128):
        raise ValueError('Original names AND labels/questions/IDs must remain an exact prefix')
    if (len({row['id'] for row in facts}) != len(facts)
            or len({row['stimulus'] for row in facts}) != len(facts)
            or {row['stimulus'] for row in facts} & {row['stimulus'] for row in world['known_items']}):
        raise ValueError('Distinct disjoint semantic introductions required')
    return facts


def final_gate(source_final, final, score):
    old_keys = {'base_capabilities', 'confirmation', 'confirmation_v2', 'skill_audit',
                'skill_transfer', 'known_facts', 'known_paraphrase'}
    if set(final) != old_keys | {'exact', 'paraphrase'}:
        raise ValueError('Full final capability/skill/known/novel suites required')
    expected_old_counts = {'base_capabilities': 100, 'confirmation': 160, 'confirmation_v2': 160,
                           'skill_audit': 96, 'skill_transfer': 96,
                           'known_facts': 128, 'known_paraphrase': 128}
    if (any(len(final[k]) != n for k, n in expected_old_counts.items())
            or any(len(final[k]) != 10240 for k in ('exact', 'paraphrase'))):
        raise ValueError('Require all21348 actual final responses, never extrapolate samples')
    old_equal = all(final[k] == source_final[k] for k in old_keys)
    for key, prior_key in [('exact', 'prior512_facts'), ('paraphrase', 'prior512_paraphrase')]:
        old_equal = old_equal and final[key][:1024] == source_final[prior_key] + source_final[key]
    cohort_recall = [{key: sum(score(r['response'], r['expected']) for r in final[key][i:i + 512]) / 512
                      for key in ('exact', 'paraphrase')} for i in range(0, 10240, 512)]
    return {'passed': old_equal and all(v >= .95 for cohort in cohort_recall for v in cohort.values()),
            'all2916_source_old_raw_responses_exactly_unchanged': old_equal,
            'each512_cohort_recall': cohort_recall, 'four_context_acquisition_must_be_separately_measured': True,
            'new_skill_sleep_causality_final_blind_audit_and_replication_pending': True,
            'overall_goal_achieved': False}
