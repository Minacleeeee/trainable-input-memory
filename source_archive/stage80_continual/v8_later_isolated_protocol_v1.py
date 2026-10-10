"""Isolated new-bank production policy, distinct from plastic sleep mechanisms."""
from v8_later_learning_contract_v1 import PROFILE as CONTRACT, fact_after_learning, skill_after_learning
from scale_protocol_v1 import validate_rows
from storage import exact_score

PROTOCOL='stage80_V8_qualified_skill_then_isolated512_v1'
PROFILE={**CONTRACT,'mode':'isolated_new_bank','teach_lr':.003,'sleep_lr':.001,
         'active_trainable_parameters':1048576,'new_bank_bytes':4194304,
         'loss':'explicit2D_masked_answer_mean_nll_v3','old_and_shared_network_frozen':True}
PREPARED='prepared_V8_later_isolated_stage_v1.json'
AUDIT='independent_V8_later_isolated_stage_audit_v1.json'
PREFLIGHT_AUDIT='independent_V8_later_isolated_runtime_audit_v1.json'


def final_gate(parent_old,parent_skill,old_before,skill_before,old_after,skill_after,
               facts,new_before,new_after,original_skill_before):
    if sum(map(len,parent_old.values()))!=21348 or sum(map(len,parent_skill.values()))!=448:
        raise ValueError('Complete qualified parent references required')
    for raw in (old_before,old_after):
        if set(raw)!=set(parent_old):raise ValueError('Complete old suites required')
        for k,rows in parent_old.items():
            items=[{'id':r['id'],'type':r['type'],'eval_q':r['question'],'answer':r['expected']} for r in rows]
            validate_rows(raw[k],items)
            if [r['id'] for r in raw[k]]!=[r['id'] for r in rows] or any(r['correct']!=exact_score(r['response'],r['expected']) for r in raw[k]):
                raise ValueError('Changed old order or fabricated score')
    retained=skill_after_learning(parent_skill,skill_after,original_skill_before)
    # Fresh expanded-model BEFORE must match source, including wrong replies.
    parity=old_before==parent_old and skill_before==parent_skill
    for raw in (new_before,new_after):
        for rows in raw.values():
            if [r['id'] for r in rows]!=[r['id'] for r in facts] or any(r['correct']!=exact_score(r['response'],r['expected']) for r in rows):
                raise ValueError('Changed fact order or fabricated score')
    memory=fact_after_learning(facts,new_before,new_after)
    old_exact=old_after==parent_old
    passed=parity and old_exact and retained['all448_source_skill_raw_exact'] and memory['minimum_each_wording_pass']
    return {'passed':bool(passed),'complete_expanded_model_old_skill_BEFORE_source_raw_exact':parity,
            'all21348_old_raw_exact_after_later_learning':old_exact,'skill_retention':retained,
            'new512_memory':memory,'isolation_NOT_sleep_causal_evidence':True,
            'reserved_independent_worlds_replication_plasticity_and_sleep_controls_pending':True,
            'overall_goal_achieved':False}
