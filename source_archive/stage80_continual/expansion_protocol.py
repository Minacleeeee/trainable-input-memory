"""CPU-only qualification and fixed-world continuation; no training or inference.

A future expansion must extend the exact immutable curriculum prefix learned
by a completed, independently audited parent. Never generate replacement labels.
"""
from storage import digest


def qualify_parent(result,audit,results_file_hash,skill_details):
    if result.get('status')!='complete': raise ValueError('Parent is incomplete')
    if not audit.get('integrity_and_budget_checks_passed') or not audit.get('pilot_joint_gate_passed'):
        raise ValueError('Parent did not pass the predeclared joint development gate')
    if audit['run_id']!=result['manifest']['args']['run_id']:
        raise ValueError('Parent audit belongs to another experiment')
    if audit['input_hashes']['results.json']!=results_file_hash:
        raise ValueError('Parent assessment is stale')
    # The minimal80% skill gate can pass while replacing many old correct
    # transfer answers. Expansion requires retaining >=90% of those answers.
    for kind in ('skill_audit','skill_transfer'):
        retention=audit['trained_parent_retention'][kind]['retention_of_previously_correct']
        if retention is None or retention<.9:
            raise ValueError('Skill retention is too fragile for expansion')
    if skill_details['mode']!='completed_final' or skill_details['run_id']!=audit['run_id']:
        raise ValueError('Need final per-operation skill assessment')
    if skill_details['input_hashes']['results.json']!=results_file_hash or skill_details['input_hashes']['baseline.json']!=audit['input_hashes']['baseline.json']:
        raise ValueError('Per-operation assessment is stale')
    if set(skill_details['per_operation_retention'])!={'skill_audit','skill_transfer'}:
        raise ValueError('Both combination and transfer skill groups are required')
    for groups in skill_details['per_operation_retention'].values():
        if set(groups)!={'mivok','tavrel'}: raise ValueError('Both individual operations are required')
        for metric in groups.values():
            if metric['after_accuracy']<.8 or metric['retention_of_previously_correct'] is None or metric['retention_of_previously_correct']<.9:
                raise ValueError('An individual skill fails the stronger expansion gate')


def prepare_expansion(curriculum,result,source_facts,source_known,new_count,block):
    if result['status']!='complete' or result['history'][-1]['facts_taught']!=len(source_facts):
        raise ValueError('Only a completed fully taught fact archive can be inherited')
    if digest(source_facts)!=result['manifest']['dataset_hash']:
        raise ValueError('Source fact archive changed')
    if digest(source_known)!=result['manifest']['known_dataset_hash']:
        raise ValueError('Source known-fact archive changed')
    novel,old=curriculum['novel_items'],curriculum['known_items']
    if digest(novel)!=curriculum['novel_dataset_hash'] or digest(old)!=curriculum['known_dataset_hash']:
        raise ValueError('Frozen curriculum changed')
    def identities(rows):
        values={r['id']:(r['stimulus'],r['train_q'],r['eval_q'],r['audit_q'],r['answer']) for r in rows}
        if len(values)!=len(rows): raise ValueError('Duplicate source identity')
        return values
    all_ids=identities(old+novel)
    if len({r['stimulus'] for r in old+novel})!=len(old+novel):
        raise ValueError('Overlapping nonce entities')
    inherited=source_known+source_facts
    known_ids=identities(inherited)
    if any(all_ids.get(key)!=value for key,value in known_ids.items()):
        raise ValueError('Inherited knowledge uses different labels or questions')
    old_ids={r['id'] for r in old}
    if not old_ids.issubset(known_ids): raise ValueError('Original known facts disappeared')
    offset=len(known_ids)-len(old_ids)
    if set(known_ids)!=old_ids|{r['id'] for r in novel[:offset]}:
        raise ValueError('Known novel items are not an exact completed curriculum prefix')
    if new_count<1 or block<1 or offset+new_count>len(novel):
        raise ValueError('Invalid extension size')
    selected=[{**r,'curriculum_cohort':r.get('curriculum_cohort',r['cohort']),
               'cohort':i//block+1} for i,r in enumerate(novel[offset:offset+new_count])]
    known=[{**r,'source_cohort':r['cohort'],
            'origin_cohort':r.get('origin_cohort',r['cohort']),'cohort':0} for r in inherited]
    return selected,known,{'curriculum_offset':offset,'new_fact_count':new_count,
                           'inherited_known_count':len(known),'cumulative_novel_count':offset+new_count,
                           'inference_policy':'No curriculum or archive access during inference'}
