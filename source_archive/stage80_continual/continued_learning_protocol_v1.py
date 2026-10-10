"""A repaired parent must recover before the next disjoint learning challenge.

This qualification supports the repair audit schema explicitly; it never
relabels a failed original pilot as a qualified parent. CPU-only provenance.
"""
import math
from expansion_protocol import prepare_expansion


def _operations(groups):
    if set(groups)!={'skill_audit','skill_transfer'}:
        raise ValueError('Both combination and vocabulary suites required')
    for operations in groups.values():
        if set(operations)!={'mivok','tavrel'}:
            raise ValueError('Both individual skills required')
        for row in operations.values():
            acc=row['after_accuracy']; retention=row['retention_of_previously_correct']
            if acc is None or retention is None or not all(math.isfinite(v) for v in (acc,retention)):
                raise ValueError('Missing/nonfinite skill evidence')
            if not .8<=acc<=1 or not .9<=retention<=1:
                raise ValueError('An individual operation failed accuracy or retention')


def _categories(suites):
    required={'base_capabilities':{'arithmetic','factual','instruction','language'},
              'confirmation':{'arithmetic','comparison','instruction','language'},
              'confirmation_v2':{'arithmetic','comparison','instruction','language'}}
    if set(suites)!=set(required): raise ValueError('All capability suites required')
    for kind,groups in suites.items():
        if set(groups)!=required[kind]: raise ValueError('A capability category disappeared')
        for row in groups.values():
            change=row['change_pp']
            if change is None or not math.isfinite(change) or change < -5:
                raise ValueError('Original capability category regressed')


def qualify_repair(result,audit,result_file_hash,skill_details,maintenance_baseline_hash):
    if result.get('status')!='complete' or result.get('novel_fact_count')!=0:
        raise ValueError('Require a completed explicitly zero-novel repair')
    if result['manifest']['protocol']!='stage80_zero_novel_skill_repair_v1':
        raise ValueError('Unexpected parent protocol')
    run_id=result['manifest']['args']['run_id']
    if audit.get('run_id')!=run_id or not audit.get('integrity_and_budget_checks_passed') or not audit.get('repair_joint_gate_passed'):
        raise ValueError('Repair integrity/joint gate failed')
    if audit['input_hashes']['results.json']!=result_file_hash:
        raise ValueError('Stale repair audit')
    if audit['maintenance_reference_baseline_hash']!=maintenance_baseline_hash:
        raise ValueError('Earlier maintenance reference changed')
    if skill_details['mode']!='completed_final' or skill_details['run_id']!=run_id:
        raise ValueError('Only the final matching operation audit qualifies')
    for key in ('results.json','baseline.json'):
        if skill_details['input_hashes'][key]!=audit['input_hashes'][key]:
            raise ValueError('Stale individual-skill audit')
    _operations(skill_details['per_operation_retention'])
    if skill_details['per_operation_retention']!=audit['parent_per_operation_retention']:
        raise ValueError('Independent skill measurements disagree')
    _operations(audit['maintenance_reference_per_operation_retention'])
    _categories(audit['retention_by_category'])
    _categories(audit['maintenance_reference_category_retention'])
    for key in ('exact','paraphrase','known_facts','known_paraphrase'):
        counts=audit['raw_answer_counts'][key]
        if counts['items']<1 or counts['correct']/counts['items']<.95:
            raise ValueError('Known mappings are insufficiently retained under both forms')
        if not math.isclose(result['final_accuracy'][key],counts['correct']/counts['items']):
            raise ValueError('Accuracy summary disagrees with raw-answer audit')


def prepare_challenge(curriculum,result,source_facts,source_known):
    facts,known,metadata=prepare_expansion(curriculum,result,source_facts,source_known,512,512)
    if metadata['curriculum_offset']!=512 or len(known)!=640:
        raise ValueError('This challenge extends the completed first512 plus old128 only')
    metadata['purpose']='Test retention through512 genuinely new facts after zero-novel repair'
    metadata['large_scale_claim']=False
    return facts,known,metadata
