"""Prospective scale qualification and diagnostics; CPU only, no answer generation."""
import math
import random
from assessment import paired
from assess_skill_retention import per_operation
from continued_learning_protocol_v1 import _operations, _categories
from expansion_protocol import prepare_expansion
from storage import exact_score

PROTOCOL='stage80_progressive_scale_v1'
PROFILE={'teach_epochs':12,'sleep_epochs':2,'teach_lr':8e-5,'sleep_lr':2.5e-5,
         'batch':8,'allocation':{'facts':2,'general':3,'skills':3},
         'prior_replay':128,'diagnostic_old_sample':128,'block':512,'new_count':4096}
CAPS=('base_capabilities','confirmation','confirmation_v2')
SKILLS=('skill_audit','skill_transfer')
COUNTS={'base_capabilities':100,'confirmation':160,'confirmation_v2':160,
        'skill_audit':96,'skill_transfer':96}


def validate_rows(rows,items,withheld=False):
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows) or len(rows)!=len(items):
        raise ValueError('Incomplete or duplicate raw questions')
    for item in items:
        row=by_id.get(item['id'])
        question=item['audit_q'] if withheld else item['eval_q']
        if row is None or row['question']!=question or row['expected']!=item['answer'] or row['type']!=item['type']:
            raise ValueError('Raw question/label/type identity changed')
        if not isinstance(row['response'],str): raise ValueError('Missing raw response')


def categories(before,after):
    return {kind:{t:paired([r for r in before[kind] if r['type']==t],
                            [r for r in after[kind] if r['type']==t])
                  for t in sorted({r['type'] for r in before[kind]})} for kind in CAPS}


def capability_gate(references,after):
    details={}
    safe=True
    for name,before in references.items():
        if any(len(before[k])!=COUNTS[k] or len(after[k])!=COUNTS[k] for k in CAPS+SKILLS):
            raise ValueError('Incomplete capability/skill suites')
        operations={kind:per_operation(before[kind],after[kind]) for kind in SKILLS}
        cat=categories(before,after)
        try:
            _operations(operations)
            _categories(cat)
        except ValueError:
            safe=False
        details[name]={'per_operation_retention':operations,'retention_by_category':cat}
    if not references: raise ValueError('No preserved-ability references')
    return {'passed':safe,'references':details}


def score(rows):
    if not rows: raise ValueError('Empty audit')
    return sum(exact_score(r['response'],r['expected']) for r in rows)/len(rows)


def qualify_challenge(result,audit,result_hash,detail,maintenance_hash):
    if result.get('status')!='complete' or result.get('novel_fact_count')!=512 or result.get('cumulative_novel_facts_taught')!=1024:
        raise ValueError('Require completed second512 challenge, not partial or repair')
    if result['manifest']['protocol']!='stage80_continued_novel512_v1' or len(result['history'])!=1:
        raise ValueError('Unexpected completed source protocol')
    run_id=result['manifest']['args']['run_id']
    if audit['run_id']!=run_id or not audit.get('integrity_and_budget_checks_passed') or not audit.get('challenge_joint_gate_passed'):
        raise ValueError('Challenge joint/integrity gate did not pass')
    if audit['input_hashes']['results.json']!=result_hash or audit['maintenance_reference_baseline_hash']!=maintenance_hash:
        raise ValueError('Stale qualification or maintenance reference')
    if detail['run_id']!=run_id or detail['mode']!='completed_final':
        raise ValueError('Need completed matching individual skill audit')
    for key in ('results.json','baseline.json'):
        if detail['input_hashes'][key]!=audit['input_hashes'][key]:
            raise ValueError('Stale per-operation evidence')
    if detail['per_operation_retention']!=audit['parent_per_operation_retention']:
        raise ValueError('Independent operation measurements disagree')
    for suite in ('parent_per_operation_retention','maintenance_reference_per_operation_retention'):
        _operations(audit[suite])
    for suite in ('retention_by_category','maintenance_reference_category_retention'):
        _categories(audit[suite])
    # Prospective expansion admission is stronger than the original80/60 pilot:
    # every inherited block should be adequately acquired before large scaling.
    for key in ('exact','paraphrase','known_facts','known_paraphrase'):
        counts=audit['raw_answer_counts'][key]
        rows=result['final_raw'][key]
        if counts['items']!=({'exact':512,'paraphrase':512,'known_facts':640,'known_paraphrase':640}[key]) or len(rows)!=counts['items'] or sum(exact_score(r['response'],r['expected']) for r in rows)!=counts['correct']:
            raise ValueError('Audited counts do not match raw responses')
        value=score(rows)
        if not math.isclose(value,result['final_accuracy'][key]) or value<.95:
            raise ValueError('Prospective scale admission requires95% each wording')
    groups=audit['known_group_retention']
    if set(groups)!={'original128','first512'}:
        raise ValueError('A known group disappeared')
    for name,kinds in groups.items():
        if set(kinds)!={'known_facts','known_paraphrase'}:
            raise ValueError('Missing known wording')
        for kind,metric in kinds.items():
            if metric['items']!={'original128':128,'first512':512}[name] or metric['after_accuracy'] is None or not .95<=metric['after_accuracy']<=1:
                raise ValueError('Inherited group insufficiently preserved')
            selected=[r for r in result['final_raw'][kind]
                      if (r['id'].startswith('novel_mem_'))==(name=='first512')]
            if len(selected)!=metric['items'] or not math.isclose(score(selected),metric['after_accuracy']):
                raise ValueError('Known group summary does not match its raw answers')


def prepare_world(curriculum,result,source_facts,source_known):
    facts,known,meta=prepare_expansion(curriculum,result,source_facts,source_known,4096,512)
    if meta['curriculum_offset']!=1024 or len(known)!=1152 or len(curriculum['novel_items'])!=5120:
        raise ValueError('Require exact original5120 world with1024 already taught')
    validate_rows(result['final_raw']['exact'],source_facts)
    validate_rows(result['final_raw']['paraphrase'],source_facts,True)
    validate_rows(result['final_raw']['known_facts'],source_known)
    validate_rows(result['final_raw']['known_paraphrase'],source_known,True)
    meta['protocol']=PROTOCOL
    meta['training_profile']=PROFILE
    return facts,known,meta


def stratified_sample(items,count,seed):
    if count<1: raise ValueError('Positive diagnostic size required')
    groups={}
    for row in items:
        if row['id'].startswith('novel_mem_'):
            group=int(row['id'].rsplit('_',1)[-1])//512
        else:
            raise ValueError('Old-novel sampling excludes original known128')
        groups.setdefault(group,[]).append(row)
    rng=random.Random(seed)
    buckets={k:rng.sample(v,len(v)) for k,v in sorted(groups.items())}
    out=[]
    while len(out)<min(count,len(items)):
        for key in buckets:
            if buckets[key] and len(out)<count: out.append(buckets[key].pop())
    return out


def diagnostic_gate(raw,references):
    ability=capability_gate(references,raw)
    required={'current_exact':.8,'current_paraphrase':.6,
              'original_known':.95,'original_known_paraphrase':.95,
              'old_novel_sample':.95,'old_novel_sample_paraphrase':.95}
    scores={key:score(raw[key]) for key in required}
    sampled={}
    for key in ('old_novel_sample','old_novel_sample_paraphrase'):
        groups={int(r['id'].rsplit('_',1)[-1])//512 for r in raw[key]}
        sampled[key]={str(group):score([r for r in raw[key] if int(r['id'].rsplit('_',1)[-1])//512==group]) for group in sorted(groups)}
    passed=(ability['passed'] and all(scores[k]>=v for k,v in required.items())
            and all(value>=.8 for groups in sampled.values() for value in groups.values()))
    return {'passed':passed,'ability':ability,'measured_accuracy':scores,
            'sampled_cohort_accuracy':sampled,'sample_is_not_exhaustive_retention':True,
            'thresholds':required,'sampled_cohort_minimum':.8}
