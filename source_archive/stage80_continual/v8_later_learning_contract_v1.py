"""Architecture-aware later-learning admission/scoring; never neural inference."""
import hashlib
import json
from pathlib import Path
from storage import digest, exact_score

PROTOCOL = 'stage80_V8_skill_then_disjoint512_development_retention_contract_v1'
MODES = ('isolated_new_bank', 'plastic_existing_shared')
SOURCE_AUDIT = 'independent_permutation_broad_skill_audit_v8.json'
PROFILE = {'new_facts':512, 'new_input_block':21, 'new_rows_start_id':59393,
           'operation_input_id':59392, 'original_output_vocabulary':49152,
           'teach_epochs':20, 'sleep_epochs':10, 'batch':8,
           'old_replay_presentations':0, 'minimum_each_fact_wording_accuracy':.95,
           'old_raw_exact_required':True, 'source_skill_raw_exact_required':True,
           'full_actual_before_and_fresh_restored_after_required':True,
           'reserved_independent_worlds_and_sleep_causality_pending':True}


def qualify(result, audit, result_sha):
    from permutation_broad_skill_protocol_v8 import PROTOCOL as parent_protocol, PROFILE as parent_profile
    m=result.get('manifest', {}); gate=result.get('joint_gate', {})
    arm=m.get('arm')
    if (result.get('status')!='complete' or m.get('protocol')!=parent_protocol
            or m.get('profile')!=parent_profile or arm not in ('input','shared')
            or m.get('input_inventory_rows')!=10241
            or m.get('active_trainable_parameters')!=(2048 if arm=='input' else 18432)
            or m.get('source_run')!='isolated_scale10240_s42_v2'
            or len(result.get('history',[]))!=1
            or result.get('full_actual_pregradient_queries')!=21796
            or result.get('full_saved_reload_final_queries')!=21796
            or result.get('actual_updates')!=1920 or result.get('actual_presentations')!=15360
            or result.get('actual_supervised_tokens')!=103920):
        raise ValueError('Complete exact V8 architecture and actual budgets required')
    if (not gate.get('passed') or not gate.get('all21348_old_raw_responses_exactly_unchanged')
            or not result.get('all_old20_neural_banks_bit_identical')
            or not m.get('source_qualification',{}).get('passed')
            or not m.get('frozen_maintenance_hashes')
            or result.get('frozen_maintenance_hashes_after')!=m.get('frozen_maintenance_hashes')):
        raise ValueError('Failed skill/old retention/ancestor/maintenance cannot seed later learning')
    if (audit.get('status')!='complete' or audit.get('run_id')!=m.get('run_id')
            or audit.get('arm')!=arm or not audit.get('integrity_raw_identity_and_actual_budget_passed')
            or not audit.get('immediate_skill_joint_gate_passed')
            or not audit.get('all_old20_banks_bit_identical')
            or not audit.get('all43592_before_and_after_result_rows_bind_complete_physical_raw_files')
            or not audit.get('maintenance_hashes_before_after_equal')
            or audit.get('input_hashes',{}).get('results.json')!=result_sha
            or audit.get('joint_gate')!=gate):
        raise ValueError('Matching complete physical independent V8 audit required')
    return {'passed':True,'arm':arm,'source_results_sha256':result_sha,
            'full_physical_admission_still_required':True,
            'no_permission_for_reserved_or_automatic_GPU_dispatch':True,'overall_goal_achieved':False}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_admission(data, source_run):
    """Rebind immutable completion, raw files and runtime-only checkpoint before restore."""
    data=Path(data)
    if Path(source_run).name!=source_run:
        raise ValueError('Simple source identifier required')
    run=data/source_run; result=read(run/'results.json'); audit=read(run/SOURCE_AUDIT)
    qualification=qualify(result,audit,sha(run/'results.json')); m=result['manifest']
    if m['run_id']!=source_run:
        raise ValueError('Source directory/manifest identity mismatch')
    prepared_path=data/'prepared_permutation_broad_skill_v8.json'; prepared=read(prepared_path)
    here=Path(__file__).resolve().parent
    if sha(prepared_path)!=m['prepared_sha256']:
        raise ValueError('Changed V8 production preparation')
    for n,h in prepared['code_hashes'].items():
        if sha(here/n)!=h:raise ValueError('Changed source code '+n)
    for n,h in prepared['external_code_hashes'].items():
        if sha(here.parent/n)!=h:raise ValueError('Changed external source code '+n)
    for n,h in prepared['data_hashes'].items():
        if sha(data/n)!=h:raise ValueError('Changed source/data '+n)
    from conditional_skill_protocol_v5 import qualify_source as qualify_scale
    ancestor=data/m['source_run']; scale=read(ancestor/'results.json')
    if (sha(ancestor/'results.json')!=m['source_results_sha256']
            or sha(ancestor/'independent_isolated_scale10240_audit.json')!=m['source_audit_sha256']
            or qualify_scale(scale,read(ancestor/'independent_isolated_scale10240_audit.json'),sha(ancestor/'results.json'))!=m['source_qualification']
            or m['code_hashes']!=prepared['code_hashes'] or m['external_code_hashes']!=prepared['external_code_hashes']):
        raise ValueError('Qualified ancestor or source implementation binding mismatch')
    for n,h in m['source_checkpoint_hashes'].items():
        if sha(Path(m['source_checkpoint'])/n)!=h:raise ValueError('Changed original ancestor checkpoint')
    if audit['audit_code_sha256']!=sha(here/'audit_permutation_broad_skill_v8.py'):
        raise ValueError('Unbound independent audit implementation')
    for n,h in audit['input_hashes'].items():
        if sha(run/n)!=h:raise ValueError('Changed source/audit input '+n)
    from audit_permutation_broad_skill_v8 import bind_physical
    old=read(run/'baseline_old.json'); before=read(run/'baseline_skill.json')
    bound=bind_physical(run,old,before,result['final_old_raw'],result['final_skill_raw'])
    if bound!=audit['physical_raw_file_hashes']:
        raise ValueError('Changed/incomplete physical before/final source answers')
    from permutation_broad_skill_protocol_v8 import validate_development_groups, gate_from_raw
    ds=read(data/'permutation_broad_vocabulary_teaching_seed20261005_v8.json')
    groups={k:ds['groups'][k] for k in ('train','dev','audit','transfer_audit')}
    validate_development_groups(groups,ds['operation_name'])
    if gate_from_raw(groups,before,result['final_skill_raw'],old==result['final_old_raw'],exact_score)!=result['joint_gate']:
        raise ValueError('Source scientific gate cannot be trusted without raw reconstruction')
    cp=Path(result['checkpoint']).resolve(); pointer=read(run/'checkpoints/active.json')
    if cp!=(run/'checkpoints'/pointer['checkpoint']).resolve():
        raise ValueError('Active source checkpoint mismatch')
    record=read(cp/'record.json')
    if (digest(record)!=pointer['record_hash'] or digest(m)!=pointer['manifest_hash']
            or record['history']!=result['history'] or not record['reload_parity']['passed']):
        raise ValueError('Incomplete or changed committed source')
    required={'input_parameters.pt','lexical_inventory.json','inference_config.json','shared_skill_config.json'}
    if m['arm']=='shared':required.add('shared_skill_parameters.pt')
    if set(result['checkpoint_hashes'])!=required:
        raise ValueError('Shared parameters may not be silently omitted')
    for n,h in result['checkpoint_hashes'].items():
        if sha(cp/n)!=h:raise ValueError('Changed source checkpoint '+n)
    from shared_skill_module_v1 import CONFIG
    expected={'arm':m['arm'],'module_config':CONFIG if m['arm']=='shared' else None}
    if read(cp/'shared_skill_config.json')!=expected:
        raise ValueError('Architecture-aware restore configuration mismatch')
    lexical=read(cp/'lexical_inventory.json')
    if lexical['new_input_rows']!=10241 or lexical['names'][-1]!=ds['operation_name'] or lexical['contains_answer_labels']:
        raise ValueError('Operation row shifted or answer-bearing lexical inventory')
    return {'qualification':qualification,'checkpoint':str(cp),'source_audit_sha256':sha(run/SOURCE_AUDIT),
            'source_checkpoint_hashes':result['checkpoint_hashes'],'physical_raw_file_hashes':bound}


def configure_later_trainable(model, embedding, modules, source_arm, mode):
    """Training-only whitelist. The existing shared module stays attached in all modes."""
    if mode not in MODES or source_arm not in ('input','shared'):
        raise ValueError('Declared source architecture and intervention required')
    if len(embedding.blocks)!=22 or any(tuple(p.shape)!=(512,2048) for p in embedding.blocks[:20]):
        raise ValueError('Exact inherited twenty memory banks and two appended blocks required')
    if tuple(embedding.blocks[20].shape)!=(1,2048) or tuple(embedding.blocks[21].shape)!=(512,2048):
        raise ValueError('Operation row and new512 bank may not be merged or shifted')
    if (source_arm=='input' and modules) or (source_arm=='shared' and set(modules)!={'q_proj','v_proj'}):
        raise ValueError('Existing shared architecture cannot be dropped or invented')
    if mode=='plastic_existing_shared' and source_arm!='shared':
        raise ValueError('Plastic shared intervention requires qualified shared source')
    registered={id(m) for m in model.modules()}
    if id(embedding) not in registered or any(id(m) not in registered for m in modules.values()):
        raise ValueError('Shared parameters must remain in the forward graph')
    from shared_skill_module_v1 import SharedProjection
    import torch
    for m in modules.values():
        if (not isinstance(m,SharedProjection) or m.rank!=2 or m.scaling!=2
                or tuple(m.skill_down.shape)!=(2,2048) or tuple(m.skill_up.shape)!=(2048,2)
                or m.skill_down.dtype!=torch.float32 or m.skill_up.dtype!=torch.float32
                or list(getattr(m.original,'active_adapters',[]))!=['single']):
            raise ValueError('Pinned active shared/maintenance architecture required')
    for p in model.parameters():p.requires_grad_(False)
    embedding.activate_block(21)
    groups={'name':[embedding.blocks[21]],'shared':[]}
    if mode=='plastic_existing_shared':
        for m in modules.values():
            for p in (m.skill_down,m.skill_up):
                p.requires_grad_(True); groups['shared'].append(p)
    expected={id(p) for ps in groups.values() for p in ps}
    actual={id(p) for p in model.parameters() if p.requires_grad}
    count=sum(p.numel() for ps in groups.values() for p in ps)
    if actual!=expected or count!=(1048576 if mode=='isolated_new_bank' else 1064960):
        raise ValueError('Incorrect later-learning gradient whitelist')
    return groups


def skill_after_learning(source_raw, later_raw, original_before):
    """Separate retained newly acquired answers from answers already correct before teaching."""
    counts={'dev':64,'audit':192,'transfer_audit':192}
    if any(set(x)!=set(counts) for x in (source_raw,later_raw,original_before)):
        raise ValueError('Complete development and both transfer suites required')
    out={}
    for k,n in counts.items():
        if any(len(x[k])!=n for x in (source_raw,later_raw,original_before)):
            raise ValueError('Partial skill retention cannot qualify')
        acquired=retained=lost=0
        for b,s,a in zip(original_before[k],source_raw[k],later_raw[k]):
            for r in (b,s,a):
                if (any(r[f]!=s[f] for f in ('id','type','question','expected'))
                        or r['correct']!=exact_score(r['response'],r['expected'])):
                    raise ValueError('Changed raw identity or fabricated score')
            was_acquired=not b['correct'] and s['correct']
            acquired+=was_acquired; retained+=was_acquired and a['correct']
            lost+=s['correct'] and not a['correct']
        out[k]={'questions':n,'acquired_after_initial_teaching':acquired,
                'still_correct_after_later_learning':retained,'source_correct_lost':lost,
                'all_source_raw_replies_exact':source_raw[k]==later_raw[k]}
    return {'all448_source_skill_raw_exact':source_raw==later_raw,'measurements':out,
            'independent_worlds_reserved_sleep_causality_pending':True,'overall_goal_achieved':False}


def fact_after_learning(facts, before, after):
    """Actual two-wording acquisition; previously correct either and forgotten excluded."""
    from scale_protocol_v1 import validate_rows
    if len(facts)!=512 or len({r['id'] for r in facts})!=512:
        raise ValueError('Complete unique new512 cohort required')
    for raw in (before,after):
        if set(raw)!={'exact','paraphrase'}:
            raise ValueError('Both actual introduction and later wordings required')
        validate_rows(raw['exact'],facts)
        validate_rows(raw['paraphrase'],facts,True)
    acquired=[]
    for i,item in enumerate(facts):
        if (not before['exact'][i]['correct'] and not before['paraphrase'][i]['correct']
                and after['exact'][i]['correct'] and after['paraphrase'][i]['correct']):
            acquired.append(item['id'])
    scores={k:sum(r['correct'] for r in after[k]) for k in ('exact','paraphrase')}
    return {'final_correct':scores,'minimum_each_wording_pass':all(n/512>=.95 for n in scores.values()),
            'conservative_acquired_ids':acquired,'conservative_acquired_facts':len(acquired),
            'compact_tokens_require_actual_runtime_tokenizer_NOT_estimated_here':True}
