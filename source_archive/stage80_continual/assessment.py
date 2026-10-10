"""Independent paired scoring of saved answers; no model/GPU or training feedback."""
import json
import hashlib
from pathlib import Path
from storage import atomic_json

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/stage80_continual'


def exact(response,answer):
    return ' '.join(response.casefold().split()).rstrip('.!')==' '.join(answer.casefold().split())


def paired(before,after,ordinary_only=False):
    left={r['id']:r for r in before}
    right={r['id']:r for r in after}
    if len(left)!=len(before) or len(right)!=len(after) or set(left)!=set(right):
        raise ValueError('Paired audit requires identical unique item ids')
    rows=[]
    for key,a in left.items():
        b=right[key]
        if a['question']!=b['question'] or a['expected']!=b['expected']:
            raise ValueError('Question/answer key changed between paired audits')
        if ordinary_only and key.startswith('prot_'):
            continue
        rows.append((exact(a['response'],a['expected']),exact(b['response'],b['expected'])))
    n=len(rows)
    old=sum(a for a,b in rows)
    new=sum(b for a,b in rows)
    retained=sum(a and b for a,b in rows)
    acquired=sum(not a and b for a,b in rows)
    return {'items':n,'before_correct':old,'after_correct':new,
            'before_accuracy':old/n if n else None,'after_accuracy':new/n if n else None,
            'change_pp':100*(new-old)/n if n else None,
            'previously_correct_retained':retained,'previously_correct_lost':old-retained,
            'previously_incorrect_acquired':acquired,'previously_incorrect_items':n-old,
            'previously_incorrect_acquisition_rate':acquired/(n-old) if n>old else None,
            'retention_of_previously_correct':retained/old if old else None}


def main():
    baseline_path=DATA/'frozen_control_baseline.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    output={'baseline_available':baseline is not None,'controls':{},'overall_goal_achieved':False}
    if baseline:
        if baseline['checkpoint'] is not None or not baseline['control_mappings']['pristine_pretraining_equivalent']:
            raise ValueError('A trained model cannot serve as the pristine baseline')
    for rid in ['control_b_s42','control_dp_s42','control_c_s42']:
        p=DATA/rid/'results.json'
        if not p.exists():continue
        d=json.loads(p.read_text())
        if d['status']!='complete':continue
        reload_path=DATA/rid/'full_reload_audit.json'
        reloaded=json.loads(reload_path.read_text()) if reload_path.exists() else None
        assessed={'complete_training':True,'full_reload_audit_available':reloaded is not None,
                  'mapping_gain':None,'base_preservation':None,
                  'joint_stage_gate_passed':False}
        if reloaded:
            assessed['reload_generations_identical']=all(not values for values in reloaded['generation_changes_after_reload'].values())
            if baseline:
                if baseline['control_mappings']['dataset_hash']!=d['manifest']['dataset_hash']:
                    raise ValueError('Pristine baseline dataset does not match control')
                assessed['mapping_gain']=paired(baseline['control_mappings']['raw_results']['audit'],
                                               reloaded['raw_results']['canonical'],True)
                before=baseline['raw_results']
                after=reloaded['raw_results']['base_capabilities']
                assessed['base_preservation']=paired(before,after)
                assessed['base_preservation_by_category']={t:paired([r for r in before if r['type']==t],
                                                                  [r for r in after if r['type']==t])
                                                          for t in sorted({r['type'] for r in before})}
        output['controls'][rid]=assessed
    skill_base=DATA/'skills_base_audit.json'
    if baseline and skill_base.exists():
        output['learned_skill_model_base_preservation']=paired(baseline['raw_results'],json.loads(skill_base.read_text())['raw_results'])
    output['pending_stage_requirements']=['>=4-context unique scale learned after reload',
                                          'new skills retained through intervening content',
                                          'joint memory/skill/base-capability thresholds',
                                          'replication across seeds']
    sources=[baseline_path,skill_base,*[DATA/rid/name for rid in output['controls']
                                      for name in ['results.json','full_reload_audit.json']]]
    output['input_hashes']={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources if p.exists()}
    output['assessment_code_hash']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(DATA/'paired_assessment.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
