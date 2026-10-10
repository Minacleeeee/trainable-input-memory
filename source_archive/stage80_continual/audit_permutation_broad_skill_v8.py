"""Independent CPU full V8 raw/physical/source/budget audit; never generates or trains."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from storage import atomic_json, digest, exact_score
from permutation_broad_skill_protocol_v8 import PROTOCOL, PROFILE, qualify_source, validate_development_groups, gate_from_raw
from scale_protocol_v1 import validate_rows
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'data/stage80_continual'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id:raise ValueError('Simple run ID')
    run=DATA/args.run_id;target=run/'independent_permutation_broad_skill_audit_v8.json'
    if target.exists():raise FileExistsError('Immutable audit')
    result=read(run/'results.json');m=result['manifest'];arm=m['arm'];expected=18432 if arm=='shared' else 2048
    assert result['status']=='complete' and m['protocol']==PROTOCOL and m['profile']==PROFILE
    assert m['run_id']==args.run_id and m['active_trainable_parameters']==expected
    admission=m['admitted_zero_update_probe'];probe_dir=DATA/admission['run_id']
    assert sha(probe_dir/'results.json')==admission['results_sha256']
    assert sha(probe_dir/'independent_permutation_broad_runtime_preflight_cpu_v8.json')==admission['audit_sha256']
    probe=read(probe_dir/'results.json');assert probe['manifest']['arm']==arm and probe['manifest']['prepared_sha256']==m['prepared_sha256']
    assert m['actual_old_pregradient_generated'] and result['full_actual_pregradient_queries']==21796
    prepared=read(DATA/'prepared_permutation_broad_skill_v8.json');assert sha(DATA/'prepared_permutation_broad_skill_v8.json')==m['prepared_sha256']
    assert m['code_hashes']==prepared['code_hashes'] and m['external_code_hashes']==prepared['external_code_hashes']
    for n,h in m['code_hashes'].items():assert sha(HERE/n)==h
    for n,h in m['external_code_hashes'].items():assert sha(HERE.parent/n)==h
    for n,h in prepared['data_hashes'].items():assert sha(DATA/n)==h
    source=DATA/m['source_run'];parent=read(source/'results.json');source_audit=read(source/'independent_isolated_scale10240_audit.json')
    assert sha(source/'results.json')==m['source_results_sha256']
    assert sha(source/'independent_isolated_scale10240_audit.json')==m['source_audit_sha256']
    assert qualify_source(parent,source_audit,m['source_results_sha256'])==m['source_qualification']
    for n,h in m['source_checkpoint_hashes'].items():assert sha(Path(m['source_checkpoint'])/n)==h
    dataset=read(DATA/'permutation_broad_vocabulary_teaching_seed20261005_v8.json')
    groups={k:dataset['groups'][k] for k in ('train','dev','audit','transfer_audit')}
    validate_development_groups(groups,dataset['operation_name'])
    assert read(run/'development_dataset.json')==groups and {k:digest(v) for k,v in groups.items()}==m['development_group_hashes']
    before=read(run/'baseline_skill.json');old_before=read(run/'baseline_old.json')
    assert old_before==parent['final_raw'] and sum(map(len,old_before.values()))==21348
    assert arm in ['input','shared']
    final_old=result['final_old_raw'];after=result['final_skill_raw']
    for k,rows in old_before.items():
        items=[{'id':r['id'],'type':r['type'],'eval_q':r['question'],'answer':r['expected']} for r in rows]
        validate_rows(final_old[k],items)
    for k in ('dev','audit','transfer_audit'):
        validate_rows(before[k],groups[k]);validate_rows(after[k],groups[k])
    gate=gate_from_raw(groups,before,after,final_old==old_before,exact_score)
    assert gate==result['joint_gate']==result['history'][0]['joint_gate']
    assert len(result['history'])==1 and sum(map(len,after.values()))==448 and result['full_saved_reload_final_queries']==21796
    phase_budgets={};totals={k:0 for k in ('steps','presentations','supervised_tokens','training_sec')}
    for phase in ('teach','sleep'):
        b=result['history'][0]['phases'][phase];epochs=PROFILE[phase+'_epochs']
        assert b['epochs']==epochs and b['seed']=={'teach':144,'sleep':145}[phase]
        assert b['steps']==64*epochs and b['presentations']==512*epochs
        assert b['presentations_by_id']=={r['id']:epochs for r in groups['train']}
        assert b['supervised_tokens']==prepared['teacher_supervised_tokens_per_epoch']*epochs
        assert b['active_trainable_parameters']==expected and b['optimizer']=='AdamWNoLerp'
        peaks={'name':PROFILE[phase+'_name_lr']};counts={'name':2048}
        if arm=='shared':peaks['shared']=PROFILE[phase+'_shared_lr'];counts['shared']=16384
        assert b['peak_lr_by_group']==peaks and b['parameters_by_group']==counts
        assert len(b['losses'])==epochs and all(math.isfinite(x) for x in b['losses']) and b['training_sec']>0
        check=b['first_batch_independent_cpu_loss_check'];assert abs(check['gpu_corrected_nll']-check['independent_cpu_nll'])<1e-4
        phase_budgets[phase]={k:b[k] for k in totals}
        for k in totals:totals[k]+=b[k]
    for key,out in [('steps','actual_updates'),('presentations','actual_presentations'),('supervised_tokens','actual_supervised_tokens'),('training_sec','actual_training_sec')]:assert totals[key]==result[out]
    assert totals['steps']==1920 and totals['presentations']==15360 and totals['supervised_tokens']==prepared['planned_supervised_tokens']
    cp=Path(result['checkpoint']);pointer=read(run/'checkpoints/active.json');record=read(cp/'record.json')
    assert cp.resolve()==(run/'checkpoints'/pointer['checkpoint']).resolve()
    assert pointer['record_hash']==digest(record) and pointer['manifest_hash']==digest(m)
    assert record['history']==result['history'] and record['reload_parity']['passed']
    assert record['reload_parity']['before']==record['reload_parity']['after_reload']
    for n,h in result['checkpoint_hashes'].items():assert sha(cp/n)==h
    assert read(cp/'inference_config.json')==read(Path(parent['checkpoint'])/'inference_config.json')
    assert result['frozen_maintenance_hashes_after']==m['frozen_maintenance_hashes']
    import torch
    torch.set_num_threads(4)
    saved=torch.load(cp/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    previous=torch.load(Path(parent['checkpoint'])/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    assert len(saved)==21 and len(previous)==20 and tuple(saved[20].shape)==(1,2048)
    assert all(torch.equal(a,b) for a,b in zip(saved[:20],previous))
    assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in saved)
    original_name=torch.load(run/'initial_operation_vector.pt',map_location='cpu',weights_only=True)
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from model_support import load_tokenizer
    snapshot=Path(snapshot_download(m['backbone']['model'],revision=m['backbone']['revision'],local_files_only=True))
    index=snapshot/'model.safetensors.index.json';key='model.embed_tokens.weight'
    weights_path=snapshot/(read(index)['weight_map'][key] if index.exists() else 'model.safetensors')
    with safe_open(str(weights_path),framework='pt',device='cpu') as f:original=f.get_tensor(key).half().float()
    tok=load_tokenizer();ids=tok(dataset['operation_name'],add_special_tokens=False)['input_ids']
    assert torch.equal(original[ids].mean(dim=0,keepdim=True),original_name)
    assert not torch.equal(saved[20],original_name)
    initial_shared=torch.load(run/'initial_shared_parameters.pt',map_location='cpu',weights_only=True)
    from shared_skill_module_v1 import CONFIG
    if arm=='shared':
        final_shared=torch.load(cp/'shared_skill_parameters.pt',map_location='cpu',weights_only=True)
        assert set(initial_shared)==set(final_shared)==set(CONFIG['targets'])
        for i,name in enumerate(CONFIG['targets']):
            gen=torch.Generator().manual_seed(CONFIG['initialization_seed']+i)
            expected_A=torch.randn(2,2048,generator=gen)*CONFIG['A_std']
            assert torch.equal(expected_A,initial_shared[name]['A']) and initial_shared[name]['B'].abs().sum().item()==0
            assert final_shared[name]['B'].abs().sum().item()>0
            for t in final_shared[name].values():assert t.dtype==torch.float32 and torch.isfinite(t).all()
        assert read(cp/'shared_skill_config.json')=={'arm':'shared','module_config':CONFIG}
    else:
        assert initial_shared=={} and not(cp/'shared_skill_parameters.pt').exists()
        assert read(cp/'shared_skill_config.json')=={'arm':'input','module_config':None}
    inventory=read(run/'trainable_tensor_inventory.json')
    assert sum(x['numel'] for x in inventory)==(18432 if arm=='shared' else 2048)
    assert len(inventory)==(5 if arm=='shared' else 1)
    physical_hashes=bind_physical(run,old_before,before,final_old,after)
    result_out={'status':'complete','run_id':args.run_id,'arm':arm,
                'integrity_raw_identity_and_actual_budget_passed':True,'immediate_skill_joint_gate_passed':gate['passed'],
                'joint_gate':gate,'actual_phase_budgets':phase_budgets,'actual_budgets':totals,
                'all_old20_banks_bit_identical':True,'full_actual_pregradient_and_restored_final_queries':21796,
                'blind_name_and_shared_initialization_independently_reconstructed':True,
                'all43592_before_and_after_result_rows_bind_complete_physical_raw_files':True,
                'physical_raw_file_hashes':physical_hashes,'numeric_branch_selection_NOT_tested':True,
                'maintenance_hashes_before_after_equal':True,'reserved_not_read':True,'model_calls_or_updates':0,
                'input_hashes':{'results.json':sha(run/'results.json'),'baseline_skill.json':sha(run/'baseline_skill.json'),
                               'baseline_old.json':sha(run/'baseline_old.json')},'audit_code_sha256':sha(Path(__file__)),
                'later_retention_reserved_replication_sleep_causality_pending':True,'overall_goal_achieved':False}
    atomic_json(target,result_out);print('V8 CPU AUDIT',args.run_id,gate['passed'],flush=True)


def bind_physical(run,old_before,before,final_old,after):
    """Bind ALL baseline/final results to complete physical generation artifacts."""
    hashes={}
    for directory,groups in [('baseline_old',old_before),('final_old',final_old),('baseline_skill',before),('final_skill',after)]:
        for key,rows in groups.items():
            paths=[run/directory/(key+'.json')]
            if directory.endswith('old') and key in ('exact','paraphrase'):
                paths=[run/directory/(key+'_source1024_serial.json'),run/directory/(key+'_source9216_batched.json')]
            physical=[]
            for p in paths:
                value=read(p);assert value['complete'];physical+=value['raw']
                hashes[str(p.relative_to(run))]=sha(p)
            assert physical==rows
    return hashes

if __name__=='__main__':main()
