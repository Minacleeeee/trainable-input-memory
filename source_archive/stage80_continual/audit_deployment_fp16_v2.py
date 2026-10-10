"""Independent CPU raw-output/source/storage audit of a completed GPU probe."""
import argparse
import datetime
from pathlib import Path
import torch
from prepare_deployment_fp16_v2 import read, sha, DATA, HERE
from storage import atomic_json, exact_score
from half_round_towards_zero_cpu_v1 import half_round_towards_zero


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id',required=True); args = parser.parse_args()
    if not args.run_id.startswith('deployment_fp16_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in args.run_id):
        raise ValueError('Invalid deployment run ID')
    target = DATA/args.run_id; output = target/'independent_deployment_fp16_audit_v2.json'
    if output.exists(): raise FileExistsError('Immutable independent audit already exists')
    result = read(target/'results.json'); manifest = result['manifest']; prepared = read(DATA/'prepared_deployment_fp16_v2.json')
    assert result['status']=='complete' and manifest['prepared_sha256']==sha(DATA/'prepared_deployment_fp16_v2.json')
    for n,h in prepared['code_hashes'].items(): assert sha(HERE/n)==h
    for n,h in prepared['external_code_hashes'].items(): assert sha(HERE.parent/n)==h
    source = DATA/prepared['source_run']; parent = read(source/'results.json')
    assert sha(source/'results.json')==prepared['source_results_sha256']
    assert sha(source/'independent_isolated_scale10240_audit.json')==prepared['source_audit_sha256']
    for n,h in prepared['source_checkpoint_hashes'].items(): assert sha(Path(parent['checkpoint'])/n)==h
    final = result['final_raw']; assert set(final)==set(parent['final_raw'])
    for key,rows in final.items():
        if key in ('exact','paraphrase'):
            physical = read(target/'final_raw'/f'{key}_source1024_serial.json')
            physical2 = read(target/'final_raw'/f'{key}_source9216_batched.json')
            assert physical['complete'] and physical2['complete'] and physical['raw']+physical2['raw']==rows
        else:
            physical = read(target/'final_raw'/f'{key}.json'); assert physical['complete'] and physical['raw']==rows
        assert len(rows)==len(parent['final_raw'][key])
        for old,new in zip(parent['final_raw'][key],rows):
            assert all(old[k]==new[k] for k in ('id','type','question','expected'))
            assert isinstance(new['response'],str) and new['correct']==exact_score(new['response'],new['expected'])
    differences = [{'suite':key,'id':row['id'],'before':before,'after':row}
                   for key,rows in final.items() for before,row in zip(parent['final_raw'][key],rows) if before!=row]
    assert differences==result['differences'] and result['deployment_gate_passed']==(not differences)
    cp = Path(result['checkpoint'])
    for n,h in result['checkpoint_hashes'].items(): assert sha(cp/n)==h
    for n in ('inference_config.json','lexical_inventory.json'):
        assert (cp/n).read_bytes()==(Path(parent['checkpoint'])/n).read_bytes()
    master = torch.load(Path(parent['checkpoint'])/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    deployed = torch.load(cp/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    assert len(master)==len(deployed)==20
    assert all(d.dtype==torch.float16 and torch.equal(half_round_towards_zero(m),d) for m,d in zip(master,deployed))
    assert sum(d.numel()*d.element_size() for d in deployed)==result['resident_bank_tensor_bytes']==41943040
    inventory = read(target/'deployment_tensor_inventory.json')
    assert len(inventory)==20 and all(r['dtype']=='torch.float16' and r['device']=='privateuseone:0' and not r['requires_grad'] for r in inventory)
    assert sum(r['bytes'] for r in inventory)==41943040
    embedding = read(target/'GPU_embedding_parity.json')
    assert [r['token_id'] for r in embedding['rows']]==[0,1,49151]+list(range(49152,59392))
    assert all(r['bit_identical'] for r in embedding['rows']) and embedding['all_bit_identical']
    assert result['training_updates']==0 and sum(len(rows) for rows in final.values())==21348
    assert result['deployment_input_file_bytes']==(cp/'input_parameters.pt').stat().st_size
    assert result['source_input_file_bytes']==(Path(parent['checkpoint'])/'input_parameters.pt').stat().st_size
    atomic_json(output,{'status':'complete','recorded_at':datetime.datetime.now().astimezone().isoformat(),
        'run_id':args.run_id,'input_results_sha256':sha(target/'results.json'),
        'audit_code_sha256':sha(Path(__file__)),'integrity_raw_identity_source_and_storage_passed':True,
        'deployment_gate_passed':not differences,'raw_differences':len(differences),
        'actual_saved_reload_questions':21348,'resident_input_parameter_bytes':41943040,
        'CPU_audit_model_queries_or_training_updates':0,
        'training_resume_fixed_capacity_compression_and_overall_goal_NOT_qualified':True})
    print('Independent deployment integrity complete; raw parity',not differences,flush=True)


if __name__ == '__main__': main()
