"""Standalone prospective deployment precision probe; shared lease, no optimizer.

Not dispatched by preparation or the current skill queue. Requires a fresh run
after process/lease inspection and enough time for complete original-layout
generation. Failed parity is saved, never promoted to a qualified deployment.
"""
import argparse
import gc
import os
import time
from pathlib import Path
from prepare_deployment_fp16_v1 import read, sha, DATA, HERE
from storage import atomic_json
from gpu_lease import gpu_lease
from deadline import check_deadline, DeadlineReached


def run(args,target):
    import torch
    from model_support import patch_causal_mask, get_directml_device
    from run_isolated_memory_pilot_v2 import restore as restore_original
    from deployment_inputs_fp16_v1 import DeploymentInputEmbeddingFP16, restore as restore_half
    from isolated_inference_audit_v1 import evaluate_scale_source
    prepared = read(DATA/'prepared_deployment_fp16_v1.json')
    for n,h in prepared['code_hashes'].items(): assert sha(HERE/n) == h
    for n,h in prepared['external_code_hashes'].items(): assert sha(HERE.parent/n) == h
    source = DATA/prepared['source_run']; parent = read(source/'results.json')
    assert sha(source/'results.json') == prepared['source_results_sha256']
    assert sha(source/'independent_isolated_scale10240_audit.json') == prepared['source_audit_sha256']
    cp = Path(parent['checkpoint'])
    assert parent['checkpoint_hashes'] == prepared['source_checkpoint_hashes']
    for n,h in prepared['source_checkpoint_hashes'].items(): assert sha(cp/n) == h
    manifest = {**prepared,'run_id':args.run_id,'stop_at':args.stop_at,
                'pid':os.getpid(),'prepared_sha256':sha(DATA/'prepared_deployment_fp16_v1.json'),
                'no_training_or_resume':True,'inference_policy':'current question and neural parameters only'}
    atomic_json(target/'manifest.json',manifest)
    atomic_json(target/'results.json',{'status':'initialization','manifest':manifest,'optimizer_updates':0})
    check_deadline(args.stop_at,{'phase':'original_restore'})
    patch_causal_mask(); device = get_directml_device(); started = time.perf_counter()
    model,tok = restore_original(device,cp); old = model.get_input_embeddings()
    saved = torch.load(cp/'input_parameters.pt',map_location='cpu',weights_only=True)
    half = DeploymentInputEmbeddingFP16(old.original,saved['blocks'])
    assert half.num_embeddings == 59392
    # Same vocabulary IDs, all introduced rows, and three original rows; no
    # teacher answer is passed to either neural embedding module.
    checked = []
    ids = [0,1,49151]+list(range(49152,59392))
    with torch.no_grad():
        for offset in range(0,len(ids),512):
            check_deadline(args.stop_at,{'phase':'GPU_embedding_parity','completed_rows':len(checked)})
            selected = ids[offset:offset+512]
            tensor = torch.tensor([selected],dtype=torch.long,device=device)
            left = old(tensor).detach().cpu(); right = half(tensor).detach().cpu()
            equal = torch.eq(left,right).all(dim=-1).reshape(-1).tolist()
            checked.extend({'token_id':i,'bit_identical':bool(v)} for i,v in zip(selected,equal))
    atomic_json(target/'GPU_embedding_parity.json',{'rows':checked,'all_bit_identical':all(r['bit_identical'] for r in checked)})
    if not all(r['bit_identical'] for r in checked): raise ValueError('DirectML input cast/selection parity failed')
    deployment = target/'deployment_checkpoint'; deployment.mkdir()
    torch.save({'blocks':[b.detach().cpu() for b in half.blocks]},deployment/'input_parameters.pt')
    # Copy only configuration and lexical IDs; never source training data.
    for name in ('inference_config.json','lexical_inventory.json'):
        (deployment/name).write_bytes((cp/name).read_bytes())
    del old,half,saved,model,tok,left,right,tensor; gc.collect()
    check_deadline(args.stop_at,{'phase':'fresh_saved_fp16_restore'})
    model,tok = restore_half(device,deployment); embedding = model.get_input_embeddings()
    details = [{'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device),
                'bytes':p.numel()*p.element_size(),'requires_grad':p.requires_grad} for p in embedding.blocks]
    assert len(details)==20 and all(r['dtype']=='torch.float16' and not r['requires_grad'] for r in details)
    assert sum(r['bytes'] for r in details) == 41943040
    atomic_json(target/'deployment_tensor_inventory.json',details)
    final = evaluate_scale_source(model,tok,parent['final_raw'],target/'final_raw',args.stop_at,
                                  parent['manifest']['memory_generation_batch'])
    differences = [{'suite':key,'id':row['id'],'before':before,'after':row}
                   for key,rows in final.items() for before,row in zip(parent['final_raw'][key],rows) if before!=row]
    count = sum(len(rows) for rows in final.values()); assert count == 21348
    atomic_json(target/'results.json',{
        'status':'complete','manifest':manifest,'checkpoint':str(deployment),
        'checkpoint_hashes':{n:sha(deployment/n) for n in ('input_parameters.pt','lexical_inventory.json','inference_config.json')},
        'final_raw':final,'full_fresh_saved_reload_queries':count,'differences':differences,
        'GPU_all_embedding_rows_bit_identical':True,'all_original_raw_replies_exact':not differences,
        'deployment_gate_passed':not differences,'resident_bank_tensor_bytes':41943040,
        'deployment_input_file_bytes':(deployment/'input_parameters.pt').stat().st_size,
        'source_input_file_bytes':(cp/'input_parameters.pt').stat().st_size,
        'original_fixedweights_lexical_metadata_and_master_checkpoint_additional':True,
        'GPU_peak_allocator_bytes':None,'GPU_peak_allocator_unavailable_not_zero':True,
        'elapsed_sec_including_load_cast_and_generation':time.perf_counter()-started,
        'training_updates':0,'original_source_files_modified':False,
        'fixed_capacity_compression_sleep_causality_and_resume_NOT_validated':True,
        'overall_goal_achieved':False})
    print('Deployment probe complete; raw parity',not differences,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stop-at',required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('deployment_fp16_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in args.run_id):
        raise ValueError('Fresh deployment_fp16_ run ID required')
    # No silent extension of the current user-authorized research deadline.
    assert args.stop_at == read(DATA/'second_window_state_v1.json')['stop_at']
    target = DATA/args.run_id
    if target.exists(): raise FileExistsError('Never overwrite or resume a deployment probe')
    check_deadline(args.stop_at,{'phase':'admission'})
    with gpu_lease(DATA,'deployment_fp16_v1'):
        target.mkdir()
        try:
            run(args,target)
        except BaseException as error:
            atomic_json(target/'failure.json',{'error':repr(error),'pid':os.getpid(),
                        'deadline_reached':isinstance(error,DeadlineReached),'training_updates':0,
                        'complete_deployment_qualification':False})
            raise


if __name__ == '__main__': main()
