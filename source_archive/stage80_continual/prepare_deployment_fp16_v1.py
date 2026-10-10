"""Pin a prospective GPU deployment parity probe without any GPU/model call."""
import datetime
import hashlib
import json
from pathlib import Path
from storage import atomic_json

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]/'data/stage80_continual'
CODE_NAMES = ['deployment_inputs_fp16_v1.py','run_deployment_fp16_v1.py',
              'audit_deployment_fp16_v1.py','prepare_deployment_fp16_v1.py',
              'audit_deployment_inputs_cpu_v1.py','run_isolated_memory_pilot_v2.py',
              'isolated_embeddings_v1.py','isolated_inputs_v2.py',
              'isolated_inference_audit_v1.py','isolated_generation_batch_v1.py',
              'engine.py','model_support.py','storage.py','gpu_lease.py','deadline.py']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    target = DATA/'prepared_deployment_fp16_v1.json'
    if target.exists(): raise FileExistsError('Pinned preparation must not be overwritten')
    source = DATA/'isolated_scale10240_s42_v2'
    result = read(source/'results.json'); audit = read(source/'independent_isolated_scale10240_audit.json')
    guard_path = DATA/'independent_deployment_inputs_cpu_guard_v1.json'; guard = read(guard_path)
    assert result['status'] == audit['status'] == 'complete' and audit['joint_memory_retention_gate_passed']
    assert audit['input_hashes']['results.json'] == guard['source_results_sha256'] == sha(source/'results.json')
    assert guard['all_CPU_forward_vectors_bit_identical'] and guard['CPU_serialization_restore_bit_identical']
    for n,h in guard['code_hashes'].items(): assert sha(HERE/n) == h
    for n,h in result['checkpoint_hashes'].items(): assert sha(Path(result['checkpoint'])/n) == h
    assert sum(len(rows) for rows in result['final_raw'].values()) == 21348
    atomic_json(target,{
        'status':'prepared_CPU_only_NOT_launched_or_GPU_qualified',
        'protocol':'stage80_deployment_fp16_v1','recorded_at':datetime.datetime.now().astimezone().isoformat(),
        'source_run':source.name,'source_results_sha256':sha(source/'results.json'),
        'source_audit_sha256':sha(source/'independent_isolated_scale10240_audit.json'),
        'source_checkpoint_hashes':result['checkpoint_hashes'],
        'CPU_guard_sha256':sha(guard_path),
        'code_hashes':{n:sha(HERE/n) for n in CODE_NAMES},
        'external_code_hashes':{'stage72_gpu/directml_patch.py':sha(HERE.parent/'stage72_gpu/directml_patch.py')},
        'required_actual_replies':21348,'required_all_learned_GPU_embedding_rows':10240,
        'required_original_GPU_embedding_rows':3,'runtime_bank_tensor_bytes_required':41943040,
        'source_generation_layout_required':True,'primary_gate_all_raw_byte_exact':True,
        'actual_model_calls':0,'actual_GPU_calls':0,'training_updates':0,
        'automatic_queue_created':False,'reserved_contents_read':False,
        'training_master_and_resume_NOT_qualified':True,'overall_goal_achieved':False})
    print('Deployment parity probe pinned; no GPU dispatch',flush=True)


if __name__ == '__main__': main()
