"""Immutable CPU implementation preparation; no selected source or GPU dispatch."""
import datetime
import json
import py_compile
import subprocess
import sys
from queue_matched_memory_v1 import HERE,ROOT,DATA,read,sha
from storage import atomic_json
from v8_later_isolated_protocol_v1 import PROTOCOL,PROFILE,PREPARED


def main():
    target=DATA/PREPARED
    if target.exists():raise FileExistsError('Immutable preparation already exists')
    contract_path=DATA/'prepared_V8_later_learning_contract_CPU_v1.json';contract=read(contract_path)
    source_path=DATA/'prepared_permutation_broad_skill_v8.json';source=read(source_path)
    for prior in (contract,source):
        for n,h in prior['code_hashes'].items():
            if sha(HERE/n)!=h:raise ValueError('Pinned dependency changed '+n)
        for n,h in prior['data_hashes'].items():
            if sha(DATA/n)!=h:raise ValueError('Pinned input changed '+n)
    for n,h in source['external_code_hashes'].items():
        if sha(HERE.parent/n)!=h:raise ValueError('Pinned external dependency changed '+n)
    new=['run_V8_later_isolated_stage_v1.py','audit_V8_later_isolated_stage_v1.py','v8_later_isolated_protocol_v1.py',
         'test_V8_later_isolated_stage_v1.py','prepare_V8_later_isolated_stage_v1.py','queue_matched_memory_v1.py']
    names=sorted(set(new)|set(contract['code_hashes'])|set(source['code_hashes']))
    for n in names:py_compile.compile(str(HERE/n),doraise=True)
    outputs=[]
    for test in ['test_V8_later_isolated_stage_v1.py','test_v8_later_learning_contract_v1.py']:
        executed=subprocess.run([sys.executable,str(HERE/test)],cwd=ROOT,capture_output=True,text=True)
        if executed.returncode:raise RuntimeError(executed.stdout+executed.stderr)
        outputs.append({'test':test,'returncode':executed.returncode,'output':executed.stdout+executed.stderr})
    if contract['teacher_answer_supervised_tokens_per_epoch']!=3456 or contract['prior_questions_tokenization_checked']!=21796:
        raise ValueError('Incorrect complete prospective CPU mask/tokenization evidence')
    data_names=set(contract['data_hashes'])|set(source['data_hashes'])|{contract_path.name,source_path.name}
    artifact={'status':'CPU_runner_independent_auditor_and_guards_prepared_actual_qualified_source_GPU_admission_pending',
        'recorded_at':datetime.datetime.now().astimezone().isoformat(),'protocol':PROTOCOL,'profile':PROFILE,
        'code_hashes':{n:sha(HERE/n) for n in names},'external_code_hashes':source['external_code_hashes'],
        'data_hashes':{n:sha(DATA/n) for n in sorted(data_names)},'CPU_test_outputs':outputs,
        'six_new_guard_tests_and_ten_existing_contract_tests_passed':True,
        'complete_prior21796_tokenization_and_new1024_mask_CPU_contract_sha256':sha(contract_path),
        'prospective_updates':3840,'prospective_presentations':30720,'prospective_supervised_tokens':103680,
        'prospective_full_actual_BEFORE_and_fresh_restored_final_queries':45640,'new512_memory_bank_bytes':4194304,
        'qualified_source_selected':None,'actual_GPU_preflight_or_production_model_calls_updates':0,
        'preflight_requires_actual_source_same_prepared_version_and_matching_physical_CPU_audit':True,
        'isolation_NOT_sleep_causal_evidence':True,'existing_shared_plasticity_matched_replay_schedules_NOT_implemented':True,
        'reserved_reads':False,'no_automatic_dispatch':True,'overall_goal_achieved':False}
    atomic_json(target,artifact);print(json.dumps({k:artifact[k] for k in ['status','recorded_at','six_new_guard_tests_and_ten_existing_contract_tests_passed','prospective_updates','prospective_full_actual_BEFORE_and_fresh_restored_final_queries']}))


if __name__=='__main__':main()
