"""Fresh-source V8 paired skills AFTER the audited memory queue has exited.

Recovery v2: fresh r1 IDs after preserved abrupt interruption. No model/GPU imports or later-retention dispatch. This controller must NOT be
launched while the strict memory queue remains alive, even as a waiting process.
"""
import argparse
import datetime
import os
import subprocess
import sys
import time
from deadline import check_deadline, DeadlineReached
from storage import atomic_json
from queue_matched_memory_v1 import HERE, ROOT, DATA, read, sha, processes, verify_free_gpu, completed_stage

PREPARED = 'prepared_broad_skill_v8_recovery_queue_v2.json'
STATE = 'queue_broad_skill_v8_recovery_s42_w2_v2.json'
MEMORY_QUEUE = 'queue_matched_memory_s42_w2_v1.json'
RUNS = {arm: ('runtime_permutation_broad_' + arm + '_s42_w2_v8r1',
               'permutation_broad_' + arm + '_s42_w2_v8r1') for arm in ('input','shared')}


def admit_prior(prior, inventory):
    if (prior.get('status') != 'complete_audited_memory_comparison_only'
            or prior.get('child_pid') is not None or prior.get('current') is not None
            or prior.get('pid') in {r['ProcessId'] for r in inventory}
            or not prior.get('completed')):
        raise ValueError('Memory queue must be fully audited COMPLETE and physically exited')


def skill_binding(result, audit, result_hash, arm):
    if (result.get('status') != 'complete' or result.get('manifest',{}).get('arm') != arm
            or result['manifest'].get('source_run') != 'isolated_scale10240_s42_v2'
            or audit.get('status') != 'complete' or not audit.get('integrity_raw_identity_and_actual_budget_passed')
            or audit.get('input_hashes',{}).get('results.json') != result_hash
            or audit.get('joint_gate') != result.get('joint_gate')
            or audit.get('immediate_skill_joint_gate_passed') != result.get('joint_gate',{}).get('passed')):
        raise ValueError('Partial, wrong-source/arm or unbound V8 result')
    return audit['immediate_skill_joint_gate_passed']


def pins(prepared):
    if prepared['stop_at'] != read(DATA/'second_window_state_v1.json')['stop_at']:
        raise ValueError('Authorized window mismatch')
    for name, expected in prepared['wrapper_code_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError('Changed queue code '+name)
    for name, expected in prepared['data_hashes'].items():
        if sha(DATA/name) != expected:
            raise ValueError('Changed prepared source/data '+name)
    production = read(DATA/'prepared_permutation_broad_skill_v8.json')
    for name, expected in production['code_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError('Changed V8 production code '+name)
    for name, expected in production['external_code_hashes'].items():
        if sha(HERE.parent/name) != expected:
            raise ValueError('Changed V8 external code '+name)
    for name, expected in production['data_hashes'].items():
        if sha(DATA/name) != expected:
            raise ValueError('Changed V8 production data '+name)


def verify_prior_results(prior):
    first_ids = ['matched_memory_compact_c1_s42_w2_v3','matched_memory_full_c1_s42_w2_v3']
    if len(prior['completed']) != len(set(prior['completed'])) or prior['completed'][:2] != first_ids:
        raise ValueError('Both complete memory first arms required in actual serial order')
    expected = list(first_ids)
    for arm,rid in zip(('compact','full'),first_ids):
        if read(DATA/rid/'results.json')['joint_gate']['passed']:
            expected.append('matched_memory_'+arm+'_c2_s42_w2_v3')
    if prior['completed'] != expected:
        raise ValueError('Qualified second stages missing or failed first stage improperly continued')
    for rid in prior['completed']:
        result = read(DATA/rid/'results.json')
        audit_path = DATA/rid/'independent_compact_memory_stage_audit_v3.json'
        audit = read(audit_path)
        manifest = result['manifest']
        arm = 'compact' if '_compact_' in rid else 'full'
        cohort = 1 if '_c1_' in rid else 2
        completed_stage(result, audit, sha(DATA/rid/'results.json'), arm, cohort)
        binding = prior['result_bindings'][rid]
        if binding['results_sha256'] != sha(DATA/rid/'results.json') or binding['audit_sha256'] != sha(audit_path):
            raise ValueError('Changed completed prior memory evidence')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-at',required=True)
    args = parser.parse_args()
    prepared = read(DATA/PREPARED)
    pins(prepared)
    if args.stop_at != prepared['stop_at'] or (DATA/STATE).exists():
        raise ValueError('Fresh queue state and exact deadline required')
    prior = read(DATA/MEMORY_QUEUE)
    if sha(DATA/'prepared_matched_memory_queue_v1.json') != prepared['prior_memory_queue_prepared_sha256']:
        raise ValueError('Prior queue preparation changed')
    admit_prior(prior,processes())
    verify_prior_results(prior)
    state = {'status':'admitted_after_complete_audited_memory_queue_exit','pid':os.getpid(),
        'current':None,'child_pid':None,'completed':[],'stop_at':args.stop_at,
        'prepared_sha256':sha(DATA/PREPARED),'prior_complete_memory_queue_sha256':sha(DATA/MEMORY_QUEUE),
        'immediate_joint_by_arm':{},'qualified_later_retention_source_NOT_selected':True,
        'no_later_retention_episodic_or_reserved_dispatch':True,'overall_goal_achieved':False}
    child = None

    def publish(**values):
        state.update(values,updated_at=datetime.datetime.now().astimezone().isoformat())
        atomic_json(DATA/STATE,state)

    def run_child(script,rid,extra,gpu):
        nonlocal child
        pins(prepared)
        check_deadline(args.stop_at,{'phase':'V8_child_admission','run_id':rid})
        if gpu:
            remaining = (datetime.datetime.fromisoformat(args.stop_at)-datetime.datetime.now().astimezone()).total_seconds()
            if remaining < prepared['per_complete_production_arm_cost_proxy_seconds_NOT_ETA']:
                raise DeadlineReached({'phase':'insufficient_fresh_arm_cost_proxy','remaining_seconds':remaining})
            inventory = processes()
            locks = [n for n in ('standalone_gpu.lock','campaign_seed42.lock','gpu_optimizer_benchmark.lock') if (DATA/n).exists()]
            verify_free_gpu(inventory,os.getpid(),locks)
            if (DATA/rid).exists():
                raise FileExistsError('No duplicate run or partial resume '+rid)
            arguments = ['--run-id',rid,'--stop-at',args.stop_at]+extra
        else:
            arguments = ['--run-id',rid]+extra
        child = subprocess.Popen([sys.executable,'-u',str(HERE/script)]+arguments,cwd=ROOT)
        publish(status='running_leased_GPU_child' if gpu else 'running_independent_CPU_audit',
                current=rid,child_pid=child.pid,child_kind='GPU' if gpu else 'CPU_audit')
        while child.poll() is None:
            check_deadline(args.stop_at,{'phase':'V8_child','run_id':rid,'pid':child.pid})
            time.sleep(3)
        code = child.returncode
        child = None
        publish(child_pid=None,child_kind=None)
        if code:
            raise RuntimeError('V8 child failed '+rid+' exit '+str(code))

    publish()
    try:
        for arm in ('input','shared'):
            probe, production = RUNS[arm]
            run_child('run_permutation_broad_skill_v8.py',probe,['--arm',arm,'--preflight-only'],True)
            run_child('audit_permutation_broad_runtime_preflight_cpu_v8.py',probe,[],False)
            r = read(DATA/probe/'results.json')
            a = read(DATA/probe/'independent_permutation_broad_runtime_preflight_cpu_v8.json')
            if (r['status'] != 'runtime_preflight_complete_no_updates' or r['manifest']['arm'] != arm
                    or a['status'] != 'complete' or a['arm'] != arm
                    or a['input_results_sha256'] != sha(DATA/probe/'results.json')
                    or not a['zero_optimizer_updates_and_old_banks_exact']
                    or not a['actual_GPU_CPU_NLL_and_gradient_group_checks_verified']):
                raise ValueError('Same-arm actual runtime preflight/audit mismatch')
            state['completed'].append(probe)
            publish()
            # Production restarts from original qualified scale, never probe weights.
            run_child('run_permutation_broad_skill_v8.py',production,['--arm',arm,'--preflight-run',probe],True)
            run_child('audit_permutation_broad_skill_v8.py',production,[],False)
            r = read(DATA/production/'results.json')
            apath = DATA/production/'independent_permutation_broad_skill_audit_v8.json'
            a = read(apath)
            passed = skill_binding(r,a,sha(DATA/production/'results.json'),arm)
            state['completed'].append(production)
            state['immediate_joint_by_arm'][arm] = passed
            state.setdefault('complete_result_bindings',{})[arm] = {'run_id':production,
                'result_sha256':sha(DATA/production/'results.json'),'audit_sha256':sha(apath)}
            publish()
        publish(status='complete_audited_V8_pair_only',current=None,child_pid=None)
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
        publish(status='stopped_at_deadline_or_cost_guard' if isinstance(error,DeadlineReached) else 'failed',
                error=repr(error),child_pid=None,current=None,failed_partial_weights_not_qualified=True)
        raise


if __name__ == '__main__':
    main()
