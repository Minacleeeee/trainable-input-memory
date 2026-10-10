"""Serial, independently audited fixed-capacity memory comparison. No GPU imports.

Wait for the already running compact first stage to exit. Run a fresh full-row
control even if compact learning fails scientifically. Each second stage requires
the SAME arm's complete audited first-stage pass. Never restart partial runs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from deadline import check_deadline, DeadlineReached
from storage import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/stage80_continual'
PREPARED = 'prepared_matched_memory_queue_v1.json'
STATE = 'queue_matched_memory_s42_w2_v1.json'
FIRST = 'matched_memory_compact_c1_s42_w2_v3'
FULL_FIRST = 'matched_memory_full_c1_s42_w2_v3'
SECOND = {arm: 'matched_memory_' + arm + '_c2_s42_w2_v3' for arm in ('compact', 'full')}
AUDIT = 'independent_compact_memory_stage_audit_v3.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def processes():
    command = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    return rows if isinstance(rows, list) else [rows]


def verify_free_gpu(rows, own_pid, locks):
    if locks or any(row['ProcessId'] != own_pid for row in rows):
        raise RuntimeError('Competing Python processes or GPU lock; no dispatch')


def completed_stage(result, audit, result_hash, arm, cohort):
    manifest = result.get('manifest', {})
    if (result.get('status') != 'complete' or manifest.get('arm') != arm
            or manifest.get('cohort') != cohort or manifest.get('source_run') != 'isolated_scale10240_s42_v2'
            or audit.get('status') != 'complete_independent_CPU_memory_stage_audit'
            or not audit.get('integrity_physical_and_actual_budget_passed')
            or audit.get('results_sha256') != result_hash
            or audit.get('joint_gate') != result.get('joint_gate')
            or audit.get('stage_joint_gate_passed') != result.get('joint_gate', {}).get('passed')):
        raise ValueError('Partial, wrong-arm or unbound independent stage result')
    return audit['stage_joint_gate_passed']


def verify_pins(prepared):
    if prepared['stop_at'] != read(DATA / 'second_window_state_v1.json')['stop_at']:
        raise ValueError('Authorized window changed')
    for name, expected in prepared['wrapper_code_hashes'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Changed wrapper code ' + name)
    for name, expected in prepared['data_hashes'].items():
        if sha(DATA / name) != expected:
            raise ValueError('Changed prepared data ' + name)
    stage = read(DATA / 'prepared_compact_memory_comparison_stage_v3.json')
    for name, expected in stage['code_hashes'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Changed production code ' + name)
    for name, expected in stage['external_code_hashes'].items():
        if sha(HERE.parent / name) != expected:
            raise ValueError('Changed external code ' + name)
    for name, expected in stage['data_hashes'].items():
        if sha(DATA / name) != expected:
            raise ValueError('Changed production data ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-at', required=True)
    args = parser.parse_args()
    prepared = read(DATA / PREPARED)
    verify_pins(prepared)
    if args.stop_at != prepared['stop_at'] or (DATA / STATE).exists():
        raise ValueError('Fresh queue ID and exact current window required')
    state = {'status': 'waiting_for_existing_compact_first_process_exit', 'pid': os.getpid(),
             'child_pid': None, 'current': FIRST, 'completed': [], 'skipped': {},
             'stop_at': args.stop_at, 'prepared_sha256': sha(DATA / PREPARED),
             'overall_goal_achieved': False, 'no_skill_episodic_or_reserved_dispatch': True}
    child = None

    def publish(**values):
        state.update(values, updated_at=datetime.datetime.now().astimezone().isoformat())
        atomic_json(DATA / STATE, state)

    def audit(rid, arm, cohort):
        nonlocal child
        check_deadline(args.stop_at, {'phase': 'CPU_audit_admission', 'run': rid})
        path = DATA / rid
        if not (path / AUDIT).exists():
            publish(status='running_independent_CPU_audit', current=rid)
            child = subprocess.Popen([sys.executable, str(HERE / 'audit_compact_memory_stage_v3.py'), '--run-id', rid], cwd=ROOT)
            publish(child_pid=child.pid, child_kind='CPU_audit')
            child.wait()
            code = child.returncode
            child = None
            publish(child_pid=None, child_kind=None)
            if code:
                raise RuntimeError('Independent CPU audit failed: ' + rid)
        result = read(path / 'results.json')
        independent = read(path / AUDIT)
        passed = completed_stage(result, independent, sha(path / 'results.json'), arm, cohort)
        if result['manifest']['prepared_sha256'] != prepared['stage_prepared_sha256']:
            raise ValueError('Different production preparation')
        state['completed'].append(rid)
        state.setdefault('result_bindings', {})[rid] = {'results_sha256': sha(path / 'results.json'),
            'audit_sha256': sha(path / AUDIT), 'joint_passed': passed,
            'acquired_count': result['joint_gate']['acquired_and_final_recalled_both'],
            'acquired_compact_tokens': result['acquired_compact_tokens_current_retained_cohorts'],
            'new_runtime_tensor_bytes': result['new_runtime_tensor_bytes_FULL_allocated_capacity'],
            'actual_updates': result['actual_updates'], 'actual_training_sec': result['actual_training_sec']}
        publish()
        return passed

    def gpu(rid, arm, cohort, parent=None):
        nonlocal child
        verify_pins(prepared)
        check_deadline(args.stop_at, {'phase': 'GPU_admission', 'run': rid})
        remaining = (datetime.datetime.fromisoformat(args.stop_at) - datetime.datetime.now().astimezone()).total_seconds()
        if remaining < prepared['per_fresh_stage_cost_proxy_seconds_NOT_ETA']:
            raise DeadlineReached({'phase': 'insufficient_fresh_full_stage_cost_proxy', 'remaining_seconds': remaining})
        lock_names = ['standalone_gpu.lock', 'campaign_seed42.lock', 'gpu_optimizer_benchmark.lock']
        inventory = processes()
        verify_free_gpu(inventory, os.getpid(), [n for n in lock_names if (DATA / n).exists()])
        if (DATA / rid).exists():
            raise FileExistsError('No duplicate/partial resume: ' + rid)
        command = [sys.executable, '-u', str(HERE / 'run_compact_memory_comparison_stage_v3.py'),
                   '--run-id', rid, '--arm', arm, '--cohort', str(cohort),
                   '--preflight-run', prepared['preflight_run'], '--stop-at', args.stop_at]
        if parent:
            command += ['--parent-run', parent]
        child = subprocess.Popen(command, cwd=ROOT)
        publish(status='running_leased_GPU_child', current=rid, child_pid=child.pid, child_kind='GPU',
                last_pre_dispatch_python_processes=inventory, remaining_seconds_at_dispatch=remaining)
        while child.poll() is None:
            check_deadline(args.stop_at, {'phase': 'GPU_child', 'run': rid, 'pid': child.pid})
            time.sleep(3)
        code = child.returncode
        child = None
        publish(child_pid=None, child_kind=None)
        if code:
            raise RuntimeError('Fresh GPU stage failed: ' + rid + ' exit ' + str(code))

    publish()
    try:
        while True:
            check_deadline(args.stop_at, {'phase': 'wait_existing_first'})
            if sha(DATA / FIRST / 'manifest.json') != prepared['existing_manifest_sha256']:
                raise ValueError('Existing run identity changed')
            if (DATA / FIRST / 'failure.json').exists():
                raise RuntimeError('Existing compact first stage failed technically; preserve it')
            inventory = processes()
            matching = [r for r in inventory if r['ProcessId'] == prepared['existing_pid']]
            if matching and matching[0]['CommandLine'] != prepared['existing_command_line']:
                raise ValueError('Existing PID reused by a different process')
            if not matching:
                if read(DATA / FIRST / 'results.json')['status'] != 'complete':
                    raise RuntimeError('Existing process exited with incomplete stage')
                break
            time.sleep(10)
        compact_passed = audit(FIRST, 'compact', 1)
        gpu(FULL_FIRST, 'full', 1)
        full_passed = audit(FULL_FIRST, 'full', 1)
        for arm, passed, parent in [('compact', compact_passed, FIRST), ('full', full_passed, FULL_FIRST)]:
            if passed:
                gpu(SECOND[arm], arm, 2, parent)
                audit(SECOND[arm], arm, 2)
            else:
                state['skipped'][SECOND[arm]] = 'Same-arm first cohort failed unchanged scientific gates'
                publish()
        publish(status='complete_audited_memory_comparison_only', current=None, child_pid=None)
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
        publish(status='stopped_at_deadline_or_cost_guard' if isinstance(error, DeadlineReached) else 'failed',
                error=repr(error), child_pid=None, current=None, failed_partial_weights_not_qualified=True)
        raise


if __name__ == '__main__':
    main()
