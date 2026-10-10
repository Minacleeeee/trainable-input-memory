"""Read-only actual GPU serial-versus-batched parity and throughput probe."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from storage import atomic_json
from gpu_lease import gpu_lease
from deadline import check_deadline

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/stage80_continual'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stop-at', default='2026-10-05T14:30:00+08:00'); args = ap.parse_args()
    target = DATA / 'isolated_generation_batch_probe_v1.json'
    if target.exists(): raise FileExistsError('Immutable probe required')
    source = DATA / 'isolated_second512_s42_v1'; result = read(source / 'results.json')
    audit = read(source / 'independent_isolated_second512_audit.json')
    assert result['status'] == audit['status'] == 'complete'
    assert audit['integrity_raw_identity_and_actual_budget_passed']
    assert audit['input_hashes']['results.json'] == sha(source / 'results.json')
    for n, h in result['checkpoint_hashes'].items(): assert sha(Path(result['checkpoint']) / n) == h
    selected = {key: result['final_raw'][key][:32] for key in ('prior512_facts', 'prior512_paraphrase', 'exact', 'paraphrase')}
    with gpu_lease(DATA, 'isolated_generation_batch_probe_v1'):
        from model_support import patch_causal_mask, get_directml_device
        from run_isolated_memory_pilot_v2 import restore
        from isolated_generation_batch_v1 import generate_equal_length
        from engine import evaluate
        patch_causal_mask(); device = get_directml_device(); model, tok = restore(device, result['checkpoint'])
        model.eval(); assert not any(p.requires_grad for p in model.parameters())
        measured = {}; started = time.perf_counter(); serial = {}
        for key, rows in selected.items():
            serial[key] = []
            for row in rows:
                check_deadline(args.stop_at, {'phase': 'serial_generation_probe', 'answered': sum(map(len, serial.values()))})
                serial[key] += evaluate(model, tok, [{'id': row['id'], 'type': row['type'], 'eval_q': row['question'], 'answer': row['expected']}])
        serial_sec = time.perf_counter() - started
        assert serial == selected, 'Actual serial restore differs from saved full source generations'
        for size in (2, 4, 8):
            started = time.perf_counter(); raw = {}
            for key, rows in selected.items():
                raw[key] = []
                for offset in range(0, len(rows), size):
                    check_deadline(args.stop_at, {'phase': 'batched_generation_probe', 'batch': size})
                    items = rows[offset:offset + size]
                    answers = generate_equal_length(model, tok, [row['question'] for row in items])
                    raw[key] += [{**row, 'response': answer, 'correct': bool(__import__('storage').exact_score(answer, row['expected']))}
                                 for row, answer in zip(items, answers)]
            elapsed = time.perf_counter() - started
            measured[str(size)] = {'raw': raw, 'all128_raw_identical_to_serial': raw == serial,
                                   'generation_sec': elapsed, 'speedup_over_serial': serial_sec / elapsed}
        passing = [int(size) for size, m in measured.items() if m['all128_raw_identical_to_serial']]
        selected_size = min(passing, key=lambda n: measured[str(n)]['generation_sec']) if passing else 1
        output = {'status': 'complete', 'source_run': source.name, 'source_results_sha256': sha(source / 'results.json'),
                  'checkpoint_hashes': result['checkpoint_hashes'], 'device': str(device),
                  'serial_questions': 128, 'serial_sec': serial_sec, 'serial_raw': serial,
                  'batches': measured, 'selected_batch_by_fastest_exact_raw_parity': selected_size,
                  'only_equal_length_memory_questions': True, 'model_training_updates': 0,
                  'reserved_final_suites_read': False, 'overall_goal_achieved': False,
                  'code_hashes': {n: sha(HERE / n) for n in ('probe_isolated_generation_batch_v1.py', 'isolated_generation_batch_v1.py',
                      'run_isolated_memory_pilot_v2.py', 'model_support.py', 'isolated_inputs_v2.py', 'isolated_embeddings_v1.py')},
                  'interpretation': 'Measured read-only generation throughput for128 questions, not optimizer or whole-training speed. Exact parity on this subset is not a universal guarantee; scale must still check all2916 actual source old responses and exhaustive fresh final responses.'}
        atomic_json(target, output)
        print(json.dumps({'status': 'complete', 'serial_sec': serial_sec, 'selected_batch': selected_size,
                          'batches': {k: {n: v for n, v in m.items() if n != 'raw'} for k, m in measured.items()}}, indent=2))


if __name__ == '__main__': main()
