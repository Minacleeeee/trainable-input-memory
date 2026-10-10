"""Fresh v2: serial old-memory verification, batched newly introduced memory.

V1 stopped before updates because one already-wrong long old response changed
under batch8 despite subset128 parity. Preserve v1; do not reuse its state.

Block commits are provisional, never advertised as qualified partial sources.
Inference restoration uses only neural parameters, lexical IDs and runtime config.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from storage import atomic_json, digest, CycleStore, exact_score
from gpu_lease import gpu_lease
from deadline import check_deadline, DeadlineReached
from isolated_scale_protocol_v1 import PROTOCOL, PROFILE, qualify_source, validate_world, final_gate

PROTOCOL = 'stage80_isolated_scale10240_v2'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/stage80_continual'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, target):
    source = DATA / 'isolated_second512_s42_v1'
    parent = read(source / 'results.json')
    qualification = qualify_source(parent, read(source / 'independent_isolated_second512_audit.json'), sha(source / 'results.json'))
    for name, fingerprint in parent['checkpoint_hashes'].items():
        assert sha(Path(parent['checkpoint']) / name) == fingerprint
    for name, fingerprint in parent['manifest']['code_hashes'].items(): assert sha(HERE / name) == fingerprint
    world_path = DATA / 'isolated_memory10240_seed20261005_v1.json'
    world = read(world_path); prefix = read(DATA / 'progressive_memory_seed20261004.json')
    facts = validate_world(world, prefix); known = world['known_items']
    assert digest(facts) == world['novel_dataset_hash'] and digest(known) == world['known_dataset_hash']
    prepared = read(DATA / 'prepared_isolated_scale10240_v1.json')
    assert prepared['profile'] == PROFILE and prepared['world_sha256'] == sha(world_path)
    for name, fingerprint in prepared['code_hashes'].items(): assert sha(HERE / name) == fingerprint
    numerical = read(DATA / 'directml_supervised_nll_probe_v3.json')
    assert numerical['actual_vocabulary_loss_AND_gradient_validated']
    for name, fingerprint in numerical['code_hashes'].items(): assert sha(HERE / name) == fingerprint
    generation_probe = read(DATA / 'isolated_generation_batch_probe_v1.json')
    assert generation_probe['status'] == 'complete' and generation_probe['source_results_sha256'] == sha(source / 'results.json')
    assert generation_probe['checkpoint_hashes'] == parent['checkpoint_hashes']
    for name, fingerprint in generation_probe['code_hashes'].items(): assert sha(HERE / name) == fingerprint
    generation_batch = generation_probe['selected_batch_by_fastest_exact_raw_parity']
    assert generation_batch in (1, 2, 4, 8)
    if generation_batch > 1: assert generation_probe['batches'][str(generation_batch)]['all128_raw_identical_to_serial']
    check_deadline(args.stop_at, {'run_id': args.run_id, 'phase': 'initialization'})
    import torch
    from model_support import load_tokenizer, patch_causal_mask, get_directml_device
    from run_isolated_memory_pilot_v2 import restore
    from isolated_inputs_v2 import add_input_names
    from isolated_embeddings_v1 import IsolatedInputEmbedding, freeze_network_except_input_block
    from isolated_extension_v1 import extend_saved_blocks
    from teacher_wording_v1 import expand
    from cached_batches import encode_items
    from isolated_training_v1 import train
    from engine import evaluate as network_evaluate
    from isolated_generation_batch_v1 import generate_equal_length
    from cumulative_memory_scale_v1 import summarize
    patch_causal_mask(); device = get_directml_device()
    model, old_tok = restore(device, parent['checkpoint'])
    old_lexical = read(Path(parent['checkpoint']) / 'lexical_inventory.json')
    old_saved = torch.load(Path(parent['checkpoint']) / 'input_parameters.pt', map_location='cpu', weights_only=True)
    old_wrapper = model.get_input_embeddings()
    names = [r['stimulus'] for r in facts]
    initial_tok = load_tokenizer()
    subtoken_ids = initial_tok(names[5120:], add_special_tokens=False)['input_ids']
    original_matrix = old_wrapper.original.weight.detach().cpu().float()
    initial_appended = torch.stack([original_matrix[ids].mean(dim=0) for ids in subtoken_ids])
    del original_matrix
    extended = extend_saved_blocks(old_saved, old_lexical, names, initial_appended)
    initial_all = [p.clone() for p in extended['blocks']]
    wrapper = IsolatedInputEmbedding(old_wrapper.original, extended['blocks'])
    model.set_input_embeddings(wrapper)
    del extended, initial_appended, old_wrapper
    tok = load_tokenizer(); lexical = add_input_names(tok, names, 49152)
    inherited_source = parent['final_raw']
    old_questions = [r['question'] for rows in inherited_source.values() for r in rows]
    assert old_tok(old_questions, add_special_tokens=False)['input_ids'] == tok(old_questions, add_special_tokens=False)['input_ids']
    del old_tok, initial_tok
    for p in model.parameters(): p.requires_grad_(False)
    model.eval(); assert model.get_output_embeddings().weight.shape[0] == 49152
    inherited_items = {key: [{'id': r['id'], 'type': r['type'], 'eval_q': r['question'], 'answer': r['expected']} for r in rows]
                       for key, rows in inherited_source.items()}
    assert sum(map(len, inherited_items.values())) == 2916
    names_code = ['run_isolated_scale10240_v2.py', 'audit_isolated_scale10240_v2.py',
                  'isolated_scale_protocol_v1.py', 'isolated_extension_v1.py', 'isolated_training_v1.py',
                  'run_isolated_memory_pilot_v2.py', 'isolated_inputs_v2.py', 'isolated_embeddings_v1.py',
                  'supervised_nll_v3.py', 'teacher_wording_v1.py', 'engine.py', 'model_support.py',
                  'cached_batches.py', 'adamw_no_lerp.py', 'storage.py', 'gpu_lease.py', 'deadline.py',
                  'cumulative_memory_scale_v1.py', 'memory_dataset.py', 'isolated_generation_batch_v1.py',
                  'probe_isolated_generation_batch_v1.py']
    manifest = {'protocol': PROTOCOL, 'run_id': args.run_id, 'profile': PROFILE, 'source_run': source.name,
                'source_results_hash': sha(source / 'results.json'), 'source_audit_hash': sha(source / 'independent_isolated_second512_audit.json'),
                'source_checkpoint': parent['checkpoint'], 'source_checkpoint_hashes': parent['checkpoint_hashes'],
                'source_qualification': qualification, 'world_path': str(world_path), 'world_hash': sha(world_path),
                'dataset_hash': digest(facts), 'known_hash': digest(known), 'prepared_hash': sha(DATA / 'prepared_isolated_scale10240_v1.json'),
                'backbone': parent['manifest']['backbone'], 'device': str(device), 'torch_version': torch.__version__,
                'fixed_input_inventory_rows': 10240, 'allocated_extra_parameters': 20971520,
                'active_trainable_parameters_per_block': 1048576, 'output_vocabulary_size': 49152,
                'new_semantic_facts_this_run': 9216, 'existing1024_not_new_this_run': True,
                'old_training_replay_presentations': 0, 'inference_policy': 'Current question and network parameters only',
                'partial_block_commits_not_qualified_sources': True, 'overall_goal_achieved': False,
                'memory_generation_batch': generation_batch, 'source_old_memory_generation': 'serial',
                'v1_failed_before_any_updates_NOT_starting_weights': True, 'generation_probe_hash': sha(DATA / 'isolated_generation_batch_probe_v1.json'),
                'code_hashes': {n: sha(HERE / n) for n in names_code},
                'external_code_hashes': {'stage72_gpu/directml_patch.py': sha(HERE.parent / 'stage72_gpu/directml_patch.py')}}
    store = CycleStore(target / 'checkpoints', manifest)
    atomic_json(target / 'execution_window.json', {'pid': os.getpid(), 'stop_at': args.stop_at})
    atomic_json(target / 'teaching_archive.json', facts)
    atomic_json(target / 'known_fact_training.json', known)
    # Initialization evidence lives outside inference assets and is never read by restore.
    initial_hashes = [hashlib.sha256(p.contiguous().numpy().tobytes()).hexdigest() for p in initial_all]
    atomic_json(target / 'initial_bank_hashes.json', {'block_sha256': initial_hashes,
                'source_first10_bit_identical': all(torch.equal(initial_all[i], old_saved['blocks'][i]) for i in range(10)),
                'last10_original_frozen_subtoken_means_without_labels': True})
    history = []
    atomic_json(target / 'results.json', {'status': 'initializing', 'manifest': manifest, 'history': history})

    def evaluate(items, path, batch_memory=False):
        rows = []; started = time.perf_counter()
        batch = generation_batch if batch_memory else 1
        try:
            for offset in range(0, len(items), batch):
                check_deadline(args.stop_at, {'run_id': args.run_id, 'phase': 'evaluation',
                                             'raw_path': str(path), 'requested': len(items), 'answered': len(rows)})
                selected = items[offset:offset + batch]
                if batch == 1:
                    rows += network_evaluate(model, tok, selected)
                else:
                    answers = generate_equal_length(model, tok, [item['eval_q'] for item in selected])
                    rows += [{'id': item['id'], 'type': item['type'], 'question': item['eval_q'],
                              'expected': item['answer'], 'response': answer,
                              'correct': exact_score(answer, item['answer'])} for item, answer in zip(selected, answers)]
                if len(rows) % 64 == 0: atomic_json(path, {'complete': False, 'raw': rows}); gc.collect()
        except BaseException:
            atomic_json(path, {'complete': False, 'raw': rows, 'elapsed_sec': time.perf_counter() - started}); raise
        atomic_json(path, {'complete': True, 'raw': rows, 'elapsed_sec': time.perf_counter() - started})
        return rows

    initial_raw = {key: evaluate(items, target / 'initial_source_audit' / (key + '.json')) for key, items in inherited_items.items()}
    assert initial_raw == inherited_source, 'Inventory extension altered actual inherited2916 responses'
    atomic_json(target / 'initial_old2916_baseline.json', initial_raw)
    for index in range(2, 20):
        check_deadline(args.stop_at, {'run_id': args.run_id, 'phase': 'next_block', 'completed_blocks': len(history)})
        block_path = target / 'blocks' / f'block_{index:02d}'; block_path.mkdir(parents=True)
        rows = facts[index * 512:(index + 1) * 512]
        forms = expand(rows); cache = encode_items(tok, forms)
        assert digest(forms) == prepared['blocks'][index - 2]['teacher_forms_hash']
        assert torch.equal(wrapper.blocks[index].detach().cpu(), initial_all[index])
        baseline = {'exact': evaluate(rows, block_path / 'baseline_exact.json', True),
                    'paraphrase': evaluate([{**r, 'eval_q': r['audit_q']} for r in rows], block_path / 'baseline_paraphrase.json', True)}
        atomic_json(block_path / 'baseline.json', baseline)
        atomic_json(block_path / 'teacher_forms.json', forms)
        params = freeze_network_except_input_block(model, wrapper, index); model.eval()
        before_inactive = {i: p.detach().cpu().clone() for i, p in enumerate(wrapper.blocks) if i != index}
        phases = train(model, tok, cache, forms, params, device, PROFILE, block_path, args.stop_at)
        wrapper.activate_block(None)
        assert all(torch.equal(wrapper.blocks[i].detach().cpu(), p) for i, p in before_inactive.items())
        assert not torch.equal(wrapper.blocks[index].detach().cpu(), initial_all[index])
        selected = rows[:2] + inherited_items['base_capabilities'][:2]
        probe_before = evaluate(selected, block_path / 'before_save_probe.json')
        pending = store.pending(index - 1)
        blocks = [p.detach().cpu() for p in wrapper.blocks]
        after_hashes = [hashlib.sha256(p.contiguous().numpy().tobytes()).hexdigest() for p in blocks]
        assert all(torch.isfinite(p).all() for p in blocks)
        torch.save({'blocks': blocks}, pending / 'input_parameters.pt')
        atomic_json(pending / 'lexical_inventory.json', lexical)
        atomic_json(pending / 'inference_config.json', read(Path(parent['checkpoint']) / 'inference_config.json'))
        reloaded = torch.load(pending / 'input_parameters.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(a, b) for a, b in zip(blocks, reloaded['blocks']))
        # Disk roundtrip in the same frozen network per block; fresh full-model
        # reconstruction and exhaustive final generation are compulsory below.
        for dst, src in zip(wrapper.blocks, reloaded['blocks']):
            with torch.no_grad(): dst.copy_(src.to(device))
        probe_after = evaluate(selected, block_path / 'after_save_probe.json')
        assert probe_before == probe_after
        record = {'cycle': index - 1, 'input_block': index, 'new_semantic_facts': 512,
                  'phases': phases, 'baseline_sha256': sha(block_path / 'baseline.json'),
                  'teacher_forms_hash': digest(forms), 'all_inactive_banks_bit_identical': True,
                  'all20_bank_sha256_after': after_hashes,
                  'disk_roundtrip_and_probe_parity': True, 'full_final_retention_pending': True,
                  'partial_checkpoint_not_a_qualified_source': True}
        history.append(record)
        cp = store.commit(pending, {'cycle': index - 1, 'history': history,
                'reload_parity': {'passed': True, 'before': probe_before, 'after_reload': probe_after},
                'reload_scope': 'input weights disk roundtrip, same frozen full model; final fresh model pending',
                'partial_checkpoint_not_a_qualified_source': True})
        atomic_json(target / 'results.json', {'status': 'running_final_audit_pending', 'manifest': manifest,
                    'history': history, 'latest_provisional_checkpoint': str(cp), 'overall_goal_achieved': False})
        print('ISOLATED SCALE BLOCK COMMIT', index, 'completed', len(history), '/18', flush=True)
        del params, cache, before_inactive, blocks, reloaded; gc.collect()

    final_probe_items = facts[:2] + facts[-2:] + inherited_items['base_capabilities'][:2]
    pre_fresh = evaluate(final_probe_items, target / 'before_fresh_model_probe.json')
    del model, wrapper, old_saved, initial_all; gc.collect()
    model, tok = restore(device, cp)
    post_fresh = evaluate(final_probe_items, target / 'after_fresh_model_probe.json')
    assert pre_fresh == post_fresh
    old_keys = ['base_capabilities', 'confirmation', 'confirmation_v2', 'skill_audit', 'skill_transfer', 'known_facts', 'known_paraphrase']
    final = {key: evaluate(inherited_items[key], target / 'final_audit' / (key + '.json')) for key in old_keys}
    final['exact'] = evaluate(facts[:1024], target / 'final_audit/exact_old1024_serial.json') + evaluate(facts[1024:], target / 'final_audit/exact_new9216_batched.json', True)
    final['paraphrase'] = evaluate([{**r, 'eval_q': r['audit_q']} for r in facts[:1024]], target / 'final_audit/paraphrase_old1024_serial.json') + evaluate([{**r, 'eval_q': r['audit_q']} for r in facts[1024:]], target / 'final_audit/paraphrase_new9216_batched.json', True)
    gate = final_gate(inherited_source, final, exact_score)
    first_baseline = read(DATA / 'isolated_first512_s42_v2/baseline.json')
    second_baseline = read(source / 'baseline.json')
    segments = [{'name': 'isolated_first512', 'facts': facts[:512], 'baseline_canonical': first_baseline['exact'], 'baseline_withheld': first_baseline['paraphrase']},
                {'name': 'isolated_second512', 'facts': facts[512:1024], 'baseline_canonical': second_baseline['exact'], 'baseline_withheld': second_baseline['paraphrase']}]
    for index in range(2, 20):
        baseline = read(target / 'blocks' / f'block_{index:02d}' / 'baseline.json')
        segments.append({'name': f'isolated_block{index}', 'facts': facts[index * 512:(index + 1) * 512],
                         'baseline_canonical': baseline['exact'], 'baseline_withheld': baseline['paraphrase']})
    scale = summarize(segments, final['exact'], final['paraphrase'], tok, 8192)
    budgets = {out: sum(h['phases'][phase][key] for h in history for phase in ('teach', 'sleep'))
               for key, out in [('steps', 'actual_updates'), ('presentations', 'actual_presentations'),
                                ('supervised_tokens', 'actual_supervised_tokens'), ('training_sec', 'actual_training_sec')]}
    result = {'status': 'complete', 'manifest': manifest, 'history': history, 'final_raw': final,
              'joint_gate': gate, 'cumulative_dual_wording_acquisition': scale,
              'memory_scale_gate_passed': scale['compact_acquired_tokens'] >= 32768,
              'fresh_full_model_probe_parity': {'passed': True, 'before': pre_fresh, 'after': post_fresh},
              'checkpoint': str(cp), 'checkpoint_hashes': {n: sha(cp / n) for n in ('input_parameters.pt', 'lexical_inventory.json', 'inference_config.json')},
              'full_saved_reload_final_queries': 21348, 'new_semantic_facts_this_run': 9216,
              'cumulative_semantic_facts_taught': 10240, 'old_training_replay_presentations': 0,
              'overall_goal_achieved': False, **budgets,
              'interpretation': 'Growing per-entity neural input capacity, full final current-question-only generations and each segment actual baseline. No new algorithm, fixed-capacity compression, sleep causal benefit, blind final confirmation or independent training replication claim.'}
    atomic_json(target / 'results.json', result)
    print('ISOLATED SCALE COMPLETE', 'gate', gate['passed'], 'compact acquired', scale['compact_acquired_tokens'], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id', required=True)
    ap.add_argument('--stop-at', default='2026-10-05T14:30:00+08:00'); args = ap.parse_args()
    if Path(args.run_id).name != args.run_id: raise ValueError('Simple fresh run-id required')
    target = DATA / args.run_id
    with gpu_lease(DATA, 'isolated_scale10240_v2'):
        if target.exists(): raise FileExistsError('Fresh run-id required; no partial-state resume')
        target.mkdir()
        try: run(args, target)
        except BaseException as exc:
            atomic_json(target / 'failure.json', {'status': 'stopped_at_deadline' if isinstance(exc, DeadlineReached) else 'failed',
                        'error': repr(exc), 'deadline_progress': getattr(exc, 'progress', None),
                        'partial_weights_not_qualified_sources': True})
            raise


if __name__ == '__main__': main()
