"""Independent CPU audit for v2 serial old-memory and batched new-memory run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from storage import atomic_json, digest, exact_score
from isolated_scale_protocol_v1 import PROTOCOL, PROFILE, qualify_source, validate_world, final_gate
from scale_protocol_v1 import validate_rows

PROTOCOL = 'stage80_isolated_scale10240_v2'

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/stage80_continual'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_hash(value): return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id', required=True); args = ap.parse_args()
    if Path(args.run_id).name != args.run_id: raise ValueError('Simple run-id required')
    run = DATA / args.run_id; target = run / 'independent_isolated_scale10240_audit.json'
    if target.exists(): raise FileExistsError('Immutable independent audit required')
    result = read(run / 'results.json'); m = result['manifest']
    assert m['source_old_memory_generation'] == 'serial' and m['v1_failed_before_any_updates_NOT_starting_weights']
    assert result['status'] == 'complete' and m['protocol'] == PROTOCOL and m['profile'] == PROFILE
    assert m['run_id'] == args.run_id and m['new_semantic_facts_this_run'] == 9216
    assert m['fixed_input_inventory_rows'] == 10240 and m['allocated_extra_parameters'] == 20971520
    assert m['active_trainable_parameters_per_block'] == 1048576 and m['output_vocabulary_size'] == 49152
    for n, h in m['code_hashes'].items(): assert sha(HERE / n) == h
    for n, h in m['external_code_hashes'].items(): assert sha(HERE.parent / n) == h
    source = DATA / m['source_run']; parent = read(source / 'results.json')
    assert sha(source / 'results.json') == m['source_results_hash']
    assert sha(source / 'independent_isolated_second512_audit.json') == m['source_audit_hash']
    assert qualify_source(parent, read(source / 'independent_isolated_second512_audit.json'), m['source_results_hash']) == m['source_qualification']
    for n, h in m['source_checkpoint_hashes'].items(): assert sha(Path(m['source_checkpoint']) / n) == h
    assert sha(DATA / 'isolated_memory10240_seed20261005_v1.json') == m['world_hash']
    assert sha(DATA / 'prepared_isolated_scale10240_v1.json') == m['prepared_hash']
    generation_probe = read(DATA / 'isolated_generation_batch_probe_v1.json')
    assert sha(DATA / 'isolated_generation_batch_probe_v1.json') == m['generation_probe_hash']
    assert generation_probe['source_results_sha256'] == m['source_results_hash']
    assert m['memory_generation_batch'] == generation_probe['selected_batch_by_fastest_exact_raw_parity']
    world = read(DATA / 'isolated_memory10240_seed20261005_v1.json')
    prefix = read(DATA / 'progressive_memory_seed20261004.json'); facts = validate_world(world, prefix)
    assert read(run / 'teaching_archive.json') == facts and digest(facts) == m['dataset_hash']
    assert read(run / 'known_fact_training.json') == world['known_items']
    assert digest(world['known_items']) == m['known_hash']
    initial = read(run / 'initial_old2916_baseline.json')
    assert initial == parent['final_raw'] and sum(map(len, initial.values())) == 2916
    final = result['final_raw']
    for key in ('base_capabilities', 'confirmation', 'confirmation_v2', 'skill_audit', 'skill_transfer', 'known_facts', 'known_paraphrase'):
        items = [{'id': r['id'], 'type': r['type'], 'eval_q': r['question'], 'answer': r['expected']} for r in initial[key]]
        validate_rows(final[key], items)
    validate_rows(final['exact'], facts); validate_rows(final['paraphrase'], facts, True)
    gate = final_gate(parent['final_raw'], final, exact_score); assert gate == result['joint_gate']
    assert sum(map(len, final.values())) == result['full_saved_reload_final_queries'] == 21348
    cp = Path(result['checkpoint']); pointer = read(run / 'checkpoints/active.json'); record = read(cp / 'record.json')
    assert cp.resolve() == (run / 'checkpoints' / pointer['checkpoint']).resolve()
    assert pointer['cycle'] == record['cycle'] == 18
    assert digest(record) == pointer['record_hash'] and digest(m) == pointer['manifest_hash']
    assert record['history'] == result['history'] and len(result['history']) == 18
    assert record['reload_parity']['passed'] and record['reload_parity']['before'] == record['reload_parity']['after_reload']
    assert result['fresh_full_model_probe_parity']['passed']
    assert result['fresh_full_model_probe_parity']['before'] == result['fresh_full_model_probe_parity']['after']
    for n, h in result['checkpoint_hashes'].items(): assert sha(cp / n) == h
    assert read(cp / 'inference_config.json') == read(Path(m['source_checkpoint']) / 'inference_config.json')
    lexical = read(cp / 'lexical_inventory.json')
    assert set(lexical) == {'original_vocab_size', 'new_input_rows', 'names', 'contains_answer_labels', 'output_vocabulary_resized'}
    assert lexical['names'] == [r['stimulus'] for r in facts] and lexical['original_vocab_size'] == 49152
    assert lexical['new_input_rows'] == 10240 and not lexical['contains_answer_labels'] and not lexical['output_vocabulary_resized']
    from model_support import load_tokenizer, format_chat
    from isolated_inputs_v2 import add_input_names
    tok = load_tokenizer(); assert add_input_names(tok, lexical['names'], 49152) == lexical
    from teacher_wording_v1 import expand
    first_baseline = read(DATA / 'isolated_first512_s42_v2/baseline.json'); second_baseline = read(source / 'baseline.json')
    segments = [{'name': 'isolated_first512', 'facts': facts[:512], 'baseline_canonical': first_baseline['exact'], 'baseline_withheld': first_baseline['paraphrase']},
                {'name': 'isolated_second512', 'facts': facts[512:1024], 'baseline_canonical': second_baseline['exact'], 'baseline_withheld': second_baseline['paraphrase']}]
    totals = {'steps': 0, 'presentations': 0, 'supervised_tokens': 0, 'training_sec': 0.0}
    bank_hashes = read(run / 'initial_bank_hashes.json')['block_sha256']
    initial_hashes = bank_hashes.copy(); assert len(initial_hashes) == 20
    budgets = []
    for index, h in enumerate(result['history'], start=2):
        block = run / 'blocks' / f'block_{index:02d}'; rows = facts[index * 512:(index + 1) * 512]
        assert h['cycle'] == index - 1 and h['input_block'] == index and h['new_semantic_facts'] == 512
        assert h['full_final_retention_pending'] and h['partial_checkpoint_not_a_qualified_source']
        baseline = read(block / 'baseline.json'); validate_rows(baseline['exact'], rows); validate_rows(baseline['paraphrase'], rows, True)
        assert sha(block / 'baseline.json') == h['baseline_sha256']
        forms = read(block / 'teacher_forms.json'); assert forms == expand(rows) and digest(forms) == h['teacher_forms_hash']
        assert not {r['train_q'] for r in forms} & {r['audit_q'] for r in facts}
        tokens = {}
        for row in forms:
            full = tok(format_chat(tok, row['train_q'], row['answer']), add_special_tokens=False)['input_ids']
            prompt = tok(format_chat(tok, row['train_q']), add_special_tokens=False)['input_ids']
            assert len(prompt) < len(full) <= 64; tokens[row['id']] = len(full) - len(prompt)
        phase_budgets = {}
        for phase in ('teach', 'sleep'):
            b = h['phases'][phase]; epochs = PROFILE[phase + '_epochs']
            assert b['presentations_by_id'] == {r['id']: epochs for r in forms}
            assert b['steps'] == 128 * epochs and b['presentations'] == 1024 * epochs
            assert b['supervised_tokens'] == epochs * sum(tokens.values()) and b['peak_lr'] == PROFILE[phase + '_lr']
            assert b['epochs'] == epochs and b['active_trainable_parameters'] == 1048576
            assert b['seed'] == {'teach': 144, 'sleep': 145}[phase] and b['optimizer'] == 'AdamWNoLerp'
            assert len(b['losses']) == epochs and all(math.isfinite(v) for v in b['losses']) and b['training_sec'] > 0
            checked = b['first_batch_independent_cpu_loss_check']
            assert abs(checked['gpu_corrected_nll'] - checked['independent_cpu_nll']) < 1e-4
            phase_budgets[phase] = {key: b[key] for key in totals}
            for key in totals: totals[key] += b[key]
        after = h['all20_bank_sha256_after']; assert len(after) == 20
        assert bank_hashes[index] == initial_hashes[index] and after[index] != bank_hashes[index]
        assert all(after[i] == bank_hashes[i] for i in range(20) if i != index)
        assert h['all_inactive_banks_bit_identical'] and h['disk_roundtrip_and_probe_parity']
        bank_hashes = after
        segments.append({'name': f'isolated_block{index}', 'facts': rows, 'baseline_canonical': baseline['exact'], 'baseline_withheld': baseline['paraphrase']})
        budgets.append({'block': index, 'phases': phase_budgets})
    for key, out in [('steps', 'actual_updates'), ('presentations', 'actual_presentations'), ('supervised_tokens', 'actual_supervised_tokens'), ('training_sec', 'actual_training_sec')]:
        assert totals[key] == result[out]
    assert totals['steps'] == 69120 and totals['presentations'] == 552960 and totals['supervised_tokens'] == 1868340
    import torch
    torch.set_num_threads(4)
    saved = torch.load(cp / 'input_parameters.pt', map_location='cpu', weights_only=True)
    source_saved = torch.load(Path(m['source_checkpoint']) / 'input_parameters.pt', map_location='cpu', weights_only=True)
    assert set(saved) == {'blocks'} and len(saved['blocks']) == 20
    assert all(tuple(p.shape) == (512, 2048) and p.dtype == torch.float32 and torch.isfinite(p).all() for p in saved['blocks'])
    assert [tensor_hash(p) for p in source_saved['blocks']] == initial_hashes[:10]
    assert [tensor_hash(p) for p in saved['blocks']] == bank_hashes
    assert all(torch.equal(saved['blocks'][i], source_saved['blocks'][i]) for i in (0, 1))
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    backbone = m['backbone']; snapshot = Path(snapshot_download(backbone['model'], revision=backbone['revision'], local_files_only=True))
    key = 'model.embed_tokens.weight'; index_path = snapshot / 'model.safetensors.index.json'
    weight_file = snapshot / (read(index_path)['weight_map'][key] if index_path.exists() else 'model.safetensors')
    with safe_open(str(weight_file), framework='pt', device='cpu') as weights:
        original = weights.get_tensor(key).half().float()
    original_tok = load_tokenizer(); subtokens = original_tok(lexical['names'][5120:], add_special_tokens=False)['input_ids']
    expected_appended = torch.stack([original[ids].mean(dim=0) for ids in subtokens]).split(512)
    assert [tensor_hash(p) for p in expected_appended] == initial_hashes[10:]
    from cumulative_memory_scale_v1 import summarize
    scale = summarize(segments, final['exact'], final['paraphrase'], tok, 8192)
    assert scale == result['cumulative_dual_wording_acquisition']
    assert result['memory_scale_gate_passed'] == (scale['compact_acquired_tokens'] >= 32768)
    assert result['old_training_replay_presentations'] == 0 and result['overall_goal_achieved'] is False
    output = {'status': 'complete', 'run_id': args.run_id, 'integrity_raw_identity_and_actual_budget_passed': True,
              'joint_memory_retention_gate_passed': gate['passed'], 'joint_gate': gate,
              'memory_scale_gate_passed': result['memory_scale_gate_passed'], 'cumulative_dual_wording_acquisition': scale,
              'actual_budgets': totals, 'per_block_budgets': budgets,
              'existing_source1024_vectors_bit_identical': True, 'appended_initialization_independently_reconstructed_without_labels': True,
              'all18_hash_chain_inactive_banks_unchanged': True,
              'source_embedding_weight_file_sha256': sha(weight_file),
              'input_hashes': {n: sha(run / n) for n in ('results.json', 'initial_old2916_baseline.json', 'initial_bank_hashes.json', 'teaching_archive.json')},
              'audit_code_sha256': sha(__file__), 'model_updates_from_this_audit': 0, 'overall_goal_achieved': False,
              'interpretation': 'Actual full final21348 responses and each segment before baseline, conservative acquired BOTH scale and neural bank isolation. No final blind skill confirmation, newly learned algorithm, sleep causal comparison or independent training replication.'}
    atomic_json(target, output)
    print(json.dumps({'status': output['status'], 'joint_memory_gate': gate['passed'],
                      'memory_scale_gate': output['memory_scale_gate_passed'], 'actual_budgets': totals}, indent=2))


if __name__ == '__main__': main()
