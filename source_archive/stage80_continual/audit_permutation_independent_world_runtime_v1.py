# Fresh version derived from audit_permutation_broad_runtime_preflight_cpu_v8.py; original production files remain immutable.
"""Independent physical/initialization audit; zero production GPU/model calls."""
import argparse
import math
from pathlib import Path
from permutation_independent_world_protocol_v1 import PROFILE, PROTOCOL, qualify_source
from prepare_permutation_independent_world_production_v1 import read, sha, DATA, HERE
from storage import atomic_json, exact_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id: raise ValueError('Simple run ID')
    run = DATA / args.run_id
    output = run / 'independent_permutation_world_runtime_cpu_v1.json'
    if output.exists(): raise FileExistsError('Fresh independent audit only')
    result = read(run / 'results.json')
    manifest = result['manifest']
    arm = manifest['arm']
    prepared_path = DATA / 'prepared_permutation_independent_world_production_v1.json'
    prepared = read(prepared_path)
    assert result['status'] == 'runtime_preflight_complete_no_updates'
    assert result['actual_training_updates'] == 0 and result['actual_generated_answers'] == 8
    assert manifest['protocol'] == PROTOCOL and manifest['profile'] == PROFILE
    assert arm in ['input', 'shared']
    assert manifest['run_id'] == args.run_id
    assert manifest['active_trainable_parameters'] == (18432 if arm == 'shared' else 2048)
    assert sha(prepared_path) == manifest['prepared_sha256']
    for n, h in prepared['code_hashes'].items(): assert sha(HERE / n) == h
    for n, h in prepared['data_hashes'].items(): assert sha(DATA / n) == h
    for n, h in prepared['external_code_hashes'].items(): assert sha(HERE.parent / n) == h
    before = read(run / 'zero_before_reload.json')
    after = read(run / 'zero_after_reload.json')
    assert before['complete'] and after['complete']
    assert before['raw'] == after['raw'] and len(before['raw']) == 4
    source = DATA / manifest['source_run']
    parent = read(source / 'results.json')
    assert sha(source / 'results.json') == manifest['source_results_sha256']
    assert qualify_source(parent, read(source / 'independent_isolated_scale10240_audit.json'), sha(source / 'results.json')) == manifest['source_qualification']
    for n, h in manifest['source_checkpoint_hashes'].items(): assert sha(Path(parent['checkpoint']) / n) == h
    assert before['raw'][:2] == parent['final_raw']['base_capabilities'][:2]
    dataset = read(DATA / 'permutation_independent_world_seed20263945_CPU_v1.json')
    from permutation_independent_world_protocol_v1 import validate_candidate, PRIOR_DATASETS
    validate_candidate(dataset, {n:read(DATA/n) for n in PRIOR_DATASETS})
    for row, item in zip(before['raw'][2:], dataset['groups']['dev'][:2]):
        assert row['id'] == item['id'] and row['question'] == item['eval_q']
        assert row['type'] == item['type'] and row['expected'] == item['answer']
        assert row['correct'] == exact_score(row['response'], row['expected'])
    assert math.isfinite(result['loss_gpu']) and math.isfinite(result['loss_cpu'])
    assert abs(result['loss_gpu'] - result['loss_cpu']) < 1e-4
    gradients = result['gradient_norms']
    assert len(gradients) == (5 if arm == 'shared' else 1)
    assert all(math.isfinite(v) and v >= 0 for v in gradients.values())
    name_gradients = [v for n, v in gradients.items() if '.blocks.20' in n]
    assert len(name_gradients) == 1 and name_gradients[0] > 0
    if arm == 'shared':
        down = [v for n, v in gradients.items() if n.endswith('skill_down')]
        up = [v for n, v in gradients.items() if n.endswith('skill_up')]
        assert len(down) == len(up) == 2 and all(v == 0 for v in down) and all(v > 0 for v in up)
    import torch
    torch.set_num_threads(2)
    cp = Path(result['preflight_checkpoint'])
    saved = torch.load(cp / 'input_parameters.pt', map_location='cpu', weights_only=True)['blocks']
    previous = torch.load(Path(parent['checkpoint']) / 'input_parameters.pt', map_location='cpu', weights_only=True)['blocks']
    initial = torch.load(run / 'initial_operation_vector.pt', map_location='cpu', weights_only=True)
    assert len(saved) == 21 and len(previous) == 20 and list(initial.shape) == [1, 2048]
    assert all(torch.equal(a, b) for a, b in zip(saved[:20], previous))
    assert torch.equal(saved[20], initial)
    assert read(cp / 'inference_config.json') == read(Path(parent['checkpoint']) / 'inference_config.json')
    lexical = read(cp / 'lexical_inventory.json')
    previous_lexical = read(Path(parent['checkpoint']) / 'lexical_inventory.json')
    assert lexical['names'] == previous_lexical['names'] + [dataset['operation_name']]
    assert not (run / 'checkpoints/active.json').exists()
    initial_shared = torch.load(run / 'initial_shared_parameters.pt', map_location='cpu', weights_only=True)
    from shared_skill_module_v1 import CONFIG
    if arm == 'shared':
        restored_shared = torch.load(cp / 'shared_skill_parameters.pt', map_location='cpu', weights_only=True)
        assert set(initial_shared) == set(restored_shared) == set(CONFIG['targets'])
        for i, name in enumerate(CONFIG['targets']):
            generator = torch.Generator().manual_seed(CONFIG['initialization_seed'] + i)
            expected_a = torch.randn(2, 2048, generator=generator) * CONFIG['A_std']
            assert torch.equal(initial_shared[name]['A'], expected_a)
            assert torch.count_nonzero(initial_shared[name]['B']).item() == 0
            for key in ['A', 'B']: assert torch.equal(initial_shared[name][key], restored_shared[name][key])
        assert read(cp / 'shared_skill_config.json') == {'arm': arm, 'module_config': CONFIG}
    else:
        assert initial_shared == {} and not (cp / 'shared_skill_parameters.pt').exists()
        assert read(cp / 'shared_skill_config.json') == {'arm': arm, 'module_config': None}
    inventory = read(run / 'trainable_tensor_inventory.json')
    assert len(inventory) == (5 if arm == 'shared' else 1)
    assert sum(r['numel'] for r in inventory) == manifest['active_trainable_parameters']
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from model_support import load_tokenizer
    root = Path(snapshot_download(manifest['backbone']['model'], revision=manifest['backbone']['revision'], local_files_only=True))
    index = root / 'model.safetensors.index.json'
    key = 'model.embed_tokens.weight'
    weights = root / (read(index)['weight_map'][key] if index.exists() else 'model.safetensors')
    with safe_open(str(weights), framework='pt', device='cpu') as handle:
        table = handle.get_tensor(key).half().float()
    ids = load_tokenizer()(dataset['operation_name'], add_special_tokens=False)['input_ids']
    assert torch.equal(table[ids].mean(0, keepdim=True), initial)
    atomic_json(output, {'status': 'complete', 'run_id': args.run_id, 'arm': arm,
        'input_results_sha256': sha(run / 'results.json'),
        'zero_optimizer_updates_and_old_banks_exact': True,
        'actual4_before4_reload_raw_exact': True,
        'actual_GPU_CPU_NLL_and_gradient_group_checks_verified': True,
        'blind_name_and_shared_initialization_independently_reconstructed': True,
        'physical_hashes': {n: sha(run / n) for n in ['zero_before_reload.json', 'zero_after_reload.json', 'initial_operation_vector.pt', 'initial_shared_parameters.pt']},
        'preflight_checkpoint_hashes': {f.name: sha(f) for f in cp.iterdir() if f.is_file()},
        'full_old_baseline_new_skill_and_retention_NOT_qualified': True,
        'CPU_production_model_calls_or_updates': 0, 'reserved_reads': False,
        'audit_code_sha256': sha(Path(__file__)), 'overall_goal_achieved': False})
    print('V8 same-arm zero-update runtime CPU audit PASS', arm)

if __name__ == '__main__': main()
