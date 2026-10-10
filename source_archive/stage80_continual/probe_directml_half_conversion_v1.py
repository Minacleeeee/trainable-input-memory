"""Independent zero-update numeric probe; no model answers or teacher labels."""
import datetime
import hashlib
import os
from pathlib import Path
import torch
import torch_directml
from torch.nn import functional as F
from gpu_lease import gpu_lease
from storage import atomic_json

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/stage80_continual'

def main():
    target = DATA / 'directml_half_conversion_diagnostic_v1.json'
    if target.exists():
        raise FileExistsError('Fresh diagnostic only')
    with gpu_lease(DATA, 'directml_half_conversion_diagnostic_v1'):
        source = DATA / 'isolated_scale10240_s42_v2/results.json'
        import json
        parent = json.loads(source.read_text(encoding='utf-8'))
        cp = Path(parent['checkpoint']) / 'input_parameters.pt'
        master = torch.load(cp, map_location='cpu', weights_only=True)['blocks'][0]
        device = torch_directml.device()
        gpu32 = master.to(device)
        cpu_half = master.half()
        device_half = gpu32.half().cpu()
        ids_cpu = torch.arange(master.shape[0]).unsqueeze(0)
        ids = ids_cpu.to(device)
        original = F.embedding(ids, gpu32).half().cpu()
        cpu_path = F.embedding(ids, cpu_half.to(device)).cpu()
        device_path = F.embedding(ids, device_half.to(device)).cpu()
        diff = device_half.float() - cpu_half.float()
        payload = {
            'recorded_at': datetime.datetime.now().astimezone().isoformat(),
            'pid': os.getpid(), 'device': str(device),
            'source_results_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_parameters_sha256': hashlib.sha256(cp.read_bytes()).hexdigest(),
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'actual_rows': master.shape[0], 'actual_coordinates': master.numel(),
            'CPU_vs_device_half_changed_coordinates': int(torch.count_nonzero(diff)),
            'CPU_vs_device_half_max_abs_difference': float(diff.abs().max()),
            'original_vs_CPU_preconverted_embedding_bit_exact': torch.equal(original, cpu_path),
            'original_vs_device_preconverted_embedding_bit_exact': torch.equal(original, device_path),
            'original_vs_device_cast_tensor_bit_exact': torch.equal(original[0], device_half),
            'CPU_cast_towards_zero_coordinate_count': int((device_half.abs() < cpu_half.abs()).sum()),
            'CPU_cast_away_from_zero_coordinate_count': int((device_half.abs() > cpu_half.abs()).sum()),
            'model_answers': 0, 'optimizer_updates': 0,
            'source_unchanged': hashlib.sha256(cp.read_bytes()).hexdigest() == parent['checkpoint_hashes']['input_parameters.pt'],
            'full_deployment_or_compact_learning_qualification': False,
        }
        atomic_json(target, payload)
        print(payload, flush=True)

if __name__ == '__main__':
    main()
