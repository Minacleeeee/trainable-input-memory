"""CPU-only fresh preparation for a device-conversion deployment repair."""
import datetime
import hashlib
import json
import py_compile
from pathlib import Path
from storage import atomic_json

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/stage80_continual'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    from half_round_towards_zero_cpu_v1 import self_check
    from conditional_skill_protocol_v5 import qualify_source
    self_check()
    target = DATA / 'prepared_deployment_fp16_v2.json'
    if target.exists(): raise FileExistsError('Fresh preparation only')
    prior = read(DATA / 'prepared_deployment_fp16_v1.json')
    # Original dependencies and failure evidence remain immutable.
    for name, expected in prior['code_hashes'].items():
        assert sha(HERE / name) == expected
    diagnostic_path = DATA / 'directml_half_conversion_diagnostic_v1.json'
    diagnostic = read(diagnostic_path)
    assert diagnostic['original_vs_device_preconverted_embedding_bit_exact']
    assert diagnostic['original_vs_device_cast_tensor_bit_exact']
    assert diagnostic['CPU_cast_away_from_zero_coordinate_count'] == 0
    assert diagnostic['CPU_vs_device_half_changed_coordinates'] > 0
    source = DATA / prior['source_run']
    parent = read(source / 'results.json')
    qualify_source(parent, read(source / 'independent_isolated_scale10240_audit.json'), sha(source / 'results.json'))
    assert sha(source / 'results.json') == diagnostic['source_results_sha256'] == prior['source_results_sha256']
    for name, expected in parent['checkpoint_hashes'].items():
        assert sha(Path(parent['checkpoint']) / name) == expected
    own = ['deployment_inputs_fp16_v2.py', 'run_deployment_fp16_v2.py',
           'audit_deployment_fp16_v2.py', 'prepare_deployment_fp16_v2.py',
           'half_round_towards_zero_cpu_v1.py', 'conditional_skill_protocol_v5.py',
           'probe_directml_half_conversion_v1.py']
    for name in own: py_compile.compile(str(HERE / name), doraise=True)
    names = sorted(set(prior['code_hashes']) | set(own))
    output = {**prior,
              'protocol': 'stage80_deployment_fp16_device_conversion_v2',
              'recorded_at': datetime.datetime.now().astimezone().isoformat(),
              'code_hashes': {name: sha(HERE / name) for name in names},
              'prior_failed_run': 'deployment_fp16_scale_s42_w2_v1',
              'failed_GPU_embedding_checks_sha256': sha(DATA / 'deployment_fp16_scale_s42_w2_v1/GPU_embedding_parity.json'),
              'prior_preparation_sha256': sha(DATA / 'prepared_deployment_fp16_v1.json'),
              'device_conversion_diagnostic_sha256': sha(diagnostic_path),
              'CPU_RTZ_reference_self_checks_passed': True,
              'source_CPU_guard_NOT_v2_GPU_generation_qualification': True,
              'conversion_policy': 'actual DirectML source fp32 embedding cast, freeze saved fp16; CPU RTZ audit plus full raw parity',
              'fresh_whole_qualified_source_not_partial_failed_checkpoint': True}
    atomic_json(target, output)
    print('V2 device-conversion repair pinned; no GPU dispatch by preparation')

if __name__ == '__main__': main()
