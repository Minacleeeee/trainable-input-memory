"""Meaningful CPU guard on all qualified learned rows, no production model call."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import torch
from torch import nn
from deployment_inputs_fp16_v1 import DeploymentInputEmbeddingFP16
from isolated_embeddings_v1 import IsolatedInputEmbedding
from storage import atomic_json

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]/'data/stage80_continual'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expect_rejected(call):
    try:
        call()
    except (ValueError, RuntimeError, IndexError):
        return
    raise AssertionError('Unsafe or inconsistent operation was accepted')


def main():
    target = DATA/'independent_deployment_inputs_cpu_guard_v1.json'
    if target.exists():
        raise FileExistsError('Immutable CPU evidence already exists')
    torch.set_num_threads(4)
    source = DATA/'isolated_scale10240_s42_v2'
    result = json.loads((source/'results.json').read_text(encoding='utf-8'))
    audit = json.loads((source/'independent_isolated_scale10240_audit.json').read_text(encoding='utf-8'))
    assert result['status'] == audit['status'] == 'complete'
    assert audit['joint_memory_retention_gate_passed']
    assert audit['input_hashes']['results.json'] == sha(source/'results.json')
    cp = Path(result['checkpoint'])
    for name, expected in result['checkpoint_hashes'].items():
        assert sha(cp/name) == expected
    saved = torch.load(cp/'input_parameters.pt', map_location='cpu', weights_only=True)
    blocks = saved['blocks']
    assert len(blocks) == 20 and all(b.shape == (512,2048) and b.dtype == torch.float32 for b in blocks)
    # A tiny synthetic frozen base exercises both original token paths without
    # loading the full backbone or claiming its responses have been measured.
    generator = torch.Generator().manual_seed(202610051)
    base = torch.randn(3,2048,generator=generator).half()
    before = IsolatedInputEmbedding(nn.Embedding.from_pretrained(base.clone(),freeze=True), blocks)
    after = DeploymentInputEmbeddingFP16(nn.Embedding.from_pretrained(base.clone(),freeze=True), blocks)
    ids = torch.arange(after.num_embeddings,dtype=torch.long).reshape(1,-1)
    with torch.no_grad():
        assert torch.equal(before(ids),after(ids))
        mixed = torch.randint(0,after.num_embeddings,(8,51),generator=generator)
        assert torch.equal(before(mixed),after(mixed))
    assert torch.equal(after.original.weight,base)
    assert all(b.dtype == torch.float16 and not b.requires_grad for b in after.blocks)
    assert all(not p.requires_grad for p in after.parameters())
    tensor_bytes = sum(b.numel()*b.element_size() for b in after.blocks)
    assert tensor_bytes == 41943040
    serialized = io.BytesIO()
    torch.save({'blocks':[b.detach().cpu() for b in after.blocks]},serialized)
    serialized.seek(0)
    reloaded = torch.load(serialized,map_location='cpu',weights_only=True)
    roundtrip = DeploymentInputEmbeddingFP16(nn.Embedding.from_pretrained(base.clone(),freeze=True),reloaded['blocks'])
    assert all(torch.equal(x,y) for x,y in zip(after.blocks,roundtrip.blocks))
    with torch.no_grad():
        assert torch.equal(roundtrip(mixed),after(mixed))
    expect_rejected(lambda: after.activate_block(0))
    expect_rejected(lambda: after(torch.tensor([[after.num_embeddings]],dtype=torch.long)))
    expect_rejected(lambda: after(torch.tensor([[-1]],dtype=torch.long)))
    expect_rejected(lambda: DeploymentInputEmbeddingFP16(nn.Embedding(3,2048),blocks))
    expect_rejected(lambda: DeploymentInputEmbeddingFP16(nn.Embedding(3,2048,dtype=torch.float16),[]))
    overflow = torch.full((1,2048),1e10)
    expect_rejected(lambda: DeploymentInputEmbeddingFP16(nn.Embedding(3,2048,dtype=torch.float16),[overflow]))
    atomic_json(target,{
        'status':'cpu_embedding_guard_complete_NOT_GPU_or_full_generation_qualified',
        'recorded_at':datetime.datetime.now().astimezone().isoformat(),
        'source_results_sha256':sha(source/'results.json'),
        'source_checkpoint_hashes':result['checkpoint_hashes'],
        'code_hashes':{n:sha(HERE/n) for n in ('deployment_inputs_fp16_v1.py','audit_deployment_inputs_cpu_v1.py','isolated_embeddings_v1.py','storage.py')},
        'all_actual_learned_rows_checked':10240,'hidden_coordinates_per_row':2048,
        'synthetic_original_rows_checked':3,'mixed_ids_shape':[8,51],
        'all_CPU_forward_vectors_bit_identical':True,
        'CPU_serialization_restore_bit_identical':True,
        'deployment_bank_dtype':'torch.float16','resident_bank_tensor_bytes':tensor_bytes,
        'serialized_input_parameter_bytes':serialized.getbuffer().nbytes,
        'training_activation_invalid_ids_base_dtype_and_overflow_rejected':True,
        'production_model_forward_calls':0,'GPU_calls':0,'optimizer_updates':0,
        'source_checkpoint_modified':False,'deployment_checkpoint_written':False,
        'DirectML_cast_full_generation_logits_and_resume_not_tested':True,
        'original_fixed_weights_and_lexical_metadata_are_additional_costs':True,
        'no_fixed_budget_consolidation_or_overall_goal_claim':True})
    print('CPU deployment embedding guard complete; GPU and full-generation parity pending',flush=True)


if __name__ == '__main__':
    main()
