"""Fresh four-slot name-only trial; actual BEFORE and restored AFTER, no numeric branch claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from storage import atomic_json, CycleStore, exact_score, digest
from deadline import check_deadline, DeadlineReached
from gpu_lease import gpu_lease
from permutation_broad_skill_protocol_v8 import PROTOCOL, PROFILE, CODE_NAMES, qualify_source, validate_development_groups, gate_from_raw
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/stage80_continual'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(args, target):
    import torch
    from model_support import load_tokenizer, patch_causal_mask, get_directml_device
    from run_isolated_memory_pilot_v2 import restore as restore_inputs
    from shared_skill_module_v1 import attach, export, restore, CONFIG, configure_trainable, maintenance_hashes
    from isolated_inputs_v2 import add_input_names
    from isolated_embeddings_v1 import IsolatedInputEmbedding
    from isolated_inference_audit_v1 import evaluate, evaluate_scale_source
    from cached_batches import encode_items, collate
    from shared_skill_training_v1 import train
    from supervised_nll_v3 import supervised_nll
    prepared_path = DATA / 'prepared_permutation_broad_skill_v8.json'; prepared = read(prepared_path)
    assert prepared['profile'] == PROFILE
    for name,h in prepared['code_hashes'].items(): assert sha(HERE/name) == h
    for name,h in prepared['data_hashes'].items(): assert sha(DATA/name) == h
    for name,h in prepared['external_code_hashes'].items(): assert sha(HERE.parent/name) == h
    source = DATA/'isolated_scale10240_s42_v2'; parent = read(source/'results.json')
    qualification = qualify_source(parent, read(source/'independent_isolated_scale10240_audit.json'), sha(source/'results.json'))
    assert sha(source/'results.json') == prepared['source_results_sha256']
    for n,h in parent['checkpoint_hashes'].items(): assert sha(Path(parent['checkpoint'])/n) == h
    for n,h in parent['manifest']['code_hashes'].items(): assert sha(HERE/n) == h
    dataset_path = DATA/'permutation_broad_vocabulary_teaching_seed20261005_v8.json'; data = read(dataset_path)
    groups = {k:data['groups'][k] for k in ('train','dev','audit','transfer_audit')}
    validate_development_groups(groups, data['operation_name'])
    admitted_probe = None
    if not args.preflight_only:
        if not args.preflight_run or Path(args.preflight_run).name != args.preflight_run:
            raise ValueError('Actual complete same-arm V8 GPU preflight and independent audit required')
        probe_dir = DATA / args.preflight_run
        probe = read(probe_dir / 'results.json')
        probe_audit_path = probe_dir / 'independent_permutation_broad_runtime_preflight_cpu_v8.json'
        probe_audit = read(probe_audit_path)
        if (probe['status'] != 'runtime_preflight_complete_no_updates'
                or probe['actual_training_updates'] != 0
                or probe['manifest']['arm'] != args.arm
                or probe['manifest']['prepared_sha256'] != sha(prepared_path)
                or probe['manifest']['dataset_sha256'] != sha(dataset_path)
                or probe_audit['input_results_sha256'] != sha(probe_dir / 'results.json')
                or not probe_audit['zero_optimizer_updates_and_old_banks_exact']):
            raise ValueError('V8 same-arm runtime admission mismatch')
        admitted_probe = {'run_id': args.preflight_run, 'results_sha256': sha(probe_dir / 'results.json'),
                          'audit_sha256': sha(probe_audit_path)}
    check_deadline(args.stop_at, {'phase':'load_fresh_qualified_parent','run_id':args.run_id})
    patch_causal_mask(); device = get_directml_device()
    model, old_tok = restore_inputs(device, parent['checkpoint'])
    parent_maintenance = maintenance_hashes(model)
    original = model.get_input_embeddings().original
    initial_tok = load_tokenizer(); ids = initial_tok(data['operation_name'],add_special_tokens=False)['input_ids']
    initial = original.weight.detach().cpu().float()[ids].mean(dim=0,keepdim=True)
    old_blocks = torch.load(Path(parent['checkpoint'])/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    lexical_old = read(Path(parent['checkpoint'])/'lexical_inventory.json')
    tok = load_tokenizer(); lexical = add_input_names(tok,lexical_old['names']+[data['operation_name']],49152)
    assert tok(data['operation_name'],add_special_tokens=False)['input_ids']==[59392]
    old_questions = [r['question'] for rows in parent['final_raw'].values() for r in rows]
    assert old_tok(old_questions,add_special_tokens=False)['input_ids']==tok(old_questions,add_special_tokens=False)['input_ids']
    embedding = IsolatedInputEmbedding(original,old_blocks+[initial]); model.set_input_embeddings(embedding)
    modules = attach(model) if args.arm=='shared' else {}
    parameter_groups = configure_trainable(model,embedding,modules,args.arm)
    assert maintenance_hashes(model)==parent_maintenance
    model.eval()
    initial_shared = export(modules)
    active = sum(p.numel() for ps in parameter_groups.values() for p in ps)
    manifest = {'protocol':PROTOCOL,'run_id':args.run_id,'arm':args.arm,'profile':PROFILE,
                'stop_at':args.stop_at,'source_run':source.name,'source_results_sha256':sha(source/'results.json'),
                'source_audit_sha256':sha(source/'independent_isolated_scale10240_audit.json'),
                'source_checkpoint':parent['checkpoint'],'source_checkpoint_hashes':parent['checkpoint_hashes'],
                'source_qualification':qualification,'admitted_zero_update_probe':admitted_probe,'backbone':parent['manifest']['backbone'],
                'prepared_sha256':sha(prepared_path),'dataset_sha256':sha(dataset_path),
                'development_group_hashes':{k:digest(v) for k,v in groups.items()},
                'active_trainable_parameters':active,'output_vocabulary_size':49152,'input_inventory_rows':10241,
                'device':str(device),'torch_version':torch.__version__,
                'frozen_maintenance_hashes':parent_maintenance,'actual_old_pregradient_generated':not args.preflight_only,
                'source_memory_generation_batch':parent['manifest']['memory_generation_batch'],
                'code_hashes':prepared['code_hashes'],'external_code_hashes':prepared['external_code_hashes'],
                'teacher_rules_never_inference_inputs':True,'inference_policy':'current question and neural parameters only',
                'reserved_not_read':True,'old_replay_presentations':0,'development_informed_design':True,
                'failed_conditional_weights_reused':False,'numeric_branch_selection_NOT_tested':True,
                'task_scope':'fresh four-slot learned command, not unprecedented algorithm',
                'overall_goal_achieved':False}
    atomic_json(target/'manifest.json',manifest)
    atomic_json(target/'execution_window.json',{'pid':os.getpid(),'stop_at':args.stop_at})
    atomic_json(target/'development_dataset.json',groups)
    atomic_json(target/'results.json',{'status':'initialization','manifest':manifest,'history':[]})
    torch.save(initial_shared,target/'initial_shared_parameters.pt')
    torch.save(initial,target/'initial_operation_vector.pt')
    atomic_json(target/'trainable_tensor_inventory.json',[
        {'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel()} for n,p in model.named_parameters() if p.requires_grad])
    store = CycleStore(target/'checkpoints',manifest)
    cache = encode_items(tok,groups['train'])
    if args.preflight_only:
        chosen = [dict(r,eval_q=r['question'],answer=r['expected']) for r in parent['final_raw']['base_capabilities'][:2]]+groups['dev'][:2]
        before = evaluate(model,tok,chosen,target/'zero_before_reload.json',args.stop_at)
        for row,reference in zip(before[:2],parent['final_raw']['base_capabilities'][:2]): assert row==reference
        ids,labels=collate(cache,[0,1],tok.pad_token_id,device)
        logits=model(input_ids=ids).logits;loss=supervised_nll(logits,labels)
        cpu=logits.detach().cpu().float()[:,:-1,:];targets=labels.detach().cpu()[:,1:];valid=targets!=-100
        reference=float(-cpu[valid].log_softmax(-1).gather(1,targets[valid].unsqueeze(1)).mean())
        gpu_loss=float(loss.detach().cpu());assert abs(gpu_loss-reference)<1e-4
        loss.backward()
        gradients={n:float(p.grad.detach().cpu().norm()) for n,p in model.named_parameters() if p.requires_grad}
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for ps in parameter_groups.values() for p in ps)
        assert float(embedding.blocks[20].grad.detach().cpu().norm())>0
        if args.arm=='shared':
            assert all(float(m.skill_down.grad.detach().cpu().norm())==0 for m in modules.values())
            assert all(float(m.skill_up.grad.detach().cpu().norm())>0 for m in modules.values())
        model.zero_grad(set_to_none=True)
        pending=store.pending(1)
        save_runtime(pending,embedding,modules,args.arm,lexical,parent)
        del model,embedding,parameter_groups,cache,logits,loss,modules;gc.collect()
        model,tok,modules=restore(device,pending)
        after=evaluate(model,tok,chosen,target/'zero_after_reload.json',args.stop_at);assert before==after
        atomic_json(target/'results.json',{'status':'runtime_preflight_complete_no_updates','manifest':manifest,
                    'preflight_checkpoint':str(pending),'small_zero_init_raw_parity':True,'actual_generated_answers':8,'actual_training_updates':0,
                    'loss_cpu':reference,'loss_gpu':gpu_loss,'gradient_norms':gradients,'full_old_baseline_NOT_measured':True,
                    'model_checkpoint_NOT_qualified':True,'overall_goal_achieved':False})
        print('V8 PREFLIGHT COMPLETE',args.arm,flush=True);return
    baseline_old=evaluate_scale_source(model,tok,parent['final_raw'],target/'baseline_old',args.stop_at,parent['manifest']['memory_generation_batch'])
    atomic_json(target/'baseline_old.json',baseline_old)
    if baseline_old!=parent['final_raw']:
        raise ValueError('Full actual zero-update old raw parity failed; no training allowed')
    baseline={k:evaluate(model,tok,groups[k],target/'baseline_skill'/ (k+'.json'),args.stop_at) for k in ('dev','audit','transfer_audit')}
    atomic_json(target/'baseline_skill.json',baseline)
    atomic_json(target/'results.json',{'status':'baseline_complete_training_pending','manifest':manifest,'history':[]})
    phases=train(model,tok,cache,groups['train'],parameter_groups,device,PROFILE,target,args.stop_at)
    embedding.activate_block(None)
    assert all(torch.equal(p.detach().cpu(),old_blocks[i]) for i,p in enumerate(embedding.blocks[:20]))
    assert maintenance_hashes(model)==parent_maintenance
    for p in model.parameters():p.requires_grad_(False)
    chosen=groups['dev'][:2]
    before=evaluate(model,tok,chosen,target/'before_reload_probe.json',args.stop_at)
    pending=store.pending(1);save_runtime(pending,embedding,modules,args.arm,lexical,parent)
    del model,embedding,parameter_groups,cache,modules;gc.collect()
    model,tok,modules=restore(device,pending)
    after=evaluate(model,tok,chosen,target/'after_reload_probe.json',args.stop_at);assert before==after
    final_old=evaluate_scale_source(model,tok,parent['final_raw'],target/'final_old',args.stop_at,parent['manifest']['memory_generation_batch'])
    final_skill={k:evaluate(model,tok,groups[k],target/'final_skill'/(k+'.json'),args.stop_at) for k in ('dev','audit','transfer_audit')}
    gate=gate_from_raw(groups,baseline,final_skill,final_old==baseline_old,exact_score)
    history=[{'cycle':1,'phases':phases,'joint_gate':gate,'final_skill_raw':final_skill}]
    checkpoint=store.commit(pending,{'cycle':1,'history':history,'reload_parity':{'passed':True,'before':before,'after_reload':after}})
    budgets={out:sum(b[k] for b in phases.values()) for k,out in [('steps','actual_updates'),('presentations','actual_presentations'),('supervised_tokens','actual_supervised_tokens'),('training_sec','actual_training_sec')]}
    files=['input_parameters.pt','lexical_inventory.json','inference_config.json','shared_skill_config.json']
    if args.arm=='shared':files+=['shared_skill_parameters.pt']
    result={'status':'complete','manifest':manifest,'history':history,'checkpoint':str(checkpoint),
            'checkpoint_hashes':{n:sha(checkpoint/n) for n in files},'final_old_raw':final_old,'final_skill_raw':final_skill,
            'joint_gate':gate,'full_actual_pregradient_queries':21796,'full_saved_reload_final_queries':21796,
            'all_old20_neural_banks_bit_identical':True,'frozen_maintenance_hashes_after':maintenance_hashes(model),
            'overall_goal_achieved':False,**budgets}
    assert result['frozen_maintenance_hashes_after']==parent_maintenance
    atomic_json(target/'results.json',result);print('V8 SKILL COMPLETE',args.arm,gate,flush=True)


def save_runtime(pending,embedding,modules,arm,lexical,parent):
    import torch
    from shared_skill_module_v1 import CONFIG,export
    torch.save({'blocks':[p.detach().cpu() for p in embedding.blocks]},pending/'input_parameters.pt')
    atomic_json(pending/'lexical_inventory.json',lexical)
    atomic_json(pending/'inference_config.json',read(Path(parent['checkpoint'])/'inference_config.json'))
    atomic_json(pending/'shared_skill_config.json',{'arm':arm,'module_config':CONFIG if arm=='shared' else None})
    if arm=='shared':torch.save(export(modules),pending/'shared_skill_parameters.pt')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--arm',choices=['input','shared'],required=True)
    ap.add_argument('--preflight-run');ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--stop-at',required=True);args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id:raise ValueError('Simple fresh run ID required')
    if args.stop_at!=read(DATA/'second_window_state_v1.json')['stop_at']:raise ValueError('Current authorized window required')
    with gpu_lease(DATA,'permutation_skill_v7_'+args.arm):
        target=DATA/args.run_id
        if target.exists():raise FileExistsError('Fresh run ID; no partial resume')
        target.mkdir()
        try:run(args,target)
        except BaseException as exc:
            atomic_json(target/'failure.json',{'status':'stopped_at_deadline' if isinstance(exc,DeadlineReached) else 'failed',
                         'error':repr(exc),'deadline_progress':getattr(exc,'progress',None),
                         'partial_weights_not_qualified_or_reusable':True});raise

if __name__=='__main__':main()
