"""Fresh qualified V8 source; full later512 BEFORE/AFTER, no automatic dispatch."""
import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
from queue_matched_memory_v1 import HERE,ROOT,DATA,read,sha
from storage import atomic_json,digest,CycleStore,exact_score
from gpu_lease import gpu_lease
from deadline import check_deadline,DeadlineReached
from v8_later_learning_contract_v1 import physical_admission,configure_later_trainable
from v8_later_isolated_protocol_v1 import PROTOCOL,PROFILE,PREPARED,PREFLIGHT_AUDIT,final_gate


def verify_prepared():
    prepared=read(DATA/PREPARED)
    if prepared['protocol']!=PROTOCOL or prepared['profile']!=PROFILE:
        raise ValueError('Exact isolated production profile required')
    for n,h in prepared['code_hashes'].items():
        if sha(HERE/n)!=h:raise ValueError('Changed production code '+n)
    for n,h in prepared['external_code_hashes'].items():
        if sha(HERE.parent/n)!=h:raise ValueError('Changed external code '+n)
    for n,h in prepared['data_hashes'].items():
        if sha(DATA/n)!=h:raise ValueError('Changed data/preparation '+n)
    return prepared


def source_and_curriculum(source_run):
    admission=physical_admission(DATA,source_run)
    source=DATA/source_run;parent=read(source/'results.json')
    world=read(DATA/'isolated_memory10240_seed20261005_v1.json')
    curriculum=read(DATA/'post_conditional_memory512_seed20261005_v1.json')
    skill=read(DATA/'permutation_broad_vocabulary_teaching_seed20261005_v8.json')
    from post_conditional_retention_protocol_v5 import validate_curriculum
    rows=validate_curriculum(curriculum,world,skill['operation_name'])
    groups={k:skill['groups'][k] for k in ('train','dev','audit','transfer_audit')}
    return admission,parent,rows,groups


def save_runtime(path,embedding,modules,lexical,parent):
    import torch
    from shared_skill_module_v1 import export
    cp=Path(parent['checkpoint'])
    torch.save({'blocks':[p.detach().cpu() for p in embedding.blocks]},path/'input_parameters.pt')
    atomic_json(path/'lexical_inventory.json',lexical)
    atomic_json(path/'inference_config.json',read(cp/'inference_config.json'))
    atomic_json(path/'shared_skill_config.json',read(cp/'shared_skill_config.json'))
    if modules:torch.save(export(modules),path/'shared_skill_parameters.pt')


def run(args,target):
    prepared=verify_prepared()
    check_deadline(args.stop_at,{'phase':'later_skill_source_admission'})
    admission,parent,rows,groups=source_and_curriculum(args.source_run)
    arm=parent['manifest']['arm']
    if not args.preflight_only:
        if not args.preflight_run or Path(args.preflight_run).name!=args.preflight_run:
            raise ValueError('Actual source-specific expanded-model runtime/audit required')
        probe=DATA/args.preflight_run;pr=read(probe/'results.json');pa=read(probe/PREFLIGHT_AUDIT)
        if (pr['status']!='runtime_complete_zero_optimizer_updates' or not pa['runtime_integrity_passed']
                or pr['manifest']['source_run']!=args.source_run or pr['manifest']['prepared_sha256']!=sha(DATA/PREPARED)
                or pa['results_sha256']!=sha(probe/'results.json') or pa['source_results_sha256']!=sha(DATA/args.source_run/'results.json')):
            raise ValueError('Partial/wrong-source/wrong-version runtime admission')
        from audit_V8_later_isolated_stage_v1 import verify_audit_binding
        verify_audit_binding(probe,pa)
        probe_binding={'run_id':args.preflight_run,'results_sha256':sha(probe/'results.json'),'audit_sha256':sha(probe/PREFLIGHT_AUDIT)}
    else:probe_binding=None
    import torch
    from model_support import load_tokenizer,patch_causal_mask,get_directml_device
    from shared_skill_module_v1 import restore,export,maintenance_hashes
    from isolated_embeddings_v1 import IsolatedInputEmbedding
    from isolated_inputs_v2 import add_input_names
    from nontruncating_chat_cache_v1 import encode_items,collate
    from teacher_wording_v1 import expand
    from isolated_inference_audit_v1 import evaluate,evaluate_scale_source
    from isolated_training_v1 import train
    from supervised_nll_v3 import supervised_nll
    patch_causal_mask();device=get_directml_device()
    model,old_tok,modules=restore(device,admission['checkpoint'])
    original=model.get_input_embeddings().original
    source_blocks=torch.load(Path(parent['checkpoint'])/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    source_shared=export(modules);maintenance=maintenance_hashes(model)
    if maintenance!=parent['frozen_maintenance_hashes_after']:raise ValueError('Source maintenance restoration changed')
    initial_tok=load_tokenizer();weight=original.weight.detach().cpu().float()
    initial=torch.stack([weight[initial_tok(r['stimulus'],add_special_tokens=False)['input_ids']].mean(0) for r in rows]);del weight
    source_lexical=read(Path(parent['checkpoint'])/'lexical_inventory.json')
    tok=load_tokenizer();lexical=add_input_names(tok,source_lexical['names']+[r['stimulus'] for r in rows],49152)
    if lexical['new_input_rows']!=10753 or tok('qezorvank',add_special_tokens=False)['input_ids']!=[59392]:
        raise ValueError('Appended lexical IDs changed')
    old_questions=[r['question'] for g in (parent['final_old_raw'],parent['final_skill_raw']) for rr in g.values() for r in rr]
    if tok(old_questions,add_special_tokens=False)['input_ids']!=old_tok(old_questions,add_special_tokens=False)['input_ids']:
        raise ValueError('Old/skill tokenization changed')
    embedding=IsolatedInputEmbedding(original,source_blocks+[initial]);model.set_input_embeddings(embedding)
    parameter_groups=configure_later_trainable(model,embedding,modules,arm,'isolated_new_bank')
    params=parameter_groups['name'];model.eval();forms=expand(rows);cache=encode_items(tok,forms,64,8192)
    supervised=sum(int((labels[1:]!=-100).sum()) for _,labels in cache)
    if supervised!=3456 or model.get_output_embeddings().weight.shape[0]!=49152:
        raise ValueError('Unexpected teacher budget/output vocabulary')
    manifest={'protocol':PROTOCOL,'profile':PROFILE,'run_id':args.run_id,'source_run':args.source_run,'source_arm':arm,
        'stop_at':args.stop_at,'prepared_sha256':sha(DATA/PREPARED),'source_results_sha256':sha(DATA/args.source_run/'results.json'),
        'source_audit_sha256':admission['source_audit_sha256'],'source_checkpoint':parent['checkpoint'],
        'source_checkpoint_hashes':parent['checkpoint_hashes'],'source_admission':admission,'backbone':parent['manifest']['backbone'],
        'code_hashes':prepared['code_hashes'],'external_code_hashes':prepared['external_code_hashes'],
        'dataset_sha256':digest(rows),'teacher_forms_sha256':digest(forms),'active_trainable_parameters':1048576,
        'input_inventory_rows':10753,'output_vocabulary_size':49152,'new_bank_tensor_bytes':4194304,
        'source_memory_generation_batch':parent['manifest']['source_memory_generation_batch'],
        'frozen_source_maintenance_hashes':maintenance,'shared_module_attached_and_frozen':arm=='shared',
        'inference_policy':'current question and neural parameters only; runtime restore never reads teachers',
        'reserved_not_read':True,'development_data_reused_not_independent_confirmation':True,
        'old_training_replay_presentations':0,'actual_source_specific_preflight':probe_binding,
        'device':str(device),'torch_version':torch.__version__,'overall_goal_achieved':False}
    atomic_json(target/'manifest.json',manifest);atomic_json(target/'execution_window.json',{'pid':os.getpid(),'stop_at':args.stop_at})
    atomic_json(target/'new512_teaching_archive.json',rows);atomic_json(target/'teacher_forms.json',forms)
    torch.save(initial,target/'initial_new512_bank.pt')
    atomic_json(target/'trainable_tensor_inventory.json',[{'name':n,'shape':list(p.shape),'numel':p.numel(),'dtype':str(p.dtype)} for n,p in model.named_parameters() if p.requires_grad])
    store=CycleStore(target/'checkpoints',manifest)
    atomic_json(target/'results.json',{'status':'fresh_expanded_model_actual_baseline_pending','manifest':manifest,'history':[]})
    if args.preflight_only:
        references=parent['final_old_raw']['base_capabilities'][:2]+parent['final_old_raw']['exact'][:2]+parent['final_skill_raw']['dev'][:2]
        chosen=[{'id':r['id'],'type':r['type'],'eval_q':r['question'],'answer':r['expected']} for r in references]+rows[:2]
        before=evaluate(model,tok,chosen,target/'runtime_before.json',args.stop_at)
        if before[:6]!=references:raise ValueError('Actual expanded runtime source raw parity failed')
        ids,labels=collate(cache,list(range(8)),tok.pad_token_id,device)
        logits=model(input_ids=ids).logits;loss=supervised_nll(logits,labels)
        cpu=logits.detach().cpu().float()[:,:-1,:];targets=labels.detach().cpu()[:,1:];valid=targets!=-100
        selected=cpu[valid].contiguous();target_labels=targets[valid].contiguous()
        reference=float(-selected.log_softmax(-1).gather(1,target_labels.unsqueeze(1)).mean())
        gpu_loss=float(loss.detach().cpu())
        if abs(reference-gpu_loss)>=1e-4:raise ValueError('Expanded-model full-vocabulary NLL mismatch')
        loss.backward();norm=float(params[0].grad.detach().cpu().norm())
        if not math.isfinite(norm) or norm<=0 or any(p.grad is not None for p in embedding.blocks[:21]):
            raise ValueError('Incorrect new-bank gradient path')
        torch.save({'answer_logits':selected,'targets':target_labels},target/'runtime_NLL_reference_inputs.pt')
        checkpoint=store.pending(1);save_runtime(checkpoint,embedding,modules,lexical,parent)
        del model,embedding,modules,params,parameter_groups,original,cache,logits,loss;gc.collect()
        model,tok,modules=restore(device,checkpoint)
        after=evaluate(model,tok,chosen,target/'runtime_after_reload.json',args.stop_at)
        if before!=after:raise ValueError('Expanded-model runtime reload changed raw output')
        atomic_json(target/'results.json',{'status':'runtime_complete_zero_optimizer_updates','manifest':manifest,
            'actual_training_updates':0,'preflight_checkpoint':str(checkpoint),'actual_generated_answers':16,
            'loss_gpu':gpu_loss,'loss_cpu':reference,'new_bank_gradient_norm':norm,
            'full_old_skill_and_new_memory_not_qualified':True,'overall_goal_achieved':False})
        return
    batch=manifest['source_memory_generation_batch']
    old_before=evaluate_scale_source(model,tok,parent['final_old_raw'],target/'baseline_old',args.stop_at,batch)
    skill_before={k:evaluate(model,tok,groups[k],target/'baseline_skill'/(k+'.json'),args.stop_at) for k in ('dev','audit','transfer_audit')}
    atomic_json(target/'baseline_old.json',old_before);atomic_json(target/'baseline_skill.json',skill_before)
    if old_before!=parent['final_old_raw'] or skill_before!=parent['final_skill_raw']:
        raise ValueError('Complete expanded-model source parity failed before teaching')
    new_before={'exact':evaluate(model,tok,rows,target/'baseline_new_exact.json',args.stop_at,batch),
                'paraphrase':evaluate(model,tok,[{**r,'eval_q':r['audit_q']} for r in rows],target/'baseline_new_paraphrase.json',args.stop_at,batch)}
    atomic_json(target/'baseline_new512.json',new_before)
    atomic_json(target/'results.json',{'status':'complete_actual22820_BEFORE_training_pending','manifest':manifest,'history':[]})
    phases=train(model,tok,cache,forms,params,device,PROFILE,target,args.stop_at)
    embedding.activate_block(None)
    if any(not torch.equal(p.detach().cpu(),source_blocks[i]) for i,p in enumerate(embedding.blocks[:21])):
        raise ValueError('Old memory/command tensors changed')
    if any(not torch.equal(export(modules)[n][k],v) for n,s in source_shared.items() for k,v in s.items()):
        raise ValueError('Frozen shared skill parameters changed')
    if maintenance_hashes(model)!=maintenance:raise ValueError('Frozen maintenance changed')
    learned=embedding.blocks[21].detach().cpu()
    if not torch.isfinite(learned).all() or torch.equal(learned,initial):
        raise ValueError('New bank nonfinite or unchanged after recorded updates')
    chosen=rows[:2]+groups['dev'][:2]
    before=evaluate(model,tok,chosen,target/'before_reload_probe.json',args.stop_at)
    pending=store.pending(1);save_runtime(pending,embedding,modules,lexical,parent)
    del model,embedding,modules,params,parameter_groups,cache,original;gc.collect()
    model,tok,modules=restore(device,pending)
    after=evaluate(model,tok,chosen,target/'after_reload_probe.json',args.stop_at)
    if before!=after:raise ValueError('Production saved-model reload parity failed')
    old_after=evaluate_scale_source(model,tok,parent['final_old_raw'],target/'final_old',args.stop_at,batch)
    skill_after={k:evaluate(model,tok,groups[k],target/'final_skill'/(k+'.json'),args.stop_at) for k in ('dev','audit','transfer_audit')}
    new_after={'exact':evaluate(model,tok,rows,target/'final_new_exact.json',args.stop_at,batch),
               'paraphrase':evaluate(model,tok,[{**r,'eval_q':r['audit_q']} for r in rows],target/'final_new_paraphrase.json',args.stop_at,batch)}
    original_skill_before=read(DATA/args.source_run/'baseline_skill.json')
    gate=final_gate(parent['final_old_raw'],parent['final_skill_raw'],old_before,skill_before,old_after,skill_after,rows,new_before,new_after,original_skill_before)
    from v8_later_learning_contract_v1 import fact_after_learning
    acquisition=fact_after_learning(rows,new_before,new_after);selected_ids=set(acquisition['conservative_acquired_ids'])
    compact_tokens=sum(len(tok(r['stimulus']+' '+r['answer'],add_special_tokens=False)['input_ids']) for r in rows if r['id'] in selected_ids)
    history=[{'cycle':1,'phases':phases,'joint_gate':gate}]
    cp=store.commit(pending,{'cycle':1,'history':history,'reload_parity':{'passed':True,'before':before,'after_reload':after}})
    files=['input_parameters.pt','lexical_inventory.json','inference_config.json','shared_skill_config.json']
    if arm=='shared':files.append('shared_skill_parameters.pt')
    budgets={out:sum(p[key] for p in phases.values()) for key,out in [('steps','actual_updates'),('presentations','actual_presentations'),('supervised_tokens','actual_supervised_tokens'),('training_sec','actual_training_sec')]}
    result={'status':'complete','manifest':manifest,'history':history,'checkpoint':str(cp),'checkpoint_hashes':{n:sha(cp/n) for n in files},
        'final_old_raw':old_after,'final_skill_raw':skill_after,'new_final_raw':new_after,'joint_gate':gate,
        'full_actual_BEFORE_queries':22820,'full_fresh_restored_final_queries':22820,
        'conservative_new_fact_acquisitions':acquisition['conservative_acquired_facts'],'new_acquired_compact_tokens':compact_tokens,
        'all_source21_banks_bit_identical':True,'shared_and_maintenance_frozen_preserved':True,**budgets,
        'inherited_scale_evidence_is_retained_NOT_new_acquisitions':True,'isolation_NOT_sleep_causal_evidence':True,'overall_goal_achieved':False}
    atomic_json(target/'results.json',result);print('V8 LATER ISOLATED COMPLETE',args.source_run,gate['passed'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--source-run',required=True)
    p.add_argument('--stop-at',required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--preflight-run');args=p.parse_args()
    if any(Path(v).name!=v for v in (args.run_id,args.source_run)):raise ValueError('Simple fresh IDs required')
    if args.stop_at!=read(DATA/'second_window_state_v1.json')['stop_at']:raise ValueError('Current authorized deadline required')
    verify_prepared();source_and_curriculum(args.source_run)
    with gpu_lease(DATA,'V8_later_isolated512_v1'):
        target=DATA/args.run_id
        if target.exists():raise FileExistsError('Fresh version/ID; no partial resume')
        target.mkdir()
        try:run(args,target)
        except BaseException as error:
            atomic_json(target/'failure.json',{'error':repr(error),'deadline_progress':getattr(error,'progress',None),
                'partial_weights_not_qualified_or_reusable':True,'overall_goal_achieved':False});raise


if __name__=='__main__':main()
