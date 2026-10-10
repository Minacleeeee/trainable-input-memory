"""Physical CPU audit; never generates answers or authorizes partial sources."""
import argparse
import datetime
import math
from pathlib import Path
from queue_matched_memory_v1 import HERE,DATA,read,sha
from storage import atomic_json,digest
from v8_later_isolated_protocol_v1 import PROTOCOL,PROFILE,PREPARED,AUDIT,PREFLIGHT_AUDIT,final_gate
from v8_later_learning_contract_v1 import physical_admission


def require(condition,message):
    if not condition:raise ValueError(message)


def verify_audit_binding(run,audit):
    require(audit.get('runtime_integrity_passed') is True,'Complete independent runtime audit required')
    require(audit['results_sha256']==sha(run/'results.json'),'Runtime result changed')
    require(bool(audit['bound_file_hashes']),'Physical binding required')
    for name,h in audit['bound_file_hashes'].items():
        path=(run/name).resolve()
        require(path.is_relative_to(run.resolve()),'Audit path escapes its run')
        require(sha(path)==h,'Changed audited physical artifact '+name)


def physical(run,name,count,hashes):
    path=run/name;value=read(path)
    require(value.get('complete') is True and len(value.get('raw',[]))==count,'Incomplete physical generation '+name)
    hashes[name]=sha(path)
    return value['raw']


def skill_raw(run,folder,hashes):
    return {k:physical(run,folder+'/'+k+'.json',n,hashes) for k,n in [('dev',64),('audit',192),('transfer_audit',192)]}


def old_raw(run,folder,parent,hashes):
    out={}
    for k,rows in parent.items():
        if k in ('exact','paraphrase'):
            out[k]=(physical(run,folder+'/'+k+'_source1024_serial.json',1024,hashes)
                    +physical(run,folder+'/'+k+'_source9216_batched.json',9216,hashes))
        else:out[k]=physical(run,folder+'/'+k+'.json',len(rows),hashes)
    return out


def check_checkpoint(run,cp,parent,rows,manifest,preflight,hashes):
    import torch
    require(cp.resolve().parent==(run/'checkpoints').resolve(),'Checkpoint outside owned run')
    names={'input_parameters.pt','lexical_inventory.json','inference_config.json','shared_skill_config.json'}
    if manifest['source_arm']=='shared':names.add('shared_skill_parameters.pt')
    for n in names:hashes[str((cp/n).relative_to(run))]=sha(cp/n)
    source=Path(parent['checkpoint'])
    for n in ['inference_config.json','shared_skill_config.json']:
        require(read(cp/n)==read(source/n),'Source runtime configuration changed')
    cfg=read(cp/'shared_skill_config.json')
    from shared_skill_module_v1 import CONFIG
    require(cfg=={'arm':manifest['source_arm'],'module_config':CONFIG if manifest['source_arm']=='shared' else None},'Shared forward configuration missing/changed')
    require((cp/'shared_skill_parameters.pt').exists()==(manifest['source_arm']=='shared'),'Unexpected/missing shared weights')
    if manifest['source_arm']=='shared':
        saved=torch.load(cp/'shared_skill_parameters.pt',map_location='cpu',weights_only=True)
        original=torch.load(source/'shared_skill_parameters.pt',map_location='cpu',weights_only=True)
        require(set(saved)==set(original)=={'q_proj','v_proj'},'Shared neural modules changed')
        for n in original:
            require(set(saved[n])=={'A','B'},'Shared parameter keys changed')
            for k,v in original[n].items():require(torch.equal(saved[n][k],v),'Frozen shared weights changed')
    source_blocks=torch.load(source/'input_parameters.pt',map_location='cpu',weights_only=True)['blocks']
    saved=torch.load(cp/'input_parameters.pt',map_location='cpu',weights_only=True)
    require(set(saved)=={'blocks'},'Unexpected neural input state')
    blocks=saved['blocks'];require(len(blocks)==22 and len(source_blocks)==21,'Wrong input bank count')
    for i,v in enumerate(blocks):
        require(v.dtype==torch.float32 and tuple(v.shape)==((1,2048) if i==20 else (512,2048)) and bool(torch.isfinite(v).all()),'Invalid neural bank')
        if i<21:require(torch.equal(v,source_blocks[i]),'Old or command bank changed')
    initial=torch.load(run/'initial_new512_bank.pt',map_location='cpu',weights_only=True)
    require(tuple(initial.shape)==(512,2048) and initial.dtype==torch.float32 and bool(torch.isfinite(initial).all()),'Invalid blind initial bank')
    require(torch.equal(blocks[21],initial)==preflight,'Zero-update parity or actual new-bank learning failed')
    require(initial.numel()*initial.element_size()==manifest['new_bank_tensor_bytes']==4194304,'Incorrect new memory allocation')
    lexical=read(cp/'lexical_inventory.json');source_lex=read(source/'lexical_inventory.json')
    require(lexical['names']==source_lex['names']+[r['stimulus'] for r in rows],'Lexical inventory contains wrong/answer entries')
    from model_support import load_tokenizer
    from isolated_inputs_v2 import add_input_names
    tok=load_tokenizer();require(add_input_names(tok,lexical['names'],49152)==lexical,'Incorrect lexical runtime IDs')
    require(tok('qezorvank',add_special_tokens=False)['input_ids']==[59392] and lexical['new_input_rows']==10753,'Old command ID shifted')
    # Reconstruct label-blind means from pinned frozen original neural weights.
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    backbone=parent['manifest']['backbone'];snapshot=Path(snapshot_download(backbone['model'],revision=backbone['revision'],local_files_only=True))
    key='model.embed_tokens.weight';index=snapshot/'model.safetensors.index.json'
    weightfile=snapshot/(read(index)['weight_map'][key] if index.exists() else 'model.safetensors')
    evidence=read(DATA/'isolated_first512_s42_v2/independent_untrained_bank_initialization_audit_v2.json')
    require(sha(weightfile)==evidence['source_embedding_weight_file_sha256'],'Frozen source neural weight fingerprint changed')
    original_tok=load_tokenizer();parts=original_tok([r['stimulus'] for r in rows],add_special_tokens=False)['input_ids']
    with safe_open(str(weightfile),framework='pt',device='cpu') as handle:weights=handle.get_tensor(key).half().float()
    rebuilt=torch.stack([weights[p].mean(0) for p in parts]);del weights
    require(torch.equal(initial,rebuilt),'Initialization not the specified label-blind original BPE mean')
    hashes['initial_new512_bank.pt']=sha(run/'initial_new512_bank.pt')
    return tok,names


def audit(run_id,preflight=False):
    import torch
    run=DATA/run_id;output=run/(PREFLIGHT_AUDIT if preflight else AUDIT)
    if output.exists():raise FileExistsError('Independent audits are immutable')
    from run_V8_later_isolated_stage_v1 import verify_prepared,source_and_curriculum
    prepared=verify_prepared();result=read(run/'results.json');m=result['manifest'];hashes={}
    require(m['protocol']==PROTOCOL and m['profile']==PROFILE and m['run_id']==run_id,'Wrong experiment/version')
    require(m==read(run/'manifest.json')==read(run/'checkpoints/manifest.json'),'Manifest disagreement')
    require(m['prepared_sha256']==sha(DATA/PREPARED) and m['code_hashes']==prepared['code_hashes'] and m['external_code_hashes']==prepared['external_code_hashes'],'Changed preparation')
    admission,parent,rows,groups=source_and_curriculum(m['source_run'])
    require(m['source_admission']==admission and m['source_results_sha256']==sha(DATA/m['source_run']/'results.json') and m['source_audit_sha256']==admission['source_audit_sha256'],'Wrong/changed qualified source')
    require(m['source_checkpoint']==parent['checkpoint'] and m['source_checkpoint_hashes']==parent['checkpoint_hashes'],'Source checkpoint changed')
    require(m['source_arm']==parent['manifest']['arm'] and m['active_trainable_parameters']==1048576 and m['output_vocabulary_size']==49152 and m['input_inventory_rows']==10753,'Wrong effective architecture')
    require(m['frozen_source_maintenance_hashes']==parent['frozen_maintenance_hashes_after'],'Maintenance fingerprint binding changed')
    from teacher_wording_v1 import expand
    forms=expand(rows)
    require(read(run/'new512_teaching_archive.json')==rows and read(run/'teacher_forms.json')==forms and m['dataset_sha256']==digest(rows) and m['teacher_forms_sha256']==digest(forms),'Teacher identity changed')
    inventory=read(run/'trainable_tensor_inventory.json')
    require(len(inventory)==1 and inventory[0]['shape']==[512,2048] and inventory[0]['numel']==1048576 and inventory[0]['dtype']=='torch.float32' and inventory[0]['name'].endswith('blocks.21'),'Incorrect trainable whitelist')
    cp=Path(result['preflight_checkpoint'] if preflight else result['checkpoint'])
    tok,files=check_checkpoint(run,cp,parent,rows,m,preflight,hashes)
    from nontruncating_chat_cache_v1 import encode_items
    encoded=encode_items(tok,forms,64,8192);per_epoch=sum(int((labels[1:]!=-100).sum()) for _,labels in encoded)
    require(len(encoded)==1024 and per_epoch==3456,'Incorrect complete answer masks')
    from scale_protocol_v1 import validate_rows
    if preflight:
        require(result['status']=='runtime_complete_zero_optimizer_updates' and result['actual_training_updates']==0 and result['actual_generated_answers']==16,'Runtime incomplete/trained')
        refs=parent['final_old_raw']['base_capabilities'][:2]+parent['final_old_raw']['exact'][:2]+parent['final_skill_raw']['dev'][:2]
        chosen=[{'id':r['id'],'type':r['type'],'eval_q':r['question'],'answer':r['expected']} for r in refs]+rows[:2]
        before=physical(run,'runtime_before.json',8,hashes);after=physical(run,'runtime_after_reload.json',8,hashes)
        validate_rows(before,chosen);validate_rows(after,chosen)
        require(before==after and before[:6]==refs,'Runtime source/reload raw parity failed')
        raw=torch.load(run/'runtime_NLL_reference_inputs.pt',map_location='cpu',weights_only=True)
        require(set(raw)=={'answer_logits','targets'} and raw['answer_logits'].shape[1]==49152 and raw['answer_logits'].dtype==torch.float32,'Incomplete full-vocabulary NLL evidence')
        targets=torch.cat([labels[1:][labels[1:]!=-100] for _,labels in encoded[:8]])
        require(torch.equal(raw['targets'],targets) and len(targets)==raw['answer_logits'].shape[0] and bool(torch.isfinite(raw['answer_logits']).all()),'Wrong actual NLL target/mask evidence')
        nll=float(-raw['answer_logits'].log_softmax(-1).gather(1,targets[:,None]).mean())
        require(math.isfinite(result['loss_gpu']) and abs(nll-result['loss_gpu'])<1e-4 and abs(nll-result['loss_cpu'])<1e-7,'CPU recomputed runtime NLL mismatch')
        require(math.isfinite(result['new_bank_gradient_norm']) and result['new_bank_gradient_norm']>0,'Missing positive real gradient')
        require(not (run/'checkpoints/active.json').exists(),'Runtime must not be committed as a scientific learned source')
        hashes['runtime_NLL_reference_inputs.pt']=sha(run/'runtime_NLL_reference_inputs.pt')
        scientific=False;actual={'updates':0,'generated_answers':16}
    else:
        require(result['status']=='complete' and len(result['history'])==1,'Incomplete production result')
        probe=DATA/m['actual_source_specific_preflight']['run_id'];pa=read(probe/PREFLIGHT_AUDIT)
        verify_audit_binding(probe,pa)
        require(pa['source_results_sha256']==m['source_results_sha256'] and sha(probe/PREFLIGHT_AUDIT)==m['actual_source_specific_preflight']['audit_sha256'],'Wrong source runtime audit')
        before=old_raw(run,'baseline_old',parent['final_old_raw'],hashes);after=old_raw(run,'final_old',parent['final_old_raw'],hashes)
        sb=skill_raw(run,'baseline_skill',hashes);sa=skill_raw(run,'final_skill',hashes)
        nb={k:physical(run,'baseline_new_'+k+'.json',512,hashes) for k in ('exact','paraphrase')}
        na={k:physical(run,'final_new_'+k+'.json',512,hashes) for k in ('exact','paraphrase')}
        require(before==read(run/'baseline_old.json') and sb==read(run/'baseline_skill.json') and nb==read(run/'baseline_new512.json'),'Physical BEFORE aggregate changed')
        require(before==parent['final_old_raw'] and sb==parent['final_skill_raw'],'Production trained from an unqualified expanded BEFORE state')
        require(after==result['final_old_raw'] and sa==result['final_skill_raw'] and na==result['new_final_raw'],'Physical final aggregate changed')
        gate=final_gate(parent['final_old_raw'],parent['final_skill_raw'],before,sb,after,sa,rows,nb,na,read(DATA/m['source_run']/'baseline_skill.json'))
        require(gate==result['joint_gate']==result['history'][0]['joint_gate'],'Recomputed scientific gate mismatch')
        phases=read(run/'in_progress_phase_budget.json')['phases']
        require(phases==result['history'][0]['phases'] and set(phases)=={'teach','sleep'},'Actual phase evidence incomplete')
        for phase,epochs,seed in [('teach',20,144),('sleep',10,145)]:
            b=phases[phase]
            require(b['steps']==128*epochs and b['presentations']==1024*epochs and b['supervised_tokens']==per_epoch*epochs and b['epochs']==epochs and b['seed']==seed,'Actual phase budgets incorrect')
            require(b['presentations_by_id']=={r['id']:epochs for r in forms},'Wrong/omitted teacher presentations')
            require(b['active_trainable_parameters']==1048576 and b['optimizer']=='AdamWNoLerp' and b['peak_lr']==PROFILE[phase+'_lr'],'Wrong optimizer/parameters')
            require(math.isfinite(b['training_sec']) and b['training_sec']>0 and len(b['losses'])==epochs and all(math.isfinite(v) for v in b['losses']),'Invalid actual phase duration/losses')
            n=b['first_batch_independent_cpu_loss_check'];require(abs(n['gpu_corrected_nll']-n['independent_cpu_nll'])<1e-4,'Actual production NLL discrepancy')
        for field,key in [('actual_updates','steps'),('actual_presentations','presentations'),('actual_supervised_tokens','supervised_tokens'),('actual_training_sec','training_sec')]:require(result[field]==sum(b[key] for b in phases.values()),'Aggregate actual budget incorrect')
        require(result['full_actual_BEFORE_queries']==result['full_fresh_restored_final_queries']==22820,'Full actual query counts required')
        ids=set(gate['new512_memory']['conservative_acquired_ids'])
        tokens=sum(len(tok(r['stimulus']+' '+r['answer'],add_special_tokens=False)['input_ids']) for r in rows if r['id'] in ids)
        require(result['conservative_new_fact_acquisitions']==len(ids) and result['new_acquired_compact_tokens']==tokens,'Retained actual acquisition count incorrect')
        rb=physical(run,'before_reload_probe.json',4,hashes);ra=physical(run,'after_reload_probe.json',4,hashes)
        validate_rows(rb,rows[:2]+groups['dev'][:2]);require(rb==ra,'Production reload parity failed')
        pointer=read(run/'checkpoints/active.json');record=read(cp/'record.json')
        require(pointer=={'cycle':1,'checkpoint':cp.name,'record_hash':digest(record),'manifest_hash':digest(m)} and record=={'cycle':1,'history':result['history'],'reload_parity':{'passed':True,'before':rb,'after_reload':ra}},'Uncommitted/stale scientific checkpoint')
        require(set(result['checkpoint_hashes'])==files and all(result['checkpoint_hashes'][n]==sha(cp/n) for n in files),'Checkpoint fingerprints incomplete/changed')
        hashes[str((cp/'record.json').relative_to(run))]=sha(cp/'record.json');hashes['checkpoints/active.json']=sha(run/'checkpoints/active.json')
        scientific=gate['passed'];actual={k:result[k] for k in ['actual_updates','actual_presentations','actual_supervised_tokens','actual_training_sec']}
    for n in ['results.json','manifest.json','checkpoints/manifest.json','new512_teaching_archive.json','teacher_forms.json','trainable_tensor_inventory.json']:hashes[n]=sha(run/n)
    atomic_json(output,{'status':'complete_independent_CPU_V8_later_'+('runtime' if preflight else 'stage')+'_audit',
        'recorded_at':datetime.datetime.now().astimezone().isoformat(),'run_id':run_id,'results_sha256':sha(run/'results.json'),
        'source_results_sha256':m['source_results_sha256'],'prepared_sha256':m['prepared_sha256'],
        'runtime_integrity_passed':preflight,'production_integrity_passed':not preflight,'scientific_joint_gate_passed':scientific,
        'all_source21_banks_and_shared_runtime_physically_bit_identical':True,'initialization_all512_label_blind_means_independently_reconstructed':True,
        'bound_file_hashes':hashes,'actual_budget':actual,'generated_before_after_queries':16 if preflight else 45640,
        'maintenance_forward_identity_is_GPU_manifest_bound_NOT_CPU_model_rerun':True,
        'positive_gradient_norm_is_actual_GPU_record_bound_NOT_CPU_backward_rerun':preflight,
        'training_NLL_checks_are_GPU_record_bound_NOT_all_logits_recomputed':not preflight,
        'CPU_model_generation_or_updates':0,'reserved_reads':False,'isolation_NOT_sleep_causal_evidence':True,'overall_goal_achieved':False})
    print('Independent later audit',run_id,'runtime' if preflight else 'production',scientific,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if Path(args.run_id).name!=args.run_id:raise ValueError('Simple run ID required')
    audit(args.run_id,args.preflight_only)
