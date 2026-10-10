"""Fresh double-budget512-association teaching/sleep trial on isolated neural lexical inputs.

Existing backbone/maintenance adapter/input rows/output head stay frozen.
Only one preallocated512-row neural input block learns. No probe-weight reuse.
All inference is normal network generation on the current question only.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from storage import atomic_json,digest,CycleStore,exact_score
from gpu_lease import gpu_lease
from deadline import check_deadline,DeadlineReached
from isolated_inputs_v2 import IsolatedInputEmbedding,freeze_network_except_input_block,add_input_names
from supervised_nll_v3 import supervised_nll
from teacher_wording_v1 import expand as teacher_forms

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'data/stage80_continual'
PROFILE={'teach_epochs':20,'sleep_epochs':10,'teach_lr':.003,'sleep_lr':.001,
         'batch':8,'teacher_forms_per_semantic_fact':2,'new_semantic_facts':512,
         'sleep_source':'all512 introduced associations, both independent teacher forms',
         'old_network_fully_frozen':True,'loss':'explicit2D_masked_answer_mean_nll_v3'}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def restore(device,checkpoint):
    """Inference restoration reads ONLY runtime config, lexical IDs and weights."""
    from engine import pin_backbone,load
    from model_support import load_tokenizer
    import torch
    cp=Path(checkpoint); config=read(cp/'inference_config.json')
    assert pin_backbone()==config['backbone']
    for n,h in config['frozen_adapter_hashes'].items(): assert sha(Path(config['frozen_adapter_path'])/n)==h
    tok=load_tokenizer(); model=load('B',device,config['frozen_adapter_path'],seed=42)
    lexical=read(cp/'lexical_inventory.json')
    assert add_input_names(tok,lexical['names'],lexical['original_vocab_size'])==lexical
    saved=torch.load(cp/'input_parameters.pt',map_location='cpu',weights_only=True)
    wrapped=IsolatedInputEmbedding(model.get_input_embeddings(),saved['blocks']); model.set_input_embeddings(wrapped)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval(); return model,tok


def run(args,run_path):
    check_deadline(args.stop_at,{'run_id':args.run_id,'phase':'initialization'})
    comparison=DATA/'isolated_first512_s42_v1'
    previous=read(comparison/'results.json'); previous_audit=read(comparison/'independent_isolated_pilot_audit.json')
    assert previous['status']=='complete' and previous_audit['integrity_raw_identity_and_actual_budget_passed']
    assert not previous['joint_gate']['passed'] and previous_audit['input_hashes']['results.json']==sha(comparison/'results.json')
    assert read(comparison/'independent_untrained_bank_initialization_audit_v1.json')['all4608_untaught_rows_exactly_label_blind_initialization']
    source=DATA/'repair_maintenance_s42_v1'; parent=read(source/'results.json')
    audit=read(source/'independent_capacity_audit.json')
    if parent['status']!='complete' or not audit['integrity_and_budget_checks_passed'] or audit['input_hashes']['results.json']!=sha(source/'results.json'):
        raise ValueError('Exact completed audited maintenance source required')
    for n,h in parent['checkpoint_hashes'].items():
        if sha(Path(parent['checkpoint'])/n)!=h: raise ValueError('Frozen source weights changed')
    numerical=read(DATA/'directml_supervised_nll_probe_v3.json')
    assert numerical['actual_vocabulary_loss_AND_gradient_validated']
    for n,h in numerical['code_hashes'].items(): assert sha(HERE/n)==h
    world=read(DATA/'progressive_memory_seed20261004.json')
    assert digest(world['novel_items'])==world['novel_dataset_hash'] and digest(world['known_items'])==world['known_dataset_hash']
    prepared=read(DATA/'prepared_isolated_inputs_v1.json'); facts=world['novel_items'][:512]; known=world['known_items']
    assert len(facts)==512 and len(known)==128
    skills=read(DATA/'protected_general1024_s42_v1/skill_dataset.json')
    from base_audit import generate as base_data
    from confirmation_dataset import generate as confirmation_data
    confirmation=read(DATA/'confirmation_probes_v2.json'); assert digest(confirmation['items'])==confirmation['dataset_hash']
    suites={'base_capabilities':base_data(),'confirmation':confirmation_data(),'confirmation_v2':confirmation['items'],
            'skill_audit':skills['audit'],'skill_transfer':skills['transfer_audit'],
            'known_facts':known,'known_paraphrase':[{**r,'eval_q':r['audit_q']} for r in known],
            'exact':facts,'paraphrase':[{**r,'eval_q':r['audit_q']} for r in facts]}
    training=teacher_forms(facts)
    heldout={r['eval_q'] for k,items in suites.items() if k not in ('exact',) for r in items}
    if {r['train_q'] for r in training}&heldout: raise ValueError('Heldout question leaked into teaching')
    import torch
    from engine import pin_backbone,load,evaluate as network_evaluate
    from model_support import load_tokenizer,patch_causal_mask,get_directml_device,format_chat,MAX_LENGTH
    from cached_batches import encode_items,collate
    from adamw_no_lerp import AdamWNoLerp
    backbone=pin_backbone(); assert backbone==parent['manifest']['backbone']==prepared['backbone']
    tok=load_tokenizer(); patch_causal_mask(); device=get_directml_device()
    model=load('B',device,parent['checkpoint'],seed=42); original=model.get_input_embeddings()
    names=prepared['tokenizer_inventory']['names']; assert names==[r['stimulus'] for r in world['novel_items']]
    old_questions=[r['eval_q'] for k,items in suites.items() if k not in ('exact','paraphrase') for r in items]
    old_ids=tok(old_questions,add_special_tokens=False)['input_ids']
    subtokens=tok(names,add_special_tokens=False)['input_ids']; matrix=original.weight.detach().cpu().float()
    initial=torch.stack([matrix[ids].mean(dim=0) for ids in subtokens]); del matrix
    lexical=add_input_names(tok,names,original.num_embeddings)
    assert tok(old_questions,add_special_tokens=False)['input_ids']==old_ids
    wrapped=IsolatedInputEmbedding(original,list(initial.split(512))); del initial
    model.set_input_embeddings(wrapped); params=freeze_network_except_input_block(model,wrapped,0); model.eval()
    assert sum(p.numel() for p in params)==1048576
    assert model.get_output_embeddings().weight.shape[0]==49152
    cache=encode_items(tok,training)
    lengths=[len(tok(format_chat(tok,r['train_q'],r['answer']),add_special_tokens=False)['input_ids']) for r in training]
    assert max(lengths)<=MAX_LENGTH
    for row,(ids,labels) in zip(training,cache):
        full=tok(format_chat(tok,row['train_q'],row['answer']),add_special_tokens=False)['input_ids']
        prompt=tok(format_chat(tok,row['train_q']),add_special_tokens=False)['input_ids']
        assert ids.tolist()==full and int((labels[1:]!=-100).sum())==len(full)-len(prompt)>0
    code_names=['run_isolated_memory_pilot_v2.py','isolated_inputs_v2.py','isolated_embeddings_v1.py','supervised_nll_v3.py',
                'teacher_wording_v1.py','engine.py','model_support.py','cached_batches.py','adamw_no_lerp.py','storage.py',
                'gpu_lease.py','deadline.py','scale_protocol_v1.py','assessment.py','assess_skill_retention.py']
    manifest={'protocol':'stage80_isolated_memory_pilot_v2','run_id':args.run_id,'profile':PROFILE,
              'comparison_run':comparison.name,'comparison_results_hash':sha(comparison/'results.json'),'comparison_audit_hash':sha(comparison/'independent_isolated_pilot_audit.json'),
              'comparison_is_not_initial_weights':True,'changed_setting':'epochs20/10 versus10/5; all other teaching/data/model/profile settings unchanged, warm/cosine schedule stretches with phase duration',
              'source_run':source.name,'source_results_hash':sha(source/'results.json'),'source_audit_hash':sha(source/'independent_capacity_audit.json'),
              'frozen_source_checkpoint':parent['checkpoint'],'frozen_source_hashes':parent['checkpoint_hashes'],
              'backbone':backbone,'device':str(device),'torch_version':torch.__version__,
              'dataset_hash':digest(facts),'known_hash':digest(known),'teacher_forms_hash':digest(training),
              'allocation_parameters':10485760,'active_trainable_parameters':1048576,
              'existing_base_lora_parameters_frozen':True,'output_vocabulary_size':49152,'fixed_input_inventory_rows':5120,
              'original_subtoken_mean_initialization_has_no_gold_labels':True,
              'inherited_previously_taught_facts':128,'previous_other_branch1024_not_inherited':True,
              'original_context_capacity':8192,'actual_runtime_tokenizer_changes_semantic_compact_size':True,
              'no_four_context_claim_possible_from5120_new_token_inventory_corpus':True,
              'code_hashes':{n:sha(HERE/n) for n in code_names},
              'external_code_hashes':{'stage72_gpu/directml_patch.py':sha(HERE.parent/'stage72_gpu/directml_patch.py')},
              'inference_policy':'only current question and model parameters; restoration reads runtime config/lexical inventory/weights',
              'sleep_causality_and_replication_pending':True,'overall_goal_achieved':False}
    store=CycleStore(run_path/'checkpoints',manifest)
    atomic_json(run_path/'execution_window.json',{'pid':os.getpid(),'stop_at':args.stop_at})
    for n,value in [('teaching_archive.json',facts),('known_fact_training.json',known),('teacher_forms.json',training)]: atomic_json(run_path/n,value)
    atomic_json(run_path/'results.json',{'status':'baseline','manifest':manifest,'history':[]})
    sequence=0
    def evaluate(model,tok,items):
        nonlocal sequence
        sequence+=1; path=run_path/'read_only_audits'/f'audit_{sequence:04d}.json'; rows=[]
        try:
            for item in items:
                check_deadline(args.stop_at,{'run_id':args.run_id,'phase':'evaluation','requested':len(items),'answered':len(rows),'raw_file':str(path)})
                rows+=network_evaluate(model,tok,[item])
                if len(rows)%20==0: atomic_json(path,{'complete':False,'raw':rows}); gc.collect()
        except DeadlineReached:
            atomic_json(path,{'complete':False,'raw':rows}); raise
        atomic_json(path,{'complete':True,'raw':rows}); return rows
    def full(model,tok): return {key:evaluate(model,tok,items) for key,items in suites.items()}
    baseline=full(model,tok)
    reference=read(DATA/'balanced_novel512_s42_v1/baseline.json')
    for key in ('base_capabilities','confirmation','confirmation_v2','skill_audit','skill_transfer'):
        if baseline[key]!=reference[key]: raise ValueError('Actual old ability baseline must exactly repeat maintenance')
    if baseline['known_facts']!=parent['final_raw']['exact'] or baseline['known_paraphrase']!=parent['final_raw']['paraphrase']:
        raise ValueError('Old128 baseline changed from maintenance')
    if baseline!=read(comparison/'baseline.json'): raise ValueError('Fresh comparison before-gradient baseline changed')
    atomic_json(run_path/'baseline.json',baseline)
    phases={}
    for phase,seed in [('teach',144),('sleep',145)]:
        epochs=PROFILE[phase+'_epochs']; steps_total=math.ceil(len(training)/8)*epochs
        optimizer=AdamWNoLerp(params,lr=PROFILE[phase+'_lr'],weight_decay=0)
        rng=random.Random(seed); steps=0; tokens=0; presentations=Counter(); losses=[]; first_check=None
        started=time.perf_counter()
        for epoch in range(epochs):
            order=list(range(len(training))); rng.shuffle(order); epoch_loss=0; batches=0
            for offset in range(0,len(order),8):
                check_deadline(args.stop_at,{'run_id':args.run_id,'phase':phase,'completed_updates':steps,'presentations':sum(presentations.values()),'supervised_tokens':tokens,'training_sec':time.perf_counter()-started})
                indices=order[offset:offset+8]; ids,labels=collate(cache,indices,tok.pad_token_id,device)
                warm=min(1.,(steps+1)/max(1,int(steps_total*.05)))
                progress=steps/max(1,steps_total-1); lr=PROFILE[phase+'_lr']*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                for group in optimizer.param_groups: group['lr']=lr
                optimizer.zero_grad(set_to_none=True); output=model(input_ids=ids); loss=supervised_nll(output.logits,labels)
                value=float(loss.detach().cpu())
                if not math.isfinite(value): raise FloatingPointError('Nonfinite corrected NLL')
                count=int((labels[:,1:]!=-100).sum().detach().cpu())
                if steps==0:
                    cpu_logits=output.logits.detach().cpu().float()[:,:-1,:]; cpu_targets=labels.detach().cpu()[:,1:]
                    valid=cpu_targets!=-100
                    cpu=float(-cpu_logits[valid].log_softmax(-1).gather(1,cpu_targets[valid].unsqueeze(1)).mean())
                    # Keep an independently computed value on actual logits.
                    if abs(value-cpu)>1e-4: raise ValueError('Actual model corrected NLL disagrees with independent CPU')
                    first_check={'gpu_corrected_nll':value,'independent_cpu_nll':cpu,'supervised_tokens':count}
                loss.backward(); norm=float(torch.nn.utils.clip_grad_norm_(params,1).detach().cpu())
                if not math.isfinite(norm): raise FloatingPointError('Nonfinite input gradient')
                optimizer.step(); steps+=1; tokens+=count; presentations.update(training[i]['id'] for i in indices)
                epoch_loss+=value; batches+=1
                if steps%100==0:
                    atomic_json(run_path/'in_progress_updates.json',{'phase':phase,'completed_phase_updates':steps,'supervised_tokens':tokens})
                    print('ISOLATED',phase,'step',steps,'/',steps_total,'nll',value,flush=True)
            losses.append(epoch_loss/batches); print('ISOLATED',phase,'epoch',epoch+1,'/',epochs,'nll',losses[-1],flush=True)
        phases[phase]={'steps':steps,'presentations':sum(presentations.values()),'supervised_tokens':tokens,
                       'presentations_by_id':dict(presentations),'epochs':epochs,'seed':seed,'losses':losses,
                       'training_sec':time.perf_counter()-started,'first_batch_independent_cpu_loss_check':first_check,
                       'active_trainable_parameters':1048576,'optimizer':'AdamWNoLerp','peak_lr':PROFILE[phase+'_lr']}
        atomic_json(run_path/'in_progress_phase_budget.json',{'phases':phases})
    wrapped.activate_block(None); model.eval()
    probes=facts[:2]+suites['base_capabilities'][:2]; before=evaluate(model,tok,probes)
    pending=store.pending(1)
    torch.save({'blocks':[p.detach().cpu() for p in wrapped.blocks]},pending/'input_parameters.pt')
    atomic_json(pending/'lexical_inventory.json',lexical)
    atomic_json(pending/'inference_config.json',{'backbone':backbone,'frozen_adapter_path':parent['checkpoint'],'frozen_adapter_hashes':parent['checkpoint_hashes']})
    del params,wrapped,original,model,output,loss,optimizer; gc.collect()
    model,tok=restore(device,pending); after_reload=evaluate(model,tok,probes)
    final=full(model,tok)
    old_keys=[key for key in suites if key not in ('exact','paraphrase')]
    old_parity=all(baseline[key]==final[key] for key in old_keys)
    recall={key:sum(exact_score(r['response'],r['expected']) for r in final[key])/len(final[key]) for key in ('exact','paraphrase')}
    gate={'passed':old_parity and all(v>=.95 for v in recall.values()),'all868_old_responses_exactly_unchanged':old_parity,
          'new512_accuracy_each_wording':recall,'overall_goal_achieved':False}
    history=[{'cycle':1,'facts_taught':512,'novel_fact_count':512,'phases':phases,'raw':final,'joint_gate':gate}]
    cp=store.commit(pending,{'cycle':1,'history':history,'reload_parity':{'passed':before==after_reload,'before':before,'after_reload':after_reload}})
    from cumulative_memory_scale_v1 import summarize
    scale=summarize([{'name':'isolated_first512','facts':facts,'baseline_canonical':baseline['exact'],'baseline_withheld':baseline['paraphrase']}],
                    final['exact'],final['paraphrase'],tok,8192)
    result={'status':'complete','manifest':manifest,'history':history,'final_raw':final,'joint_gate':gate,'checkpoint':str(cp),
            'checkpoint_hashes':{n:sha(cp/n) for n in ('input_parameters.pt','lexical_inventory.json','inference_config.json')},
            'cumulative_dual_wording_acquisition':scale,'actual_updates':sum(b['steps'] for b in phases.values()),
            'actual_presentations':sum(b['presentations'] for b in phases.values()),
            'actual_supervised_tokens':sum(b['supervised_tokens'] for b in phases.values()),
            'actual_training_sec':sum(b['training_sec'] for b in phases.values()),'novel_fact_count':512,
            'full_saved_reload_final_queries':1892,'actual_original_ability_baseline_repeats_maintenance':True,
            'overall_goal_achieved':False,'interpretation':'New512 associations on fresh label-blind neural input vectors, previous other-branch1024 weights NOT inherited. More allocated input parameters/changed lexical representation, not fixed-rank compressed learning. Skills inherited from maintenance, no new algorithm claim. Whole model fresh reconstruction before final1892 answers. Per-segment dual-form baseline and actual new-tokenizer compact acquisition govern scale;5120 corpus under this tokenizer is only2.734 contexts, so no four-context claim. Sleep causality/independent final confirmation/replication pending.'}
    atomic_json(run_path/'results.json',result); print('ISOLATED PILOT COMPLETE',recall,'gate',gate['passed'],flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True)
    ap.add_argument('--stop-at',default='2026-10-05T14:30:00+08:00'); args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id: raise ValueError('Simple ID required')
    target=DATA/args.run_id
    with gpu_lease(DATA,'isolated_memory_pilot_v2'):
        if target.exists(): raise FileExistsError('Fresh run ID required; no partial probe or training reuse')
        target.mkdir()
        try: run(args,target)
        except BaseException as exc:
            atomic_json(target/'failure.json',{'status':'stopped_at_deadline' if isinstance(exc,DeadlineReached) else 'failed',
                        'error':repr(exc),'deadline_progress':getattr(exc,'progress',None),'partial_training_must_not_be_reused':True})
            raise


if __name__=='__main__': main()
