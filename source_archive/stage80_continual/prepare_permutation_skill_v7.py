"""CPU pinning and independent mask/budget preflight; no reserved reads/model forwards."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from storage import atomic_json,digest
from permutation_skill_protocol_v7 import PROTOCOL,PROFILE,CODE_NAMES,qualify_source,validate_development_groups
HERE=Path(__file__).resolve().parent;DATA=HERE.parents[1]/'data/stage80_continual'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target=DATA/'prepared_permutation_skill_v7.json'
    if target.exists():raise FileExistsError('Immutable preparation')
    for n in CODE_NAMES:
        p=HERE/n;compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    source=DATA/'isolated_scale10240_s42_v2';result=read(source/'results.json')
    qualification=qualify_source(result,read(source/'independent_isolated_scale10240_audit.json'),sha(source/'results.json'))
    data_path=DATA/'permutation_skill_teaching_seed20261005_v7.json';data=read(data_path)
    groups={k:data['groups'][k] for k in ('train','dev','audit','transfer_audit')};validate_development_groups(groups,data['operation_name'])
    from model_support import load_tokenizer,format_chat
    from isolated_inputs_v2 import add_input_names
    from cached_batches import encode_items
    lexical=read(Path(result['checkpoint'])/'lexical_inventory.json')
    old_tok=load_tokenizer();add_input_names(old_tok,lexical['names'],49152)
    tok=load_tokenizer();add_input_names(tok,lexical['names']+[data['operation_name']],49152)
    questions=[r['question'] for rows in result['final_raw'].values() for r in rows]
    questions += [r['eval_q'] for k in ('dev','audit','transfer_audit') for r in groups[k]]
    old_ids=old_tok([format_chat(old_tok,q) for q in questions[:21348]],add_special_tokens=False)['input_ids']
    new_ids=tok([format_chat(tok,q) for q in questions[:21348]],add_special_tokens=False)['input_ids']
    assert old_ids==new_ids
    cache=encode_items(tok,groups['train']);tokens=0;maximum=0
    for row,(ids,labels) in zip(groups['train'],cache):
        full=tok(format_chat(tok,row['train_q'],row['answer']),add_special_tokens=False)['input_ids']
        prompt=tok(format_chat(tok,row['train_q']),add_special_tokens=False)['input_ids']
        n=int((labels[1:]!=-100).sum());assert ids.tolist()==full and n==len(full)-len(prompt)>0
        assert len(full)<=64 and all(v<49152 for v in full[len(prompt):])
        tokens+=n;maximum=max(maximum,len(full))
    assert tokens>0
    done=subprocess.run([sys.executable,str(HERE/'test_permutation_skill_protocol_v7.py')],capture_output=True,text=True)
    if done.returncode:raise RuntimeError(done.stdout+done.stderr)
    numeric=read(DATA/'directml_supervised_nll_probe_v3.json');assert numeric['actual_vocabulary_loss_AND_gradient_validated']
    for n,h in numeric['code_hashes'].items():assert sha(HERE/n)==h
    pinned_data=['permutation_skill_teaching_seed20261005_v7.json','directml_supervised_nll_probe_v3.json',
                 'isolated_scale10240_s42_v2/results.json',
                 'isolated_scale10240_s42_v2/independent_isolated_scale10240_audit.json']
    out={'status':'prepared_cpu_only','protocol':PROTOCOL,'profile':PROFILE,'qualified_source':qualification,
         'source_results_sha256':sha(source/'results.json'),'source_audit_sha256':sha(source/'independent_isolated_scale10240_audit.json'),
         'code_hashes':{n:sha(HERE/n) for n in CODE_NAMES},'data_hashes':{n:sha(DATA/n) for n in pinned_data},
         'external_code_hashes':{'stage72_gpu/directml_patch.py':sha(HERE.parent/'stage72_gpu/directml_patch.py')},
         'teacher_supervised_tokens_per_epoch':tokens,'maximum_teacher_sequence_tokens':maximum,
         'planned_updates':960,'planned_presentations':7680,'planned_supervised_tokens':tokens*30,
         'actual_old_chat_tokenizations_checked':len(old_ids),'unit_test_count':5,'unit_test_output':done.stderr+done.stdout,
         'actual_full_before_and_after_required':21796,'reserved_read':False,'model_forwards_or_updates':0,
         'overall_goal_achieved':False}
    atomic_json(target,out);print(json.dumps({k:v for k,v in out.items() if k not in ('code_hashes','data_hashes','unit_test_output')},indent=2))

if __name__=='__main__':main()
