"""Pin admission/gradient/scoring contracts and complete prospective teaching masks."""
import datetime
import json
import subprocess
import sys
from pathlib import Path
from storage import atomic_json,digest
from queue_matched_memory_v1 import HERE,ROOT,DATA,read,sha
from queue_broad_skill_v8_recovery_v2 import pins


def main():
    target=DATA/'prepared_V8_later_learning_contract_CPU_v1.json'
    if target.exists():raise FileExistsError('Fresh immutable CPU preparation required')
    pins(read(DATA/'prepared_broad_skill_v8_recovery_queue_v2.json'))
    test=subprocess.run([sys.executable,str(HERE/'test_v8_later_learning_contract_v1.py')],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    from model_support import load_tokenizer,format_chat
    from isolated_inputs_v2 import add_input_names
    from post_conditional_retention_protocol_v5 import validate_curriculum
    from teacher_wording_v1 import expand
    from nontruncating_chat_cache_v1 import encode_items
    from v8_later_learning_contract_v1 import PROFILE
    world_path=DATA/'isolated_memory10240_seed20261005_v1.json';world=read(world_path)
    curriculum_path=DATA/'post_conditional_memory512_seed20261005_v1.json';curriculum=read(curriculum_path)
    skill_path=DATA/'permutation_broad_vocabulary_teaching_seed20261005_v8.json';skill=read(skill_path)
    rows=validate_curriculum(curriculum,world,skill['operation_name'])
    source_names=[r['stimulus'] for r in world['novel_items']]+[skill['operation_name']]
    before=load_tokenizer();add_input_names(before,source_names,49152)
    tok=load_tokenizer();lexical=add_input_names(tok,source_names+[r['stimulus'] for r in rows],49152)
    if tok(skill['operation_name'],add_special_tokens=False)['input_ids']!=[59392]:raise ValueError('Operation ID shifted')
    if tok([r['stimulus'] for r in rows],add_special_tokens=False)['input_ids']!=[[59393+i] for i in range(512)]:raise ValueError('Later names shifted')
    scale=read(DATA/'isolated_scale10240_s42_v2/results.json')
    questions=[r['question'] for rr in scale['final_raw'].values() for r in rr]
    questions+=[r['eval_q'] for k in ('dev','audit','transfer_audit') for r in skill['groups'][k]]
    if len(questions)!=21796 or before(questions,add_special_tokens=False)['input_ids']!=tok(questions,add_special_tokens=False)['input_ids']:
        raise ValueError('Later lexical expansion changed any prior memory/skill tokenization')
    forms=expand(rows);cache=encode_items(tok,forms,64,8192)
    if len(forms)!=1024 or len(cache)!=1024:raise ValueError('Complete teacher masks required')
    total=0;maximum=0
    for item,encoded in zip(forms,cache):
        full=tok(format_chat(tok,item['train_q'],item['answer']),add_special_tokens=False)['input_ids']
        prompt=tok(format_chat(tok,item['train_q']),add_special_tokens=False)['input_ids']
        if encoded[0].tolist()!=full or len(full)>64 or len(prompt)>=len(full):
            raise ValueError('Truncated teaching row')
        labels=encoded[1].tolist()
        if labels!=[-100]*len(prompt)+full[len(prompt):] or any(x>=49152 for x in full[len(prompt):]):
            raise ValueError('Wrong supervision mask or resized output vocabulary')
        total+=len(full)-len(prompt);maximum=max(maximum,len(full))
    if {r['train_q'] for r in forms}&{r['audit_q'] for r in rows}:raise ValueError('Withheld wording leaked into teachers')
    code_names=['v8_later_learning_contract_v1.py','test_v8_later_learning_contract_v1.py',Path(__file__).name,
                'post_conditional_retention_protocol_v5.py','teacher_wording_v1.py','shared_skill_module_v1.py',
                'permutation_broad_skill_protocol_v8.py','audit_permutation_broad_skill_v8.py',
                'cached_batches.py','nontruncating_chat_cache_v1.py','isolated_inputs_v2.py','model_support.py','storage.py']
    artifact={'status':'CPU_contract_masks_and_gradient_fixtures_prepared_NOT_production_ready',
        'recorded_at':datetime.datetime.now().astimezone().isoformat(),'stop_at':read(DATA/'second_window_state_v1.json')['stop_at'],
        'profile':PROFILE,'code_hashes':{n:sha(HERE/n) for n in code_names},
        'data_hashes':{str(p.relative_to(DATA)):sha(p) for p in (world_path,curriculum_path,skill_path,DATA/'prepared_permutation_broad_skill_v8.json')},
        'ten_CPU_admission_gradient_retention_scoring_tests_passed':True,'actual_test_output':test.stdout+test.stderr,
        'prior_questions_tokenization_checked':len(questions),'new_fact_rows':len(rows),'teacher_forms':len(forms),
        'new_fact_dataset_sha256':digest(rows),'teacher_forms_sha256':digest(forms),'new_lexical_rows':lexical['new_input_rows'],
        'teacher_max_sequence_tokens':maximum,'teacher_answer_supervised_tokens_per_epoch':total,
        'prospective_actual_budgets':{'updates':3840,'presentations':30720,'supervised_tokens':30*total},
        'new512_bank_tensor_bytes':4194304,'inherited_source_memory_skill_backbone_lexical_training_archives_additional':True,
        'new_curriculum_is_preexisting_development_data_NOT_independent_confirmation':True,
        'qualified_source_selected':None,'actual_source_restore_or_GPU_preflight_or_production_teaching':False,
        'production_runner_independent_final_auditor_and_admission_probe_still_required':True,
        'shared_plasticity_matched_replay_schedules_NOT_implemented':True,
        'production_model_calls_GPU_calls_updates_or_reserved_reads':0,'overall_goal_achieved':False}
    pins(read(DATA/'prepared_broad_skill_v8_recovery_queue_v2.json'))
    atomic_json(target,artifact)
    print(json.dumps({k:artifact[k] for k in ('status','prior_questions_tokenization_checked','teacher_forms','teacher_max_sequence_tokens','teacher_answer_supervised_tokens_per_epoch','prospective_actual_budgets')}))


if __name__=='__main__':main()
