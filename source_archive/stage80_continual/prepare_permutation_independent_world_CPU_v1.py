"""New-world/seed replication candidate; no reserved reads, GPU or inference."""
import collections
import datetime
import random
from pathlib import Path
from queue_matched_memory_v1 import HERE,DATA,read,sha
from storage import atomic_json,digest

SEED=20261006+2939
NAME='qezorvank'
ORDER=(2,0,3,1)
TRAIN_WORDS='''piano violin guitar trumpet flute cello harp drum
tiger lion zebra camel rabbit mouse eagle owl
shirt jacket sock shoe hat scarf glove belt
paper pencil eraser ruler book notebook envelope stamp
train bus truck bicycle airplane helicopter subway ferry
rain snow hail fog wind storm thunder cloud
oak pine maple poplar fir birch ash elm
keyboard monitor printer camera radio phone tablet cable'''.split()
TRANSFER_WORDS='''beaver weasel mongoose meerkat compass anchor turbine piston
slate basalt limestone sandstone parsley basil thyme mint'''.split()


def row(group,index,values):
    q=NAME+': '+' '.join(values)
    return {'id':f'permutation_independent_world_v1_{group}_{index}','type':'four_slot_permutation_skill',
            'skill':NAME,'train_q':q,'eval_q':q,'answer':' '.join(values[i] for i in ORDER)}


def main():
    target=DATA/'permutation_independent_world_seed20263945_CPU_v1.json'
    if target.exists():raise FileExistsError('Immutable unqueried candidate already exists')
    from run_V8_later_isolated_stage_v1 import verify_prepared
    verify_prepared()
    prior_paths=[DATA/'permutation_skill_teaching_seed20261005_v7.json',DATA/'permutation_broad_vocabulary_teaching_seed20261005_v8.json']
    prior_words=set()
    for path in prior_paths:
        prior=read(path);prior_words.update(prior['train_vocabulary']);prior_words.update(prior['transfer_vocabulary'])
    if len(set(TRAIN_WORDS))!=64 or len(set(TRANSFER_WORDS))!=16 or set(TRAIN_WORDS)&set(TRANSFER_WORDS) or prior_words&(set(TRAIN_WORDS)|set(TRANSFER_WORDS)):
        raise ValueError('Independent candidate vocabulary overlaps existing development domains')
    rng=random.Random(SEED);seen=set();groups={'train':[]}
    for _ in range(8):
        while True:
            words=TRAIN_WORDS.copy();rng.shuffle(words);shifts=rng.sample(range(64),4)
            tuples=[tuple(words[(i+s)%64] for s in shifts) for i in range(64)]
            if len(set(tuples))==64 and not set(tuples)&seen:break
        for values in tuples:
            seen.add(values);groups['train'].append(row('train',len(groups['train']),values))
    for name,n,vocabulary in [('dev',64,TRAIN_WORDS),('audit',192,TRAIN_WORDS),('transfer_audit',192,TRANSFER_WORDS)]:
        groups[name]=[]
        while len(groups[name])<n:
            values=tuple(rng.sample(vocabulary,4))
            if values in seen:continue
            seen.add(values);groups[name].append(row(name,len(groups[name]),values))
    from permutation_broad_skill_protocol_v8 import validate_development_groups
    validate_development_groups(groups,NAME)
    if len(seen)!=960:raise ValueError('Repeated tuples/split overlap')
    source=read(DATA/'isolated_scale10240_s42_v2/results.json')
    from conditional_skill_protocol_v5 import qualify_source
    qualify_source(source,read(DATA/'isolated_scale10240_s42_v2/independent_isolated_scale10240_audit.json'),sha(DATA/'isolated_scale10240_s42_v2/results.json'))
    from model_support import load_tokenizer,format_chat
    from isolated_inputs_v2 import add_input_names
    from nontruncating_chat_cache_v1 import encode_items
    lexical=read(Path(source['checkpoint'])/'lexical_inventory.json')
    old=load_tokenizer();add_input_names(old,lexical['names'],49152)
    tok=load_tokenizer();add_input_names(tok,lexical['names']+[NAME],49152)
    questions=[r['question'] for rr in source['final_raw'].values() for r in rr]
    chats=[format_chat(old,q) for q in questions]
    if old(chats,add_special_tokens=False)['input_ids']!=tok(chats,add_special_tokens=False)['input_ids']:raise ValueError('Changed prior old question tokenization')
    cache=encode_items(tok,groups['train'],64,8192)
    tokens=maximum=0
    for r,(ids,labels) in zip(groups['train'],cache):
        full=tok(format_chat(tok,r['train_q'],r['answer']),add_special_tokens=False)['input_ids']
        prompt=tok(format_chat(tok,r['train_q']),add_special_tokens=False)['input_ids']
        if ids.tolist()!=full or labels.tolist()!=[-100]*len(prompt)+full[len(prompt):] or len(full)>64 or any(t>=49152 for t in full[len(prompt):]):raise ValueError('Wrong complete teacher mask/output vocabulary')
        maximum=max(maximum,len(full));tokens+=len(full)-len(prompt)
    artifact={'status':'unqueried_new_world_CPU_replication_candidate_NOT_admitted_or_trained',
        'recorded_at':datetime.datetime.now().astimezone().isoformat(),'seed':SEED,'operation_name':NAME,
        'groups':groups,'group_hashes':{k:digest(v) for k,v in groups.items()},
        'train_vocabulary':TRAIN_WORDS,'transfer_vocabulary':TRANSFER_WORDS,
        'teacher_rule_training_and_scoring_only':{'zero_based_output_position_order':list(ORDER)},
        'all80_words_disjoint_from_prior_V7_V8_development_vocabulary':True,'all960_tuples_unique_no_split_overlap':True,
        'each_teacher_word_presentations_per_position_per_epoch':8,'teacher_masks_checked':512,
        'teacher_max_sequence_tokens':maximum,'teacher_answer_supervised_tokens_per_epoch':tokens,
        'prospective_updates':1920,'prospective_presentations':15360,'prospective_supervised_tokens':tokens*30,
        'same_epochs_batch_optimizer_but_new_vocabulary_token_budget_NOT_pure_compute_matched_ablation':True,
        'all21348_prior_questions_tokenization_preserved':True,
        'proposed_source_must_be_original_qualified_scale_NOT_trained_V8_skill_weights':True,
        'requires_new_data_aware_versioned_runner_auditor_preparation_and_actual_runtime_admission':True,
        'current_V8_immutable_production_cannot_be_overridden_for_this_file':True,
        'actual_new_skill_BEFORE_after_and_later_retention_NOT_measured':True,
        'candidate_preregistered_after_development_NOT_algorithm_blind_design':True,
        'existing_reserved_question_files_NOT_read':True,'failed_answers_NOT_recycled_into_teachers':True,
        'production_GPU_model_answers_or_training_updates':0,'no_automatic_GPU_dispatch':True,
        'source_results_sha256':sha(DATA/'isolated_scale10240_s42_v2/results.json'),
        'prior_development_vocabulary_files_sha256':{p.name:sha(p) for p in prior_paths},
        'generator_code_sha256':sha(Path(__file__)),
        'dependencies_sha256':{n:sha(HERE/n) for n in ['model_support.py','nontruncating_chat_cache_v1.py','isolated_inputs_v2.py','storage.py','permutation_broad_skill_protocol_v8.py']},
        'overall_goal_achieved':False}
    atomic_json(target,artifact)
    print({'candidate':str(target),'seed':SEED,'teachers':512,'max_tokens':maximum,'supervised_tokens_per_epoch':tokens,'GPU_dispatch':False})


if __name__=='__main__':main()
