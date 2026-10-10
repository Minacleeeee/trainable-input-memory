"""Development-only four-slot command gates; never imported by neural inference."""
from conditional_skill_protocol_v5 import qualify_source
from shared_skill_module_v1 import CONFIG

PROTOCOL = 'stage80_fresh_four_slot_permutation_input_v7'
PROFILE = {'teach_epochs':20,'sleep_epochs':10,'batch':8,'teacher_rows':256,
           'teach_name_lr':.003,'sleep_name_lr':.001,
           'teach_shared_lr':.0003,'sleep_shared_lr':.0001,
           'shared_module_config':CONFIG,'old_training_replay_presentations':0,
           'loss':'explicit2D_masked_answer_mean_nll_v3',
           'minimum_audit_and_transfer_accuracy':.8,
           'minimum_absolute_gain_on_each_audit':.2,
           'initial_old_full_physical_generation_required':True,
           'primary_old_raw_exact_gate_unchanged':True,
           'allowed_arm':'input','numeric_branch_skill_NOT_established':True}
CODE_NAMES = ['permutation_skill_protocol_v7.py','run_permutation_skill_v7.py',
              'audit_permutation_skill_v7.py','prepare_permutation_skill_v7.py',
              'test_permutation_skill_protocol_v7.py','prepare_permutation_skill_data_v7.py',
              'shared_skill_module_v1.py','shared_skill_training_v1.py',
              'conditional_skill_protocol_v5.py','scale_protocol_v1.py',
              'isolated_inference_audit_v1.py','run_isolated_memory_pilot_v2.py',
              'isolated_training_v1.py','isolated_inputs_v2.py','isolated_embeddings_v1.py',
              'isolated_generation_batch_v1.py','model_support.py','engine.py','storage.py',
              'cached_batches.py','supervised_nll_v3.py','adamw_no_lerp.py','gpu_lease.py','deadline.py']


def validate_development_groups(groups,name):
    required={'train':256,'dev':64,'audit':192,'transfer_audit':192}
    if set(groups)!=set(required) or name!='qezorvank':
        raise ValueError('Exact development-only task required')
    ids=set();tuples=set();vocab={}
    for key,n in required.items():
        if len(groups[key])!=n:raise ValueError('Incomplete split')
        vocab[key]=set()
        for row in groups[key]:
            prefix=name+': '
            if not row['eval_q'].startswith(prefix):raise ValueError('Wrong command')
            values=tuple(row['eval_q'][len(prefix):].split())
            if (len(values)!=4 or len(set(values))!=4 or values in tuples or row['id'] in ids
                    or row['skill']!=name or row['type']!='four_slot_permutation_skill'
                    or row['train_q']!=row['eval_q']
                    or row['answer']!=' '.join(values[i] for i in (2,0,3,1))):
                raise ValueError('Tuple/target/identity mismatch or inference rule leakage')
            ids.add(row['id']);tuples.add(values);vocab[key].update(values)
    if (len(vocab['train'])!=16 or len(vocab['transfer_audit'])!=16
            or vocab['train']&vocab['transfer_audit']
            or not vocab['audit']<=vocab['train'] or not vocab['dev']<=vocab['train']):
        raise ValueError('Unseen vocabulary split contaminated')
    return True


def gate_from_raw(groups,baseline,final,old_raw_equal,score):
    """Actual pre-teaching versus fresh restored replies; no generated replacement."""
    if set(baseline)!={'dev','audit','transfer_audit'} or set(final)!=set(baseline):
        raise ValueError('Complete actual baseline and final suites required')
    measurements={}
    for key,before in baseline.items():
        after=final[key];items=groups[key]
        if len(before)!=len(items) or len(after)!=len(items):raise ValueError('Incomplete raw suite')
        for item,b,a in zip(items,before,after):
            for row in (b,a):
                if (row['id']!=item['id'] or row['type']!=item['type']
                        or row['question']!=item['eval_q'] or row['expected']!=item['answer']
                        or row['correct']!=score(row['response'],row['expected'])):
                    raise ValueError('Changed raw identity or score')
        bcount=sum(score(r['response'],r['expected']) for r in before)
        acount=sum(score(r['response'],r['expected']) for r in after)
        newly=sum(not score(b['response'],b['expected']) and score(a['response'],a['expected']) for b,a in zip(before,after))
        lost=sum(score(b['response'],b['expected']) and not score(a['response'],a['expected']) for b,a in zip(before,after))
        measurements[key]={'questions':len(items),'before_correct':bcount,'after_correct':acount,
            'after_accuracy':acount/len(items),'absolute_gain':(acount-bcount)/len(items),
            'newly_correct':newly,'initially_correct_lost':lost}
    passed=bool(old_raw_equal) and all(measurements[k]['after_accuracy']>=.8
        and measurements[k]['absolute_gain']>=.2 for k in ('audit','transfer_audit'))
    return {'passed':passed,'all21348_old_raw_responses_exactly_unchanged':bool(old_raw_equal),
            'new_task_measurements':measurements,'numeric_branch_selection_NOT_tested':True,
            'later_new_fact_learning_retention_final_blind_audit_and_replication_pending':True,
            'overall_goal_achieved':False}
