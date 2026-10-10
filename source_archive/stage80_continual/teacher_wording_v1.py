"""Training-only alternate question wording, independent of withheld audit text.

This expands teaching FORMS, never distinct semantic facts or inference memory.
No inference path imports this module. Any use requires a fresh protocol/run-id.
"""
from storage import digest


def expand(facts):
    if not facts or len({r['id'] for r in facts})!=len(facts):
        raise ValueError('Nonempty unique semantic facts required')
    if len({r['stimulus'] for r in facts})!=len(facts):
        raise ValueError('Repeated entity association')
    if any(r['type']!='fact' for r in facts): raise ValueError('Memory facts only')
    result=[]
    for row in facts:
        canonical={**row,'semantic_id':row['id'],'teaching_form':0,'teacher_only':True}
        # This template is independent of audit_q and of the gold color.
        question=f"Name the assigned color for {row['stimulus']}."
        if question in (row['train_q'],row['eval_q'],row['audit_q']):
            raise ValueError('Alternative teacher form must remain distinct')
        alternate={**row,'id':row['id']+'_teacher_form1','semantic_id':row['id'],
                   'train_q':question,'eval_q':question,'teaching_form':1,'teacher_only':True}
        result.extend([canonical,alternate])
    if len({r['id'] for r in result})!=len(result): raise ValueError('Teaching form IDs collide')
    return result


def verify(original,variants,heldout_questions):
    if variants!=expand(original): raise ValueError('Teaching variants changed or copied audit wording')
    if {r['train_q'] for r in variants}&set(heldout_questions):
        raise ValueError('Teacher form overlaps held-out evaluation')
    return {'semantic_associations':len(original),'teaching_form_rows':len(variants),
            'original_semantic_dataset_hash':digest(original),'variant_dataset_hash':digest(variants),
            'withheld_wording_used_as_teacher':False,'inference_access_allowed':False,
            'semantic_count_must_not_include_form_duplicates':True}
