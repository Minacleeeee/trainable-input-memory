"""Count distinct baseline-failed associations still recalled under both forms.

Evaluation-only: tokenization/scoring never supplies an inference response.
Reject overlapping segments, changed labels/questions and incomplete responses.
"""
from memory_dataset import compact_knowledge
from storage import digest,exact_score


def summarize(segments,final_canonical,final_withheld,tokenizer,context_capacity):
    if not segments or context_capacity<1: raise ValueError('Nonempty segments and context required')
    def indexed(rows):
        values={r['id']:r for r in rows}
        if len(values)!=len(rows): raise ValueError('Duplicate response identities')
        return values
    canonical=indexed(final_canonical); withheld=indexed(final_withheld)
    all_items=[]; acquired=[]; seen=set(); reports=[]
    for segment in segments:
        facts=segment['facts']; ids={r['id'] for r in facts}
        if not facts or len(ids)!=len(facts) or ids&seen:
            raise ValueError('Novel segments must have disjoint unique identities')
        seen.update(ids)
        baseline_c=indexed(segment['baseline_canonical'])
        baseline_p=indexed(segment['baseline_withheld'])
        if set(baseline_c)!=ids or set(baseline_p)!=ids:
            raise ValueError('Each segment needs its own complete pre-introduction baseline')
        before_any=set(); after_both=set()
        for fact in facts:
            key=fact['id']
            if key not in canonical or key not in withheld:
                raise ValueError('Missing final canonical or withheld generation')
            for row,query_key in ((baseline_c[key],'eval_q'),(canonical[key],'eval_q'),
                                  (baseline_p[key],'audit_q'),(withheld[key],'audit_q')):
                if row['expected']!=fact['answer'] or row['question']!=fact[query_key]:
                    raise ValueError('Changed label/question in cumulative comparison')
            if exact_score(baseline_c[key]['response'],fact['answer']) or exact_score(baseline_p[key]['response'],fact['answer']):
                before_any.add(key)
            if exact_score(canonical[key]['response'],fact['answer']) and exact_score(withheld[key]['response'],fact['answer']):
                after_both.add(key)
        new_ids=after_both-before_any
        selected=[r for r in facts if r['id'] in new_ids]
        all_items+=facts; acquired+=selected
        reports.append({'segment_name':segment['name'],'unique_associations':len(facts),
                        'baseline_correct_either':len(before_any),'baseline_incorrect_both':len(facts)-len(before_any),
                        'final_recalled_both':len(after_both),'acquired_and_final_recalled_both':len(selected),
                        'fact_hash':digest(facts),'baseline_canonical_hash':digest(segment['baseline_canonical']),
                        'baseline_withheld_hash':digest(segment['baseline_withheld'])})
    if len({r['stimulus'] for r in all_items})!=len(all_items):
        raise ValueError('Nonce entities repeat across purportedly novel segments')
    compact=lambda rows:len(tokenizer(compact_knowledge(rows),add_special_tokens=False)['input_ids']) if rows else 0
    acquired_tokens=compact(acquired)
    return {'context_capacity':context_capacity,'segments':reports,
            'taught_distinct_associations':len(all_items),'taught_compact_tokens':compact(all_items),
            'acquired_and_final_recalled_both':len(acquired),'compact_acquired_tokens':acquired_tokens,
            'compact_acquired_context_ratio':acquired_tokens/context_capacity,
            'acquired_four_contexts':acquired_tokens>=4*context_capacity,
            'acquired_ids':[r['id'] for r in acquired],
            'interpretation':'Cumulative distinct novel associations, initially wrong under BOTH measured forms and still correct under BOTH at the final checkpoint. Excludes old128, repeat epochs, replay and intermediate successes subsequently lost. Extra final known items are not counted. Tokenized compact representation size is empirical, not an information-theoretic bound. This scale flag alone is not joint-stage success.'}
