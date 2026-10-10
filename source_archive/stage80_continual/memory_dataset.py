"""Scale arbitrary factual mappings without putting answers in inference context."""
import random
from wordgen import generate_distinct_words

LABELS=['red','blue','green','yellow','purple','orange','pink','brown',
        'gray','black','white','amber','crimson','violet','emerald','jade']


def generate(seed=42,n=4096,cohort_size=32):
    if n<1 or cohort_size<1:
        raise ValueError('Positive dataset sizes required')
    words=generate_distinct_words(n,random.Random(seed+11))
    labels=[LABELS[i%len(LABELS)] for i in range(n)]
    random.Random(seed+13).shuffle(labels)
    items=[{'id':f'mem_{i}','type':'fact','cohort':i//cohort_size+1,
            'stimulus':word,'train_q':f'What color is a {word}?',
            'eval_q':f'What color is a {word}?','answer':label,
            'audit_q':f'Tell me the color of a {word}.'}
           for i,(word,label) in enumerate(zip(words,labels))]
    assert len({it['stimulus'] for it in items})==n
    return items


def compact_knowledge(items):
    # A conservative representation, excluding chat framing and repeats.
    return '\n'.join(it['stimulus']+':'+it['answer'] for it in items)
