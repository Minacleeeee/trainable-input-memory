"""Additional capability questions frozen before repair results are available.

Never imported by a training runner. Covers two newly probed task families
(subtraction and comparison) alongside pluralization and exact instructions.
This complements, rather than replaces, the original factual/base audit.
"""
import hashlib
from pathlib import Path
from base_audit import generate as old_probes
from preservation_dataset import generate as anchors
from storage import atomic_json,digest


def generate():
    rows=[]
    def add(kind,q,a):
        rows.append({'id':f'confirmation_{kind}_{len(rows)}','type':kind,'eval_q':q,'answer':a})
    for i in range(20):
        a,b=28+i,21+(i*3)%28
        add('arithmetic',f'What is {a} plus {b}? Reply with only the number.',str(a+b))
        a,b=61+i,12+(i*5)%32
        add('arithmetic',f'What is {a} minus {b}? Reply with only the number.',str(a-b))
    for i in range(40):
        a,b=100+i*3,90+(i*11)%67
        if a==b:b+=1
        larger=i%2==0
        word='larger' if larger else 'smaller'
        add('comparison',f'Which number is {word}, {a} or {b}? Reply with only that number.',
            str(max(a,b) if larger else min(a,b)))
    nouns=[('beach','beaches'),('peach','peaches'),('brush','brushes'),('dish','dishes'),
           ('match','matches'),('watch','watches'),('kiss','kisses'),('class','classes'),
           ('city','cities'),('baby','babies'),('lady','ladies'),('story','stories'),
           ('family','families'),('puppy','puppies'),('candy','candies'),('country','countries'),
           ('toy','toys'),('boy','boys'),('day','days'),('key','keys'),('monkey','monkeys'),
           ('valley','valleys'),('potato','potatoes'),('tomato','tomatoes'),('hero','heroes'),
           ('echo','echoes'),('life','lives'),('wife','wives'),('shelf','shelves'),('calf','calves'),
           ('half','halves'),('loaf','loaves'),('thief','thieves'),('deer','deer'),('sheep','sheep'),
           ('fish','fish'),('species','species'),('aircraft','aircraft'),('criterion','criteria'),('ox','oxen')]
    for noun,plural in nouns:
        add('language',f'Give the English plural of {noun}. Reply with only that word.',plural)
    words=['canyon','forest','desert','island','harbor','bridge','tunnel','village','castle','palace',
           'market','museum','library','theater','stadium','airport','station','office','factory','hospital',
           'lunch','dinner','breakfast','supper','coffee','juice','bread','butter','cheese','rice',
           'wheat','corn','pepper','salt','sugar','honey','lemon','melon','grape','cherry']
    for word in words:
        add('instruction',f'Reply with exactly the word {word} and nothing else.',word)
    assert len(rows)==len({r['eval_q'] for r in rows})==160
    excluded={r['eval_q'] for r in old_probes()+anchors()}
    assert not excluded.intersection(r['eval_q'] for r in rows)
    return rows


if __name__=='__main__':
    target=Path(__file__).resolve().parents[2]/'data/stage80_continual/confirmation_probes_v1.json'
    if target.exists():
        raise FileExistsError('Frozen confirmation questions must not be overwritten')
    rows=generate()
    atomic_json(target,{'protocol':'post-repair-independent-confirmation-v1',
                        'declared_before_repair_results':True,'training_forbidden':True,
                        'items':rows,'dataset_hash':digest(rows),
                        'generator_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'interpretation':'Pair pristine and trained answers; report category/per-item retention. No comprehensive intelligence claim.'})
    print('Frozen independent confirmation questions:',len(rows),digest(rows))
