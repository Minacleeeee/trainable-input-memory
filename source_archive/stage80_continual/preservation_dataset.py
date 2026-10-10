"""General teaching anchors; held-out audit questions/arguments never trained.

This is supervised rehearsal of representative capabilities, not an inference
router or proof that all original knowledge is retained.
"""
from base_audit import generate as base_audit


def generate():
    items=[]
    def add(kind, question, answer):
        items.append({'id':f'anchor_{kind}_{len(items)}','type':kind,
                      'train_q':question,'eval_q':question,'answer':answer})
    for i in range(32):
        a,b=50+i,51+(i*11)%43
        add('arithmetic',f'What is {a} plus {b}? Reply with only the number.',str(a+b))
    capitals=[('Canada','ottawa'),('Portugal','lisbon'),('Greece','athens'),
              ('Norway','oslo'),('Sweden','stockholm'),('Finland','helsinki'),
              ('Denmark','copenhagen'),('Poland','warsaw'),('Austria','vienna'),
              ('Hungary','budapest'),('Romania','bucharest'),('Egypt','cairo'),
              ('Thailand','bangkok'),('Vietnam','hanoi'),('South Korea','seoul'),
              ('Netherlands','amsterdam')]
    for country,capital in capitals:
        add('factual',f'What is the capital of {country}? Reply with one word.',capital)
        add('factual',f'Name the capital city of {country}. Give only the city name.',capital)
    nouns=[('sofa','sofas'),('table','tables'),('school','schools'),('shop','shops'),
           ('garden','gardens'),('river','rivers'),('lake','lakes'),('mountain','mountains'),
           ('ocean','oceans'),('planet','planets'),('train','trains'),('truck','trucks'),
           ('boat','boats'),('bottle','bottles'),('plate','plates'),('spoon','spoons'),
           ('fork','forks'),('pencil','pencils'),('letter','letters'),('song','songs'),
           ('picture','pictures'),('camera','cameras'),('phone','phones'),('computer','computers'),
           ('man','men'),('foot','feet'),('goose','geese'),('person','people'),
           ('knife','knives'),('leaf','leaves'),('wolf','wolves'),('berry','berries')]
    for word,plural in nouns:
        add('language',f'Give the English plural of {word}. Reply with only that word.',plural)
    words=['sun','ocean','sand','leaf','stone','music','paper','glass','silver','gold',
           'copper','orange','purple','violet','happy','sad','warm','cold','near','far',
           'left','right','up','down','summer','winter','spring','autumn','morning',
           'evening','today','tomorrow']
    for word in words:
        add('instruction',f'Reply with exactly the word {word} and nothing else.',word)
    assert len(items)==128
    audit={it['eval_q'] for it in base_audit()}
    assert len({it['train_q'] for it in items})==len(items)
    assert not audit.intersection(it['train_q'] for it in items)
    return items
