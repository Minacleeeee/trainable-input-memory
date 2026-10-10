"""Independent base-capability probes, never supplied to sleep training."""


def generate():
    items=[]
    for i in range(25):
        a,b=i+3,(i*7)%19+2
        items.append({'id':f'base_math_{i}','type':'arithmetic',
                      'eval_q':f'What is {a} plus {b}? Reply with only the number.',
                      'answer':str(a+b)})
    facts=[('capital of France','paris'),('capital of Japan','tokyo'),
           ('capital of Italy','rome'),('capital of Germany','berlin'),
           ('capital of China','beijing'),('capital of Spain','madrid'),
           ('chemical symbol for gold','au'),('chemical symbol for iron','fe'),
           ('chemical symbol for oxygen','o'),('chemical symbol for carbon','c'),
           ('largest planet in the solar system','jupiter'),
           ('planet known as the Red Planet','mars'),
           ('closest star to Earth','sun'),('largest ocean on Earth','pacific'),
           ('largest living mammal','whale'),('main language spoken in Brazil','portuguese'),
           ('currency of Japan','yen'),('natural satellite of Earth','moon'),
           ('animal famous for eating bamboo','panda'),('insect that makes honey','bee'),
           ('color of fresh snow','white'),('usual color of healthy grass','green'),
           ('usual color of coal','black'),('frozen form of water','ice'),
           ('color of a ripe banana','yellow')]
    for i,(question,answer) in enumerate(facts):
        items.append({'id':f'base_fact_{i}','type':'factual',
                      'eval_q':f'What is the {question}? Reply with one word.',
                      'answer':answer})
    nouns=[('cat','cats'),('dog','dogs'),('bird','birds'),('book','books'),
           ('car','cars'),('tree','trees'),('apple','apples'),('chair','chairs'),
           ('desk','desks'),('pen','pens'),('house','houses'),('cup','cups'),
           ('shoe','shoes'),('hat','hats'),('road','roads'),('door','doors'),
           ('window','windows'),('flower','flowers'),('student','students'),
           ('box','boxes'),('bus','buses'),('child','children'),('woman','women'),
           ('tooth','teeth'),('mouse','mice')]
    for i,(word,plural) in enumerate(nouns):
        items.append({'id':f'base_language_{i}','type':'language',
                      'eval_q':f'Give the English plural of {word}. Reply with only that word.',
                      'answer':plural})
    words=['apple','pear','cat','book','tree','house','rain','moon','star','water',
           'green','red','blue','white','black','hello','quiet','fast','slow','small',
           'large','bird','fish','clock','door']
    for i,word in enumerate(words):
        items.append({'id':f'base_instruction_{i}','type':'instruction',
                      'eval_q':f'Reply with exactly the word {word} and nothing else.',
                      'answer':word})
    assert len(items)==len({it['eval_q'] for it in items})==100
    return items
