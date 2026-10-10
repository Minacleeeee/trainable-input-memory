"""Raw generation audit; expected answers never enter the generation function."""
import gc
import time
from storage import atomic_json, exact_score
from deadline import check_deadline


def evaluate(model, tok, items, path, stop_at, batch=1):
    from engine import evaluate as serial_evaluate
    from isolated_generation_batch_v1 import generate_equal_length
    if batch not in (1, 2, 4, 8): raise ValueError('Declared generation batch required')
    rows = []; started = time.perf_counter()
    try:
        for offset in range(0, len(items), batch):
            check_deadline(stop_at, {'phase': 'raw_evaluation', 'path': str(path),
                                     'requested': len(items), 'answered': len(rows)})
            selected = items[offset:offset + batch]
            if batch == 1:
                rows += serial_evaluate(model, tok, selected)
            else:
                answers = generate_equal_length(model, tok, [r['eval_q'] for r in selected])
                rows += [{'id': item['id'], 'type': item['type'], 'question': item['eval_q'],
                          'expected': item['answer'], 'response': answer,
                          'correct': exact_score(answer, item['answer'])} for item, answer in zip(selected, answers)]
            if len(rows) % 64 == 0:
                atomic_json(path, {'complete': False, 'raw': rows}); gc.collect()
    except BaseException:
        atomic_json(path, {'complete': False, 'raw': rows, 'elapsed_sec': time.perf_counter() - started}); raise
    atomic_json(path, {'complete': True, 'raw': rows, 'elapsed_sec': time.perf_counter() - started})
    return rows


def evaluate_scale_source(model, tok, source_final, directory, stop_at, memory_batch):
    """Exactly match scale-v2 generation layout, including serial original1024."""
    items = {key: [{'id': r['id'], 'type': r['type'], 'eval_q': r['question'], 'answer': r['expected']}
                   for r in rows] for key, rows in source_final.items()}
    raw = {}
    for key, group in items.items():
        if key in ('exact', 'paraphrase'):
            raw[key] = (evaluate(model, tok, group[:1024], directory / (key + '_source1024_serial.json'), stop_at)
                        + evaluate(model, tok, group[1024:], directory / (key + '_source9216_batched.json'), stop_at, memory_batch))
        else:
            raw[key] = evaluate(model, tok, group, directory / (key + '.json'), stop_at)
    return raw
