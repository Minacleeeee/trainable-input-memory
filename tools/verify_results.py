"""Recompute paper counts from saved responses using only Python's standard library.

This is evidence-level reproduction, not fresh neural inference or historical audits.
Run from any directory: python tools/verify_results.py
"""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def score(response, expected):
    def norm(s):
        return ' '.join(s.lower().split()).strip('.! ')
    return norm(response) == norm(expected)

def check(rows):
    assert len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        assert score(r['response'], r['expected']) == r['correct'], r['id']
    return sum(score(r['response'], r['expected']) for r in rows)

def paired(before, after):
    assert set(before) == set(after)
    changed = lost = count = 0
    for suite in before:
        b, a = before[suite], after[suite]
        check(b); check(a)
        assert len(b) == len(a)
        for x, y in zip(b, a):
            assert all(x[k] == y[k] for k in ('id', 'question', 'expected'))
            count += 1
            changed += x['response'] != y['response']
            lost += score(x['response'], x['expected']) and not score(y['response'], y['expected'])
    return {'count': count, 'raw_changed': changed, 'correct_lost': lost}

def main():
    provenance = json.loads((ROOT / 'data/provenance.json').read_text())
    data = {}
    for item in provenance['exports']:
        p = ROOT / item['export_file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['export_sha256']
        data[item['alias']] = json.loads(gzip.decompress(p.read_bytes()))
    result = {'scope': 'Saved-response and split verification; no model calls', 'skills': {}}
    for alias, counts in [('input_a', [59,190,155]), ('shared_a',[64,187,145]), ('input_b',[64,191,171])]:
        d = data[alias]; final = d['result']['final_skill_raw']; baseline = d['baseline_skill']
        for (suite, rows), expected in zip(final.items(), counts):
            assert len(rows) == (64 if suite == 'dev' else 192)
            assert check(rows) == expected and check(baseline[suite]) == 0
        skill_pair = paired(baseline, final)
        old = paired(d['baseline_old'], d['result']['final_old_raw'])
        assert old['count'] == 21348
        assert (old['raw_changed'], old['correct_lost']) == ((1788,92) if alias == 'shared_a' else (0,0))
        dataset = d['development_dataset']
        assert [len(dataset[k]) for k in ('train','dev','audit','transfer_audit')] == [512,64,192,192]
        all_q = [r['eval_q'] for rows in dataset.values() for r in rows]
        assert len(set(all_q)) == 960
        words = lambda rows: set(w for r in rows for w in r['eval_q'].split(': ',1)[1].split())
        train_words, transfer_words = words(dataset['train']), words(dataset['transfer_audit'])
        assert len(train_words) == 64 and len(transfer_words) == 16 and not train_words & transfer_words
        for suite, rows in dataset.items():
            for row in rows:
                w = row['eval_q'].split(': ',1)[1].split()
                assert len(w) == len(set(w)) == 4
                assert row['answer'] == ' '.join(w[i] for i in (2,0,3,1))
            if suite != 'train':
                assert [(r['id'],r['eval_q'],r['answer']) for r in rows] == [(r['id'],r['question'],r['expected']) for r in final[suite]]
        from collections import Counter
        for pos in range(4):
            c = Counter(r['eval_q'].split(': ',1)[1].split()[pos] for r in dataset['train'])
            assert set(c.values()) == {8} and len(c) == 64
        result['skills'][alias] = {'before':[check(baseline[k]) for k in final], 'after':counts, 'old':old,
            'updates':d['result']['actual_updates'], 'presentations':d['result']['actual_presentations']}
        assert d['result']['actual_updates'] == 1920 and d['result']['actual_presentations'] == 15360
    a_words = {w for rows in data['input_a']['development_dataset'].values() for r in rows for w in r['eval_q'].split(': ',1)[1].split()}
    b_words = {w for rows in data['input_b']['development_dataset'].values() for r in rows for w in r['eval_q'].split(': ',1)[1].split()}
    assert not a_words & b_words
    d = data['scale']; final = d['result']['final_raw']
    assert check(final['exact']) == 10182 and check(final['paraphrase']) == 9963
    assert len(final['exact']) == len(final['paraphrase']) == 10240
    acquired = 0
    for block, baseline in enumerate(d['introduction_baselines']):
        b1,b2 = baseline['exact'],baseline['paraphrase']; check(b1);check(b2)
        a1 = final['exact'][block*512:(block+1)*512];a2 = final['paraphrase'][block*512:(block+1)*512]
        assert len(b1)==len(b2)==len(a1)==len(a2)==512
        for x,y,u,v in zip(b1,b2,a1,a2):
            assert (x['id'],x['question'],x['expected']) == (u['id'],u['question'],u['expected'])
            assert (y['id'],y['question'],y['expected']) == (v['id'],v['question'],v['expected'])
            acquired += not x['correct'] and not y['correct'] and u['correct'] and v['correct']
    assert acquired == 9089
    old2916=d['initial_old2916_baseline']
    matched_old={}
    for k,rows in old2916.items():
        if k=='prior512_facts': target=final['exact'][:512]
        elif k=='prior512_paraphrase': target=final['paraphrase'][:512]
        elif k in ('exact','paraphrase'): target=final[k][512:1024]
        else: target=final[k]
        matched_old[k]=target
    assert paired(old2916,matched_old)=={'count':2916,'raw_changed':0,'correct_lost':0}
    assert data['input_a']['development_dataset']==data['shared_a']['development_dataset']
    result['scale']={'canonical':10182,'withheld':9963,'conservative_acquired':acquired,'old_unchanged':2916}
    for alias in ['input_a','shared_a','input_b']:
        assert paired(final, data[alias]['baseline_old'])['raw_changed'] == 0
    later=data['later']; r=later['result']; new=r['new_final_raw']
    assert check(new['exact'])==508 and check(new['paraphrase'])==496
    assert paired(later['baseline_old'],r['final_old_raw'])['raw_changed']==0
    assert paired(data['input_a']['result']['final_skill_raw'],r['final_skill_raw']) == {'count':448,'raw_changed':0,'correct_lost':0}
    bn=later['baseline_new512']
    assert paired(bn,new)['count']==1024
    acquired_later=sum(not x['correct'] and not y['correct'] and u['correct'] and v['correct'] for x,y,u,v in zip(bn['exact'],bn['paraphrase'],new['exact'],new['paraphrase']))
    assert acquired_later==447
    result['later']={'canonical':508,'withheld':496,'conservative_acquired':acquired_later,'skill_raw_unchanged':448}
    assert paired(final,data['fp16']['result']['final_raw'])=={'count':21348,'raw_changed':0,'correct_lost':0}
    result['fp16']={'unchanged':21348,'tensor_bytes':data['fp16']['result']['resident_bank_tensor_bytes']}
    newline_map = json.loads((ROOT/'data/source_newlines.json').read_text())
    source_exact = source_reconstructed = 0
    for item in provenance['source_archive']:
        if item['included']:
            payload = (ROOT/'source_archive'/item['file']).read_bytes()
            expected = item['sha256']
            if hashlib.sha256(payload).hexdigest() == expected:
                source_exact += 1
            else:
                # Restore only recorded newline layout, including mixed CRLF/LF.
                # The ORIGINAL full-byte SHA is still mandatory; no edits ignored.
                layout = newline_map[item['file']]
                lines = payload.replace(b'\r\n', b'\n').split(b'\n')
                assert len(lines)-1 == layout['newline_count'], item['file']
                crlf = set(layout['crlf_newline_indices'])
                assert all(type(i) is int and 0 <= i < len(lines)-1 for i in crlf)
                restored = b'\n'.join(line + (b'\r' if i in crlf else b'')
                                      for i,line in enumerate(lines))
                assert hashlib.sha256(restored).hexdigest() == expected, item['file']
                source_reconstructed += 1
    result['source_integrity'] = {
        'original_bytes_exact': source_exact,
        'original_hash_reconstructed_from_newline_layout': source_reconstructed,
        'historical_hashes_modified': False,
    }
    print(json.dumps(result,indent=2))
    print('PASS: all released primary counts, split checks, pairings and source hashes verified (newline reconstruction reported separately).')

if __name__ == '__main__':
    main()
