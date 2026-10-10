"""CPU-only per-operation preservation; never average away a failing skill."""
import argparse
import hashlib
import json
from pathlib import Path
from assessment import paired
from storage import atomic_json,digest

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/stage80_continual'


def per_operation(before,after):
    expected={'mivok','tavrel'}
    if {r['id'].split('_',1)[0] for r in before+after}!=expected:
        raise ValueError('Unexpected operation identifiers')
    return {name:paired([r for r in before if r['id'].startswith(name+'_')],
                        [r for r in after if r['id'].startswith(name+'_')]) for name in sorted(expected)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--cycle',type=int)
    args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id: raise ValueError('Simple run id required')
    run=DATA/args.run_id
    paths={'results.json':run/'results.json','baseline.json':run/'baseline.json'}
    result=json.loads(paths['results.json'].read_text())
    baseline=json.loads(paths['baseline.json'].read_text())
    if args.cycle is not None:
        if args.cycle<1: raise ValueError('Positive committed cycle required')
        paths['record.json']=run/f'checkpoints/cycle_{args.cycle:04d}/record.json'
        record=json.loads(paths['record.json'].read_text())
        if record['cycle']!=args.cycle or not record['reload_parity']['passed']:
            raise ValueError('Invalid committed cycle')
        after=record['history'][-1]
        target=run/f'skill_retention_cycle_{args.cycle:04d}.json'
        mode='committed_cycle'
    else:
        if result['status']!='complete': raise ValueError('Only complete results qualify for expansion')
        after=result['final_raw']
        target=run/'per_operation_skill_retention.json'
        mode='completed_final'
    stats={kind:per_operation(baseline[kind],after[kind]) for kind in ('skill_audit','skill_transfer')}
    safe=all(p['after_accuracy']>=.8 and p['retention_of_previously_correct'] is not None and
             p['retention_of_previously_correct']>=.9 for groups in stats.values() for p in groups.values())
    out={'run_id':args.run_id,'mode':mode,'per_operation_retention':stats,
         'safe_for_expansion':safe,'overall_goal_achieved':False,
         'input_hashes':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()},
         'code_hashes':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in ('assess_skill_retention.py','assessment.py')},
         'interpretation':'Keep original minimal pilot gate separate. Before large expansion require each operation >=80% and retention >=90% of its formerly correct answers. This criterion is motivated by observed first-cycle transfer loss and is not retroactively the original pilot gate.'}
    if target.exists(): raise FileExistsError('Per-operation assessment already exists')
    atomic_json(target,out)
    print(json.dumps({'mode':mode,'safe_for_expansion':safe,'per_operation_retention':stats},indent=2))


if __name__=='__main__': main()
