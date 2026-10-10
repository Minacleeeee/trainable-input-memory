"""Synthetic admission/scoring safety checks, not production skill evidence."""
import copy
import json
from pathlib import Path
import unittest
from permutation_broad_skill_protocol_v8 import validate_development_groups,gate_from_raw
from storage import exact_score
DATA=Path(__file__).resolve().parents[2]/'data/stage80_continual'

class Guards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((DATA/'permutation_broad_vocabulary_teaching_seed20261005_v8.json').read_text(encoding='utf-8'))
    def test_actual_disjoint_data(self):
        self.assertTrue(validate_development_groups(self.data['groups'],self.data['operation_name']))
    def test_duplicate_tuple_rejected(self):
        g=copy.deepcopy(self.data['groups']);g['audit'][0]=dict(g['train'][0],id='other')
        with self.assertRaises(ValueError):validate_development_groups(g,self.data['operation_name'])
    def test_changed_teacher_answer_rejected(self):
        g=copy.deepcopy(self.data['groups']);g['train'][0]['answer']='wrong'
        with self.assertRaises(ValueError):validate_development_groups(g,self.data['operation_name'])
    def test_reserved_group_rejected(self):
        g=copy.deepcopy(self.data['groups']);g['reserved']=[]
        with self.assertRaises(ValueError):validate_development_groups(g,self.data['operation_name'])
    def test_synthetic_baseline_gain_and_old_gate(self):
        g=self.data['groups']
        good={k:[{'id':r['id'],'type':r['type'],'question':r['eval_q'],'expected':r['answer'],
                  'response':r['answer'],'correct':True} for r in g[k]] for k in ('dev','audit','transfer_audit')}
        bad=copy.deepcopy(good)
        for rows in bad.values():
            for r in rows:r.update(response='wrong',correct=False)
        self.assertTrue(gate_from_raw(g,bad,good,True,exact_score)['passed'])
        self.assertFalse(gate_from_raw(g,good,good,True,exact_score)['passed'])
        self.assertFalse(gate_from_raw(g,bad,good,False,exact_score)['passed'])
        good['audit'][0]['correct']=False
        with self.assertRaises(ValueError):gate_from_raw(g,bad,good,True,exact_score)

if __name__=='__main__':unittest.main()
