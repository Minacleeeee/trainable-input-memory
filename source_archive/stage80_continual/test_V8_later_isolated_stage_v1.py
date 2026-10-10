"""Adversarial physical binding and complete joint-retention guard fixtures."""
import copy
import tempfile
import unittest
from pathlib import Path
from storage import atomic_json
from queue_matched_memory_v1 import sha
from audit_V8_later_isolated_stage_v1 import physical,verify_audit_binding
from v8_later_isolated_protocol_v1 import final_gate


class Joint(unittest.TestCase):
    def fixture(self):
        from test_v8_later_learning_contract_v1 import RetentionScoring
        original,skill,_=RetentionScoring().fixture()
        old={'fixture_old':[{'id':'old'+str(i),'type':'old','question':'old '+str(i),'expected':'yes','response':'yes','correct':True} for i in range(21348)]}
        facts=[{'id':'fact'+str(i),'type':'fact','eval_q':'canonical '+str(i),'audit_q':'withheld '+str(i),'answer':'yes'} for i in range(512)]
        after={k:[{'id':r['id'],'type':r['type'],'question':r['eval_q'] if k=='exact' else r['audit_q'],'expected':'yes','response':'yes','correct':True} for r in facts] for k in ('exact','paraphrase')}
        before=copy.deepcopy(after)
        for rows in before.values():
            for r in rows:r.update(response='no',correct=False)
        return [old,skill,copy.deepcopy(old),copy.deepcopy(skill),copy.deepcopy(old),copy.deepcopy(skill),facts,before,after,original]

    def test_joint_fails_one_old_reply_even_with_all_new_answers_correct(self):
        values=self.fixture();self.assertTrue(final_gate(*values)['passed'])
        values[4]['fixture_old'][0].update(response='no',correct=False)
        self.assertFalse(final_gate(*values)['passed'])

    def test_joint_fails_one_new_skill_lost_and_wrong_before_source(self):
        values=self.fixture();values[5]['transfer_audit'][0].update(response='no',correct=False)
        self.assertFalse(final_gate(*values)['passed'])
        values=self.fixture();values[3]['dev'][0].update(response='no',correct=False)
        self.assertFalse(final_gate(*values)['passed'])

    def test_partial_and_fabricated_scores_cannot_be_accepted(self):
        values=self.fixture();values[8]['paraphrase'].pop()
        with self.assertRaises(ValueError):final_gate(*values)
        values=self.fixture();values[8]['exact'][0]['response']='no'
        with self.assertRaises(ValueError):final_gate(*values)


class Physical(unittest.TestCase):
    def test_partial_physical_file_rejected_even_when_count_is_complete(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);atomic_json(run/'raw.json',{'complete':False,'raw':[{}]})
            with self.assertRaises(ValueError):physical(run,'raw.json',1,{})
            atomic_json(run/'raw.json',{'complete':True,'raw':[{}]})
            with self.assertRaises(ValueError):physical(run,'raw.json',2,{})

    def test_runtime_tensor_tampering_and_directory_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';run.mkdir();atomic_json(run/'results.json',{'fixture':True})
            (run/'tensor.pt').write_bytes(b'fixture weights')
            a={'runtime_integrity_passed':True,'results_sha256':sha(run/'results.json'),'bound_file_hashes':{'tensor.pt':sha(run/'tensor.pt')}}
            verify_audit_binding(run,a)
            (run/'tensor.pt').write_bytes(b'changed weights')
            with self.assertRaises(ValueError):verify_audit_binding(run,a)
            a['bound_file_hashes']={'../outside':sha(run/'tensor.pt')}
            with self.assertRaises(ValueError):verify_audit_binding(run,a)

    def test_unqualified_or_result_tampered_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);atomic_json(run/'results.json',{'fixture':True})
            a={'runtime_integrity_passed':False,'results_sha256':sha(run/'results.json'),'bound_file_hashes':{'results.json':sha(run/'results.json')}}
            with self.assertRaises(ValueError):verify_audit_binding(run,a)
            a['runtime_integrity_passed']=True;atomic_json(run/'results.json',{'fixture':'changed'})
            with self.assertRaises(ValueError):verify_audit_binding(run,a)


if __name__=='__main__':unittest.main()
