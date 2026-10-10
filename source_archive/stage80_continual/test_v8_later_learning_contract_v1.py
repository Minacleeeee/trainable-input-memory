"""CPU admission, parameter-path and scoring fixtures; no production model calls."""
import copy
import unittest
import torch
from torch import nn
from v8_later_learning_contract_v1 import qualify, physical_admission, configure_later_trainable, skill_after_learning, fact_after_learning
from permutation_broad_skill_protocol_v8 import PROTOCOL, PROFILE
from shared_skill_module_v1 import SharedProjection, export


class Admission(unittest.TestCase):
    def fixture(self,arm='input'):
        gate={'passed':True,'all21348_old_raw_responses_exactly_unchanged':True}
        maintenance={'synthetic_lora':'bound'}
        r={'status':'complete','manifest':{'protocol':PROTOCOL,'profile':copy.deepcopy(PROFILE),
            'run_id':'synthetic_completed_V8','arm':arm,'input_inventory_rows':10241,
            'active_trainable_parameters':2048 if arm=='input' else 18432,
            'source_run':'isolated_scale10240_s42_v2','source_qualification':{'passed':True},
            'frozen_maintenance_hashes':maintenance},'history':[{'synthetic':True}],
            'full_actual_pregradient_queries':21796,'full_saved_reload_final_queries':21796,
            'actual_updates':1920,'actual_presentations':15360,'actual_supervised_tokens':103920,
            'all_old20_neural_banks_bit_identical':True,'frozen_maintenance_hashes_after':maintenance,
            'joint_gate':gate}
        a={'status':'complete','run_id':'synthetic_completed_V8','arm':arm,
            'integrity_raw_identity_and_actual_budget_passed':True,'immediate_skill_joint_gate_passed':True,
            'all_old20_banks_bit_identical':True,'all43592_before_and_after_result_rows_bind_complete_physical_raw_files':True,
            'maintenance_hashes_before_after_equal':True,'joint_gate':copy.deepcopy(gate),
            'input_hashes':{'results.json':'synthetic_bound'}}
        return r,a

    def test_both_architectures_require_same_complete_contract(self):
        for arm in ('input','shared'):
            r,a=self.fixture(arm)
            self.assertEqual(qualify(r,a,'synthetic_bound')['arm'],arm)

    def test_failed_partial_wrong_protocol_budget_and_maintenance_rejected(self):
        for key,value in [('status','initialization'),('actual_updates',960),('actual_supervised_tokens',69120),
                          ('all_old20_neural_banks_bit_identical',False),('frozen_maintenance_hashes_after',{})]:
            r,a=self.fixture();r[key]=value
            with self.assertRaises(ValueError):qualify(r,a,'synthetic_bound')
        for key,value in [('protocol','stage80_isolated_conditional_skill_v5'),('arm','invented'),
                          ('active_trainable_parameters',18432),('input_inventory_rows',10753)]:
            r,a=self.fixture();r['manifest'][key]=value
            with self.assertRaises(ValueError):qualify(r,a,'synthetic_bound')
        r,a=self.fixture();r['joint_gate']['passed']=False;a['joint_gate']=copy.deepcopy(r['joint_gate'])
        with self.assertRaises(ValueError):qualify(r,a,'synthetic_bound')

    def test_unbound_incomplete_or_wrong_arm_audit_rejected(self):
        for key,value in [('status','partial'),('arm','shared'),('all43592_before_and_after_result_rows_bind_complete_physical_raw_files',False),
                          ('immediate_skill_joint_gate_passed',False),('input_hashes',{'results.json':'stale'})]:
            r,a=self.fixture();a[key]=value
            with self.assertRaises(ValueError):qualify(r,a,'synthetic_bound')

    def test_actual_incomplete_recovery_and_failed_historical_sources_rejected(self):
        from pathlib import Path
        data=Path(__file__).resolve().parents[2]/'data/stage80_continual'
        for rid in ('permutation_broad_input_s42_w2_v8','permutation_input_s42_w2_v7','conditional_shared_s42_w2_v6'):
            with self.assertRaises((FileNotFoundError,ValueError)):physical_admission(data,rid)
        with self.assertRaises(ValueError):physical_admission(data,'../invalid')


class GradientPaths(unittest.TestCase):
    def fixture(self,shared):
        model=nn.Module();embedding=nn.Module()
        # Expanded frozen zero fixtures avoid allocating a production-size ancestor.
        embedding.blocks=nn.ParameterList([nn.Parameter(torch.zeros(1).expand(512,2048),requires_grad=False) for _ in range(20)]
            +[nn.Parameter(torch.full((1,2048),.02),requires_grad=False),nn.Parameter(torch.full((512,2048),.01))])
        def activate(i):
            for j,p in enumerate(embedding.blocks):p.requires_grad_(j==i)
        embedding.activate_block=activate;model.embedding=embedding
        model.frozen=nn.Parameter(torch.ones(1),requires_grad=False)
        modules={}
        if shared:
            for i,name in enumerate(('q_proj','v_proj')):
                original=nn.Linear(2048,2048,bias=False);original.active_adapters=['single']
                module=SharedProjection(original,2,4,50+i,.01)
                with torch.no_grad():module.skill_up.fill_(.002)
                setattr(model,name,module);modules[name]=module
        return model,embedding,modules

    def test_isolation_updates_only_new_bank_preserves_command_shared_and_old_output(self):
        for shared in (False,True):
            model,emb,modules=self.fixture(shared)
            params=configure_later_trainable(model,emb,modules,'shared' if shared else 'input','isolated_new_bank')
            self.assertEqual(sum(p.numel() for ps in params.values() for p in ps),1048576)
            operation=emb.blocks[20].detach().clone();saved=export(modules)
            x=torch.ones(1,2048)*.02
            before={n:m(x).detach().clone() for n,m in modules.items()}
            optimizer=torch.optim.SGD(params['name'],lr=.1)
            loss=emb.blocks[21].sum();loss.backward();optimizer.step()
            self.assertFalse(torch.equal(emb.blocks[21],torch.full((512,2048),.01)))
            self.assertTrue(torch.equal(emb.blocks[20],operation))
            self.assertFalse(any(p.requires_grad for p in emb.blocks[:21]))
            for n,m in modules.items():
                self.assertTrue(torch.equal(m(x).detach(),before[n]))
                self.assertTrue(torch.equal(export(modules)[n]['A'],saved[n]['A']))
                self.assertTrue(torch.equal(export(modules)[n]['B'],saved[n]['B']))

    def test_plastic_shared_keeps_module_in_graph_and_receives_actual_gradients(self):
        model,emb,modules=self.fixture(True)
        params=configure_later_trainable(model,emb,modules,'shared','plastic_existing_shared')
        self.assertEqual(sum(p.numel() for ps in params.values() for p in ps),1064960)
        initial=export(modules)
        optimizer=torch.optim.SGD([p for ps in params.values() for p in ps],lr=.01)
        x=emb.blocks[21][:1]
        loss=sum(m(x).square().mean() for m in modules.values())
        loss.backward()
        for p in params['name']+params['shared']:
            self.assertGreater(float(p.grad.abs().sum()),0)
        optimizer.step()
        for n in modules:
            self.assertFalse(torch.equal(export(modules)[n]['B'],initial[n]['B']))
        self.assertFalse(any(p.requires_grad for p in emb.blocks[:21]))

    def test_wrong_mode_bank_missing_detached_shared_or_input_plastic_rejected(self):
        model,emb,modules=self.fixture(True)
        for arm,mode,payload in [('input','isolated_new_bank',modules),('input','plastic_existing_shared',{}),
                                 ('shared','isolated_new_bank',{}),('shared','unknown',modules)]:
            with self.assertRaises(ValueError):configure_later_trainable(model,emb,payload,arm,mode)
        detached=copy.copy(modules);detached['q_proj']=self.fixture(True)[2]['q_proj']
        with self.assertRaises(ValueError):configure_later_trainable(model,emb,detached,'shared','isolated_new_bank')
        emb.blocks[20]=nn.Parameter(torch.zeros(2,2048))
        with self.assertRaises(ValueError):configure_later_trainable(model,emb,modules,'shared','isolated_new_bank')


class RetentionScoring(unittest.TestCase):
    def fixture(self):
        source={k:[{'id':k+str(i),'type':'synthetic_skill','question':'fixture '+k+str(i),
                    'expected':'yes','response':'yes','correct':True} for i in range(n)]
                for k,n in [('dev',64),('audit',192),('transfer_audit',192)]}
        before=copy.deepcopy(source)
        for rows in before.values():
            for row in rows:row.update(response='no',correct=False)
            rows[0].update(response='yes',correct=True)
        return before,source,copy.deepcopy(source)

    def test_baseline_correct_excluded_and_later_forgotten_excluded(self):
        b,s,a=self.fixture();a['transfer_audit'][1].update(response='no',correct=False)
        r=skill_after_learning(s,a,b)
        self.assertEqual(r['measurements']['transfer_audit']['acquired_after_initial_teaching'],191)
        self.assertEqual(r['measurements']['transfer_audit']['still_correct_after_later_learning'],190)
        self.assertEqual(r['measurements']['transfer_audit']['source_correct_lost'],1)
        self.assertFalse(r['all448_source_skill_raw_exact'])

    def test_partial_changed_identity_and_fabricated_score_rejected(self):
        for kind in ('partial','identity','score'):
            b,s,a=self.fixture()
            if kind=='partial':a['audit'].pop()
            if kind=='identity':a['dev'][0]['question']='changed'
            if kind=='score':a['dev'][0]['response']='no'
            with self.assertRaises(ValueError):skill_after_learning(s,a,b)

    def test_fact_acquisition_excludes_either_before_and_later_forgetting(self):
        facts=[{'id':'fact'+str(i),'type':'fact','eval_q':'canonical '+str(i),
                'audit_q':'withheld '+str(i),'answer':'yes'} for i in range(512)]
        after={k:[{'id':r['id'],'type':r['type'],'question':r['eval_q'] if k=='exact' else r['audit_q'],
                   'expected':'yes','response':'yes','correct':True} for r in facts] for k in ('exact','paraphrase')}
        before=copy.deepcopy(after)
        for rows in before.values():
            for r in rows:r.update(response='no',correct=False)
        before['exact'][0].update(response='yes',correct=True)
        before['paraphrase'][1].update(response='yes',correct=True)
        after['paraphrase'][2].update(response='no',correct=False)
        result=fact_after_learning(facts,before,after)
        self.assertEqual(result['conservative_acquired_facts'],509)
        self.assertTrue(result['minimum_each_wording_pass'])
        after['exact'].pop()
        with self.assertRaises(ValueError):fact_after_learning(facts,before,after)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
