"""Adversarial admission checks; no neural forwards, GPU or reserved data."""
import ast
import copy
import json
from pathlib import Path
import unittest
from permutation_independent_world_protocol_v1 import (
    validate_candidate, validate_development_groups, gate_from_raw, PROFILE, DATASET_NAME, PRIOR_DATASETS)
from storage import exact_score, digest

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/stage80_continual'


class Guards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads((DATA / DATASET_NAME).read_text(encoding='utf-8'))
        cls.previous = {n: json.loads((DATA / n).read_text(encoding='utf-8')) for n in PRIOR_DATASETS}

    def test_fresh_disjoint_candidate(self):
        self.assertTrue(validate_candidate(self.candidate, self.previous))

    def test_old_development_world_rejected(self):
        with self.assertRaises(ValueError):
            validate_development_groups(self.previous[PRIOR_DATASETS[1]]['groups'], 'qezorvank')

    def test_changed_seed_source_or_previously_queried_candidate_rejected(self):
        for key, value in [('seed', 42), ('source_results_sha256', 'failed_skill_source'),
                           ('production_GPU_model_answers_or_training_updates', 1)]:
            c = copy.deepcopy(self.candidate); c[key] = value
            with self.assertRaises(ValueError): validate_candidate(c, self.previous)

    def test_overlap_hidden_by_metadata_flag_rejected(self):
        previous = copy.deepcopy(self.previous)
        previous[PRIOR_DATASETS[0]]['train_vocabulary'][0] = self.candidate['train_vocabulary'][0]
        with self.assertRaises(ValueError): validate_candidate(self.candidate, previous)

    def test_duplicate_or_wrong_teacher_answer_rejected(self):
        c = copy.deepcopy(self.candidate)
        c['groups']['audit'][0] = dict(c['groups']['train'][0], id='different_id')
        with self.assertRaises(ValueError): validate_candidate(c, self.previous)
        c = copy.deepcopy(self.candidate); c['groups']['train'][0]['answer'] = 'fabricated'
        with self.assertRaises(ValueError): validate_candidate(c, self.previous)

    def test_reserved_split_and_changed_budget_rejected(self):
        c = copy.deepcopy(self.candidate); c['groups']['reserved'] = []
        with self.assertRaises(ValueError): validate_candidate(c, self.previous)
        c = copy.deepcopy(self.candidate); c['prospective_supervised_tokens'] = 103920
        with self.assertRaises(ValueError): validate_candidate(c, self.previous)

    def test_score_fabrication_and_missing_actual_gain_rejected(self):
        g = self.candidate['groups']
        good = {k: [dict(id=r['id'], type=r['type'], question=r['eval_q'], expected=r['answer'],
                         response=r['answer'], correct=True) for r in g[k]]
                for k in ('dev', 'audit', 'transfer_audit')}
        bad = copy.deepcopy(good)
        for rows in bad.values():
            for row in rows: row.update(response='wrong', correct=False)
        self.assertTrue(gate_from_raw(g, bad, good, True, exact_score)['passed'])
        self.assertFalse(gate_from_raw(g, good, good, True, exact_score)['passed'])
        self.assertFalse(gate_from_raw(g, bad, good, False, exact_score)['passed'])
        good['audit'][0]['correct'] = False
        with self.assertRaises(ValueError): gate_from_raw(g, bad, good, True, exact_score)

    def test_new_training_seeds_and_entry_points_use_new_versions(self):
        self.assertEqual(PROFILE['training_seeds'], {'teach': 20264046, 'sleep': 20264047})
        for name in ['run_permutation_independent_world_v1.py',
                     'audit_permutation_independent_world_v1.py',
                     'audit_permutation_independent_world_runtime_v1.py']:
            source = (HERE / name).read_text(encoding='utf-8'); ast.parse(source)
            self.assertNotIn('prepared_permutation_broad_skill_v8.json', source)
            self.assertNotIn('permutation_broad_vocabulary_teaching_seed20261005_v8.json', source)
            self.assertIn('permutation_independent_world_protocol_v1', source)


if __name__ == '__main__': unittest.main()
