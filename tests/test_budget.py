import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_budget_dr5 import parser, train, evaluate
from src.dr5.budget import (BudgetFusion, MachineAdapter, acquire_budget, confusion_posterior,
                           expected_information, fit_fusion, fit_machine, fuse, normalized_features)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.y = np.repeat(np.arange(5), 40)
        self.x = self.rng.normal(size=(200, 8))
        self.h = self.y.copy()
        self.h[::3] = (self.h[::3] + 1) % 5
        self.p = self.rng.dirichlet(np.ones(5), size=200)

    def test_smoothed_likelihood_is_positive_with_missing_grades(self):
        posterior = confusion_posterior(np.array([0, 0]), np.array([0, 1]))
        self.assertTrue(np.all(posterior > 0))
        q = posterior/posterior.sum(1, keepdims=True)
        pred = fuse(self.p[:2], self.h[:2], q)
        np.testing.assert_allclose(pred.sum(1), 1.)
        np.testing.assert_allclose(fuse(self.p[:2], self.h[:2], q, 0), self.p[:2])

    def test_leave_one_out_score_excludes_held_out_response_from_prior(self):
        ids = np.arange(0, 200, 5)
        fitted = fit_fusion(self.p[ids], self.y[ids], self.h[ids])
        losses = []
        for j, held in enumerate(ids):
            others = np.delete(ids, j)
            counts = confusion_posterior(self.y[others], self.h[others], fitted.mass)
            pred = fuse(self.p[[held]], self.h[[held]], counts/counts.sum(1, keepdims=True), fitted.strength)
            losses.append(-np.log(pred[0, self.y[held]]))
        self.assertAlmostEqual(fitted.loo_nll, np.mean(losses))
        restored = BudgetFusion.from_dict(json.loads(json.dumps(fitted.to_dict())))
        np.testing.assert_allclose(fitted.predict(self.p, self.h), restored.predict(self.p, self.h))

    def test_acquisition_balanced_unique_deterministic_no_human_argument(self):
        pool = np.arange(200)
        selected = [int(i) for c in range(5) for i in np.flatnonzero(self.y == c)[:2]]
        model = fit_fusion(self.p[selected], self.y[selected], self.h[selected])
        z = normalized_features(self.x, pool)
        self.assertNotIn('human', inspect.signature(acquire_budget).parameters)
        self.assertNotIn('human', inspect.signature(expected_information).parameters)
        for mode in ('random', 'diverse'):
            args = (model, z, self.y, self.p, pool, selected)
            first, _ = acquire_budget(*args, np.random.default_rng(5), mode, exploration=0.)
            again, _ = acquire_budget(*args, np.random.default_rng(5), mode, exploration=0.)
            self.assertEqual(first, again)
            self.assertEqual(self.y[first].tolist(), list(range(5)))
            self.assertFalse(set(first) & set(selected))
        information = expected_information(model, self.p, self.y)
        self.assertTrue(np.isfinite(information).all())
        self.assertTrue((information >= 0).all())

    def test_probe_normalization_and_serialization(self):
        cal = np.arange(0, 200, 5)
        pool = np.setdiff1d(np.arange(200), cal)
        adapter, oof, details = fit_machine(self.x, self.y, self.p, pool, cal, 11)
        np.testing.assert_allclose(adapter.mean, self.x[pool].mean(0))
        np.testing.assert_allclose(oof.sum(1), 1.)
        self.assertEqual(details['extra_human_queries'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'adapter.npz'
            adapter.save(path)
            np.testing.assert_allclose(adapter.predict(self.x, self.p), MachineAdapter.load(path).predict(self.x, self.p))

    def test_full_40_query_run_ignores_all_unqueried_humans_and_never_loads_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            frame = pd.DataFrame({'Logits': [','.join(map(str, row)) for row in self.p],
                                  'True Label': self.y, 'Human Label': self.h,
                                  'Feat': [','.join(map(str, row)) for row in self.x]})
            csv = data/'model_output_0_train.csv'
            frame.to_csv(csv, index=False)
            # No test CSV exists: training must still complete.
            for name in ('first', 'poisoned'):
                args = parser().parse_args(['--data-dir', str(data), '--output', str(root/name),
                        '--folds', '0', '--seeds', '1', '--machine', 'frozen', '--acquisition', 'diverse'])
                with patch('builtins.print'):
                    train(args)
                record = json.loads((root/name/'frozen_diverse_fold0_seed1.json').read_text())
                if name == 'first':
                    original = record
                    selected = record['rounds'][-1]['selected_ids']
                    self.assertEqual(len(selected), 40)
                    self.assertEqual(len(set(selected)), 40)
                    np.testing.assert_array_equal(np.bincount(self.y[selected], minlength=5), np.full(5, 8))
                    unqueried = ~frame.index.isin(selected)
                    frame.loc[unqueried, 'Human Label'] = (frame.loc[unqueried, 'Human Label'] + 2) % 5
                    frame.to_csv(csv, index=False)
                else:
                    self.assertEqual(record, original)
            # Evaluation rejects tampering BEFORE trying to load any test file.
            artifact = root/'first'/'frozen_diverse_fold0_seed1.json'
            artifact.write_text(artifact.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'changed after sealing'):
                evaluate(root/'first')


if __name__ == '__main__':
    unittest.main()
