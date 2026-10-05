import unittest
import numpy as np
import torch
from src.dr5.core import (Evaluator, acquire, fit_temperature, fuse_log_probs, metrics,
                         predict_samples, seed_everything, temperature_scale)
from run_dr5 import split_ids


class DR5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(3)

    def test_product_loss_and_gradients(self):
        ai = torch.softmax(torch.randn(9, 5), -1)
        logits = torch.randn(9, 5, requires_grad=True)
        lp = fuse_log_probs(ai, logits)
        product = ai * logits.softmax(-1)
        torch.testing.assert_close(lp.exp(), product/product.sum(-1, keepdim=True))
        loss = torch.nn.functional.nll_loss(lp, torch.arange(9) % 5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)

    def test_mc_stochastic_and_restores_rng_and_mode(self):
        m = Evaluator(8, .5).train()
        x, h, ai = torch.randn(20, 8), torch.arange(20)%5, torch.ones(20,5)/5
        state = torch.get_rng_state().clone()
        p = predict_samples(m, x, h, ai, 10, 12)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(m.training and m.dropout.training)
        self.assertGreater(p.var(0).sum(), 0)
        torch.testing.assert_close(p.sum(-1), torch.ones(10,20))
        torch.testing.assert_close(p, predict_samples(m,x,h,ai,10,12))

    def test_calibration_preserves_argmax_and_improves_fit_nll(self):
        p = torch.tensor([[.99,.0025,.0025,.0025,.0025]]).repeat(100,1)
        y = torch.arange(100)%5
        t = fit_temperature(p,y)
        calibrated = temperature_scale(p,t)
        self.assertTrue(torch.equal(p.argmax(1),calibrated.argmax(1)))
        self.assertLess(metrics(calibrated,y)['nll'],metrics(p,y)['nll'])

    def test_ece_counts_unit_confidence(self):
        p = torch.eye(5)
        self.assertEqual(metrics(p,torch.arange(5))['ece'],0.)
        self.assertEqual(metrics(p,(torch.arange(5)+1)%5)['ece'],1.)

    def test_acquisition_disjoint_balanced_deterministic(self):
        y = torch.arange(5).repeat_interleave(12)
        cal,val,pool = split_ids(y,2,2,123)
        self.assertFalse(set(cal)&set(val)|set(cal)&set(pool)|set(val)&set(pool))
        m,x,ai = Evaluator(8),torch.randn(60,8),torch.ones(60,5)/5
        selected = [int(pool[y[pool]==c][0]) for c in range(5)]
        for mode in ['baseline','mc','reliability','random']:
            a,_ = acquire(m,x,y,ai,pool,selected,np.random.default_rng(4),mode,passes=5)
            b,_ = acquire(m,x,y,ai,pool,selected,np.random.default_rng(4),mode,passes=5)
            self.assertEqual(a,b)
            self.assertEqual(y[a].tolist(),list(range(5)))
            self.assertFalse(set(a)&set(selected))
            self.assertTrue(set(a).issubset(set(pool)))


if __name__ == '__main__':
    unittest.main()
