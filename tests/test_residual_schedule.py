"""Different data sizes must not silently get different optimizer budgets."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.job_schedule import segments,schedule_steps


class Schedules(unittest.TestCase):
    def test_epoch_order_matches_original_protocol(self):
        fits=list(range(11));train={'batch_size':4,'max_epochs':3}
        for unit,batches in segments(fits,123,train):
            self.assertEqual([i for batch in batches for i in batch],np.random.default_rng(123+unit-1).permutation(fits).tolist())
            self.assertEqual([len(x) for x in batches],[4,4,3])

    def test_fixed_budget_matches_steps_and_exposures_at_all_sizes(self):
        train={'batch_size':8,'max_epochs':80,'max_steps':51,'evaluate_every_steps':20}
        for count in (5,20,57):
            result=list(segments(list(range(count)),123,train))
            batches=[b for _,part in result for b in part]
            self.assertEqual(len(batches),51)
            self.assertEqual(sum(map(len,batches)),408)
            self.assertEqual([len(part) for _,part in result],[20,20,11])
            self.assertTrue(all(len(b)==8 for b in batches))
            flat=[i for b in batches for i in b]
            for offset in range(0,len(flat)-count+1,count):self.assertEqual(set(flat[offset:offset+count]),set(range(count)))

    def test_short_refit_preserves_original_cosine_horizon(self):
        train={'batch_size':8,'max_epochs':24,'schedule_epochs':80}
        self.assertEqual(schedule_steps(list(range(795)),train),8000)
        self.assertEqual(len(list(segments(list(range(795)),1,train))),24)


if __name__=='__main__':unittest.main()
