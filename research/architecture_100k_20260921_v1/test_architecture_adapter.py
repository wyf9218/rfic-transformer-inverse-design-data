#!/usr/bin/env python3
"""Targeted new wrapper checks; synthetic optimizer only, no formal training."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import train_architectures as a


class ArchitectureChecks(unittest.TestCase):
    def test_all_arms_change_only_width_arguments(self):
        runtime=a.c.load_runtime()
        contract={'mother':{'path':'/READ_ONLY_NOT_ACCESSED/mother.csv'}}
        base=vars(runtime._parse_args(a.d.trainer_argv(contract,Path('/NOT_CREATED/arm'))))
        allowed={'forward_depth','forward_width','forward_hidden_widths','inverse_depth','inverse_width','inverse_hidden_widths'}
        for arm,width in a.WIDTHS.items():
            args=vars(runtime._parse_args(a.architecture_argv(contract,Path('/NOT_CREATED/arm'),arm)))
            self.assertEqual({k for k in base if base[k]!=args[k] and not (isinstance(base[k],float) and np.isnan(base[k]) and np.isnan(args[k]))}-allowed,set())
            self.assertEqual(args['forward_hidden_widths'],','.join([str(width)]*3))
            self.assertEqual(args['inverse_hidden_widths'],','.join([str(width)]*3))
            self.assertEqual(args['seed'],20260713)
            self.assertEqual((args['forward_max_optimizer_updates'],args['inverse_max_optimizer_updates']),(3600,3600))
            self.assertEqual(args['q_target_semantics'],'minimum')
            self.assertEqual(args['q_minimum_margin_physical'],2)

    def test_architecture_shapes_and_counts_without_training(self):
        for arm in a.WIDTHS:
            layout=a.architecture(arm)
            for stage in ('forward','inverse'):
                shape=layout[stage]
                count=sum((x+1)*y for x,y in zip(shape,shape[1:]))
                self.assertEqual(count,layout[stage+'_parameters'])
            self.assertEqual(len(layout['forward']),5)
            self.assertEqual(len(layout['inverse']),5)

    def test_observer_retains_bitwise_original_adam_and_return(self):
        runtime=a.c.load_runtime();raw=runtime._adam_step
        w=[np.array([[.3,-.2],[.1,.4]])];b=[np.array([.02,.03])]
        gw=[np.array([[.01,-.04],[.2,-.1]])];gb=[np.array([.01,-.05])]
        expected_w=copy.deepcopy(w);expected_b=copy.deepcopy(b)
        expected_state=runtime._init_adam(expected_w,expected_b)
        for unused in range(3):raw(expected_w,expected_b,gw,gb,expected_state,.001)
        state=runtime._init_adam(w,b)
        sentinel_history=[{'optimizer_updates':3,'validation_normalized_rmse':.7}]
        sentinel_best={'optimizer_updates':2}
        def synthetic_stage():
            for unused in range(3):runtime._adam_step(w,b,gw,gb,state,.001)
            return sentinel_history,sentinel_best
        runtime._train_forward=synthetic_stage
        runtime._train_inverse=synthetic_stage
        with tempfile.TemporaryDirectory(prefix='architecture_observer_synthetic_') as folder:
            out=Path(folder);a.install_progress_observer(runtime,out)
            result=runtime._train_forward()
            self.assertIs(result[0],sentinel_history);self.assertIs(result[1],sentinel_best)
            progress=a.c.read(out/'PROGRESS.json')
            self.assertEqual(progress['stage'],'forward_training')
            self.assertEqual(progress['actual_optimizer_updates'],3)
            self.assertEqual(progress['status'],'STAGE_TRAINING_COMPLETE')
            self.assertEqual(progress['best_optimizer_update'],2)
            for actual,expected in zip(w+b,expected_w+expected_b):np.testing.assert_array_equal(actual,expected)
            for key in ('mw','vw','mb','vb'):
                for actual,expected in zip(state[key],expected_state[key]):np.testing.assert_array_equal(actual,expected)
            self.assertEqual(state['step'],expected_state['step'])


if __name__=='__main__':unittest.main(verbosity=2)
