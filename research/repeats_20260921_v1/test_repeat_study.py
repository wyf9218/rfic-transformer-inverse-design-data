"""Local plumbing checks only: no training, inference, EM or data-table read."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
import numpy as np
import repeat_study as r

W = Path(__file__).resolve().parents[1]
ROOT = W.parents[1]
TOOLKIT = ROOT/'reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model'


def actual_modules():
    paths = {
        'architecture_adapter': W/'architecture_100k_20260921_v1/train_architectures.py',
        'dataset_adapter': W/'dataset_size_811_20260921_v1/train_dataset_sizes.py',
        'core': W/'datasize_20260921_v1/train_original100k_nested.py',
        'trainer': TOOLKIT/'trainer_runtime.py',
        'splitter': ROOT/'github_worktrees/eucap15-mlp-capacity-20260909/rfic_transformer_inverse_design/model_splitting.py',
        'original_summary': TOOLKIT/'physical_feature_tandem_inverse_summary.json',
    }
    cfg = {'sources': {k: r.pin(p) for k,p in paths.items()}}
    for key, value in cfg['sources'].items():
        if value['sha256'] != r.PINS[key]: raise ValueError('Actual pinned source changed: '+key)
    return r.training_modules(cfg)


class RepeatPlumbing(unittest.TestCase):
    def test_actual_seed_scratch_validation_only_and_train_normalizer(self):
        a = actual_modules(); runtime = a.c.load_runtime()
        contract = {'mother': {'path': '/NOT_READ/original100k.csv'}}
        for seed in r.SEEDS:
            for family, arm in [('dataset','D10k'),('dataset','D50k'),('dataset','D100k'),
                                ('architecture','Small'),('architecture','Medium'),('architecture','Large')]:
                job = {'family':family,'arm':arm,'model_seed':seed,'run_dir':'/NOT_CREATED/check'}
                parsed = runtime._parse_args(r.training_argv(a,contract,job))
                r.check_recipe(parsed,job)
                self.assertEqual(parsed.split_seed,20260711)
                if family == 'architecture':
                    self.assertEqual(parsed.forward_hidden_widths, ','.join([str(a.WIDTHS[arm])]*3))
                    self.assertEqual(parsed.inverse_hidden_widths,parsed.forward_hidden_widths)
                else:
                    self.assertEqual(parsed.forward_hidden_widths,'256,256,128')
                    self.assertEqual(parsed.inverse_hidden_widths,'512,512,256')
        # Invoke the unchanged normalizer on tiny synthetic rows only. Extreme
        # held-out rows must not alter fitted means/scales/bounds.
        x=np.array([[1.,2.],[3.,4.],[999.,999.]]); y=x*2
        matrix={'x':x,'y':y,'split_x':x,'source_indices':[0,1,2],
                'source_evaluations':['a','b','c'],'source_geometry_identities':['a','b','c']}
        split={'train':np.array([0,1]),'validation':np.array([2]),'test':np.array([],dtype=int)}
        norm=runtime._normalize(matrix,split,1e-8)['normalization']
        np.testing.assert_array_equal(norm['x_mean'],[2.,3.]);np.testing.assert_array_equal(norm['y_mean'],[4.,6.])
        np.testing.assert_array_equal(norm['feature_upper'],[3.,4.])

    def test_new_d100_no_original_shortcut_and_actual_seed_gate(self):
        e = r.module(W/'dataset_size_811_20260921_v1/analysis/evaluate_completed_arm.py','test_repeat_evaluator')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); indices={'D100k':{'train':[0],'validation':[1],'test':[2]}}
            job={'kind':'new-trained','family':'dataset','arm':'D100k','model_seed':29,
                 'job_id':'dataset__D100k__seed29','partition':'D100k','run_dir':td}
            self.assertEqual(r.evaluation_job(job)['kind'],'dataset')
            with self.assertRaises(ValueError): r.evaluation_job({**job,'kind':'original'})
            ident={'schema':r.ENTRY_SCHEMA,**job,'dataset_contract_sha256':'contract',
                   'global_split_index_sha256':{p:e.b.ish(indices['D100k'][p]) for p in r.PARTS}}
            r.exclusive(root/'ENTRY_IDENTITY.json',ident);(root/'.run.lock').touch()
            sp=root/'physical_feature_tandem_inverse_summary.json'
            summary={'arguments':{'seed':29,'evaluation_mode':'validation_only'},
                     'optimizer_budget_contract':{'realized':{'exact_update_budget_pass':True}}}
            r.exclusive(sp,summary)
            term={'status':'COMPLETE','model_seed':29,'summary_sha256':r.sha(sp)}
            r.exclusive(root/'ENTRY_TERMINAL.json',term)
            self.assertEqual(r.repeat_gate(e,job,'contract',indices)['status'],'READY')
            summary['arguments']['seed']=20260713;r.atomic(sp,summary)
            term['summary_sha256']=r.sha(sp);r.atomic(root/'ENTRY_TERMINAL.json',term)
            with self.assertRaisesRegex(ValueError,'Actual trained seed'):r.repeat_gate(e,job,'contract',indices)

    def test_deadline_blocks_only_new_stage_no_optimizer_change(self):
        calls=[]
        runtime=types.SimpleNamespace(_train_forward=lambda *a,**k:calls.append('F'),_train_inverse=lambda *a,**k:calls.append('I'))
        r.install_deadline(runtime,time.time()-1)
        with self.assertRaises(r.StageDeadline):runtime._train_forward()
        self.assertEqual(calls,[])
        runtime=types.SimpleNamespace(_train_forward=lambda *a,**k:('history','best'),_train_inverse=lambda *a,**k:None)
        r.install_deadline(runtime,time.time()+60)
        self.assertEqual(runtime._train_forward(),('history','best'))


if __name__=='__main__':unittest.main()
