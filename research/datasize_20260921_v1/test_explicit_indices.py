#!/usr/bin/env python3
"""Only small synthetic adapter checks. No _train_forward/_train_inverse call."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
import numpy as np
import train_original100k_nested as a


class AdapterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime=a.load_runtime()
        cls.indices={'train_2':[1,0],'train_4':[1,0,3,2],
                     'train_6':[1,0,3,2,4,5],'validation':[6],'test':[7]}

    def matrix(self):
        x=np.column_stack([np.linspace(.7,1.4,8),np.linspace(.8,1.5,8),np.linspace(8,12,8),np.linspace(.2,.5,8)])
        y=np.arange(80,dtype=float).reshape(8,10)+200
        return {'count':8,'x':x,'y':y,'split_x':x.copy(),
                'split_reference_columns':['input__lp_nh_center','input__ls_nh_center','input__q_center','input__k_abs_center'],
                'source_indices':list(range(8)),'source_evaluations':[f'synthetic{i}' for i in range(8)],
                'source_geometry_identities':[f'{i:064x}' for i in range(8)]}

    def test_nested_and_disjoint(self):
        a.validate_indices(self.indices,6,8,(2,4,6))
        bad=copy.deepcopy(self.indices);bad['test']=[6]
        with self.assertRaises(ValueError):a.validate_indices(bad,6,8,(2,4,6))
        bad=copy.deepcopy(self.indices);bad['train_2']=[0,1]
        with self.assertRaises(ValueError):a.validate_indices(bad,6,8,(2,4,6))

    def test_source_mapping_all_assigned_and_train_only_normalizer(self):
        m=a.select_matrix(self.matrix(),self.indices,2,8)
        split,audit=a.explicit_split(m,self.indices,2,'synthetic')
        self.assertEqual(m['source_indices'],[1,0,6,7])
        isolation=self.runtime._evaluation_isolation_contract(m,split,'validation_only')
        self.assertEqual(isolation['overall_status'],'PASS')
        data=self.runtime._normalize(m,split,1e-12)
        np.testing.assert_array_equal(data['normalization']['x_mean'],m['x'][:2].mean(axis=0))
        self.assertFalse(np.array_equal(data['normalization']['x_mean'],m['x'].mean(axis=0)))
        self.assertFalse(audit['formal_cell_ood'])

    def test_nonfinite_row_drop_and_duplicate_identity_fail(self):
        m=self.matrix();m['source_indices'][0]=1
        with self.assertRaises(ValueError):a.select_matrix(m,self.indices,2,8)
        m=self.matrix();m['source_geometry_identities'][6]=m['source_geometry_identities'][1]
        with self.assertRaises(ValueError):a.select_matrix(m,self.indices,2,8)

    def test_missing_qualification_rejected(self):
        c={'combined_csv':{'sha256':'a'},'mother':{'sha256':'b'},'indices':{'sha256':'c'}}
        with self.assertRaisesRegex(ValueError,'not PASS'):a.require_qualification(c,{'status':'UNKNOWN'})
        with self.assertRaisesRegex(ValueError,'does not bind'):a.require_qualification(c,{'status':'PASS'})

    def test_family_assessment_cannot_be_unknown(self):
        c={'combined_csv':{'sha256':'a'},'mother':{'sha256':'b'},'indices':{'sha256':'c'}}
        checks={key:True for key in ['physical_compatibility','unique_geometry','mother_prefix_exact',
                'holdout_disjoint_from_mother','actual_15ghz_labels','holdout_frozen_before_current_study_training',
                'holdout_not_used_for_current_study_gradient_or_tuning']}
        q={'status':'PASS','combined_csv_sha256':'a','mother_csv_sha256':'b','indices_sha256':'c',
           'checks':checks,'independence_scope':'CURRENT_STUDY_ONLY','historical_exposure_status':'UNKNOWN',
           'family_lineage_status':'GLOBAL_LHS_SUBSET_ONLY','known_family_overlap':False,
           'source_method_family_assessment':{'status':'UNKNOWN','evidence':[]}}
        with self.assertRaisesRegex(ValueError,'UNKNOWN does not pass'):a.require_qualification(c,q)
        q['source_method_family_assessment']={'status':'PASS_INDEPENDENT_GEOMETRY_PROPOSALS','evidence':['SYNTHETIC_TEST_ONLY']}
        a.require_qualification(c,q)
        q['known_family_overlap']=True
        with self.assertRaisesRegex(ValueError,'known-family overlap'):a.require_qualification(c,q)

    def test_completed_callback_on_synthetic_summaries_only(self):
        import run_all
        with tempfile.TemporaryDirectory(prefix='eucap_adapter_callback_test_') as folder:
            root=Path(folder)
            for n in a.COUNTS:
                out=root/'runs'/f'train_{n}';out.mkdir(parents=True)
                summary={'split_audit':{'row_counts':{'train':n,'validation':2,'test':2},
                         'global_split_index_sha256':{'validation':'same-v','test':'same-t'}},
                         'optimizer_budget_contract':{'realized':{'exact_update_budget_pass':True,
                          'forward_optimizer_updates':3600,'inverse_optimizer_updates':3600}},
                         'validation_metrics':{'per_feature_forward_proxy_physical_mae':{'q':.1},
                           'per_feature_physical_mae':{'q':.2},
                           'forward_proxy':{'validation_range_normalized_rmse':.3},
                           'tandem_inverse':{'validation_response_range_normalized_rmse':.4}},
                         'input_columns':['q'],'best_optimizer_updates':{'forward_proxy':100,'tandem_inverse':120},
                         'weights_npz':'SYNTHETIC_ONLY','weights_npz_sha256':'synthetic',
                         'validation_predictions_csv':'SYNTHETIC_ONLY','validation_predictions_csv_sha256':'synthetic',
                         'history_csv':'SYNTHETIC_ONLY','history_csv_sha256':'synthetic'}
                sp=out/'physical_feature_tandem_inverse_summary.json';sp.write_text(json.dumps(summary))
                (out/'ENTRY_TERMINAL.json').write_text(json.dumps({'status':'COMPLETE','summary_sha256':a.sha(sp)}))
                (out/'ENTRY_IDENTITY.json').write_text(json.dumps({'independence_scope':'SYNTHETIC_ONLY',
                  'historical_exposure_status':'SYNTHETIC_ONLY','family_lineage_status':'SYNTHETIC_ONLY'}))
            dest=Path(run_all.aggregate(root))
            self.assertTrue((dest/'TRAIN_SIZE_VALIDATION_COMPARISON.csv').is_file())
            self.assertEqual(a.read(dest/'COMPLETION.json')['csv_rows'],3)
            with self.assertRaises(FileExistsError):run_all.aggregate(root)

    def test_exact_recipe_parse_and_honest_split_label(self):
        runtime=a.load_runtime()
        c={'combined_csv':{'path':'/synthetic/not-read.csv'}}
        argv=a.trainer_argv(c,Path('/synthetic/not-created'))
        a.install_adapter(runtime,self.indices,2,8,'synthetic')
        args=runtime._parse_args(argv)
        self.assertEqual(args.forward_hidden_widths,'256,256,128')
        self.assertEqual(args.inverse_hidden_widths,'512,512,256')
        self.assertEqual(args.inverse_geometry_projection,'hard_feasible_topology_v1')
        self.assertEqual(args.q_target_semantics,'minimum')
        self.assertEqual(args.q_minimum_margin_physical,2)
        self.assertEqual(args.inverse_checkpoint_selection,'physical_cell_tail_strict_pass_rate')
        self.assertEqual(args.forward_max_optimizer_updates,3600)
        self.assertEqual(args.inverse_max_optimizer_updates,3600)
        self.assertEqual(args.evaluation_mode,'validation_only')
        self.assertEqual(args.split_mode,a.SPLIT_MODE)

    def test_validation_cell_objective_unchanged_without_false_ood(self):
        m=a.select_matrix(self.matrix(),self.indices,2,8)
        split,audit=a.explicit_split(m,self.indices,2,'synthetic')
        legacy_audit=dict(audit,split_mode='physical_cell_grouped')
        target=m['x'][split['validation']];passed=np.array([False])
        expected=self.runtime._physical_cell_strict_pass_summary(target,passed,legacy_audit)
        runtime=a.load_runtime();a.install_adapter(runtime,self.indices,2,8,'synthetic')
        actual=runtime._physical_cell_strict_pass_summary(target,passed,audit)
        self.assertEqual(actual,expected)
        self.assertEqual(audit['split_mode'],a.SPLIT_MODE)

    def test_frozen_forward_gradient_retained_no_weights_changed(self):
        # Synthetic network only; no optimizer or training is called.
        runtime=self.runtime
        weights=[np.array([[.2,-.1,.3],[.3,.2,-.2]]),np.array([[.2],[.4],[-.1]])]
        biases=[np.zeros(3),np.zeros(1)]
        before=[w.copy() for w in weights]
        x=np.array([[.15,-.3]])
        y,_,pre=runtime._forward_with_cache(x,weights,biases)
        gradient=runtime._backward_input_gradient(np.ones_like(y),weights,pre)
        numeric=np.zeros_like(x)
        eps=1e-6
        for j in range(x.shape[1]):
            plus=x.copy();minus=x.copy();plus[0,j]+=eps;minus[0,j]-=eps
            numeric[0,j]=(runtime._predict(plus,weights,biases).sum()-runtime._predict(minus,weights,biases).sum())/(2*eps)
        np.testing.assert_allclose(gradient,numeric,rtol=1e-5,atol=1e-8)
        self.assertGreater(np.abs(gradient).max(),0)
        for b,w in zip(before,weights):np.testing.assert_array_equal(b,w)


if __name__=='__main__':
    unittest.main(verbosity=2)
