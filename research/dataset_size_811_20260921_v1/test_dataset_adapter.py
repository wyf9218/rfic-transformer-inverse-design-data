#!/usr/bin/env python3
"""Three targeted read-only adapter checks; no training or inference call."""
import copy
from pathlib import Path
import unittest
import numpy as np
import train_dataset_sizes as a


class DatasetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract,cls.indices,cls.q=a.load_contract(a.STUDY/'data/DATASET_SIZE_CONTRACT.json')
        cls.runtime=a.c.load_runtime()
        cls.argv=a.trainer_argv(cls.contract,Path('/not-created/adapter-check'))
        cls.args=cls.runtime._parse_args(cls.argv)
        rows=cls.runtime._read_rows(Path(cls.contract['mother']['path']))
        args=cls.args
        inputs=[x.strip() for x in args.input_columns.split(',') if x.strip()]
        refs=[x.strip() for x in (args.split_reference_columns or args.input_columns).split(',') if x.strip()]
        geometry=cls.runtime._resolve_geometry_columns(rows,args.geometry_columns,args.geometry_prefix)
        cls.full=cls.runtime._build_matrix(rows,inputs,geometry,refs)

    def test_reject_cross_partition_and_non_nested_indices(self):
        bad=copy.deepcopy(self.indices);bad['D10k']['train'][0]=bad['D10k']['test'][0]
        with self.assertRaises(ValueError):a.validate_indices(bad)
        bad=copy.deepcopy(self.indices);bad['common_test']=list(reversed(bad['common_test']))
        with self.assertRaises(ValueError):a.validate_indices(bad)

    def test_actual_matrix_mapping_partition_cells_train_only_normalization(self):
        for arm,counts in a.ARMS.items():
            matrix=a.select_matrix(self.full,self.indices,arm)
            split,audit=a.explicit_split(matrix,self.indices,arm,'TARGETED_READ_ONLY_CHECK')
            self.assertEqual(matrix['count'],sum(counts))
            self.assertEqual(tuple(audit['row_counts'][p] for p in a.PARTS),counts)
            self.assertEqual(audit['physical_cell_overlap_count'],0)
            for part in a.PARTS:
                self.assertEqual([matrix['source_indices'][i] for i in split[part]],self.indices[arm][part])
            isolation=self.runtime._evaluation_isolation_contract(matrix,split,'validation_only')
            self.assertEqual(isolation['overall_status'],'PASS')
            data=self.runtime._normalize(matrix,split,self.args.normalization_floor)
            np.testing.assert_allclose(data['normalization']['x_mean'],matrix['x'][split['train']].mean(axis=0),rtol=0,atol=0)
            np.testing.assert_allclose(data['normalization']['y_mean'],matrix['y'][split['train']].mean(axis=0),rtol=0,atol=0)

    def test_original_scientific_recipe_retained(self):
        for arm,counts in a.ARMS.items():
            runtime=a.c.load_runtime();a.install_adapter(runtime,self.indices,arm,'TARGETED_READ_ONLY_CHECK')
            args=runtime._parse_args(self.argv)
            self.assertEqual(args.forward_hidden_widths,'256,256,128')
            self.assertEqual(args.inverse_hidden_widths,'512,512,256')
            self.assertEqual(args.seed,20260713)
            self.assertEqual(args.q_target_semantics,'minimum')
            self.assertEqual(args.q_minimum_margin_physical,2)
            self.assertEqual(args.inverse_geometry_projection,'hard_feasible_topology_v1')
            self.assertEqual(args.inverse_checkpoint_selection,'physical_cell_tail_strict_pass_rate')
            self.assertEqual((args.forward_max_optimizer_updates,args.inverse_max_optimizer_updates),(3600,3600))
            self.assertEqual(args.evaluation_mode,'validation_only')
            self.assertEqual(args.stage_checkpoint_mode,'resume_exact')
            self.assertAlmostEqual(args.validation_fraction,counts[1]/sum(counts))
            self.assertAlmostEqual(args.test_fraction,counts[2]/sum(counts))


if __name__=='__main__':unittest.main(verbosity=2)
