"""Bounded synthetic unit checks; never loads or evaluates a trained model."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import postprocess as p


class PostprocessChecks(unittest.TestCase):
    def test_symmetric_q_and_exact_tie(self):
        target = np.array([[1., 1., 15., .5], [1., 1., 15., .5]])
        proxy = target.copy()
        proxy[:, 2] += [-1., 1.]
        np.testing.assert_array_equal(p.score_symmetric(proxy, target), [.025, .025])
        targets = np.repeat(target[:1, None], 11, axis=1)
        _, finite, chosen = p.select_candidates(np.ones((1, 11, 10)), targets.copy(), targets)
        self.assertTrue(finite.all())
        self.assertEqual(int(p.Q[chosen[0]]), 10)

    def test_invalid_candidates_do_not_win_and_failed_rows_remain(self):
        targets = np.ones((2, 11, 4))
        geometry = np.ones((2, 11, 10))
        proxy = targets.copy()
        geometry[0, 0] = np.nan
        proxy[1] = np.nan
        _, finite, chosen = p.select_candidates(geometry, proxy, targets)
        self.assertEqual(chosen.tolist(), [1, 0])
        self.assertEqual(finite[np.arange(2), chosen].tolist(), [True, False])
        selected = np.array([[1., 1., 11., .5], [1., 1., 10., .5]])
        rows, matrices = p.grid_rows(selected, np.array([True, False]))
        for matrix in matrices.values():
            self.assertEqual(int(matrix[2].sum()), 2)
            self.assertEqual(int(matrix[3].sum()), 1)

    def test_absent_or_failed_terminal_prevents_input_access(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(p.postprocess(td, {})["status"], "NOT_READY")
            Path(td, "ENTRY_TERMINAL.json").write_text(json.dumps({"status": "FAIL"}))
            self.assertEqual(p.postprocess(td, {})["status"], "NOT_READY")
            self.assertFalse(Path(td, "analysis").exists())

    def test_binding_and_wrong_recipe_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td, "input.csv")
            src.write_text("frozen input\n")
            with self.assertRaises(ValueError):
                p.pin_check({"path": str(src), "sha256": "0"*64})
            self.assertEqual(p.pin_check({"path": str(src), "sha256": p.sha(src)}), src.resolve())
            summary = Path(td, "summary.json")
            summary.write_text(json.dumps({"arguments": {"seed": 20260711, "inverse_geometry_projection": "independent_sigmoid", "q_target_semantics": "minimum", "q_minimum_margin_physical": 2., "geometry_anchor_weight": .01, "response_loss_family": "mse", "response_loss_scaling": "declared_range", "topology_feasibility_weight": 0}}))
            with self.assertRaisesRegex(ValueError, "Recipe mismatch"):
                p.load_model(Path(td), Path(td, "NO_LOADER"), summary, Path(td, "NO_WEIGHTS"))
            self.assertFalse(Path(td, "model_manifest.json").exists())

    def test_terminal_summary_does_not_authorize_mismatched_weights(self):
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td, "physical_feature_tandem_inverse_summary.json")
            summary.write_text(json.dumps({"weights_npz_sha256": "0"*64}))
            Path(td, "physical_feature_tandem_inverse_weights.npz").write_bytes(b"synthetic wrong weight identity")
            Path(td, "ENTRY_TERMINAL.json").write_text(json.dumps({"status": "COMPLETE", "summary_sha256": p.sha(summary)}))
            with self.assertRaisesRegex(ValueError, "weights_npz_sha256 mismatch"):
                p.postprocess(td, {})
            self.assertFalse(Path(td, "analysis").exists())


if __name__ == "__main__":
    unittest.main()
