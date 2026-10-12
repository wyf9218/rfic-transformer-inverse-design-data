"""Bounded non-BLAS arithmetic cross-check of observed NumPy matmul warnings."""
import importlib.util
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import run_qscan as run


def main():
    out = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('qscan_loader_numeric_check', run.LOADER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.FrozenInverseModel.load(run.BUNDLE)
    data = np.load(out / 'all_candidates.npz', allow_pickle=False)
    indices = np.array([0, 1, 137, 512, 999, 2500, 4999, 7500, 9876, 9999])
    targets = data['requests'][indices].reshape(-1, 4)
    n = model.model['normalization']
    def einsum_predict(values, weights, biases):
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            values = np.einsum('ij,jk->ik', values, weight, optimize=False) + bias
            if i != len(weights)-1:
                values = module._gelu(values)
        return values
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all='raise'):
            raw = einsum_predict((targets-n['x_mean'])/n['x_scale'],model.model['inverse_weights'],model.model['inverse_biases'])
            normg = module._project_inverse_geometry(raw,n['geometry_lower'],n['geometry_upper'],
                mode=model.model['projection_mode'],normalization=n,topology_contract=model.model['topology_contract'])
            geometry = normg*n['y_scale']+n['y_mean']
            proxy = einsum_predict(normg,model.model['forward_weights'],model.model['forward_biases'])*n['x_scale']+n['x_mean']
    oldg = data['geometry'][indices].reshape(-1,10)
    oldp = data['proxy'][indices].reshape(-1,4)
    np.testing.assert_allclose(geometry,oldg,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(proxy,oldp,rtol=1e-11,atol=1e-11)
    selected = np.argmin(run.score_symmetric(proxy,targets).reshape(-1,11),axis=1)
    np.testing.assert_array_equal(selected,data['selected_index'][indices])
    result = dict(status='PASS',utc=run.utc(),script_sha256=run.sha(__file__),
        archive_sha256=run.sha(out / 'all_candidates.npz'),
        observation='Original NumPy matmul emitted divide/overflow/invalid RuntimeWarnings. Retained here, not silently suppressed.',
        full_archive_all_finite=bool(np.isfinite(data['geometry']).all() and np.isfinite(data['proxy']).all() and np.isfinite(data['scores']).all()),
        check_scope='110 candidate computations only, deterministic 10 requests x 11 Q; not full inference rerun',
        sampled_request_indices=indices.tolist(),method='np.einsum optimize=False bypasses matmul/BLAS; original GELU/decoder retained; errstate all=raise',
        warnings=[str(x.message) for x in caught],max_abs_geometry_difference=float(np.max(abs(geometry-oldg))),
        max_abs_proxy_difference=float(np.max(abs(proxy-oldp))),sample_selected_Q_exact=True,
        interpretation='No numerical divergence found in sampled alternative arithmetic or nonfinite values in full outputs. Underlying backend warning cause not independently established.')
    run.write_json(out / 'NUMERIC_BACKEND_CHECK.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
