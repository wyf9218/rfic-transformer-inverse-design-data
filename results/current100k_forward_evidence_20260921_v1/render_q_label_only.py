"""Re-use the frozen plotting implementation, changing display notation only."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
source = root / 'plot_em_errors.py'
code = source.read_text()
assert code.count("r'$Q_{min}$'") == 1
assert code.count("ROOT/'em_error_labels_v1'") == 1
code = code.replace("r'$Q_{min}$'", "r'$Q$'")
code = code.replace("ROOT/'em_error_labels_v1'", "ROOT/'em_error_labels_q_v2'")
# The original source file and its historical products remain untouched.
exec(compile(code, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
out = root / 'em_error_labels_q_v2'
old = root / 'em_error_labels_v1'
assert (out/'forward_vs_em_errors.csv').read_bytes() == (old/'forward_vs_em_errors.csv').read_bytes()
svg = (out/'forward_prediction_vs_em_errors.svg').read_text()
assert 'Q_{min}' not in svg and '>min</tspan>' not in svg
assert '<!-- EM label: $Q$ -->' in svg and '<!-- Forward prediction: $Q$ -->' in svg
receipt = {'change': 'Display Q instead of Qmin; labels only',
    'definition_unchanged': 'Q = min(Qp, Qs); exact target matching, not a minimum-Q requirement',
    'raw_schema_unchanged': True, 'metrics_csv_byte_identical': True,
    'source_script': str(source), 'source_script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'revision_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'new_training_inference_emx': 0}
with (out/'LABEL_CHANGE.json').open('x') as f: json.dump(receipt, f, indent=2)
