"""Private, no-clobber figure/data delivery. Does not infer or render."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP = Path('/Users/wyf/Downloads/EuCAP_100K_Qscan_10000_Hotspots_20260921.zip')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification = json.loads((ROOT/'VERIFICATION.json').read_text())
    if verification['status'] != 'PASS':
        raise ValueError('Independent saved-data verification not PASS')
    visual = json.loads((ROOT/'figures/VISUAL_REVIEW.json').read_text())
    for name, expected in visual['accepted'].items():
        assert sha(ROOT/'figures'/name) == expected
    names = ['README_CN.md','CONTRACT.json','SUMMARY.json','STARTED.json',
             'IMPLEMENTATION_CHECK.json','NUMERIC_BACKEND_CHECK.json','VERIFICATION.json',
             'VERIFICATION_ATTEMPT_01_FAIL.json','run_qscan.py','check_numeric_backend.py',
             'verify_saved_results.py','plot_qscan.py','package_delivery.py',
             'selected_10000_results.csv','all_candidates.npz','hotspot_bins.csv',
             'physical_error_summary.csv','selected_Q_counts.csv',
             'figures/CAPTIONS_EN.md','figures/VISUAL_REVIEW.json','figures/RENDER_RECEIPT.json']
    for stem in ['Qscan100k_target_hotspots','Qscan100k_target_counts','Qscan100k_performance_summary']:
        names += ['figures/'+stem+ext for ext in ['.png','.svg']]
    names += ['figures/selected_q_layout_v2/'+x for x in
              ['SelectedQ_K_hotspot.png','SelectedQ_K_hotspot.svg','CAPTIONS_EN.md','RENDER_RECEIPT.json']]
    sources = {name:sha(ROOT/name) for name in sorted(names)}
    with (ROOT/'DELIVERY_SHA256SUMS').open('x') as f:
        f.write(''.join(f'{digest}  {name}\n' for name,digest in sources.items()))
    names.append('DELIVERY_SHA256SUMS')
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(names):
            z.write(ROOT/name,arcname=name)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name, digest in sources.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    receipt=dict(status='COMPLETE',path=str(ZIP),sha256=sha(ZIP),bytes=ZIP.stat().st_size,
                 files=len(names),crc_and_member_hashes='PASS',
                 inference='110000 candidates for 10000 preserved triplets',
                 new_EMX=0,public_upload=False,excluded='Large weights, raw100K, duplicate all-candidate CSV.gz, superseded overlapping Q chart, credentials and PDK')
    with (ROOT/'DELIVERY.json').open('x') as f:
        json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
