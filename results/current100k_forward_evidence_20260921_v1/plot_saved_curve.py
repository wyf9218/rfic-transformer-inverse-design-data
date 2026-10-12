"""Plot saved 100K forward history only; no inference or training."""
import csv, hashlib, json, runpy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
HELPER=ROOT.parent/'legacy10k_forward_evidence_20260921_v1/plot_saved_forward.py'
save=runpy.run_path(str(HELPER))['save']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
summary_path=ROOT/'physical_feature_tandem_inverse_summary.json'
history_path=ROOT/'physical_feature_tandem_inverse_history.csv'
s=json.loads(summary_path.read_text())
assert sha(summary_path)=='dd2773ba3eba618d7628ac22ddb9b7d99ed55742c5786799a4eb2b54a1dae76b'
assert sha(history_path)==s['history_csv_sha256']
receipt=ROOT.parent/'legacy10k_recipe_on_original100k_20260921_v1/received_report_20260921T2011Z/COMPLETE.json'
c=json.loads(receipt.read_text())
assert c['identity']['summary_sha256']==sha(summary_path)
assert c['identity']['weights_sha256']==s['weights_npz_sha256']
for ext in ['png','svg']:
    assert sha(ROOT/('forward_scatter.'+ext))==c['outputs']['common_test11369/forward_scatter.'+ext]
with history_path.open(newline='') as f: h=[r for r in csv.DictReader(f) if r['stage']=='forward_proxy']
u=np.array([int(r['optimizer_updates']) for r in h])
t=np.array([float(r['train_feature_balanced_normalized_rmse']) for r in h])
v=np.array([float(r['validation_feature_balanced_normalized_rmse']) for r in h])
best=s['best_optimizer_updates']['forward_proxy']
assert len(h)==60 and u[-1]==1200 and best==u[v.argmin()]
assert np.isfinite(t).all() and np.isfinite(v).all()
out=ROOT/'output';out.mkdir(exist_ok=False)
with (out/'forward_learning_curve.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['optimizer_updates','train_feature_balanced_nrmse','validation_feature_balanced_nrmse']);w.writerows(zip(u,t,v))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':'white','figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'current100k-forward-v1'})
fig,ax=plt.subplots(figsize=(10,6))
fig.subplots_adjust(left=.12,right=.965,bottom=.235,top=.76)
fig.text(.12,.925,'100K model: forward-network learning curve',fontsize=17,weight='bold',color='#142738')
fig.text(.12,.87,'21 Sep 2026 recipe reconstruction | seed20260711 | 15 GHz | 3 × 256, GELU',fontsize=11,color='#52616c')
ax.plot(u,t,color='#1765ad',lw=2,label='Training (N=78,891)')
ax.plot(u,v,color='#cf7320',ls='--',lw=2,label='Validation (N=9,740)')
ax.axvline(best,color='#777777',lw=.9,ls=':')
ax.scatter([best,best],[t[-1],v[-1]],c=['#1765ad','#cf7320'],s=35,zorder=5)
ax.set(xlim=(0,1240),ylim=(0,.52),xlabel='Optimizer updates',ylabel='Feature-balanced normalized RMSE')
ax.grid(color='#e8ebee',lw=.6);ax.legend(frameon=False,loc='upper right')
for spine in ['top','right']:ax.spines[spine].set_visible(False)
ax.text(.43,.82,f'Selected checkpoint: update {best:,}\nTrain: {t[-1]:.4f}  |  Validation: {v[-1]:.4f}',transform=ax.transAxes,fontsize=10.5,va='top',color='#334455')
fig.text(.12,.115,'Source table: 100,000 rows; test: 11,369. Test was not used for checkpoint selection.',fontsize=10,color='#52616c')
fig.text(.12,.07,'Saved logs only. Lower is better; not accuracy %. The 1,200-update budget is not proof of convergence.',fontsize=9.5,color='#52616c')
save(fig,out,'forward_learning_curve')
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),model_id='legacyrecipe_D100k_seed20260711',model_weights_sha256=s['weights_npz_sha256'],source_train_validation_test=[100000,78891,9740,11369],best_forward_update=best,final_train_nrmse=float(t[-1]),final_validation_nrmse=float(v[-1]),metric='original logged feature-balanced normalized RMSE',sources={str(p):sha(p) for p in [summary_path,history_path,receipt,HELPER,Path(__file__),ROOT/'forward_scatter.png',ROOT/'forward_scatter.svg']},new_training_inference_emx=0,scatter_evidence='EXISTING_REAL_EM_LABEL_HOLDOUT_NOT_FRESH_EM',text_bounds='PASS',visual_review='PENDING')
with (out/'MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
with (out/'SHA256SUMS').open('x') as f:
    for p in sorted(out.iterdir()):
        if p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
print(json.dumps({'output':str(out),'train_nrmse':float(t[-1]),'validation_nrmse':float(v[-1]),'best_update':best}))
