"""Presentation-only revision from the frozen forward-curve CSV."""
import csv, hashlib, json, runpy
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
source=ROOT/'output/forward_learning_curve.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='e1d46e48426735b746ba935be1d8c1ea52f3ffc6f8ee9fcd4fab31d37d02d538'
with source.open(newline='') as f: rows=list(csv.DictReader(f))
u=np.array([int(r['optimizer_updates']) for r in rows])
t=np.array([float(r['train_feature_balanced_nrmse']) for r in rows])
v=np.array([float(r['validation_feature_balanced_nrmse']) for r in rows])
assert len(rows)==60 and u[-1]==1200 and np.isfinite(t).all() and np.isfinite(v).all()
out=ROOT/'clean_v2';out.mkdir(exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.facecolor':'white','figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'100k-forward-clean-v2'})
fig,ax=plt.subplots(figsize=(10,5.7))
fig.subplots_adjust(left=.115,right=.94,bottom=.145,top=.87)
ax.set_title('100K forward model: learning curve',loc='left',fontsize=19,pad=18,weight='bold',color='#142738')
ax.plot(u,t,color='#1765ad',lw=2.5,label='Training')
ax.plot(u,v,color='#cf7320',lw=2.5,ls='--',label='Validation')
ax.scatter([u[-1],u[-1]],[t[-1],v[-1]],c=['#1765ad','#cf7320'],s=48,zorder=5)
ax.annotate(f'{v[-1]:.4f}',(u[-1],v[-1]),xytext=(9,10),textcoords='offset points',fontsize=16,weight='bold',color='#b55e10',va='center')
ax.annotate(f'{t[-1]:.4f}',(u[-1],t[-1]),xytext=(9,-10),textcoords='offset points',fontsize=16,weight='bold',color='#1765ad',va='center')
ax.set(xlim=(0,1370),ylim=(0,.52),xlabel='Optimizer updates',ylabel='Feature-balanced normalized RMSE')
ax.set_xticks(np.arange(0,1201,200))
ax.grid(color='#e8ebee',lw=.7)
ax.legend(frameon=False,loc='upper right',fontsize=12)
for spine in ['top','right']:ax.spines[spine].set_visible(False)
helper=ROOT.parent/'legacy10k_forward_evidence_20260921_v1/plot_saved_forward.py'
runpy.run_path(str(helper))['save'](fig,out,'forward_learning_curve_clean')
m=json.loads((ROOT/'output/MANIFEST.json').read_text())
meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),model_id=m['model_id'],weights_sha256=m['model_weights_sha256'],source_csv=str(source),source_sha256=sha(source),parent_manifest_sha256=sha(ROOT/'output/MANIFEST.json'),script_sha256=sha(Path(__file__)),helper_sha256=sha(helper),source_train_validation_test=m['source_train_validation_test'],metric='Feature-balanced normalized RMSE; not MSE or accuracy',first_logged_update=int(u[0]),last_update=int(u[-1]),first_train=float(t[0]),final_train=float(t[-1]),first_validation=float(v[0]),final_validation=float(v[-1]),change='Only presentation: remove metadata/footer/checkpoint text; label final curve endpoints. All 60 data points unchanged.',new_inference_training_emx=0,visual_review='PENDING')
with (out/'MANIFEST.json').open('x') as f:json.dump(meta,f,indent=2)
with (out/'SHA256SUMS').open('x') as f:
    for p in sorted(out.iterdir()):
        if p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
print(json.dumps({'output':str(out),'train':float(t[-1]),'validation':float(v[-1])}))
