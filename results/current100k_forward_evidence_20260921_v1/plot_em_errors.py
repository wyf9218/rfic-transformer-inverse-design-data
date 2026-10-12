"""Annotate saved forward predictions against held-out EM labels, without inference."""
import csv, hashlib, json, runpy
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'legacy10k_recipe_on_original100k_20260921_v1/received_report_20260921T2011Z'
SOURCE=ROOT/'forward_test_predictions.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
c=json.loads((OLD/'COMPLETE.json').read_text())
assert c['identity']['weights_sha256']=='ddc3476184911007132728722ae5fb0a00d1183ee867af80cbc6ccaf9dfc805b'
assert sha(SOURCE)==c['outputs']['common_test11369/predictions.csv']
assert sha(OLD/'common_test11369/errors.csv')==c['outputs']['common_test11369/errors.csv']
rows=read(SOURCE);expected=read(OLD/'common_test11369/errors.csv')
assert len(rows)==11369 and len({r['source_row_index_0based'] for r in rows})==11369
keys=['lp_nh','ls_nh','q_scalar','k_abs']
labels=[r'$L_p$ (nH)',r'$L_s$ (nH)',r'$Q_{min}$',r'$|k|$']
x=np.array([[float(r['truth_'+k]) for k in keys] for r in rows])
y=np.array([[float(r['prediction_'+k]) for k in keys] for r in rows])
assert np.isfinite(x).all() and np.isfinite(y).all()
assert (x!=0).all(), 'Do not silently include zero labels in relative-error means'
metrics=[]
for j,k in enumerate(keys):
    error=y[:,j]-x[:,j];relative=np.abs(error)/np.abs(x[:,j])*100
    m=dict(feature=k,unit='nH' if j<2 else 'dimensionless',n=11369,n_relative=11369,
        MAE=float(np.mean(abs(error))),RMSE=float(np.sqrt(np.mean(error**2))),
        MAPE_pct=float(relative.mean()),P95_relative_error_pct=float(np.percentile(relative,95)),
        P95_absolute_error=float(np.percentile(abs(error),95)),hits_10pct=int((relative<=10).sum()))
    for dest,src in [('MAE','MAE'),('RMSE','RMSE'),('MAPE_pct','mean_relative_error_pct'),('P95_relative_error_pct','P95_relative_error_pct'),('P95_absolute_error','P95_absolute_error'),('hits_10pct','hits_10pct')]:
        assert np.isclose(m[dest],float(expected[j][src]),rtol=1e-12,atol=1e-12),(k,dest)
    metrics.append(m)
out=ROOT/'em_error_labels_v1';out.mkdir(exist_ok=False)
with (out/'forward_vs_em_errors.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':'white','figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'100k-forward-em-errors-v1'})
fig,axs=plt.subplots(2,2,figsize=(10.8,9.8))
fig.subplots_adjust(left=.10,right=.965,bottom=.09,top=.84,wspace=.34,hspace=.36)
fig.text(.10,.945,'100K forward model: prediction vs. EM labels',fontsize=18,weight='bold',color='#142738')
fig.text(.10,.9,'Held-out test: N = 11,369  |  Dashed diagonal: perfect prediction',fontsize=12,color='#52616c')
for j,ax in enumerate(axs.flat):
    lo=min(x[:,j].min(),y[:,j].min());hi=max(x[:,j].max(),y[:,j].max());pad=(hi-lo)*.06
    ax.scatter(x[:,j],y[:,j],s=5,color='#1765ad',alpha=.22,edgecolors='none',rasterized=True,zorder=2)
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#303030',ls='--',lw=1.2,zorder=3)
    ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='EM label: '+labels[j],ylabel='Forward prediction: '+labels[j])
    ax.set_aspect('equal',adjustable='box')
    m=metrics[j];unit=' nH' if j<2 else ''
    ax.text(.04,.96,f'MAPE = {m["MAPE_pct"]:.2f}%',transform=ax.transAxes,va='top',fontsize=14,weight='bold',color='#123b61',bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
    ax.text(.04,.855,f'MAE = {m["MAE"]:.4f}{unit}\nRMSE = {m["RMSE"]:.4f}{unit}',transform=ax.transAxes,va='top',fontsize=10.5,color='#303030',bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
    ax.grid(color='#e8ebee',lw=.6)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
helper=ROOT.parent/'legacy10k_forward_evidence_20260921_v1/plot_saved_forward.py'
runpy.run_path(str(helper))['save'](fig,out,'forward_prediction_vs_em_errors')
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),model_id='legacyrecipe_D100k_seed20260711',weights_sha256=c['identity']['weights_sha256'],sources={str(p):sha(p) for p in [SOURCE,OLD/'COMPLETE.json',OLD/'common_test11369/errors.csv',helper,Path(__file__)]},metrics=metrics,MAPE_definition='100 * mean(abs(prediction - EM_label) / abs(EM_label)); no zero denominators in this data',evidence='EXISTING_REAL_EM_LABEL_HOLDOUT_NOT_FRESH_EM',new_inference_training_emx=0,visual_review='PENDING')
with (out/'MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
with (out/'CAPTION_EN.md').open('x') as f:f.write('Forward predictions versus existing EM labels on 11,369 held-out geometries for the 100K recipe-reconstruction model (seed20260711). MAE and RMSE use the displayed physical units; MAPE is mean absolute percentage error relative to the EM label. All rows are finite and nonzero. This is not fresh-EMX validation of newly generated inverse geometries. Source/train/validation/test: 100000/78891/9740/11369. The original 64 physical cases belong to a different model and are not combined here.\n')
with (out/'SHA256SUMS').open('x') as f:
    for p in sorted(out.iterdir()):
        if p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
print(json.dumps({'output':str(out),'metrics':metrics}))
