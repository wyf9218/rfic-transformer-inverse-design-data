#!/usr/bin/env python3
"""Ordinary editable SVG and PNG from source-backed method nodes; no models run."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent.parent
WIDTH_MM=184.2734722222222
W=WIDTH_MM/25.4*72
H=423
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'svg.fonttype':'none','savefig.facecolor':'white'})
fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
ink='#202B35';line='#536372';blue='#EAF1F8';frozen='#E8F1EB';loss='#FFF0DF';grad='#A33931';nodes={};texts=[]

def text(x,y,s,size=8,color=ink,weight='normal',ha='center'):
    obj=ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',linespacing=1.14)
    texts.append(obj);return obj

def box(key,x,y,w,h,label,fill=blue):
    nodes[key]=(x,y,w,h)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',facecolor=fill,edgecolor=line,linewidth=.65))
    text(x+w/2,y+h/2,label)

def port(key,side):
    x,y,w,h=nodes[key]
    return {'left':(x,y+h/2),'right':(x+w,y+h/2),'top':(x+w/2,y),'bottom':(x+w/2,y+h)}[side]

def arrow(a,b,color=line,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=6,linewidth=.7,color=color,linestyle=(0,(3,2)) if dashed else '-'))

def link(a,b):arrow(port(a,'right'),port(b,'left'))
def route(points,color=line,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.7,ls=(0,(3,2)) if dashed else '-')
    arrow(points[-2],points[-1],color,dashed)

def panel(y,label):
    text(7,y,label,8.5,weight='bold',ha='left')
    ax.plot([7,W-7],[y+9,y+9],color='#CED5DC',lw=.45)

panel(12,'(a) Existing data and fixed preprocessing')
box('source',7,29,160,31,'Original 100K geometries\nExisting 15-GHz EM labels')
box('split',183,29,159,31,'Preserved geometry partition\nTrain / validation / test')
box('norm',358,29,W-365,31,'Train-fitted scalers Sg, Sr\nPaired geometry / response')
link('source','split');link('split','norm')

panel(79,'(b) Forward training: update F, then select and freeze using validation')
box('fg',7,96,70,33,'Real geometry g\n10D (µm)')
box('sg',88,96,71,33,'Sg\nStandardize')
box('f',170,96,127,33,'Forward F\n10–256–256–128–4')
box('fr',308,96,83,33,'Sr inverse\nFour responses')
box('fl',402,96,W-409,33,'Relative-MSE\nStored EM labels',loss)
for a,b in [('fg','sg'),('sg','f'),('f','fr'),('fr','fl')]:link(a,b)
route([port('fl','bottom'),(457,145),(233.5,145),port('f','bottom')],grad,True)
text(343,141,'Adam updates F',8,grad)
text(7,156,'F fits all four EM labels, including exact Qmin.  No EM solver is called in a gradient update.',8,ha='left')

panel(174,'(c) Inverse training: update I through the frozen F input derivative')
box('it',7,191,88,35,'Training targets\nLp, Ls, Qreq, |k|')
box('i',107,191,128,35,'Sr, then inverse I\n4–512–512–256–10')
box('dec',247,191,84,35,'Original decoder\nNormalized g',blue)
box('if',343,191,70,35,'Frozen F\nfrom (b)',frozen)
box('il',425,191,W-432,35,'Sr inverse\nResponse loss',loss)
for a,b in [('it','i'),('i','dec'),('dec','if'),('if','il')]:link(a,b)
box('anchor',139,242,137,29,'Geometry anchor: 0.01\nPaired original geometry',loss)
box('topology',290,242,127,29,'Topology penalty: 0.02\nLabel-free constraints',loss)
route([port('dec','bottom'),(289,235),(207.5,235),port('anchor','top')])
route([port('dec','bottom'),(289,235),(353.5,235),port('topology','top')])
text(7,250,'Q loss only:\nshortfall to\nQreq + 2',8,ha='left')
route([port('it','top'),(51,186),(471,186),port('il','top')],line,True)
route([port('il','bottom'),(471,282),(119,282),(119,233),(171,233),port('i','bottom')],grad,True)
arrow(port('anchor','bottom'),(207.5,282),grad,True)
arrow(port('topology','bottom'),(353.5,282),grad,True)
text(325,288,'Combined gradient to I; F weights stay frozen',8,grad)
text(7,301,'I objective: λ(t)Lresponse + 0.01Lanchor + 0.02Ltopology; only I weights/biases are updated.',8,ha='left')

panel(316,'(d) One-shot design and separate physical verification')
box('ut',7,333,86,33,'Requested target\nQreq is minimum')
box('ui',104,333,94,33,'Saved Sr\nFrozen inverse I',frozen)
box('ud',209,333,107,33,'Original decoder\nSg inverse to µm')
box('ug',327,333,85,33,'One geometry\nNo search')
box('uf',425,333,W-432,33,'Frozen-F\ndiagnostic',frozen)
for a,b in [('ut','ui'),('ui','ud'),('ud','ug')]:link(a,b)
arrow(port('ug','right'),port('uf','left'),color='#668773',dashed=True)
box('layout',109,386,104,29,'Original layout / GDS')
box('drc',225,386,64,29,'DRC gate')
box('em',301,386,79,29,'EM validation')
box('compare',392,386,W-399,29,'Compare to target\nRetain failures')
route([port('ug','bottom'),(369.5,376),(161,376),port('layout','top')])
for a,b in [('layout','drc'),('drc','em'),('em','compare')]:link(a,b)
text(7,401,'No Q scan\nor extra +2\nat inference',8,ha='left')

fig.canvas.draw()
renderer=fig.canvas.get_renderer();bounds=fig.bbox
outside=[]
for obj in texts:
    t=obj.get_window_extent(renderer)
    if t.x0<bounds.x0-1 or t.x1>bounds.x1+1 or t.y0<bounds.y0-1 or t.y1>bounds.y1+1:outside.append(obj.get_text())
if outside:raise RuntimeError('Out-of-frame text: '+repr(outside))
for ext,dpi in [('svg',None),('png',450)]:
    p=OUT/('Fig2_ActualMethodFlow.'+ext)
    if p.exists():raise FileExistsError(p)
    fig.savefig(p,dpi=dpi)
preview=OUT/'Fig2_ActualMethodFlow_PaperSize.png'
if preview.exists():raise FileExistsError(preview)
fig.savefig(preview,dpi=144)
plt.close(fig)
qa={'schema':'method_figure_layout.v1','width_mm':WIDTH_MM,'height_mm':H/72*25.4,'body_font_pt':8,'panel_font_pt':8.5,'font_family':'Times New Roman','vector_text_editable':True,'new_training':0,'new_inference':0,'new_EM':0,'text_in_frame':not outside,'visual_review':'PENDING','source_id':'current_foundry_qmin_response_only_seed20260713','decoder':'hard_feasible_topology_v1','layout_notes':'Four stages; dashed red derivative route updates only designated neural weights. Green diagnostic branch does not select or modify geometry.'}
(OUT/'LAYOUT_CHECK.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa))
