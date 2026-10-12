#!/usr/bin/env python3
"""Small artifact/source checks only; never opens model weights or the100K CSV."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent.parent
R=OUT.parents[2]
FIG1=R/'reports/eucap15ghz_20260908T220300Z/fig1_motivation_revision_20260921T071650Z/delivery_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
declared={line.split(None,1)[1].strip():line.split()[0] for line in (FIG1/'SHA256SUMS.txt').read_text().splitlines() if line.strip()}
refs=[]
for name in ['Fig1_DesignMotivation.svg','Fig1_DesignMotivation.png','Method_Table.md','DELIVERY_RECEIPT.json']:
    p=FIG1/name;expected=declared.get(name) or declared.get('./'+name)
    if expected is not None and sha(p)!=expected:raise ValueError('Original Fig1 changed: '+name)
    refs.append({'path':str(p),'sha256':sha(p),'matches_existing_checksum':expected is not None,'action':'READ_ONLY_REUSE_REFERENCE'})
with (OUT/'workflow_nodes.csv').open() as f:nodes=list(csv.DictReader(f))
with (OUT/'workflow_edges.csv').open() as f:edges=list(csv.DictReader(f))
ids={n['node_id'] for n in nodes}
assert len(nodes)==35 and len(ids)==35 and len(edges)==42
assert all(e['source_node'] in ids and e['target_node'] in ids for e in edges)
assert any(e['source_node']=='I_TOTAL' and e['target_node']=='I_NET' and e['edge_kind']=='gradient' for e in edges)
svg=ET.parse(OUT/'Fig2_ActualMethodFlow.svg').getroot()
assert len(svg.findall('.//{http://www.w3.org/2000/svg}text'))>=30
layout=json.loads((OUT/'LAYOUT_CHECK.json').read_text())
assert layout['body_font_pt']==8 and abs(layout['width_mm']-184.2734722222222)<1e-8 and layout['text_in_frame']
receipt={'status':'DELIVERED_SOURCE_BOUND_FIGURE','utc':datetime.now(timezone.utc).isoformat(),'model_id':'current_foundry_qmin_response_only_seed20260713','model_weight_sha256_existing_identity':'c61a93d56136ad21686863aae72984c49eb0d33ce766923c8ec2d03d9c3177dd','weight_rehash_this_work_package':False,'Fig1':refs,'Fig2':{'width_mm':layout['width_mm'],'height_mm':layout['height_mm'],'font_pt':8,'nodes':35,'edges':42,'editable_SVG_text':True,'actual_PNG_visually_reviewed':True,'visual_review_note':'Final 144dpi paper-width preview inspected after rerouting the inverse derivative and moving its label below the shared branch; no label or arrow crosses a node caption. Full450dpi PNG generated from the same vector scene.'},'new_training':0,'new_inference':0,'new_native_solver_calls':0,'remote_calls':0,'active_files_modified':False,'scientific_boundary':'Method diagram, not experiment completion, target-hit or acceleration evidence. Historical seed13 loss/decoder preserved; separate dataset-total and width-only protocols in caption.'}
with (OUT/'DELIVERY_CHECK.json').open('x') as f:json.dump(receipt,f,indent=2,ensure_ascii=False);f.write('\n')
names=['README.md','Fig2_Caption_and_Methods.md','Fig2_ActualMethodFlow.svg','Fig2_ActualMethodFlow.png','Fig2_ActualMethodFlow_PaperSize.png','workflow_nodes.csv','workflow_edges.csv','METHOD_SOURCES.json','LAYOUT_CHECK.json','DELIVERY_CHECK.json','scripts/build_workflow_tables.mjs','scripts/draw_method_workflow.py','scripts/finalize_method_delivery.py']
with (OUT/'SHA256SUMS').open('x') as f:
    for name in names:f.write(sha(OUT/name)+'  '+name+'\n')
print(json.dumps({'status':receipt['status'],'out':str(OUT),'nodes':len(nodes),'edges':len(edges),'files_hashed':len(names),'receipt_sha256':sha(OUT/'DELIVERY_CHECK.json'),'svg_sha256':sha(OUT/'Fig2_ActualMethodFlow.svg')}))
