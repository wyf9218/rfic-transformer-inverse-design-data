"""Deterministically compose the two approved scientific figures; no recomputation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parent
names=['clean_v2/forward_learning_curve_clean','em_error_labels_q_v2/forward_prediction_vs_em_errors']
expected={
    names[0]+'.png':'eef54a17596999e689b03c2cc01afa74a84d06f973fae550d6d08f516077c1cc',
    names[0]+'.svg':'fcbec64ebe80072f813150d2ea6e9bdba0cd4f9308ed2d15809e10c051173c80',
    names[1]+'.png':'2ebe21e25886aed4a7943755eee5345bc80ecf3a8d6978f680ff218dd8354a51',
    names[1]+'.svg':'1fc0eec7b13d4e4767bbca159cc50916eb803e94e4c165312e2c2d9e7f61523f'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rel,digest in expected.items():assert sha(ROOT/rel)==digest,rel
out=ROOT/'combined_v1';out.mkdir(exist_ok=False)
images=[Image.open(ROOT/(n+'.png')).convert('RGB') for n in names]
width=max(i.width for i in images)
images=[i.resize((width,round(i.height*width/i.width)),Image.Resampling.LANCZOS) if i.width!=width else i for i in images]
canvas=Image.new('RGB',(width,sum(i.height for i in images)),'white')
y=0
for im in images:canvas.paste(im,(0,y));y+=im.height
canvas.save(out/'100k_forward_training_and_em_errors.png',dpi=(190,190))

# Keep the existing SVG text/paths and embedded scatter layers editable.
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
parts=[ET.parse(ROOT/(n+'.svg')).getroot() for n in names]
sizes=[tuple(float(v) for v in part.attrib['viewBox'].split()) for part in parts]
w=max(s[2] for s in sizes);heights=[s[3]*w/s[2] for s in sizes]
combined=ET.Element('{'+NS+'}svg',{'width':f'{w}pt','height':f'{sum(heights)}pt','viewBox':f'0 0 {w} {sum(heights)}'})
y=0
for k,(part,height) in enumerate(zip(parts,heights)):
    ids={e.attrib['id']:f'panel{k}_{e.attrib["id"]}' for e in part.iter() if 'id' in e.attrib}
    for e in part.iter():
        for attr,val in list(e.attrib.items()):
            if attr=='id':e.set(attr,ids[val]);continue
            if val.startswith('#') and val[1:] in ids:val='#'+ids[val[1:]]
            if 'url(#' in val:
                for old,new in ids.items():val=val.replace('url(#'+old+')','url(#'+new+')')
            e.set(attr,val)
    part.set('x','0');part.set('y',str(y));part.set('width',str(w));part.set('height',str(height))
    combined.append(part);y+=height
ET.ElementTree(combined).write(out/'100k_forward_training_and_em_errors.svg',encoding='utf-8',xml_declaration=True)
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'model_id':'legacyrecipe_D100k_seed20260711',
    'operation':'Compose approved original figures vertically; presentation only',
    'sources':{str(ROOT/r):s for r,s in expected.items()},'script_sha256':sha(Path(__file__)),
    'pixel_dimensions':canvas.size,'new_inference_training_emx':0,
    'evidence':'Existing 11369 held-out EM labels, not fresh-EMX validation of inverse designs',
    'Q_notation':'Q; underlying definition unchanged','visual_review':'PENDING'}
with (out/'MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
with (out/'SHA256SUMS').open('x') as f:
    for p in sorted(out.iterdir()):
        if p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
print(json.dumps({'output':str(out),'pixels':canvas.size}))
