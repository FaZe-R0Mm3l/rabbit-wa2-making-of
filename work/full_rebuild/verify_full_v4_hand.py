"""Check palm-out portrait replacement and unchanged full-film synchronization."""
from pathlib import Path
import json,hashlib,itertools
import av,cv2,numpy as np
import revise_full_v4_hand as edit
OUT=edit.OUT;WORK=edit.WORK;full=edit.full
old_path=OUT/'rabbit_白色相簿2_全片_v3六图闪回版.mp4'
records=json.loads((OUT/'全片v4_逐帧时间表.json').read_text(encoding='utf-8'))
old_records=edit.old_records
assert len(records)==len(old_records)==3502
assert [r['pts'] for r in records]==[r['pts'] for r in old_records]
assert [r['black'] for r in records]==[r['black'] for r in old_records]
for old,new in zip(old_records,records):
 assert new['shot']==('m_intro_setsuna_palm_out' if old['shot']=='m_intro_setsuna' else old['shot'])
changed=[(name,list(g)) for name,g in itertools.groupby(records,key=lambda r:r['shot']) if name=='m_intro_setsuna_palm_out']
assert len(changed)==3 and sum(len(g) for _,g in changed)==1040
flashplan=json.loads((OUT/'全片v3_六图闪回与卡点表.json').read_text(encoding='utf-8'))
wanted=[full.snap(t) for t in [5,74,130,20,29,39,49,59,69,79.8,90,99,109,119,141]]
wanted.extend(c['visible_start'] for c in flashplan['six_flashbacks'])
need={round(t*16000) for t in wanted};pictures={};pts=[];blacks={}
c=av.open(str(edit.TARGET));stream=c.streams.video[0]
meta={'width':stream.width,'height':stream.height,'frames':stream.frames,'duration':c.duration/av.time_base}
for i,f in enumerate(c.decode(video=0)):
 p=round(float(f.time)*16000);pts.append(p);a=f.to_ndarray(format='bgr24')
 blacks[p]=float(a[84:996].mean())<8
 assert blacks[p]==records[i]['black']
 if p in need:pictures[p]=a
c.close()
assert pts==[m['pts'] for m in full.FRAMES]
assert len(pictures)==len(need)
def audio(path):
 c=av.open(str(path));h=hashlib.sha256();samples=0;first=None;last=None
 for f in c.decode(audio=0):
  h.update(f.to_ndarray().tobytes());samples+=f.samples
  if first is None:first=float(f.time)
  last=float(f.time)
 c.close();return {'sha256':h.hexdigest(),'samples':samples,'first_pts':first,'last_pts':last}
current_audio=audio(edit.TARGET);assert current_audio==audio(edit.base.SOURCE)
print('Full-film frame PTS, blackouts and original audio passed',flush=True)
replacement=[]
for t in [5,74,130]:
 p=round(full.snap(t)*16000);reference=cv2.imdecode(np.fromfile(WORK/f'preview_{t}.png',np.uint8),cv2.IMREAD_COLOR)
 mae=float(np.abs(pictures[p].astype(np.int16)-reference.astype(np.int16)).mean())
 assert mae<5,(t,mae)
 cv2.imencode('.png',pictures[p])[1].tofile(WORK/f'encoded_{t}.png')
 replacement.append({'time':p/16000,'corrected_composite_mae':round(mae,5)})
old=av.open(str(old_path));s=old.streams.video[0];unchanged=[]
for t in wanted:
 p=round(t*16000)
 if p in edit.CHANGED_PTS:continue
 old.seek(max(0,p-800),stream=s)
 for f in old.decode(s):
  q=round(float(f.time)*16000)
  if q==p:
   a=f.to_ndarray(format='bgr24');mae=float(np.abs(pictures[p].astype(np.int16)-a.astype(np.int16)).mean())
   assert mae<6,(t,mae)
   unchanged.append({'time':t,'mae':round(mae,5)});break
  if q>p:raise AssertionError(('Missing previous frame',t))
old.close()
flash_checks=[]
for cue in flashplan['six_flashbacks']:
 p=round(cue['visible_start']*16000);idx=pts.index(p)
 assert not blacks[p] and all(blacks[q] for q in pts[idx-3:idx])
 assert records[idx]['shot']=='user_flash_'+str(cue['attachment_index']+1)
 flash_checks.append({'time':p/16000,'unchanged':True,'preceding_black_frames':3})
for path,digest in edit.HASHES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
result={'status':'verified','output':str(edit.TARGET),'metadata':meta,'corrected_ranges':[{'first':g[0]['t'],'last':g[-1]['t'],'frames':len(g)} for _,g in changed],'changed_frames':1040,'original_pose_with_palm_facing_outward':True,'replacement_samples':replacement,'unchanged_picture_samples':unchanged,'all_3502_frame_timestamps_match_original':True,'blackouts_unchanged':True,'six_flashbacks':flash_checks,'audio_identical_to_original':True,'audio':current_audio,'original_and_fixed_PNG_hashes':edit.HASHES}
(OUT/'全片v4_手部与同步校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['unchanged_picture_samples','original_and_fixed_PNG_hashes']},ensure_ascii=False,indent=2),flush=True)
