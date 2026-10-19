from pathlib import Path
from collections import Counter
import av,numpy as np,json,hashlib,itertools
R=Path(__file__).resolve().parents[2];O=R/'outputs';W=R/'work/full_revision_v2'
video=O/'rabbit_白色相簿2_全片_v2少复用版.mp4'
sample=O/'rabbit_白色相簿2_33-54秒_v2蓝花版.mp4'
source=Path(r'LOCAL_PATH/【初音ミク】rabbit【john】.mp4')
meta=json.loads((R/'work/full_rebuild/source_frames.json').read_text())
records=json.loads((O/'全片v2_逐帧时间表.json').read_text(encoding='utf-8'))
cue=json.loads((O/'全片v2_镜头与卡点表.json').read_text(encoding='utf-8'))
reference={}
c=av.open(str(sample))
for i,f in enumerate(c.decode(video=0)):
 p=round(float(f.time)*16000)+528000
 if 528000<=p<800000 and i%24==1:
  reference[p]=f.to_ndarray(format='bgr24')[660:996:2,::2].astype(np.int16)
c.close()
pts=[];blacks=[];errors=[];saved=[];want=[29,33.05,34.1,39,41,43.4,46,49.9,50.1,51.2,52.1,111,122.5]
c=av.open(str(video));s=c.streams.video[0]
metadata={'width':s.width,'height':s.height,'duration':c.duration/av.time_base,'frame_count':s.frames,'time_base':str(s.time_base)}
for f in c.decode(video=0):
 t=float(f.time);p=round(t*16000);pts.append(p)
 a=f.to_ndarray(format='bgr24');blacks.append(float(a[84:996].mean())<8)
 if p in reference:
  errors.append({'t':t,'body_mean_absolute_pixel_difference':round(float(np.abs(a[660:996:2,::2].astype(np.int16)-reference[p]).mean()),5)})
 for requested in want:
  if requested not in saved and t>=requested:
   f.to_image().save(W/f'encoded_{requested}.png');saved.append(requested)
c.close()
def audio(path):
 h=hashlib.sha256();count=0;first=None;last=None
 c=av.open(str(path))
 for f in c.decode(audio=0):
  h.update(f.to_ndarray().tobytes());count+=f.samples
  if first is None:first=float(f.time)
  last=float(f.time)
 c.close();return {'pcm_sha256':h.hexdigest(),'samples':count,'first_pts':first,'last_pts':last}
original_audio=audio(source);output_audio=audio(video)
frame_pts=[m['pts'] for m in meta['frames']]
assert pts==frame_pts
assert blacks==[m['black'] for m in meta['frames']]
assert original_audio==output_audio
assert len(reference)==len(errors) and errors
assert max(e['body_mean_absolute_pixel_difference'] for e in errors)<5
groups=[(n,list(g)) for n,g in itertools.groupby(records,key=lambda f:f['shot'])]
reuse={}
for n in ['f16_haruki_guitar','f37_kazusa_solo']:
 uses=[g for k,g in groups if k==n]
 assert len(uses)==1
 reuse[n]={'appearances':len(uses),'start':uses[0][0]['t'],'last_frame':uses[0][-1]['t'],'frames':len(uses[0])}
flashes=[];hashes=[]
for a,b,n,word in cue['flash']:
 group=[r for r in records if a<=r['t']<b and not r['black']]
 assert group and group[0]['t']==a and all(r['shot']==n for r in group)
 path=(O/'assets' if n in ['setsuna_1','setsuna_2','kazusa_1','kazusa_2'] else O/'assets/full_pv')/(n+'.png')
 hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
 flashes.append({'start':a,'end':b,'asset':n,'lyric':word,'frames':len(group)})
assert len(set(hashes))==21
result={'output':str(video),'metadata':metadata,'source_frame_count':len(pts),'max_pts_error_seconds':0,'black_frame_mismatches':0,'audio_identical':True,'audio':output_audio,'performance_image_uses':reuse,'restored_sample_body_comparison':{'range':[33,50],'samples_compared':len(errors),'max_mean_pixel_difference':max(e['body_mean_absolute_pixel_difference'] for e in errors),'details':errors,'note':'Native lower image region compared; corrected subtitles and continuous snow clock intentionally excluded.'},'flash_timing':flashes,'unique_flash_images':21,'status':'verified'}
(O/'全片v2_校验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['flash_timing','restored_sample_body_comparison']},ensure_ascii=False,indent=2))
print('Sample comparison maximum MAE:',result['restored_sample_body_comparison']['max_mean_pixel_difference'],flush=True)
