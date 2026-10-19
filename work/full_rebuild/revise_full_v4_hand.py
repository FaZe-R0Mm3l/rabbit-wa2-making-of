"""Replace all uses of the original Setsuna bouquet portrait with palm-out grip."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,subprocess,sys
import av,cv2,numpy as np,imageio_ffmpeg
import revise_full_v3_flash as previous
full=previous.full;base=previous.base;ROOT=previous.ROOT;OUT=previous.OUT
WORK=ROOT/'work/full_revision_v4';WORK.mkdir(exist_ok=True)
TARGET=OUT/'rabbit_白色相簿2_全片_v4雪菜手部修正版.mp4'
FIXED=OUT/'assets/full_revision_v4/雪菜背身持花_手心朝外.png'
ORIGINAL=OUT/'assets/full_pv/m_intro_setsuna.png'
HASHES={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [FIXED,ORIGINAL]}
old_records=json.loads((OUT/'全片v3_逐帧时间表.json').read_text(encoding='utf-8'))
CHANGED_PTS={r['pts'] for r in old_records if r['shot']=='m_intro_setsuna'}
a=cv2.imdecode(np.fromfile(FIXED,np.uint8),cv2.IMREAD_UNCHANGED)
assert a.shape==(941,1672,4) and a[:,:,3].min()==0
ys,xs=np.where(a[:,:,3]>4)
full.LAYERS['m_intro_setsuna']=a[max(0,ys.min()-3):min(a.shape[0],ys.max()+4),max(0,xs.min()-3):min(a.shape[1],xs.max()+4)]

def compose(m,source=None):
 frame,name=previous.compose(m,source)
 return frame,('m_intro_setsuna_palm_out' if name=='m_intro_setsuna' else name)

def main():
 if '--preview' in sys.argv:
  source=av.open(str(base.SOURCE))
  for t in [5,74,130]:
   m=min(full.FRAMES,key=lambda f:abs(f['t']-t));raw=None
   if t==74:
    source.seek(round((t-.15)*16000),stream=source.streams.video[0])
    for f in source.decode(video=0):
     if f.pts==m['pts']:raw=f.to_ndarray(format='bgr24');break
   frame,name=compose(m,raw);cv2.imencode('.png',frame)[1].tofile(WORK/f'preview_{t}.png')
  source.close();print('Preview ready',flush=True);return
 durations={m['pts']:m['duration'] for m in full.FRAMES}
 silent=WORK/'全片v4_无声.mp4';out=av.open(str(silent),'w')
 s=out.add_stream('libx264',rate=Fraction(24000,1001));s.width=1920;s.height=1080;s.pix_fmt='yuv420p'
 s.time_base=full.TB;s.codec_context.time_base=full.TB;s.options={'crf':'14','preset':'fast','bf':'0'}
 records=[];source=av.open(str(base.SOURCE))
 for i,f in enumerate(source.decode(video=0)):
  m=full.FRAMES[i];assert f.pts==m['pts'];t=m['t']
  raw=f.to_ndarray(format='bgr24') if 18.64<t<33.91 or 70.44<t<87.68 else None
  frame,name=compose(m,raw)
  v=av.VideoFrame.from_ndarray(frame,format='bgr24');v.pts=m['pts'];v.duration=m['duration'];v.time_base=full.TB
  for p in s.encode(v):
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  records.append({'pts':m['pts'],'t':t,'black':bool(not np.any(frame)),'shot':name})
  if i%240==0:print(f'Render v4 {i}/{len(full.FRAMES)} {t:.2f}s',flush=True)
 for p in s.encode():
  if p.pts in durations:p.duration=durations[p.pts]
  out.mux(p)
 out.close();source.close()
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(silent),'-i',str(base.SOURCE),'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(TARGET)],check=True)
 for path,digest in HASHES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 (OUT/'全片v4_逐帧时间表.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'全片v4_立绘替换记录.json').write_text(json.dumps({'video':str(TARGET),'original':str(ORIGINAL),'replacement':str(FIXED),'updated_frames':len(CHANGED_PTS),'hashes':HASHES,'correction':'Original pose; palm facing outward; reverse underhand grip. Previous forward-grip alternatives discarded.'},ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',str(TARGET),flush=True)
if __name__=='__main__':main()
