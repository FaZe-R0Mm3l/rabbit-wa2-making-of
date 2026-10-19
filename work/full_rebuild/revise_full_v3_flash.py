"""User-selected six flashbacks in the full film, replacing the abandoned rope.
Input PNGs are copied byte-for-byte and composited at their native pixel sizes.
"""
from pathlib import Path
from fractions import Fraction
import json,math,hashlib,shutil,subprocess,sys
import av,cv2,numpy as np,imageio_ffmpeg
import revise_full_v2 as rev
full=rev.full;base=full.base;ROOT=full.ROOT;OUT=full.OUT
WORK=ROOT/'work/full_revision_v3';WORK.mkdir(exist_ok=True)
ASSETS=OUT/'assets/full_revision_v3';ASSETS.mkdir(exist_ok=True)
TARGET=OUT/'rabbit_白色相簿2_全片_v3六图闪回版.mp4'
TB=full.TB;W,H=full.W,full.H
ATTACHMENTS=[
 ('codex-clipboard-87a785ca-59df-410a-8bb0-863443f97001.png','闪回1_雪菜舞台.png'),
 ('codex-clipboard-e1d10f16-123e-4038-af13-75bace037ff2.png','闪回2_和纱钢琴.png'),
 ('codex-clipboard-b52af3bb-83d7-460b-a773-6cbeb4dcda4f.png','闪回3_春希吉他.png'),
 ('codex-clipboard-156c39ae-2c9c-4758-830f-aec330632250.png','闪回4_和纱抱毯.png'),
 ('codex-clipboard-1bdbcd32-2c3d-4a5f-b56a-8c12388c2d89.png','闪回5_雪菜电话.png'),
 ('codex-clipboard-4e6bc568-8e22-4356-9f95-6280409d15de.png','闪回6_春希背影.png')]
SOURCE_DIR=Path(r'LOCAL_PATH/Temp')
IMAGES=[];MANIFEST=[]
for original,name in ATTACHMENTS:
 dst=ASSETS/name;src=dst if dst.exists() else SOURCE_DIR/original
 digest=hashlib.sha256(src.read_bytes()).hexdigest()
 if dst.exists():assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
 else:shutil.copy2(src,dst)
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
 a=cv2.imdecode(np.fromfile(dst,np.uint8),cv2.IMREAD_UNCHANGED)
 assert a.shape[2]==4 and a[:,:,3].min()==0,(name,a.shape)
 IMAGES.append(a)
 MANIFEST.append({'source':str(src),'copy':str(dst),'sha256':digest,'native_size':[a.shape[1],a.shape[0]]})

# Original audio spectral-flux peaks. Each visible onset is the closest video frame.
PEAKS=[[34.113015873015875,34.658684807256236,35.2043537414966],
       [86.47691609977325,87.0225850340136,87.5653514739229]]
HITS=[[full.snap(t) for t in peaks] for peaks in PEAKS]
ENDS=[35.8691875,88.2131875]
TIMES=[f['t'] for f in full.FRAMES]
FLASH=[]
for section,hits in enumerate(HITS):
 for j,onset in enumerate(hits):
  idx=TIMES.index(onset);black_start=TIMES[idx-3]
  if j<2:
   end=TIMES[TIMES.index(hits[j+1])-3]
  else:end=ENDS[section]
  FLASH.append({'section':section,'attachment_index':section*3+j,'black_start':black_start,
   'visible_start':onset,'visible_end':end,'audio_peak':PEAKS[section][j],
   'asset':MANIFEST[section*3+j]['copy'],'offset_ms':round((onset-PEAKS[section][j])*1000,4)})
RANGES=[(33.9089375,ENDS[0]),(FLASH[3]['black_start'],ENDS[1])]

def picture(m,cue):
 t=m['t'];age=t-cue['visible_start'];a=IMAGES[cue['attachment_index']]
 frame=np.zeros((H,W,3),np.uint8);content=base.scene('yellow_paper')
 kick=round(6*math.sin(age*29)*math.exp(-age*15))
 x=(W-a.shape[1])//2+kick
 # Preserve the head and native pixels; only the source's lower canvas may leave
 # the 912 px composition stage, as with existing approved full-film portraits.
 y=max(0,(912-a.shape[0])//2)
 base.overlay_native(content,a,x,y)
 frame[84:996]=content;base.snow(frame,t)
 return frame

def compose(m,source=None):
 t=m['t']
 for cue in FLASH:
  if cue['black_start']<=t<cue['visible_start']:
   return np.zeros((H,W,3),np.uint8),'flash_black'
  if cue['visible_start']<=t<cue['visible_end']:
   return picture(m,cue),'user_flash_'+str(cue['attachment_index']+1)
 # Hold the preceding complete illustration until the newly aligned first blackout.
 if RANGES[0][0]<=t<FLASH[0]['black_start']:
  frame=np.zeros((H,W,3),np.uint8)
  frame[84:996]=base.scene('kazusa_piano_yellow',dx=round(3*math.sin(t*2)))
  base.snow(frame,t);return frame,'kazusa_piano_yellow'
 # Retain the requested blue bouquet at 1:28, including its brief ink transition.
 if 88.3383125<=t<92.7594375:
  frame=np.zeros((H,W,3),np.uint8)
  content=base.scene('kazusa_bouquet_teal',dx=round(-16+(t-88.4634375)*8))
  full.petals(content,t,True)
  if t<88.4634375:
   _,g=cv2.threshold(cv2.cvtColor(content,cv2.COLOR_BGR2GRAY),130,255,cv2.THRESH_BINARY)
   content=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
  frame[84:996]=content
  if t>=88.4634375:rev.caption(frame,t)
  base.snow(frame,t);return frame,'blue_bouquet_ink' if t<88.4634375 else 'kazusa_bouquet_teal_blue'
 return rev.compose(m,source)

def write_plan():
 plan={'output':str(TARGET),'six_flashbacks':FLASH,'changed_bridge_ranges':RANGES,
       'blue_bouquet_range':[88.3383125,92.7594375],'blackout_frames_before_each_image':3,
       'assets':MANIFEST,'audio':'original source AAC stream copy','rope_design':'discarded',
       'image_policy':'copy2 byte-identical originals; native-pixel alpha compositing, no resampling'}
 (OUT/'全片v3_六图闪回与卡点表.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 write_plan()
 if '--preview' in sys.argv:
  for cue in FLASH:
   m=min(full.FRAMES,key=lambda f:abs(f['t']-cue['visible_start']-.0834))
   a,n=compose(m);cv2.imencode('.png',a)[1].tofile(WORK/f"preview_flash_{cue['attachment_index']+1}.png")
  print(json.dumps({'flashbacks':FLASH,'native_sizes':[a['native_size'] for a in MANIFEST]},ensure_ascii=False,indent=2),flush=True)
  return
 durations={m['pts']:m['duration'] for m in full.FRAMES}
 silent=WORK/'全片v3_无声.mp4';out=av.open(str(silent),'w')
 s=out.add_stream('libx264',rate=Fraction(24000,1001));s.width=W;s.height=H;s.pix_fmt='yuv420p'
 s.time_base=TB;s.codec_context.time_base=TB;s.options={'crf':'14','preset':'fast','bf':'0'}
 records=[];source=av.open(str(base.SOURCE))
 for i,f in enumerate(source.decode(video=0)):
  m=full.FRAMES[i];assert f.pts==m['pts'];t=m['t']
  raw=f.to_ndarray(format='bgr24') if 18.64<t<33.91 or 70.44<t<87.68 else None
  a,name=compose(m,raw)
  v=av.VideoFrame.from_ndarray(a,format='bgr24');v.pts=m['pts'];v.duration=m['duration'];v.time_base=TB
  for p in s.encode(v):
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  records.append({'pts':m['pts'],'t':t,'black':bool(not np.any(a)),'shot':name})
  if i%240==0:print(f'Render full v3 {i}/{len(full.FRAMES)} {t:.2f}s',flush=True)
 for p in s.encode():
  if p.pts in durations:p.duration=durations[p.pts]
  out.mux(p)
 out.close();source.close()
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(silent),'-i',str(base.SOURCE),'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(TARGET)],check=True)
 (OUT/'全片v3_逐帧时间表.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',str(TARGET),flush=True)
if __name__=='__main__':main()
