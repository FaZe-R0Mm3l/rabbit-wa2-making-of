"""Two short alternatives. Existing PNG assets remain byte-identical.
Procedural strokes and native-pixel shadow layers are rendered only to video.
"""
from pathlib import Path
from fractions import Fraction
import sys,json,math,subprocess,hashlib
import av,cv2,numpy as np,imageio_ffmpeg
import revise_full_v2 as rev
full=rev.full;base=full.base
ROOT=full.ROOT;OUT=ROOT/'outputs';WORK=ROOT/'work/bridge_samples'
WORK.mkdir(exist_ok=True)
TB=Fraction(1,16000);W,H=1920,1080
REF=OUT/'rabbit_白色相簿2_全片_v2少复用版.mp4'
RANGES=[(full.snap(33.033),full.snap(36.745)),(full.snap(85.502),full.snap(89.715))]
# High-frequency spectral-flux peaks from original audio, snapped to source frames.
PEAKS=[[34.113015873,34.658684807,35.204353741],[86.476916100,87.022585034,87.565351474]]
HITS=[[full.snap(t) for t in group] for group in PEAKS]
BRIDGES=[(33.9089375,35.8691875),(85.91925,88.2131875)]
SHADOW_PATH=OUT/'assets/三人影子_原图.png'
SHADOW_HASH=hashlib.sha256(SHADOW_PATH.read_bytes()).hexdigest()
sheet=cv2.imdecode(np.fromfile(SHADOW_PATH,np.uint8),cv2.IMREAD_UNCHANGED)
assert sheet.shape[2]==4 and sheet[:,:,3].min()==0
_,labels,stats,_=cv2.connectedComponentsWithStats((sheet[:,:,3]>14).astype(np.uint8))
boxes=sorted([s for s in stats[1:] if s[4]>10000],key=lambda s:s[0])
assert len(boxes)==3,boxes
SHADOWS=[]
for x,y,w,h,area in boxes:
 # Read-only native rectangular crop: no extraction, resampling or saved PNG edit.
 x0=max(0,x-4);y0=max(0,y-4)
 SHADOWS.append(sheet[y0:min(sheet.shape[0],y+h+4),x0:min(sheet.shape[1],x+w+4)])

COLORS=[(24,22,21),(53,99,158),(110,63,34)] # BGR: Haruki, Setsuna, Kazusa

def ease(x):
 x=max(0.,min(1.,x));return 1-(1-x)**3

def stage(t,section):
 hits=HITS[section];k=sum(t>=h for h in hits)-1
 age=max(0,t-hits[k]) if k>=0 else 1
 return k,age

def rough_curve(dst,pts,color,width=14,seed=0):
 pts=np.asarray(pts,dtype=np.float32)
 j=np.arange(len(pts),dtype=np.float32)
 off=np.column_stack((.65*np.sin(j*1.27+seed),1.25*np.sin(j*.61+seed)))
 p=np.rint(pts+off).astype(np.int32)
 cv2.polylines(dst,[p],False,tuple(int(c*.62) for c in color),width+3,cv2.LINE_AA)
 cv2.polylines(dst,[p],False,color,width,cv2.LINE_AA)
 # A narrow discontinuous pencil highlight leaves a coarse, hand-drawn edge.
 hi=tuple(min(255,c+29) for c in color)
 for a in range(0,len(p)-5,19):
  q=p[a:a+12]+np.array([0,-3],np.int32)
  cv2.polylines(dst,[q],False,hi,2,cv2.LINE_AA)

def braided(dst,x0,x1,y0,amplitude=24,phase=0,tilt=0):
 x=np.linspace(x0,x1,240);u=np.linspace(0,1,240)
 for i,c in [(0,COLORS[0]),(2,COLORS[2])]:
  y=y0+tilt*(u-.5)+amplitude*np.sin(u*math.pi*3.5+phase+(0 if i==0 else math.pi))
  rough_curve(dst,np.column_stack((x,y)),c,14,i+2)

def intact(dst,t,tight=False):
 x=np.linspace(365,1555,240);u=np.linspace(0,1,240)
 a=32 if tight else 86
 for i,c in [(0,COLORS[0]),(1,COLORS[1]),(2,COLORS[2])]:
  y=447+a*np.sin(u*math.pi*3.2+i*2.094)+10*np.sin(u*math.pi)
  rough_curve(dst,np.column_stack((x,y)),c,14,i+2)

def separated(dst,age,wide=False,fade=False):
 e=ease(age/.16);g=(145 if wide else 65)*e
 braided(dst,870+g*.55,1580+g*.15,482,22,tilt=-25)
 x=np.linspace(290-g*.45,790-g*.4,160);u=np.linspace(0,1,160)
 y=410-63*np.sin(u*math.pi)+14*u
 layer=dst.copy()
 rough_curve(layer,np.column_stack((x,y)),COLORS[1],14,4)
 # Broken end frays and short recoil marks are visible on the cut's first frames.
 end=np.array([x[-1],y[-1]])
 for j in range(3):
  p=np.array([end+[0,j*3-3],end+[20+j*6,(j-1)*8],end+[38+j*5,(j-1)*13]],np.int32)
  cv2.polylines(layer,[p],False,COLORS[1],3,cv2.LINE_AA)
 alpha=max(.20,1-age*.65) if fade else 1
 cv2.addWeighted(layer,alpha,dst,1-alpha,0,dst)
 if age<.11:
  for j in range(4):
   a=j*math.pi/2+.3;cx,cy=850,440
   cv2.line(dst,(round(cx+25*math.cos(a)),round(cy+25*math.sin(a))),(round(cx+48*math.cos(a)),round(cy+48*math.sin(a))),COLORS[1],3,cv2.LINE_AA)

def lines(dst,t,section):
 k,age=stage(t,section)
 if section==0:
  if k<0:intact(dst,t)
  elif k==0:intact(dst,t,True)
  elif k==1:
   braided(dst,380,1560,472,25)
   e=ease(age/.13);gap=35+90*e
   x=np.linspace(370,960-gap,130);u=np.linspace(0,1,130)
   rough_curve(dst,np.column_stack((x,405+35*np.sin(u*math.pi))),COLORS[1],14,4)
   x=np.linspace(960+gap,1550,100);u=np.linspace(0,1,100)
   rough_curve(dst,np.column_stack((x,470+35*np.sin(u*math.pi))),COLORS[1],14,4)
  else:separated(dst,age,True)
 else:
  if k<0:separated(dst,1,True)
  elif k==0:
   braided(dst,780,1575,480,25)
   x=np.linspace(380,945,160);u=np.linspace(0,1,160)
   rough_curve(dst,np.column_stack((x,411+58*u+32*np.sin(u*math.pi))),COLORS[1],14,4)
  elif k==1:
   braided(dst,745,1575,480,25)
   x=np.linspace(390,986,180);u=np.linspace(0,1,180)
   rough_curve(dst,np.column_stack((x,410+62*u+26*np.sin(u*math.pi))),COLORS[1],14,4)
   # The brown end touches the pair, but curls away instead of joining.
   a=np.linspace(0,math.pi*1.2,70)
   rough_curve(dst,np.column_stack((986+35*np.sin(a),472-35*(1-np.cos(a)))),COLORS[1],14,6)
  else:separated(dst,age,True,True)

def overlay_alpha(dst,src,x,y,opacity):
 h,w=src.shape[:2];x0=max(0,x);x1=min(W,x+w);y0=max(0,y);y1=min(912,y+h)
 if x0>=x1 or y0>=y1:return
 p=src[y0-y:y1-y,x0-x:x1-x];a=p[:,:,3:4].astype(np.float32)*(opacity/255)
 dst[y0:y1,x0:x1]=(dst[y0:y1,x0:x1]*(1-a)+p[:,:,:3]*a).astype(np.uint8)

def shadows(dst,t,section):
 k,age=stage(t,section);e=ease(age/.15)
 # Shadows retain every native source pixel; only position and video opacity vary.
 if section==0:
  before=[560,855,1055];targets=[520,855,1055] if k==0 else [360,925,1110] if k==1 else [175,1000,1180]
  if k<0:positions=before
  else:
   previous=before if k==0 else [520,855,1055] if k==1 else [360,925,1110]
   positions=[round(a+(b-a)*e) for a,b in zip(previous,targets)]
  opacity=[.88,.88,.88]
 else:
  previous=[175,1000,1180] if k<=0 else [410,920,1100] if k==1 else [525,865,1050]
  targets=[410,920,1100] if k==0 else [525,865,1050] if k==1 else [170,1000,1180]
  positions=[175,1000,1180] if k<0 else [round(a+(b-a)*e) for a,b in zip(previous,targets)]
  opacity=[max(.17,.88-age*.9) if k==2 else .88,.88,.88]
 for i in [1,2,0]:
  layer=SHADOWS[i];overlay_alpha(dst,layer,positions[i],max(6,(912-layer.shape[0])//2),opacity[i])

def compose(raw,m,kind,section):
 t=m['t'];a,b=BRIDGES[section]
 if a<=t<b:
  content=base.scene('yellow_paper')
  (lines if kind=='lines' else shadows)(content,t,section)
  # Short decaying impact shake, shared by both alternatives.
  k,age=stage(t,section)
  if k>=0 and age<.13:
   dx=round(7*math.sin(age*85)*math.exp(-age*25))
   content=np.roll(content,dx,axis=1)
  frame=np.zeros((H,W,3),np.uint8);frame[84:996]=content;base.snow(frame,t)
  return frame
 if section==1 and 88.3383125<=t<92.7594375:
  frame=np.zeros((H,W,3),np.uint8)
  content=base.scene('kazusa_bouquet_teal',dx=round(-16+(t-88.4634375)*8))
  full.petals(content,t,True)
  if t<88.4634375:
   _,g=cv2.threshold(cv2.cvtColor(content,cv2.COLOR_BGR2GRAY),130,255,cv2.THRESH_BINARY)
   content=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
  frame[84:996]=content
  if t>=88.4634375:rev.caption(frame,t)
  base.snow(frame,t);return frame
 return raw.copy()

def load_context(a,b):
 c=av.open(str(REF));s=c.streams.video[0];c.seek(max(0,int((a-.5)/float(s.time_base))),stream=s)
 result={}
 for f in c.decode(s):
  t=float(f.pts*s.time_base)
  if t<a-1e-5:continue
  if t>=b-1e-5:break
  result[round(t*16000)]=f.to_ndarray(format='bgr24')
 c.close();return result

def main():
 contexts=[load_context(a,b) for a,b in RANGES]
 meta=[ [m for m in full.FRAMES if a<=m['t']<b] for a,b in RANGES]
 report={'ranges':RANGES,'audio_peaks':PEAKS,'visual_hits':HITS,'hit_errors_ms':[[round(1000*(v-a),3) for v,a in zip(vs,aa)] for vs,aa in zip(HITS,PEAKS)],'shadow_sha256':SHADOW_HASH,'native_shadow_bounds':[s.tolist() for s in boxes],'outputs':[]}
 for kind,label in [('lines','三色细线'),('shadows','三人影子')]:
  silent=WORK/(kind+'_silent.mp4');out=av.open(str(silent),'w')
  stream=out.add_stream('libx264',rate=Fraction(24000,1001));stream.width=W;stream.height=H
  stream.pix_fmt='yuv420p';stream.time_base=TB;stream.codec_context.time_base=TB
  stream.options={'crf':'12','preset':'fast','bf':'0'}
  durations={};records=[];offset=0;frame_count=0
  def encode(image,pts,duration):
   nonlocal frame_count
   durations[pts]=duration
   v=av.VideoFrame.from_ndarray(image,format='bgr24');v.pts=pts;v.duration=duration;v.time_base=TB
   for p in stream.encode(v):
    if p.pts in durations:p.duration=durations[p.pts]
    out.mux(p)
   frame_count+=1
  for section,((a,b),group,context) in enumerate(zip(RANGES,meta,contexts)):
   first=round(a*16000)
   for m in group:
    image=compose(context[m['pts']],m,kind,section)
    encode(image,m['pts']-first+offset,m['duration'])
    records.append({'output_pts':m['pts']-first+offset,'source_time':m['t'],'section':section})
   offset+=round((b-a)*16000)
   if section==0:
    for j in range(12):encode(np.zeros((H,W,3),np.uint8),offset+j*667,667)
    offset+=12*667
  for p in stream.encode():
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  out.close()
  dest=OUT/('rabbit_关系破裂_'+label+'_对比样片.mp4')
  a0,b0=RANGES[0];a1,b1=RANGES[1];gap=12*667/16000
  filt=f'[1:a]atrim=start={a0}:end={b0},asetpts=PTS-STARTPTS[a0];[1:a]atrim=start={a1}:end={b1},asetpts=PTS-STARTPTS[a1];anullsrc=r=44100:cl=stereo,atrim=duration={gap}[gap];[a0][gap][a1]concat=n=3:v=0:a=1[a]'
  subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(silent),'-i',str(base.SOURCE),'-filter_complex',filt,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-b:a','320k','-movflags','+faststart',str(dest)],check=True)
  for section in range(2):
   for j,t in enumerate(HITS[section]):
    m=min(full.FRAMES,key=lambda f:abs(f['t']-(t+.13)))
    image=compose(contexts[section][m['pts']],m,kind,section)
    cv2.imencode('.png',image)[1].tofile(WORK/f'{kind}_{section}_{j}.png')
  chk=av.open(str(dest));frames=list(chk.decode(video=0));duration=chk.duration/av.time_base;chk.close()
  assert len(frames)==frame_count
  assert abs(duration-offset/16000)<.05,(duration,offset/16000)
  assert all(frames[i].pts<frames[i+1].pts for i in range(len(frames)-1))
  report['outputs'].append({'file':str(dest),'frames':frame_count,'duration':duration,'video_end':offset/16000,'bytes':dest.stat().st_size})
  (WORK/(kind+'_timeline.json')).write_text(json.dumps(records,indent=2))
  print('DONE',label,duration,frame_count,flush=True)
 assert hashlib.sha256(SHADOW_PATH.read_bytes()).hexdigest()==SHADOW_HASH
 (OUT/'关系破裂_两版样片_卡点校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
