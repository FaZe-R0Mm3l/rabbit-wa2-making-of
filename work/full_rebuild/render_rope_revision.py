"""Repaint the existing procedural video strokes as coarse twisted hemp.
No raster input is resized, re-encoded, warped, or overwritten.
"""
from pathlib import Path
from fractions import Fraction
import json,math,subprocess,hashlib,sys
import av,cv2,numpy as np,imageio_ffmpeg
import render_bridge_samples as prev
ROOT=prev.ROOT;OUT=prev.OUT;WORK=ROOT/'work/rope_revision'
WORK.mkdir(exist_ok=True)
W,H=1920,1080;TB=Fraction(1,16000)
TARGET=OUT/'rabbit_关系破裂_手绘麻绳_v2样片.mp4'
OLD=OUT/'rabbit_关系破裂_三色细线_对比样片.mp4'
X=np.arange(W,dtype=np.float32)
NOISE=prev.full.grain.astype(np.float32)
COLORS=[np.array([29,28,27],np.float32),np.array([49,91,139],np.float32),np.array([88,52,31],np.float32)]
TEXTURE_PATH=OUT/'assets/三色麻绳_手绘纹理_原图.png'
TEXTURE_HASH=hashlib.sha256(TEXTURE_PATH.read_bytes()).hexdigest()
atlas=cv2.imdecode(np.fromfile(TEXTURE_PATH,np.uint8),cv2.IMREAD_UNCHANGED)
assert atlas.shape[2]==4 and atlas[:,:,3].min()==0
ROPE_TEXTURES=[]
for a,b in [(100,350),(350,620),(620,880)]:
 p=atlas[a:b]
 mass=p[:,:,3].sum(axis=1).astype(float)
 cy=int(round(np.average(np.arange(len(p)),weights=mass)))
 ROPE_TEXTURES.append((p,cy))

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)

def geometry(t,section):
 k,age=prev.stage(t,section)
 pulse=math.exp(-age*23)*math.sin(age*65)*3 if k>=0 else 0
 if section==0:
  amp=128 if k<0 else 97 if k==0 else 103
  gap=0 if k<1 else 100+95*prev.ease(age/.12) if k==1 else 350+180*prev.ease(age/.14)
  lift=0 if k<1 else 105 if k==1 else 215
 else:
  amp=103
  gap=510 if k<0 else 200-100*prev.ease(age/.14) if k==0 else 38 if k==1 else 300+240*prev.ease(age/.15)
  lift=205 if k<0 else 125 if k==0 else 68 if k==1 else 220
 y=[]
 for i in range(3):
  y.append(457+amp*np.sin(X/172+i*2.094)+pulse)
 active=[np.ones(W,np.float32) for _ in range(3)]
 ends=[]
 if gap:
  left=960-gap/2;right=960+gap/2
  # Two ends remain anchored beyond opposite screen edges.
  left_y=450+20*np.sin(X/195+2)-lift*smooth((X+80)/(left+80))
  right_y=455+20*np.sin(X/195+2)+lift*smooth((W+80-X)/(W+80-right))
  y[1]=np.where(X<960,left_y,right_y)
  # Fiber bundles break at slightly different lengths across the rope diameter.
  active[1]=((X<left)|(X>right)).astype(np.float32)
  ends=[(left,float(np.interp(left,X,left_y)),-1),(right,float(np.interp(right,X,right_y)),1)]
 # Only brown is removed from the crossing. Black and navy stay across the frame.
 return y,active,ends,k,age

def rope_layer(center,color,active,identity):
 p,cy=ROPE_TEXTURES[identity];ph,pw=p.shape[:2]
 y0=max(0,int(center.min())-cy-2);y1=min(912,int(center.max())+ph-cy+2)
 # Video displacement uses integer column translations only: no source resampling.
 yy=np.arange(y0,y1,dtype=np.int32)[:,None]
 sy=yy-np.rint(center).astype(np.int32)[None,:]+cy
 sx=np.arange(W,dtype=np.int32)%(2*pw-2)
 sx=np.where(sx<pw,sx,2*pw-2-sx)
 pixel=p[np.clip(sy,0,ph-1),sx[None,:]]
 edge=pixel[:,:,3].astype(np.float32)/255*((sy>=0)&(sy<ph))
 if identity==1 and active.min()==0:
  left=int(np.flatnonzero(np.diff(active)<0)[0])+1
  right=int(np.flatnonzero(np.diff(active)>0)[0])+1
  n=yy-center[None,:]
  jag=9*np.sin(n*.31)+6*np.sin(n*.77)+4*np.sin(n*1.8)
  edge*=((X[None,:]<left+jag)|(X[None,:]>right+jag))
 return y0,y1,pixel[:,:,:3],edge

def draw_fibers(dst,center,color,identity,active):
 c=tuple(int(min(255,v+16)) for v in color)
 # Sparse short loose hemp fibers escape the outline; no heavy cartoon border.
 for x in range(8+identity*11,W,37):
  if not active[x]:continue
  sign=1 if (x//37+identity)%2 else -1
  y=center[x]+sign*(21+math.sin(x*.7))
  pts=np.array([[x-8,round(y-sign*2)],[x,round(y)],[x+6,round(y+sign*4)],[x+10,round(y+sign*5)]],np.int32)
  cv2.polylines(dst,[pts],False,c,1,cv2.LINE_AA)

def frayed(dst,x,y,direction,color,t):
 c=np.asarray(color)
 for j in range(17):
  v=j/16*2-1;length=27+29*(.5+.5*math.sin(j*7.31))
  # Fibers emerge from within the broken shaft, with ragged individual ends.
  startx=x+direction*(5+7*math.sin(j*2.1))
  pts=[]
  for u in np.linspace(0,1,22):
   px=startx-direction*length*u
   py=y+v*56+v*(11+6*math.sin(j))*u*u+math.sin(u*3+j)*2
   pts.append([round(px),round(py)])
  col=tuple(int(z) for z in np.clip(c+(j%4-1)*9,0,255))
  cv2.polylines(dst,[np.array(pts,np.int32)],False,col,2 if j%3 else 3,cv2.LINE_AA)

def ropes(dst,t,section):
 centers,active,ends,k,age=geometry(t,section)
 # Detached brown fibers belong behind the still-connected black/navy pair.
 for x,y,direction in ends:frayed(dst,x,y,direction,COLORS[1],t)
 layers=[rope_layer(centers[i],COLORS[i],active[i],i) for i in range(3)]
 # Cylindrical surface depth resolves crossings per pixel, avoiding a straight
 # compositing seam when two thick ropes exchange foreground/background.
 z=np.full((3,912,W),-1e6,np.float32)
 for i,(y0,y1,rgb,alpha) in enumerate(layers):
  n=np.arange(y0,y1,dtype=np.float32)[:,None]-centers[i][None,:]
  surface=np.sqrt(np.maximum(0,68**2-n*n))
  d=128*np.cos(X/172+i*2.094)[None,:]+surface
  if i==1 and ends:d=np.full_like(d,-220)+surface
  z[i,y0:y1]=np.where(alpha>.003,d,-1e6)
 order=np.argsort(z,axis=0)
 for rank in range(3):
  for i,(y0,y1,rgb,alpha) in enumerate(layers):
   a=alpha*(order[rank,y0:y1]==i)
   region=dst[y0:y1];a=a[:,:,None]
   region[:]=(region*(1-a)+rgb*a).astype(np.uint8)
 # Fine loose fibers are already painted in the original generated texture.

prev.lines=ropes

def render():
 contexts=[prev.load_context(a,b) for a,b in prev.RANGES]
 groups=[[m for m in prev.full.FRAMES if a<=m['t']<b] for a,b in prev.RANGES]
 # Native stills of the animated video geometry, for visual inspection.
 for section in range(2):
  for j,t in enumerate([prev.BRIDGES[section][0]+.08]+[h+.15 for h in prev.HITS[section]]):
   m=min(groups[section],key=lambda f:abs(f['t']-t))
   image=prev.compose(contexts[section][m['pts']],m,'lines',section)
   cv2.imencode('.png',image)[1].tofile(WORK/f'preview_{section}_{j}.png')
 if '--preview' in sys.argv:return
 silent=WORK/'rope_silent.mp4';out=av.open(str(silent),'w')
 stream=out.add_stream('libx264',rate=Fraction(24000,1001));stream.width=W;stream.height=H
 stream.pix_fmt='yuv420p';stream.time_base=TB;stream.codec_context.time_base=TB
 stream.options={'crf':'12','preset':'fast','bf':'0'}
 durations={};count=0;offset=0
 def encode(image,pts,duration):
  nonlocal count
  durations[pts]=duration
  f=av.VideoFrame.from_ndarray(image,format='bgr24');f.pts=pts;f.duration=duration;f.time_base=TB
  for p in stream.encode(f):
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  count+=1
 for section,((a,b),group,context) in enumerate(zip(prev.RANGES,groups,contexts)):
  for m in group:
   image=prev.compose(context[m['pts']],m,'lines',section)
   encode(image,m['pts']-round(a*16000)+offset,m['duration'])
  offset+=round((b-a)*16000)
  if section==0:
   for j in range(12):encode(np.zeros((H,W,3),np.uint8),offset+j*667,667)
   offset+=12*667
  print('Rendered segment',section+1,flush=True)
 for p in stream.encode():
  if p.pts in durations:p.duration=durations[p.pts]
  out.mux(p)
 out.close()
 # The already-verified audio from the previous short sample is stream-copied.
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(silent),'-i',str(OLD),'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(TARGET)],check=True)
 report={'file':str(TARGET),'ranges':prev.RANGES,'visual_hits':prev.HITS,'rope_texture':str(TEXTURE_PATH),'rope_texture_sha256':TEXTURE_HASH,'texture_method':'native-pixel integer column translation and alpha compositing','full_width':True,'original_images_modified':False,'frames':count}
 c=av.open(str(TARGET));frames=list(c.decode(video=0));report['duration']=c.duration/av.time_base;c.close()
 assert len(frames)==202 and count==202
 old=av.open(str(OLD));oldpts=[f.pts for f in old.decode(video=0)];old.close()
 assert [f.pts for f in frames]==oldpts
 def audiohash(path):
  pcm=subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-vn','-f','s16le','-'])
  return hashlib.sha256(pcm).hexdigest()
 assert audiohash(TARGET)==audiohash(OLD)
 assert hashlib.sha256(TEXTURE_PATH.read_bytes()).hexdigest()==TEXTURE_HASH
 report['audio_decoded_identical_to_previous_sample']=True
 (OUT/'手绘麻绳_v2样片_校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',str(TARGET),flush=True)
if __name__=='__main__':render()
