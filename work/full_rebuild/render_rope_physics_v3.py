"""Thinner rope in the video; damped spring-chain recoil, reused in both passages.
Original PNG texture is read-only and is verified by SHA256 after rendering.
"""
from pathlib import Path
from fractions import Fraction
import json,math,subprocess,hashlib,sys
import av,cv2,numpy as np,imageio_ffmpeg
import render_rope_revision as art
prev=art.prev;ROOT=art.ROOT;OUT=art.OUT
WORK=ROOT/'work/rope_physics_v3';WORK.mkdir(exist_ok=True)
TARGET=OUT/'rabbit_关系破裂_麻绳弹断_v3样片.mp4'
OLD=OUT/'rabbit_关系破裂_手绘麻绳_v2样片.mp4'
W,H=1920,1080;TB=Fraction(1,16000);X=np.arange(W,dtype=np.float32)
RADIUS=39.;WIDTH_SCALE=.58
BREAK=prev.HITS[0][1];PULSE=prev.HITS[0][2]-BREAK
DT=1/480;SIM_DURATION=1.65;N=65

def base_y(x,amp=29):return 456+amp*np.sin((x-960)/165)

def simulate(direction):
 # Each half is anchored offscreen; its newly free end stores tensile energy.
 xs=np.linspace(-150,960,N) if direction<0 else np.linspace(2070,960,N)
 p=np.column_stack((xs,base_y(xs))).astype(np.float64)
 rest=np.linalg.norm(np.diff(p,axis=0),axis=1)*.975
 anchor=p[0].copy();u=np.linspace(0,1,N)
 velocity=np.column_stack((direction*160*u**5,direction*1340*u**5))
 previous=p-velocity*DT;states=[p.copy()];impulsed=False
 for step in range(1,round(SIM_DURATION/DT)+1):
  t=step*DT
  v=(p-previous)*math.exp(-1.35*DT)
  if t>=PULSE and not impulsed:
   # A second, smaller travelling recoil on the next musical accent.
   v[:,0]+=direction*150*u**7*DT;v[:,1]+=direction*290*u**7*DT
   impulsed=True
  candidate=p+v+np.array([0,220])*DT**2
  previous=p.copy();p=candidate;p[0]=anchor
  # Position-based distance constraints. Both released ends remain completely free.
  for _ in range(10):
   for parity in [0,1]:
    ids=np.arange(parity,N-1,2)
    d=p[ids+1]-p[ids];length=np.linalg.norm(d,axis=1)
    correction=d*((length-rest[ids])/np.maximum(length,1e-9))[:,None]
    weights=np.full(len(ids),.5);weights[ids==0]=0
    p[ids]+=correction*weights[:,None]
    p[ids+1]-=correction*(1-weights[:,None]);p[0]=anchor
  # Mild bending stiffness suppresses an unrealistic angular kink.
  mid=(p[:-2]+p[2:])*.5
  p[1:-1]+=(mid-p[1:-1])*.012
  p[0]=anchor;states.append(p.copy())
 return np.array(states,np.float32),rest

LEFT,REST_L=simulate(-1);RIGHT,REST_R=simulate(1)

def sample_state(states,t):
 q=np.clip(t/DT,0,len(states)-1);i=int(q);j=min(i+1,len(states)-1)
 return states[i]*(1-(q-i))+states[j]*(q-i)

def phase_time(t,section):return t-(prev.HITS[section][0]-prev.HITS[0][0])

def straight_texture(center,identity):
 p,cy=art.ROPE_TEXTURES[identity];ph,pw=p.shape[:2]
 y0=max(0,int(center.min())-100);y1=min(912,int(center.max())+100)
 yy=np.arange(y0,y1,dtype=np.float32)[:,None]
 # Cross-section width is a requested video geometry change; the source PNG
 # remains untouched. Texture sampling occurs only in the video compositor.
 sy=(yy-center[None,:])/WIDTH_SCALE+cy
 sx=np.arange(W,dtype=np.float32)%(2*pw-2)
 sx=np.where(sx<pw,sx,2*pw-2-sx)
 mx=np.broadcast_to(sx[None,:],sy.shape).astype(np.float32)
 pixel=cv2.remap(p,mx,sy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 return y0,y1,pixel[:,:,:3],pixel[:,:,3].astype(np.float32)/255

def path_texture(dst,path,identity,fray=True):
 # A textured tube follows the simulated chain, including non-monotonic curls.
 p,cy=art.ROPE_TEXTURES[identity];ph,pw=p.shape[:2]
 pad=RADIUS+15
 x0=max(0,math.floor(path[:,0].min()-pad));x1=min(W,math.ceil(path[:,0].max()+pad))
 y0=max(0,math.floor(path[:,1].min()-pad));y1=min(912,math.ceil(path[:,1].max()+pad))
 if x0>=x1 or y0>=y1:return
 shape=(y1-y0,x1-x0)
 best=np.full(shape,1e10,np.float32)
 smap=np.zeros(shape,np.float32);nmap=np.zeros(shape,np.float32)
 lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);cumulative=np.r_[0,np.cumsum(lengths)]
 for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
  lo=np.maximum([x0,y0],np.floor(np.minimum(a,b)-pad).astype(int))
  hi=np.minimum([x1,y1],np.ceil(np.maximum(a,b)+pad).astype(int))
  lx,ly=lo;hx,hy=hi
  if lx>=hx or ly>=hy:continue
  xx=np.arange(lx,hx,dtype=np.float32)[None,:];yy=np.arange(ly,hy,dtype=np.float32)[:,None]
  delta=b-a;length=max(float(lengths[i]),1e-5)
  q=np.clip(((xx-a[0])*delta[0]+(yy-a[1])*delta[1])/(length*length),0,1)
  dx=xx-(a[0]+q*delta[0]);dy=yy-(a[1]+q*delta[1]);distance=dx*dx+dy*dy
  view=best[ly-y0:hy-y0,lx-x0:hx-x0];take=distance<view
  view[take]=distance[take]
  ss=smap[ly-y0:hy-y0,lx-x0:hx-x0];nn=nmap[ly-y0:hy-y0,lx-x0:hx-x0]
  ss[take]=(cumulative[i]+q*length)[take]
  normal=(-delta[1]*(xx-a[0])+delta[0]*(yy-a[1]))/length
  nn[take]=np.broadcast_to(normal,q.shape)[take]
 orientation=1 if path[0,0]<960 else -1
 material_x=np.linspace(-150,960,len(path)) if orientation==1 else np.linspace(2070,960,len(path))
 uv=np.interp(smap.ravel(),cumulative,material_x).reshape(shape)
 mx=uv%(2*pw-2);mx=np.where(mx<pw,mx,2*pw-2-mx)
 my=nmap*orientation/WIDTH_SCALE+cy
 pixel=cv2.remap(p,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 alpha=pixel[:,:,3].astype(np.float32)/255*(best<(RADIUS+12)**2)
 # The tube is cut at its free end, with non-uniform fiber lengths.
 end=path[-1];tan=path[-1]-path[-3];tan/=max(np.linalg.norm(tan),1e-6)
 xx=np.arange(x0,x1,dtype=np.float32)[None,:]-end[0]
 yy=np.arange(y0,y1,dtype=np.float32)[:,None]-end[1]
 longitudinal=xx*tan[0]+yy*tan[1]
 across=-xx*tan[1]+yy*tan[0]
 jag=5*np.sin(across*.47)+3*np.sin(across*1.2)
 alpha*=longitudinal<jag
 roi=dst[y0:y1,x0:x1];a=alpha[:,:,None]
 roi[:]=(roi*(1-a)+pixel[:,:,:3]*a).astype(np.uint8)
 if fray:
  normal=np.array([-tan[1],tan[0]])
  # Fine recoil-bent fiber tips, aligned to the actual moving tangent.
  for j in range(23):
   v=j/22*2-1;length=10+19*(.5+.5*math.sin(j*5.3))
   points=[]
   for f in np.linspace(0,1,14):
    q=end+tan*(length*f-5)+normal*(v*31+v*9*f*f+2*math.sin(f*3+j))
    points.append(np.rint(q).astype(int))
   col=tuple(int(z) for z in np.clip(art.COLORS[1]+(j%5-2)*9,0,255))
   cv2.polylines(dst,[np.array(points,np.int32)],False,col,1 if j%3 else 2,cv2.LINE_AA)

def paint_pair(dst,centers,amp):
 layers=[straight_texture(centers[i],i) for i in [0,2]]
 z=np.full((2,912,W),-1e6,np.float32)
 for j,i in enumerate([0,2]):
  y0,y1,rgb,a=layers[j]
  n=np.arange(y0,y1,dtype=np.float32)[:,None]-centers[i][None,:]
  depth=amp*np.cos(X/172+i*2.094)[None,:]+np.sqrt(np.maximum(0,RADIUS**2-n*n))
  z[j,y0:y1]=np.where(a>.003,depth,-1e6)
 order=np.argsort(z,axis=0)
 for rank in range(2):
  for j,(y0,y1,rgb,a) in enumerate(layers):
   alpha=(a*(order[rank,y0:y1]==j))[:,:,None];roi=dst[y0:y1]
   roi[:]=(roi*(1-alpha)+rgb*alpha).astype(np.uint8)

def paint_intact(dst,t):
 tighten=prev.ease((t-prev.HITS[0][0])/.16)
 amp=140-42*tighten
 bases=[432,456,480]
 centers=[bases[i]+amp*np.sin(X/172+i*2.094) for i in range(3)]
 # Brown straightens under tension before it snaps, without a position jump.
 centers[1]=base_y(X,29+(1-tighten)*40)
 layers=[straight_texture(centers[i],i) for i in range(3)]
 z=np.full((3,912,W),-1e6,np.float32)
 for i,(y0,y1,rgb,a) in enumerate(layers):
  n=np.arange(y0,y1,dtype=np.float32)[:,None]-centers[i][None,:]
  d=amp*np.cos(X/172+i*2.094)[None,:]+np.sqrt(np.maximum(0,RADIUS**2-n*n))
  z[i,y0:y1]=np.where(a>.003,d,-1e6)
 order=np.argsort(z,axis=0)
 for rank in range(3):
  for i,(y0,y1,rgb,a) in enumerate(layers):
   alpha=(a*(order[rank,y0:y1]==i))[:,:,None];roi=dst[y0:y1]
   roi[:]=(roi*(1-alpha)+rgb*alpha).astype(np.uint8)

CACHE={}
def ropes(dst,t,section):
 phase=phase_time(t,section);tick=round(phase*16000)
 if tick in CACHE:dst[:]=CACHE[tick];return
 if phase<BREAK:paint_intact(dst,phase)
 else:
  age=phase-BREAK
  left=sample_state(LEFT,age);right=sample_state(RIGHT,age)
  path_texture(dst,left,1);path_texture(dst,right,1)
  settle=math.exp(-age*3.1)*math.sin(age*23)*5
  amp=98+settle
  bases=[432,456,480]
  centers=[bases[i]+amp*np.sin(X/172+i*2.094) for i in range(3)]
  paint_pair(dst,centers,amp)
 CACHE[tick]=dst.copy()

prev.lines=ropes

def physics_report():
 report={}
 for label,states,rest in [('left',LEFT,REST_L),('right',RIGHT,REST_R)]:
  tips=states[:,-1];vel=np.linalg.norm(np.diff(tips,axis=0),axis=1)/DT
  errors=np.abs(np.linalg.norm(np.diff(states,axis=1),axis=2)-rest[None,:])/rest[None,:]
  report[label]={'start_tip':tips[0].tolist(),'min_tip_y':float(tips[:,1].min()),'max_tip_y':float(tips[:,1].max()),'tip_at_200ms':tips[round(.2/DT)].tolist(),'tip_at_500ms':tips[round(.5/DT)].tolist(),'max_tip_speed':float(vel.max()),'mean_length_error':float(errors[10:].mean())}
 assert report['left']['min_tip_y']<130,report
 assert report['right']['max_tip_y']>780,report
 assert max(report[s]['mean_length_error'] for s in report)<.035
 return report

def render():
 info=physics_report();print(json.dumps(info),flush=True)
 contexts=[prev.load_context(a,b) for a,b in prev.RANGES]
 groups=[[m for m in prev.full.FRAMES if a<=m['t']<b] for a,b in prev.RANGES]
 for section in range(2):
  for j,age in enumerate([-.12,.0,.10,.25,.50,.85,1.15]):
   t=prev.HITS[section][1]+age;m=min(groups[section],key=lambda f:abs(f['t']-t))
   image=prev.compose(contexts[section][m['pts']],m,'lines',section)
   cv2.imencode('.png',image)[1].tofile(WORK/f'preview_{section}_{j}.png')
 if '--preview' in sys.argv:return
 silent=WORK/'rope_physics_silent.mp4';out=av.open(str(silent),'w')
 s=out.add_stream('libx264',rate=Fraction(24000,1001));s.width=W;s.height=H;s.pix_fmt='yuv420p';s.time_base=TB;s.codec_context.time_base=TB
 s.options={'crf':'12','preset':'fast','bf':'0'}
 durations={};count=0;offset=0
 def encode(a,pts,d):
  nonlocal count
  durations[pts]=d;f=av.VideoFrame.from_ndarray(a,format='bgr24');f.pts=pts;f.duration=d;f.time_base=TB
  for p in s.encode(f):
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  count+=1
 for section,((a,b),group,context) in enumerate(zip(prev.RANGES,groups,contexts)):
  for m in group:encode(prev.compose(context[m['pts']],m,'lines',section),m['pts']-round(a*16000)+offset,m['duration'])
  offset+=round((b-a)*16000)
  if section==0:
   for j in range(12):encode(np.zeros((H,W,3),np.uint8),offset+j*667,667)
   offset+=12*667
  print('Rendered',section+1,flush=True)
 for p in s.encode():
  if p.pts in durations:p.duration=durations[p.pts]
  out.mux(p)
 out.close()
 ff=imageio_ffmpeg.get_ffmpeg_exe()
 subprocess.run([ff,'-y','-v','error','-i',str(silent),'-i',str(OLD),'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(TARGET)],check=True)
 c=av.open(str(TARGET));frames=list(c.decode(video=0));duration=c.duration/av.time_base;c.close()
 c=av.open(str(OLD));oldpts=[f.pts for f in c.decode(video=0)];c.close()
 assert len(frames)==count==202 and [f.pts for f in frames]==oldpts
 def ah(p):return hashlib.sha256(subprocess.check_output([ff,'-v','error','-i',str(p),'-vn','-f','s16le','-'])).hexdigest()
 assert ah(TARGET)==ah(OLD)
 assert hashlib.sha256(art.TEXTURE_PATH.read_bytes()).hexdigest()==art.TEXTURE_HASH
 # Both passages use the same cached animation coordinates and spring simulation.
 for age in [.0,.1,.25,.5,.9]:
  a=phase_time(prev.HITS[0][1]+age,0);b=phase_time(prev.HITS[1][1]+age,1)
  assert abs(a-b)<1e-8
 report={'file':str(TARGET),'duration':duration,'frames':count,'width_scale_in_video':WIDTH_SCALE,'pair_wave_amplitude_pixels':98,'pair_vertical_center_spacing_pixels':48,'original_texture_unchanged':True,'physics':info,'same_recoil_reused_in_second_passage':True,'reconciliation_removed':True,'visual_hits':prev.HITS,'audio_identical_to_previous_sample':True}
 (OUT/'麻绳弹断_v3_校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',str(TARGET),flush=True)
if __name__=='__main__':render()
