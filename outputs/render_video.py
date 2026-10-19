from pathlib import Path
import av, cv2, numpy as np, math, random, subprocess, imageio_ffmpeg, json
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
SOURCE=Path(__import__('os').environ.get('RABBIT_SOURCE', str(Path(__file__).resolve().parents[1]/'input/source.mp4')))
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
W,H,FPS=1920,1080,24
# Absolute source time, matched to the reference's actual color-change frames.
CUTS=[(42.0,43.293,'setsuna_1','yellow'),(43.293,43.752,'kazusa_1','monkey'),(43.752,44.336,'setsuna_2','drunk'),(44.336,44.753,'kazusa_2','baby')]
imgs={}
for _,_,name,_ in CUTS:
 a=cv2.imdecode(np.fromfile(ROOT/'assets'/f'{name}.png',dtype=np.uint8),cv2.IMREAD_COLOR)
 if a is None:raise RuntimeError('Missing asset: '+name)
 # Compose at native pixel size. Never resize or overwrite input images.
 ih,iw=a.shape[:2]; ch=min(912,ih); cw=min(W,iw)
 crop=a[(ih-ch)//2:(ih-ch)//2+ch,(iw-cw)//2:(iw-cw)//2+cw]
 fitted=cv2.copyMakeBorder(crop,0,912-ch,(W-cw)//2,W-cw-(W-cw)//2,cv2.BORDER_REFLECT_101)
 canvas=np.zeros((H,W,3),np.uint8);canvas[84:996]=fitted
 imgs[name]=canvas
rng=random.Random(20261018)
flakes=[]
for i in range(48):
 flakes.append((rng.uniform(0,W),rng.uniform(-430,430),rng.uniform(25,62),rng.uniform(5,15),rng.uniform(0,6.28),rng.uniform(-.5,.5)))
def snow(frame,t):
 # Hand-drawn six-branch flakes drift from the top; strongest in upper third.
 overlay=frame.copy()
 fade=min(1,max(0,(20.4-t)/.8))
 for i,(sx,sy,speed,r,phase,spin) in enumerate(flakes):
  y=((sy+t*speed+140)%650)-140
  if y<4 or y>455:continue
  x=(sx+24*math.sin(t*.7+phase)+t*8)%W
  alpha=.8*min(1,y/35)*max(0,1-y/470)*fade
  color=(int(180+65*alpha),int(190+60*alpha),int(200+50*alpha))
  angle=phase+t*spin
  for j in range(6):
   a=angle+j*math.pi/3;rr=r*(1+.07*math.sin(j*3+i))
   tip=(round(x+rr*math.cos(a)),round(y+rr*math.sin(a)))
   center=(round(x),round(y));cv2.line(overlay,center,tip,color,1 if r<10 else 2,cv2.LINE_AA)
   for off in [-.72,.72]:
    bx=x+rr*.59*math.cos(a);by=y+rr*.59*math.sin(a)
    end=(round(bx+rr*.29*math.cos(a+math.pi+off)),round(by+rr*.29*math.sin(a+math.pi+off)))
    cv2.line(overlay,(round(bx),round(by)),end,color,1,cv2.LINE_AA)
  if alpha<.8:
   # Composite only the flake's local area for depth and a soft lower boundary.
   xa=max(0,int(x-r-3));xb=min(W,int(x+r+4));ya=max(0,int(y-r-3));yb=min(H,int(y+r+4))
   if xb>xa and yb>ya:overlay[ya:yb,xa:xb]=cv2.addWeighted(overlay[ya:yb,xa:xb],alpha,frame[ya:yb,xa:xb],1-alpha,0)
 return overlay
silent=ROOT.parent/'work'/'render_silent.mp4'
cmd=[FFMPEG,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','15','-pix_fmt','yuv420p',str(silent)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
c=av.open(str(SOURCE));c.seek(int(32*av.time_base));decoder=c.decode(video=0)
current=next(decoder);nxt=next(decoder)
for idx in range(21*FPS):
 t=idx/FPS;st=33+t
 while float(nxt.time)<st:
  current=nxt;nxt=next(decoder)
 nearest=current if abs(float(current.time)-st)<abs(float(nxt.time)-st) else nxt
 frame=nearest.to_ndarray(format='bgr24')
 for start,end,name,word in CUTS:
  if start<=st<end:
   frame=imgs[name].copy()
   # Small native-pixel jump on cut; no image scaling.
   age=st-start
   dx=round(5*math.sin(age*24)*math.exp(-age*10));dy=round(3*math.cos(age*21)*math.exp(-age*10))
   frame=np.roll(frame,(dy,dx),(0,1));frame[:84]=0;frame[996:]=0
   if st>=42.709:
    layer=Image.new('RGBA',(W,H),(0,0,0,0));d=ImageDraw.Draw(layer)
    font=ImageFont.truetype(r'LOCAL_PATH/GOTHIC.TTF',78)
    box=d.textbbox((0,0),word,font=font);tw=box[2]-box[0]
    x=(W-tw)//2;y=476
    d.text((x+4,y+5),word,font=font,fill=(10,10,10,230))
    d.text((x-2,y),word,font=font,fill=(70,220,230,255))
    d.text((x+2,y+1),word,font=font,fill=(240,200,75,255))
    d.text((x,y),word,font=font,fill=(250,250,245,255))
    pixels=np.asarray(layer);alpha=pixels[:,:,3:4].astype(np.float32)/255
    frame=(frame*(1-alpha)+pixels[:,:,:3][:,:,::-1]*alpha).astype(np.uint8)
   break
 frame=snow(frame,t)
 p.stdin.write(frame.tobytes())
 if idx%120==0:print(f'Render {idx}/{21*FPS}',flush=True)
p.stdin.close()
if p.wait()!=0:raise RuntimeError('Encoder failed')
out=ROOT/'rabbit_雪菜和纱_33-54秒_试作.mp4'
subprocess.run([FFMPEG,'-y','-loglevel','error','-i',str(silent),'-ss','33','-t','21','-i',str(SOURCE),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','320k','-t','21','-movflags','+faststart',str(out)],check=True)
print(str(out),flush=True)
