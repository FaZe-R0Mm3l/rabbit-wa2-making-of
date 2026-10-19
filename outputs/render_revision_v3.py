"""Rebuild every visual in source 33-54 seconds; preserve original frame PTS.
Source images are read at native pixels, never resized or overwritten.
"""
from pathlib import Path
from fractions import Fraction
import av, cv2, numpy as np, math, random, subprocess, json, imageio_ffmpeg
from PIL import Image,ImageFont,ImageDraw
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
WORK=ROOT.parent/'work'
SOURCE=Path(__import__('os').environ.get('RABBIT_SOURCE', str(Path(__file__).resolve().parents[1]/'input/source.mp4')))
START,END=33,54
W,H=1920,1080
TB=Fraction(1,16000)
FONTJP=ASSETS/'NotoSerifCJKjp-Regular.otf'
FONTEN=Path(__import__('os').environ.get('RABBIT_FONT',str(ASSETS/'NotoSerifCJKjp-Regular.otf')))
# Every cue is taken from a decoded source frame's PTS, not rounded seconds.
CUES=[
 (36.328,'rabbit!'),(37.1621875,'狂気なんだ 今夜の夜景は綺麗か？'),
 (40.4154375,'音楽も耳障りだ'),(42.709375,'yellow'),(43.2933125,'monkey'),
 (43.7520625,'drunk'),(44.336,'baby'),(44.8781875,'割と簡単なんだ'),
 (45.0450625,'狂気なんだ 今夜の夜景は綺麗か？'),(47.0053125,'空の青さを翳るのは'),
 (49.2575625,'辞めてくれよ 紛い物よ'),(51.4264375,'noisy'),(51.968625,'消えろ'),
 (52.5525625,'minority'),(53.59525,'')]
FLASH=[(42.709375,43.2933125,'setsuna_1'),(43.2933125,43.7520625,'kazusa_1'),
 (43.7520625,44.336,'setsuna_2'),(44.336,44.7530625,'kazusa_2')]

ASSET_REMAP={'kazusa_bouquet_teal':'v3/kazusa_bouquet_teal_blue','kazusa_yellow_bouquet':'v3/kazusa_yellow_bouquet_blue','bouquet_toss':'v3/bouquet_toss_dual','flash_microphone':'v3/flash_microphone_small','flash_piano':'v3/flash_broken_keys','flash_torn_score':'v3/flash_torn_score_small'}
def load(name):
 name=ASSET_REMAP.get(name,name)
 a=cv2.imdecode(np.fromfile(ASSETS/(name+'.png'),np.uint8),cv2.IMREAD_UNCHANGED)
 if a is None:raise FileNotFoundError(name)
 return a

def fit_native(a):
 ih,iw=a.shape[:2];ch=min(912,ih);cw=min(W,iw)
 a=a[(ih-ch)//2:(ih-ch)//2+ch,(iw-cw)//2:(iw-cw)//2+cw]
 return cv2.copyMakeBorder(a,(912-ch)//2,912-ch-(912-ch)//2,(W-cw)//2,W-cw-(W-cw)//2,cv2.BORDER_REFLECT_101)

names=['kazusa_piano_yellow','setsuna_bouquet_teal','kazusa_bouquet_teal','setsuna_yellow_bouquet','kazusa_yellow_bouquet','bouquet_toss','setsuna_shoes','setsuna_1','setsuna_2','kazusa_1','kazusa_2','yellow_paper','flash_microphone','flash_piano','flash_torn_score']
# Native video compositing: put every illustration onto a textured background plate.
# Fill outer canvas with paper instead of reflecting character hands and hair.
blank=fit_native(load('yellow_paper')[:,:,:3])
blank_hsv=cv2.cvtColor(blank,cv2.COLOR_BGR2HSV).astype(np.float32)
blank_top=float(np.median(blank_hsv[:100,:200,2]))
plates={}
for n in names:
 raw=load(n)[:,:,:3]
 ih,iw=raw.shape[:2];ch=min(912,ih);cw=min(W,iw)
 shot=raw[(ih-ch)//2:(ih-ch)//2+ch,(iw-cw)//2:(iw-cw)//2+cw]
 sample=cv2.cvtColor(raw[:100,:200],cv2.COLOR_BGR2HSV)
 hue,sat,value=np.median(sample.reshape(-1,3),axis=0)
 # Match the actual paper color at each row; keep native texture in the margins.
 source_hsv=cv2.cvtColor(shot,cv2.COLOR_BGR2HSV)
 colors=[]
 lastcolor=np.array([hue,sat,value],np.float32)
 for row in range(ch):
  vals=source_hsv[row].astype(np.float32)
  hdist=np.minimum(abs(vals[:,0]-hue),180-abs(vals[:,0]-hue))
  ok=(hdist<9)&(vals[:,1]>max(50,float(sat)*.62))
  if int(ok.sum())>30:lastcolor=np.median(vals[ok],axis=0)
  colors.append(lastcolor.copy())
 colors=np.array(colors,np.float32)
 colors=cv2.GaussianBlur(colors[:,None,:],(1,21),0)[:,0,:]
 if ch<912:colors=np.pad(colors,(((912-ch)//2,912-ch-(912-ch)//2),(0,0)),mode='edge')
 hsv=np.empty_like(blank_hsv)
 hsv[:,:,0]=colors[:,0,None];hsv[:,:,1]=colors[:,1,None]
 texture=blank_hsv[:,:,2]-cv2.GaussianBlur(blank_hsv[:,:,2],(15,15),0)
 hsv[:,:,2]=np.clip(colors[:,2,None]+texture*.65,0,255)
 background=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
 x=(W-cw)//2;y=(912-ch)//2
 alpha=np.ones((ch,cw,1),np.float32)
 if cw<W:
  for j in range(22):alpha[:,j]=j/22;alpha[:,-j-1]=j/22
 background[y:y+ch,x:x+cw]=(background[y:y+ch,x:x+cw]*(1-alpha)+shot*alpha).astype(np.uint8)
 plates[n]=background if n!='yellow_paper' else blank
# Enlarge only the video canvas by reflected padding, retaining native source pixels.
padded={n:cv2.copyMakeBorder(a,80,80,100,100,cv2.BORDER_REFLECT_101) for n,a in plates.items()}
cutout=load('kazusa_piano_cutout')
if cutout.shape[2]!=4:raise ValueError('Opening animation requires genuine alpha')

def scene(name,dx=0,dy=0,mirror=False):
 a=padded[name];x=100-int(dx);y=80-int(dy)
 x=max(0,min(a.shape[1]-W,x));y=max(0,min(a.shape[0]-912,y))
 return a[y:y+912,x:x+W].copy()[:,::-1].copy() if mirror else a[y:y+912,x:x+W].copy()

def flash_scene(name,age,sign=1):
 content=scene(name)
 decay=math.exp(-max(0,age)*8)
 angle=sign*(1.8*math.cos(age*37)*decay+.18*math.sin(age*19))
 dx=sign*11*math.sin(age*47)*decay
 dy=7*math.cos(age*43)*decay
 matrix=cv2.getRotationMatrix2D((W/2,456),angle,1.0)
 matrix[0,2]+=dx;matrix[1,2]+=dy
 return cv2.warpAffine(content,matrix,(W,912),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

def overlay_native(dst,src,x,y):
 sh,sw=src.shape[:2];x0=max(0,x);x1=min(dst.shape[1],x+sw);y0=max(0,y);y1=min(dst.shape[0],y+sh)
 if x1<=x0 or y1<=y0:return
 part=src[y0-y:y1-y,x0-x:x1-x]
 alpha=part[:,:,3:4].astype(np.float32)/255
 dst[y0:y1,x0:x1]=(dst[y0:y1,x0:x1]*(1-alpha)+part[:,:,:3]*alpha).astype(np.uint8)

# Pre-render all typography once; retain the source's centered baseline and side panels.
text_cache={}
def text_layer(text):
 if text in text_cache:return text_cache[text]
 isjp=any(ord(ch)>255 for ch in text)
 size=82 if isjp else 78
 font=ImageFont.truetype(str(FONTJP if isjp else FONTEN),size)
 box=font.getbbox(text);tw=box[2]-box[0];th=box[3]-box[1]
 layer=Image.new('RGBA',(min(W,tw+28),th+28),(0,0,0,0));d=ImageDraw.Draw(layer)
 origin=(14-box[0],12-box[1])
 for dx,dy,col in [(4,4,(10,10,12,245)),(-2,0,(55,210,225,255)),(2,1,(245,196,65,255)),(0,0,(249,249,243,255))]:
  d.text((origin[0]+dx,origin[1]+dy),text,font=font,fill=col)
 a=np.array(layer);a[:,:,:3]=a[:,:,:3][:,:,::-1]
 text_cache[text]=a
 return a
for _,s in CUES:
 if s:text_layer(s)

def put_caption(frame,text):
 a=text_layer(text);overlay_native(frame,a,(W-a.shape[1])//2,(H-a.shape[0])//2)

def panel(columns,x,y,width,height):
 layer=Image.new('RGBA',(width,height),(18,18,16,225));d=ImageDraw.Draw(layer)
 font=ImageFont.truetype(str(FONTJP),48)
 for ci,col in enumerate(columns):
  cx=width-63-ci*53
  for j,ch in enumerate(col):
   px=cx;py=30+j*50
   d.text((px+3,py+3),ch,font=font,fill=(4,4,4,255))
   d.text((px-1,py),ch,font=font,fill=(60,210,225,255))
   d.text((px+1,py+1),ch,font=font,fill=(240,195,80,255))
   d.text((px,py),ch,font=font,fill=(246,246,235,255))
 a=np.array(layer);a[:,:,:3]=a[:,:,:3][:,:,::-1]
 return (a,x,y)
panels=[panel(['これまた','立派な欲望で','偽の数字で','喜んで'],1535,272,312,536),panel(['お前さん','とっても滑稽','なんだよ'],75,272,230,536)]

rng=random.Random(20261018)
flakes=[(rng.uniform(0,W),rng.uniform(-430,430),rng.uniform(25,62),rng.uniform(5,15),rng.uniform(0,6.28),rng.uniform(-.5,.5)) for i in range(48)]
def snow(frame,t):
 # All flakes are replaced by round white dots.
 for i,(sx,sy,speed,r,phase,spin) in enumerate(flakes):
  y=((sy+t*speed+140)%650)-140
  if y<4 or y>455:continue
  x=(sx+24*math.sin(t*.7+phase)+t*8)%W
  alpha=.8*min(1,y/35)*max(0,1-y/470)
  radius=max(2,round(r*.33))
  xa=max(0,int(x-radius-3));xb=min(W,int(x+radius+4));ya=max(0,int(y-radius-3));yb=min(H,int(y+radius+4))
  roi=frame[ya:yb,xa:xb];overlay=roi.copy()
  cv2.circle(overlay,(round(x)-xa,round(y)-ya),radius,(242,246,250),-1,cv2.LINE_AA)
  cv2.addWeighted(overlay,alpha,roi,1-alpha,0,roi)

# Moving petals are native vector video overlays; the character images remain untouched.
petals=[(rng.uniform(-100,600),rng.uniform(100,980),rng.uniform(7,16),rng.uniform(0,6.28)) for _ in range(12)]
def falling_petals(content,st):
 for j,(x,y,r,p) in enumerate(petals):
  yy=(y+(st-36)*25+j*10)%930;xx=x+18*math.sin(st+p)
  angle=p+st*.3
  pts=[]
  for k in range(7):
   a=angle+k*2*math.pi/7;rad=r*(1+.2*math.sin(k*2+j))
   pts.append([round(xx+rad*math.cos(a)),round(yy+rad*.7*math.sin(a))])
  pts=np.array(pts,np.int32)
  color=(213,127,56) if st>=44.8781875 else (164,86,176)
  outline=(90,55,32) if st>=44.8781875 else (59,47,69)
  cv2.fillPoly(content,[pts],color,lineType=cv2.LINE_AA);cv2.polylines(content,[pts],True,outline,1,cv2.LINE_AA)

# Ending animation: two-colored petals continuously morph into tiny score pages.
mrng=random.Random(20261019)
ending_particles=[]
for i in range(20):
 ending_particles.append((i*.009,mrng.uniform(-40,35),mrng.uniform(-65,65),mrng.uniform(160,420),mrng.uniform(-230,150),mrng.uniform(10,17),mrng.uniform(-1.4,1.4),i%2))
flower_hsv=cv2.cvtColor(plates['bouquet_toss'],cv2.COLOR_BGR2HSV)
flower_mask=((flower_hsv[:,:,0]>92)&(flower_hsv[:,:,1]>90)).astype(np.uint8)
flower_mask[500:]=0
nlab,labels,stats,centroids=cv2.connectedComponentsWithStats(flower_mask)
if nlab>1:
 k=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);flower_origin=centroids[k]+np.array([30.,0.])
else:flower_origin=np.array([1100.,230.])
def petal_scores(content,st):
 age_total=st-52.5525625
 for delay,ox,oy,vx,vy,r,theta,color_id in ending_particles:
  age=age_total-delay
  if age<0:continue
  x=flower_origin[0]+ox+vx*age
  y=flower_origin[1]+oy+vy*age+50*age*age
  if st>=53.59525:y+=35*(st-53.59525)
  morph=float(np.clip((age-.08)/.55,0,1));morph=morph*morph*(3-2*morph)
  size=r*(1+.85*morph);angle=theta+age*.65
  ca,sa=math.cos(angle),math.sin(angle)
  def xy(px,py):return (round(x+size*(px*ca-py*sa)),round(y+size*(px*sa+py*ca)))
  leaf=np.array([[0,-1],[.6,-.65],[.85,0],[.4,.65],[0,1],[-.55,.6],[-.8,0],[-.35,-.6]])
  page=np.array([[0,-1],[.72,-1],[.72,0],[.72,1],[0,1],[-.72,1],[-.72,0],[-.72,-1]])
  shape=leaf*(1-morph)+page*morph;poly=np.array([xy(*q) for q in shape],np.int32)
  original=np.array((213,127,56) if color_id==0 else (164,86,176),float)
  fill=tuple(int(v) for v in original*(1-morph)+np.array([224,240,249])*morph)
  cv2.fillPoly(content,[poly],fill,cv2.LINE_AA);cv2.polylines(content,[poly],True,(60,51,49),1,cv2.LINE_AA)
  if morph>.27:
   opacity=np.clip((morph-.27)/.48,0,1)
   ink=tuple(int(v) for v in np.array(fill)*(1-opacity)+np.array([55,52,49])*opacity)
   for base in [-.45,.25]:
    for row in range(5):
     yy=base+row*.105
     cv2.line(content,xy(-.52,yy),xy(.52,yy),ink,1,cv2.LINE_AA)
    for j in range(3):
     nx=-.32+j*.3;ny=base+.1+(j%2)*.13
     center=xy(nx,ny)
     cv2.ellipse(content,center,(max(1,round(size*.075)),max(1,round(size*.045))),angle*180/math.pi,0,360,ink,-1,cv2.LINE_AA)
     cv2.line(content,xy(nx+.06,ny),xy(nx+.06,ny-.3),ink,1,cv2.LINE_AA)

def compose(st,black):
 frame=np.zeros((H,W,3),np.uint8)
 # Exact reference black frames remain fully black so the impact is not softened.
 if black:return frame,'black'
 text='';name=''
 if st<33.9089375:
  name='kazusa_piano_yellow';content=scene(name,dx=round(3*math.sin(st*2)))
 elif st<35.8691875:
  if st<34.326:name='flash_microphone';cue=34.0340625;sign=1
  elif st<34.951625:name='flash_piano';cue=34.451125;sign=-1
  else:name='flash_torn_score';cue=35.07675;sign=1
  content=flash_scene(name,max(0,st-cue),sign)
 elif st<36.1194375:
  name='ink_flash';content=scene('setsuna_bouquet_teal',dx=-25)
  gray=cv2.cvtColor(content,cv2.COLOR_BGR2GRAY)
  _,gray=cv2.threshold(gray,130,255,cv2.THRESH_BINARY)
  content=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
 elif st<42.709375:
  name='setsuna_bouquet_teal';age=st-36.1194375
  content=scene(name,dx=round(-32+age*8),dy=round(3*math.sin(age*1.4)))
  falling_petals(content,st)
 elif st<44.7530625:
  for a,b,n in FLASH:
   if a-.000001<=st<b-.000001:
    name=n;age=st-a;content=scene(name,dx=round(5*math.sin(age*25)*math.exp(-age*12)));break
 elif st<51.3013125:
  name='kazusa_bouquet_teal';age=st-44.8781875
  content=scene(name,dx=round(-24+age*8),dy=round(3*math.sin(age*1.4)))
  falling_petals(content,st)
 elif st<51.8435:
  name='kazusa_yellow_bouquet';content=scene(name,dx=-15)
 elif st<52.4274375:
  name='setsuna_yellow_bouquet';content=scene(name,dx=15,mirror=True)
 elif st<53.470125:
  name='bouquet_toss';content=scene(name,dy=round(-35*(st-52.5525625)))
 else:
  name='setsuna_shoes';content=scene(name,dx=round(3*math.sin(st*2)))
 if st>=52.5525625:petal_scores(content,st)
 frame[84:996]=content
 if st<33.9089375:
  for a,x,y in panels:overlay_native(frame,a,x,y)
 elif st>=36.328:
  for a,b in CUES:
   if st>=a-.000001:text=b
   else:break
  if text:put_caption(frame,text)
 snow(frame,st-START)
 return frame,name

def main():
 WORK.mkdir(exist_ok=True)
 source=av.open(str(SOURCE));source.seek(int((START-1)*av.time_base))
 # Record source PTS and black-frame flag. Read no source pixels into output.
 metadata=[];last_before=None
 for f in source.decode(video=0):
  st=float(f.time)
  if st<START:last_before=(f.pts,st,float(f.to_ndarray(format='gray')[84:996].mean())<8);continue
  if st>=END:break
  if not metadata:
   q=last_before;metadata.append({'source_pts':q[0],'source_time':q[1],'output_pts':0,'black':q[2]})
  metadata.append({'source_pts':f.pts,'source_time':st,'output_pts':f.pts-START*16000,'black':float(f.to_ndarray(format='gray')[84:996].mean())<8})
 source.close()
 silent=WORK/'revision_v3_silent.mp4'
 out=av.open(str(silent),'w')
 stream=out.add_stream('libx264',rate=Fraction(24000,1001))
 stream.width=W;stream.height=H;stream.pix_fmt='yuv420p';stream.time_base=TB;stream.codec_context.time_base=TB
 stream.options={'crf':'15','preset':'fast','bf':'0'}
 durations={}
 records=[]
 for i,m in enumerate(metadata):
  st=m['source_time'] if i else START
  frame,name=compose(st,m['black'])
  v=av.VideoFrame.from_ndarray(frame,format='bgr24');v.pts=m['output_pts'];v.time_base=TB
  nextpts=metadata[i+1]['output_pts'] if i+1<len(metadata) else (END-START)*16000
  v.duration=nextpts-v.pts;durations[v.pts]=v.duration
  for packet in stream.encode(v):
   if packet.pts in durations:packet.duration=durations[packet.pts]
   out.mux(packet)
  records.append({**m,'shot':name,'expected_seconds':m['output_pts']/16000})
  if i%120==0:print(f'Render {i}/{len(metadata)}',flush=True)
 for packet in stream.encode():
  if packet.pts in durations:packet.duration=durations[packet.pts]
  out.mux(packet)
 out.close()
 path=ROOT/'rabbit_白色相簿2_33-54秒_蓝花乐谱_v3.mp4'
 ff=imageio_ffmpeg.get_ffmpeg_exe()
 subprocess.run([ff,'-y','-v','error','-i',str(silent),'-i',str(SOURCE),'-filter_complex','[1:a]atrim=start=33:end=54,asetpts=PTS-STARTPTS[a]','-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-b:a','320k','-t','21','-movflags','+faststart',str(path)],check=True)
 (ROOT/'蓝花乐谱_v3_帧时间表.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print(str(path),flush=True)
if __name__=='__main__':main()
