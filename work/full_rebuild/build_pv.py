"""Full native-pixel PV composition. Image originals are never written or resized."""
from pathlib import Path
from fractions import Fraction
import sys,json,math,subprocess,hashlib
import av,cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageFont,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs'))
import render_v2_blue as base
W,H=1920,1080
TB=Fraction(1,16000)
OUT=ROOT/'outputs'
WORK=ROOT/'work/full_rebuild'
AS=OUT/'assets/full_pv'
META=json.loads((WORK/'source_frames.json').read_text())
FRAMES=META['frames']
FLASH=[
 (42.709375,43.2933125,'f11_rooftop','yellow'),(43.2933125,43.7520625,'f12_classroom','monkey'),
 (43.7520625,44.336,'f13_karaoke','drunk'),(44.336,44.7530625,'f14_sick','baby'),
 (51.4264375,51.8435,'f15_depart_class','消えろ'),(51.968625,52.4274375,'f16_haruki_guitar','noisy'),(52.5525625,53.470125,'f17_rehearsal','minority'),
 (95.053375,95.6373125,'f21_festival','yellow'),(95.6373125,96.0960625,'f22_haruki_run','monkey'),
 (96.0960625,96.68,'f23_christmas','drunk'),(96.68,97.0970625,'f24_snow_cry','baby'),
 (103.7704375,104.1875,'f25_airport_turn','消えろ'),(104.312625,104.7714375,'f26_stage_kazusa','noisy'),(104.8965625,105.814125,'f27_airport_setsuna','minority'),
 (112.4875,113.071375,'f31_concert_bow','yellow'),(113.071375,113.5301875,'f32_forced_smile','monkey'),
 (113.5301875,114.114125,'f33_setsuna_hug','drunk'),(114.114125,114.5311875,'f34_kazusa_lean','baby'),
 (121.2045625,121.621625,'f35_haruki_leave','消えろ'),(121.74675,122.2055625,'f36_setsuna_phone','noisy'),(122.3306875,123.24825,'f37_kazusa_solo','minority')]
MAINS=['m_intro_setsuna','m_setsuna_night','m_kazusa_night','m_haruki_grief','m_kazusa_release']
LAYERS={}
TIMING_PREVIEW='--timing-preview' in sys.argv
for n in sorted(set([s[2] for s in FLASH]+MAINS)):
 p=AS/(n+'.png')
 if not p.exists():continue
 a=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_UNCHANGED)
 if a.ndim!=3 or a.shape[2]!=4 or a[:,:,3].min()!=0:raise ValueError(f'No true transparent alpha: {n}')
 # Crop only empty canvas margins at native pixels, preserving every subject pixel.
 ys,xs=np.where(a[:,:,3]>4)
 LAYERS[n]=a[max(0,ys.min()-3):min(a.shape[0],ys.max()+4),max(0,xs.min()-3):min(a.shape[1],xs.max()+4)]

def snap(t):return min(FRAMES,key=lambda m:abs(m['t']-t))['t']
# Original shot boundaries are snapped to actual source frames, including all black gaps.
FLASH=[(snap(a),snap(b),n,s) for a,b,n,s in FLASH]
assert len(set(n for a,b,n,s in FLASH))==21
TEXT=[(36.328,'rabbit!'),(37.1621875,'狂気なんだ 今夜の夜景は綺麗か？'),(40.4154375,'音楽も耳障りだ'),
 (44.8781875,'割と簡単なんだ'),(47.0053125,'空の青さを見るのは'),(49.2575625,'辞めてくれよ 紛い物よ'),(53.59525,''),
 (88.672,'rabbit!'),(89.5061875,'狂気なんだ 今夜の夜景は綺麗か？'),(92.7594375,'音楽も耳障りだ'),
 (97.2221875,'割と簡単なんだ'),(99.3493125,'空の青さを見るのは'),(101.6015625,'辞めてくれよ 紛い物よ'),(105.93925,''),
 (106.106125,'rabbit!'),(106.94025,'狂気なんだ 今夜の夜景は綺麗か？'),(110.1935,'音楽も耳障りだ'),
 (114.6563125,'割と簡単なんだ'),(116.7834375,'空の青さを見るのは'),(119.0356875,'辞めてくれよ 紛い物よ'),(123.373375,'')]
TEXT=sorted([(snap(a),s) for a,s in TEXT]+[(a,s) for a,b,n,s in FLASH])
for _,s in TEXT:
 if s:base.text_layer(s)
TITLE=None
def title(frame):
 global TITLE
 if TITLE is None:
  font=ImageFont.truetype(str(base.FONTEN),360)
  box=font.getbbox('rabbit');tw=box[2]-box[0];th=box[3]-box[1]
  l=Image.new('RGBA',(tw+36,th+36));d=ImageDraw.Draw(l)
  x=14-box[0];y=12-box[1]
  for dx,dy,c in [(9,10,(0,0,0,255)),(-4,0,(50,210,235,255)),(4,2,(245,200,45,255)),(0,0,(252,252,249,255))]:d.text((x+dx,y+dy),'rabbit',font=font,fill=c)
  TITLE=np.array(l)[:,:,[2,1,0,3]]
 base.overlay_native(frame,TITLE,(W-TITLE.shape[1])//2,(H-TITLE.shape[0])//2)

# Paper grain is sampled at native resolution; no image asset is changed.
blank=base.blank.astype(np.float32)
gray=cv2.cvtColor(blank.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
grain=gray-cv2.GaussianBlur(gray,(19,19),0)
y=np.linspace(0,1,912,dtype=np.float32)[:,None,None]
x=np.linspace(-1,1,W,dtype=np.float32)[None,:,None]
shade=np.clip(1-.13*x*x,0,1)
BG_CACHE={}
def background(t):
 # Backgrounds change on the original cut; hold paper color stable within a shot.
 if t<.417:key=0
 elif t<.834:key=1
 elif t<1.251:key=2
 elif 36.119<t<42.709 or 44.878<t<51.301 or 88.463<t<95.053 or 97.222<t<103.645:key=3
 elif 105.939<t<114.531 or 114.656<t<123.248:key=4
 else:key=5
 for a,b,n,s in FLASH:
  if a<=t<b and s in ['yellow','monkey','drunk','baby']:
   key=({'yellow':4,'monkey':6,'drunk':6,'baby':7} if a>110 else {'yellow':5,'monkey':4,'drunk':1,'baby':2})[s]
 if key not in BG_CACHE:
  top,bottom={0:([41,68,112],[21,22,30]),1:([65,182,135],[28,53,38]),2:([175,118,168],[45,29,43]),3:([194,160,33],[49,40,20]),4:([57,127,184],[22,28,44]),5:([20,225,237],[24,30,40]),6:([210,210,212],[67,65,68]),7:([0,0,0],[0,0,0])}[key]
  cols=np.array(top,np.float32)[None,None,:]*(1-y)+np.array(bottom,np.float32)[None,None,:]*y
  a=cols*shade+grain[:,:,None]*(0 if key==7 else .84)
  BG_CACHE[key]=np.clip(a,0,255).astype(np.uint8)
 return BG_CACHE[key].copy()

def person(content,n,t,start,dx=0,dy=0,flash=False):
 if n not in LAYERS:
  if not TIMING_PREVIEW:raise RuntimeError('Missing character layer '+n)
  fallback={'m_intro_setsuna':'setsuna_yellow_bouquet','m_setsuna_night':'setsuna_bouquet_teal','m_kazusa_night':'kazusa_bouquet_teal','m_haruki_grief':'kazusa_piano_yellow','m_kazusa_release':'kazusa_bouquet_teal','f14_sick':'kazusa_2','f15_depart_class':'kazusa_2','f16_haruki_guitar':'kazusa_piano_yellow','f17_rehearsal':'setsuna_2','f21_festival':'setsuna_1','f22_haruki_run':'kazusa_piano_yellow','f23_christmas':'setsuna_2','f24_snow_cry':'kazusa_2','f25_airport_turn':'kazusa_2','f26_stage_kazusa':'kazusa_1','f27_airport_setsuna':'setsuna_2','f31_concert_bow':'kazusa_1','f32_forced_smile':'setsuna_2','f33_setsuna_hug':'setsuna_1','f34_kazusa_lean':'kazusa_2','f35_haruki_leave':'kazusa_piano_yellow','f36_setsuna_phone':'setsuna_2','f37_kazusa_solo':'kazusa_1'}[n]
  content[:]=base.scene(fallback,dx=round(3*math.sin((t-start)*2)))
  return
 a=LAYERS[n]
 age=max(0,t-start)
 # Native translation only. Keep the full top of the hair in frame.
 shake=round(6*math.sin(age*28)*math.exp(-age*13)) if flash else round(18*math.sin(age*.35)+3*math.sin(age*1.5))
 xx=(W-a.shape[1])//2+dx+shake
 yy=max(2,(912-a.shape[0])//2)+dy
 base.overlay_native(content,a,xx,yy)

def petals(content,t,blue=False,amount=18):
 for j in range(amount):
  phase=j*2.399
  xx=(j*137+t*18+30*math.sin(t*.7+phase))%(W+80)-40
  yy=(j*173+t*(22+j%5*4))%990-40
  r=8+j%7*2;ang=t*.7+phase
  pts=np.array([[round(xx+r*math.cos(ang+k*math.pi/3)),round(yy+r*.65*math.sin(ang+k*math.pi/3))] for k in range(6)],np.int32)
  col=(204,112,45) if blue else (179,99,179)
  cv2.fillPoly(content,[pts],col,lineType=cv2.LINE_AA);cv2.polylines(content,[pts],True,(72,48,65),1,cv2.LINE_AA)

def compose(m,source=None):
 t=m['t'];frame=np.zeros((H,W,3),np.uint8)
 if m['black']:return frame,'black'
 content=background(t);name='paper';cap=''
 flash=next(((a,b,n,s) for a,b,n,s in FLASH if a<=t<b),None)
 if flash:
  a,b,name,cap=flash;person(content,name,t,a,flash=True)
  if a>110:
   if cap=='baby':
    gray=cv2.cvtColor(content,cv2.COLOR_BGR2GRAY)
    _,gray=cv2.threshold(gray,140,255,cv2.THRESH_BINARY)
    content=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
   petals(content,t,False,28)
 elif t<1.251:pass
 elif t<18.5185:name='m_intro_setsuna';person(content,name,t,1.251,dy=15)
 elif t<27.3606875:name='f12_classroom';person(content,name,t,18.643625,dy=25)
 elif t<31.6149375:name='f16_haruki_guitar';person(content,name,t,27.3606875,dy=10)
 elif t<33.9089375:name='kazusa_piano_yellow';content=base.scene(name,dx=round(3*math.sin(t*2)))
 elif t<35.8691875:
  name='piano_drop';content=base.scene('yellow_paper')
  drop=635 if t<34.326 else 735 if t<34.951625 else 825
  base.overlay_native(content,base.cutout,(W-base.cutout.shape[1])//2,drop)
 elif t<36.1194375:
  name='ink_flash';content=base.scene('setsuna_bouquet_teal',dx=-25)
  _,g=cv2.threshold(cv2.cvtColor(content,cv2.COLOR_BGR2GRAY),130,255,cv2.THRESH_BINARY);content=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
 elif t<40.4154375:name='m_setsuna_night';person(content,name,t,36.1194375);petals(content,t)
 elif t<42.709375:name='f16_haruki_guitar';person(content,name,t,40.4154375);petals(content,t)
 elif t<51.3013125:name='kazusa_bouquet_teal';content=base.scene(name,dx=round(-24+(t-44.8781875)*8),dy=round(3*math.sin(t)));base.falling_petals(content,t)
 elif t<53.470125:raise RuntimeError(f'Uncovered flash frame {t}')
 elif t<70.3203125:name='setsuna_shoes';content=base.scene(name,dx=round(3*math.sin(t*2)));petals(content,t,False,12)
 elif t<79.7046875:name='m_intro_setsuna';person(content,name,t,70.4454375,dy=15)
 elif t<87.671:name='m_kazusa_night';person(content,name,t,79.7046875)
 elif t<88.2131875:name='piano_drop';base.overlay_native(content,base.cutout,(W-base.cutout.shape[1])//2,735)
 elif t<88.4634375:
  name='ink_flash';content=base.scene('setsuna_bouquet_teal',dx=-25)
  _,g=cv2.threshold(cv2.cvtColor(content,cv2.COLOR_BGR2GRAY),130,255,cv2.THRESH_BINARY);content=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
 elif t<92.7594375:name='m_kazusa_night';person(content,name,t,88.4634375);petals(content,t,True)
 elif t<95.053375:name='f26_stage_kazusa';person(content,name,t,92.7594375);petals(content,t,True)
 elif t<103.6453125:name='f27_airport_setsuna';person(content,name,t,97.2221875);petals(content,t)
 elif t<105.814125:raise RuntimeError(f'Uncovered flash frame {t}')
 elif t<110.1935:name='m_haruki_grief';person(content,name,t,105.93925);petals(content,t)
 elif t<112.4875:name='f37_kazusa_solo';person(content,name,t,110.1935);petals(content,t,True)
 elif t<121.0794375:name='m_kazusa_release';person(content,name,t,114.6563125);petals(content,t,True,26)
 elif t<123.24825:raise RuntimeError(f'Uncovered flash frame {t}')
 elif t<140.224:name='m_intro_setsuna';person(content,name,t,123.373375,dy=15);petals(content,t)
 else:name='end_title'
 frame[84:996]=content
 if (1.251<=t<18.5185) or t>=140.3486875:title(frame)
 if (18.643625<=t<33.9089375) or (70.4454375<=t<87.671):
  # Keep the original PV's vertically written text, including its exact reveal timing.
  if source is not None:
   for x0,x1 in [(75,305),(1535,1847)]:
    if source[275:290,x0+5:x0+20].mean()<60:frame[272:808,x0:x1]=source[272:808,x0:x1]
 if not cap and ((36.1194375<=t<53.470125) or (88.4634375<=t<123.24825)):
  for a,s in TEXT:
   if a<=t:cap=s
   else:break
 if cap:base.put_caption(frame,cap)
 base.snow(frame,t)
 if TIMING_PREVIEW:
  cv2.putText(frame,'TIMING PREVIEW / CHARACTER ASSETS IN PROGRESS',(34,1045),cv2.FONT_HERSHEY_SIMPLEX,.65,(210,210,210),1,cv2.LINE_AA)
 return frame,name

def main():
 missing=sorted(set([s[2] for s in FLASH]+MAINS)-LAYERS.keys())
 if missing and not TIMING_PREVIEW:raise RuntimeError('Missing final character layers: '+', '.join(missing))
 (OUT/'全片复刻_卡点与素材表.json').write_text(json.dumps({'flash':FLASH,'captions':TEXT,'frames':len(FRAMES),'time_base':str(TB),'audio':'original stream copy','image_transform':'native pixel crop/translation/alpha composite only'},ensure_ascii=False,indent=2),encoding='utf-8')
 if '--preview' in sys.argv:
  for t in [3,20,29,33.2,34.1,36.5,39,41,42.8,43.4,43.9,44.5,46,51.5,52.1,52.8,57,72,81,90,93,95.2,95.8,96.3,96.9,99,103.9,104.4,105,108,111,112.6,113.2,113.7,114.3,117,121.3,121.9,122.5,130,143]:
   m=min(FRAMES,key=lambda f:abs(f['t']-t));a,n=compose(m)
   prefix='preview_timing' if TIMING_PREVIEW else 'preview_final'
   cv2.imencode('.png',a)[1].tofile(WORK/f'{prefix}_{t}.png')
  print('PREVIEWS DONE',flush=True);return
 silent=WORK/('全片_卡点预览_无声.mp4' if TIMING_PREVIEW else '全片_无声.mp4');out=av.open(str(silent),'w')
 stream=out.add_stream('libx264',rate=Fraction(24000,1001));stream.width=W;stream.height=H;stream.pix_fmt='yuv420p';stream.time_base=TB;stream.codec_context.time_base=TB
 stream.options={'crf':'14','preset':'fast','bf':'0'}
 durations={m['pts']:m['duration'] for m in FRAMES};records=[]
 src=av.open(str(base.SOURCE))
 for i,f in enumerate(src.decode(video=0)):
  m=FRAMES[i]
  assert f.pts==m['pts']
  t=m['t'];a,n=compose(m,f.to_ndarray(format='bgr24') if 18.64<t<33.91 or 70.44<t<87.68 else None)
  v=av.VideoFrame.from_ndarray(a,format='bgr24');v.pts=m['pts'];v.time_base=TB;v.duration=m['duration']
  for p in stream.encode(v):
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  records.append({'pts':m['pts'],'t':t,'black':m['black'],'shot':n})
  if i%120==0:print(f'Render {i}/{len(FRAMES)} {t:.2f}s',flush=True)
 for p in stream.encode():
  if p.pts in durations:p.duration=durations[p.pts]
  out.mux(p)
 out.close();src.close()
 path=OUT/('rabbit_白色相簿2_全片卡点预览_待补图.mp4' if TIMING_PREVIEW else 'rabbit_白色相簿2_全片_v1.mp4')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(silent),'-i',str(base.SOURCE),'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(path)],check=True)
 (OUT/'全片复刻_逐帧时间表.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',path,flush=True)
if __name__=='__main__':main()
