"""Use the approved 21 s sample and reduce duplicate performance illustrations.
All input PNGs are read at native pixels, never resized or overwritten.
"""
from pathlib import Path
from fractions import Fraction
import json,math,subprocess,sys
import av,cv2,numpy as np,imageio_ffmpeg
import build_pv as full
base=full.base
ROOT=full.ROOT;OUT=ROOT/'outputs';WORK=ROOT/'work/full_revision_v2'
WORK.mkdir(exist_ok=True)
TARGET=OUT/'rabbit_白色相簿2_全片_v2少复用版.mp4'
W,H=full.W,full.H;TB=full.TB
RESTORE_START=33.0
# Continue the same blue-bouquet shot past 50 s until its existing cut.
RESTORE_END=51.3013125
LEGACY_NAMES=['setsuna_1','kazusa_1','setsuna_2','kazusa_2']
FLASH=[(a,b,LEGACY_NAMES[i] if i<4 else n,s) for i,(a,b,n,s) in enumerate(full.FLASH)]
def caption(frame,t):
 text=''
 for a,s in full.TEXT:
  if a<=t:text=s
  else:break
 if text:base.put_caption(frame,text)

def source_panels(frame,source):
 if source is None:return
 for x0,x1 in [(75,305),(1535,1847)]:
  if source[275:290,x0+5:x0+20].mean()<60:
   frame[272:808,x0:x1]=source[272:808,x0:x1]

def approved_sample_visual(t,black):
 # Call the unchanged sample compositor, with text handled by the full-song
 # timeline and snow by one continuous clock across the entire revised video.
 saved_cues,saved_snow=base.CUES,base.snow
 try:
  base.CUES=[]
  base.snow=lambda frame,time:None
  frame,name=base.compose(t,black)
 finally:
  base.CUES=saved_cues;base.snow=saved_snow
 if t>=36.1194375:caption(frame,t)
 saved_snow(frame,t)
 return frame,name

def compose(m,source=None):
 t=m['t']
 if m['black']:return np.zeros((H,W,3),np.uint8),'black'
 if RESTORE_START<=t<RESTORE_END:
  return approved_sample_visual(t,False)
 if 27.3606875<=t<31.6149375:
  frame=np.zeros((H,W,3),np.uint8)
  frame[84:996]=base.scene('setsuna_2',dx=round(3*math.sin((t-27.3606875)*2)))
  source_panels(frame,source);base.snow(frame,t)
  return frame,'setsuna_2'
 if 110.1935<=t<112.4875:
  frame=np.zeros((H,W,3),np.uint8)
  content=base.scene('kazusa_1',dx=round(3*math.sin((t-110.1935)*2)))
  full.petals(content,t,True)
  frame[84:996]=content;caption(frame,t);base.snow(frame,t)
  return frame,'kazusa_1'
 return full.compose(m,source)

def asset_path(name):
 if name in LEGACY_NAMES:return OUT/'assets'/(name+'.png')
 return OUT/'assets/full_pv'/(name+'.png')

def main():
 if '--preview' in sys.argv:
  source=av.open(str(base.SOURCE))
  for t in [29,32.99,33.04,34.1,36.5,39,41,42.8,43.4,43.9,44.5,46,49.9,50.1,51.2,52.1,111,122.5]:
   m=min(full.FRAMES,key=lambda f:abs(f['t']-t));raw=None
   if 18.64<m['t']<33.91:
    source.seek(int((m['t']-.1)*av.time_base))
    for f in source.decode(video=0):
     if f.pts==m['pts']:raw=f.to_ndarray(format='bgr24');break
   a,n=compose(m,raw);cv2.imencode('.png',a)[1].tofile(WORK/f'preview_{t}.png')
  source.close();print('Revised previews ready',flush=True);return
 (OUT/'全片v2_镜头与卡点表.json').write_text(json.dumps({'flash':FLASH,'restore_visual_range':[RESTORE_START,RESTORE_END],'requested_restore_range':[33,50],'replacements':[{'range':[27.3606875,31.6149375],'from':'f16_haruki_guitar','to':'setsuna_2'},{'range':[40.4154375,42.709375],'from':'f16_haruki_guitar','to':'setsuna_bouquet_teal','method':'approved sample sequence'},{'range':[110.1935,112.4875],'from':'f37_kazusa_solo','to':'kazusa_1'}],'audio':'original stream copy','original_sample':str(OUT/'rabbit_白色相簿2_33-54秒_v2蓝花版.mp4'),'caption_policy':'retain corrected full-song timing and wording','snow':'continuous full-song clock; approved snowflake design','native_image_pixels':True},ensure_ascii=False,indent=2),encoding='utf-8')
 durations={m['pts']:m['duration'] for m in full.FRAMES}
 silent=WORK/'全片v2_无声.mp4';out=av.open(str(silent),'w')
 stream=out.add_stream('libx264',rate=Fraction(24000,1001))
 stream.width=W;stream.height=H;stream.pix_fmt='yuv420p';stream.time_base=TB;stream.codec_context.time_base=TB
 stream.options={'crf':'14','preset':'fast','bf':'0'}
 records=[];source=av.open(str(base.SOURCE))
 for i,f in enumerate(source.decode(video=0)):
  m=full.FRAMES[i];assert f.pts==m['pts'];t=m['t']
  a,n=compose(m,f.to_ndarray(format='bgr24') if 18.64<t<33.91 or 70.44<t<87.68 else None)
  v=av.VideoFrame.from_ndarray(a,format='bgr24');v.pts=m['pts'];v.time_base=TB;v.duration=m['duration']
  for p in stream.encode(v):
   if p.pts in durations:p.duration=durations[p.pts]
   out.mux(p)
  records.append({'pts':m['pts'],'t':t,'black':m['black'],'shot':n})
  if i%240==0:print(f'Render v2 {i}/{len(full.FRAMES)} {t:.2f}s',flush=True)
 for p in stream.encode():
  if p.pts in durations:p.duration=durations[p.pts]
  out.mux(p)
 out.close();source.close()
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(silent),'-i',str(base.SOURCE),'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(TARGET)],check=True)
 (OUT/'全片v2_逐帧时间表.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',str(TARGET),flush=True)
if __name__=='__main__':main()
