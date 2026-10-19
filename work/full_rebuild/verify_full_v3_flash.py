"""Verify encoded full film: actual blackout/onset frames, music peaks, original audio."""
from pathlib import Path
import json,hashlib,subprocess
import av,cv2,numpy as np,imageio_ffmpeg
import revise_full_v3_flash as edit
ROOT=edit.ROOT;OUT=edit.OUT;WORK=edit.WORK;full=edit.full
video=edit.TARGET;source=edit.base.SOURCE
plan=json.loads((OUT/'全片v3_六图闪回与卡点表.json').read_text(encoding='utf-8'))
records=json.loads((OUT/'全片v3_逐帧时间表.json').read_text(encoding='utf-8'))
source_pts=[m['pts'] for m in full.FRAMES]
by_pts={m['pts']:m for m in full.FRAMES}
pts=[];blacks={};captured={}
need={round(c['visible_start']*16000) for c in plan['six_flashbacks']}
need.add(round(full.snap(90)*16000))
regular=[full.snap(t) for t in [10,20,29,39,49,59,69,79,99,109,119,129,139,145]]
need.update(round(t*16000) for t in regular)
c=av.open(str(video));s=c.streams.video[0]
metadata={'width':s.width,'height':s.height,'frames':s.frames,'duration':c.duration/av.time_base,'time_base':str(s.time_base)}
for i,f in enumerate(c.decode(video=0)):
 p=round(float(f.time)*16000);pts.append(p);a=f.to_ndarray(format='bgr24')
 black=float(a[84:996].mean())<8;blacks[p]=black
 assert black==records[i]['black'],(f.time,black,records[i])
 if p in need:captured[p]=a
c.close()
assert pts==source_pts and len(pts)==3502
assert (metadata['width'],metadata['height'])==(1920,1080)
assert all(r['shot']!='piano_drop' for r in records)
def audio(path):
 h=hashlib.sha256();samples=0;first=None;last=None
 c=av.open(str(path))
 for f in c.decode(audio=0):
  h.update(f.to_ndarray().tobytes());samples+=f.samples
  if first is None:first=float(f.time)
  last=float(f.time)
 c.close();return {'sha256':h.hexdigest(),'samples':samples,'first_pts':first,'last_pts':last}
original_audio=audio(source);final_audio=audio(video)
assert original_audio==final_audio
print('Full frame timestamps, black frames and original audio verified',flush=True)

# Re-detect the nearest actual percussive peak from the encoded full film's audio.
ff=imageio_ffmpeg.get_ffmpeg_exe()
raw=subprocess.check_output([ff,'-v','error','-i',str(video),'-vn','-ac','1','-ar','22050','-f','f32le','-'])
signal=np.frombuffer(raw,np.float32);n=256;hop=64
win=np.lib.stride_tricks.sliding_window_view(signal,n)[::hop]
spec=np.abs(np.fft.rfft(win*np.hanning(n),axis=1))
freq=np.fft.rfftfreq(n,1/22050)
flux=np.maximum(np.diff(np.log1p(spec*12),axis=0),0)[:,freq>900].sum(axis=1)
times=(np.arange(len(flux))*hop+n/2+hop)/22050
checks=[]
for cue in plan['six_flashbacks']:
 p=round(cue['visible_start']*16000);idx=pts.index(p)
 prior=pts[idx-3:idx]
 assert len(prior)==3 and all(blacks[q] for q in prior)
 assert not blacks[p]
 visible=[r for r in records if cue['visible_start']<=r['t']<cue['visible_end']]
 assert visible and all(r['shot']=='user_flash_'+str(cue['attachment_index']+1) and not r['black'] for r in visible)
 near=np.flatnonzero(abs(times-cue['visible_start'])<.10)
 peak=float(times[near[np.argmax(flux[near])]])
 error=(cue['visible_start']-peak)*1000
 assert abs(error)<21,(cue,error)
 expected=edit.picture(by_pts[p],cue)
 mae=float(np.abs(expected.astype(np.int16)-captured[p].astype(np.int16)).mean())
 assert mae<5,(cue,mae)
 cv2.imencode('.png',captured[p])[1].tofile(WORK/f"encoded_flash_{cue['attachment_index']+1}.png")
 checks.append({'attachment':cue['attachment_index']+1,'actual_visible_start':p/16000,'audio_peak_redetected':peak,'error_ms':round(error,4),'black_frames_before':3,'black_duration_ms':round((p-prior[0])/16,4),'visible_frames':len(visible),'native_composite_vs_encoded_mae':round(mae,5)})
assert max(abs(x['error_ms']) for x in checks)<21

# Existing lyric flashbacks remain on their approved source frame timestamps.
lyric_checks=[]
for a,b,name,word in edit.rev.FLASH:
 group=[r for r in records if a<=r['t']<b and not r['black']]
 assert group and group[0]['t']==a and all(r['shot']==name for r in group)
 lyric_checks.append({'start':a,'asset':name,'lyric':word})
assert len(lyric_checks)==21

# A sampling of unaffected portions must retain the previous full film's pictures.
old=av.open(str(OUT/'rabbit_白色相簿2_全片_v2少复用版.mp4'));s=old.streams.video[0]
unchanged=[]
for t in regular:
 p=round(t*16000);old.seek(max(0,p-800),stream=s)
 for f in old.decode(s):
  q=round(float(f.time)*16000)
  if q==p:
   reference=f.to_ndarray(format='bgr24')
   mae=float(np.abs(reference.astype(np.int16)-captured[p].astype(np.int16)).mean())
   assert mae<6,(t,mae)
   unchanged.append({'time':t,'mae':round(mae,5)});break
  if q>p:raise AssertionError(('Missing comparison frame',t))
old.close()
assert len(unchanged)==len(regular)
bluep=round(full.snap(90)*16000);blue,shot=edit.compose(by_pts[bluep])
assert shot=='kazusa_bouquet_teal_blue'
blue_mae=float(np.abs(blue.astype(np.int16)-captured[bluep].astype(np.int16)).mean())
assert blue_mae<5
cv2.imencode('.png',captured[bluep])[1].tofile(WORK/'encoded_blue_bouquet_90.png')
for asset in plan['assets']:
 assert hashlib.sha256(Path(asset['source']).read_bytes()).hexdigest()==asset['sha256']
 assert hashlib.sha256(Path(asset['copy']).read_bytes()).hexdigest()==asset['sha256']

result={'status':'verified','output':str(video),'metadata':metadata,'all_3502_frame_timestamps_match_original':True,'max_frame_timestamp_error_seconds':0,'audio_identical_to_original':True,'audio':final_audio,'six_flashback_checks':checks,'maximum_flash_onset_error_ms':max(abs(x['error_ms']) for x in checks),'blackout_frames_each':3,'approved_21_lyric_flashbacks_unchanged':lyric_checks,'unaffected_picture_comparisons':unchanged,'blue_bouquet_at_90s_verified':True,'blue_bouquet_composite_mae':round(blue_mae,5),'six_input_PNGs_byte_identical':True,'rope_and_head_drop_absent':True}
(OUT/'全片v3_成片卡点校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['approved_21_lyric_flashbacks_unchanged','unaffected_picture_comparisons']},ensure_ascii=False,indent=2),flush=True)
