from pathlib import Path
import subprocess,json
import numpy as np
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'LOCAL_PATH/【初音ミク】rabbit【john】.mp4'
raw=subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',SOURCE,'-vn','-ac','1','-ar','22050','-f','f32le','-'])
x=np.frombuffer(raw,np.float32);n=256;hop=64
frames=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]
spec=np.abs(np.fft.rfft(frames*np.hanning(n),axis=1));freq=np.fft.rfftfreq(n,1/22050)
flux=np.maximum(np.diff(np.log1p(spec*12),axis=0),0)
onset=flux[:,freq>900].sum(axis=1);times=(np.arange(len(onset))*hop+n/2+hop)/22050
result=[]
for a,b in [(33.5,35.9),(85.7,88.5)]:
 idx=np.flatnonzero((times>=a)&(times<=b));peaks=[]
 for i in idx:
  if onset[i]>=onset[max(0,i-8):i+9].max():peaks.append((float(times[i]),float(onset[i])))
 peaks=sorted(peaks,key=lambda z:z[1],reverse=True)[:16]
 result.append({'range':[a,b],'peaks':sorted(peaks)})
(ROOT/'work/full_rebuild/bridge_audio_onsets.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
