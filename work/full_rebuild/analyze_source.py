from pathlib import Path
import av,cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/full_rebuild'
SOURCE=Path(__import__('os').environ.get('RABBIT_SOURCE', str(Path(__file__).resolve().parents[2]/'input/source.mp4')))
WORK.mkdir(exist_ok=True)
records=[];events=[];last=None
c=av.open(str(SOURCE));s=c.streams.video[0]
for i,f in enumerate(c.decode(video=0)):
 t=float(f.time);a=f.to_ndarray(format='bgr24');roi=a[84:996]
 mean=float(roi.mean());black=mean<8
 # Take source-native pixels only for analysis, never alter an image asset.
 col=np.median(np.concatenate([roi[15:100,20:100].reshape(-1,3),roi[15:100,-100:-20].reshape(-1,3)]),axis=0)
 edges=np.concatenate([roi[:,18:40:4],roi[:,-40:-18:4]],axis=1)
 rows=np.median(edges,axis=1)
 bgtop=np.median(rows[20:180],axis=0);bgbottom=np.median(rows[-180:-20],axis=0)
 tiny=roi[::12,::12].astype(float)
 dif=0 if last is None else float(np.abs(tiny-last).mean());last=tiny
 # Subtitles are centered in these chorus portions. Capture changed native frames for manual checking.
 center=a[470:610,220:1700:3].astype(float)
 if i:
  cd=float(np.abs(center-prevcenter).mean())
  if cd>5 and ((18<t<36) or (70<t<89) or (105.8<t<124)):
   events.append({'t':t,'diff':round(cd,2)})
   if cd>7:f.to_image().save(WORK/f'cue_{t:.3f}.png')
 prevcenter=center
 records.append({'pts':f.pts,'t':t,'black':black,'bg':[round(float(v),2) for v in col],'top':[round(float(v),2) for v in bgtop],'bottom':[round(float(v),2) for v in bgbottom]})
 if i%600==0:print('Analyzed',i,round(t,2),flush=True)
endpts=s.duration
for i,r in enumerate(records):r['duration']=(records[i+1]['pts'] if i+1<len(records) else endpts)-r['pts']
(WORK/'source_frames.json').write_text(json.dumps({'duration':float(s.duration*s.time_base),'time_base':str(s.time_base),'end_pts':endpts,'frames':records},indent=2),encoding='utf-8')
(WORK/'cue_changes.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
print('DONE',len(records),'end',endpts,'cue candidates',len(events),flush=True)
