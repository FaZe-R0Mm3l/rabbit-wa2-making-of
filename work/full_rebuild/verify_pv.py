from pathlib import Path
import av,hashlib,json,numpy as np,sys
R=Path(__file__).resolve().parents[2]
source=Path(r'LOCAL_PATH/【初音ミク】rabbit【john】.mp4')
final='--final' in sys.argv
video=R/'outputs'/('rabbit_白色相簿2_全片_v1.mp4' if final else 'rabbit_白色相簿2_全片卡点预览_待补图.mp4')
meta=json.loads((R/'work/full_rebuild/source_frames.json').read_text())
def video_data(path):
 c=av.open(str(path));s=c.streams.video[0]
 pts=[];black=[]
 for f in c.decode(video=0):
  pts.append(float(f.pts*f.time_base));black.append(float(f.to_ndarray(format='gray')[84:996].mean())<8)
 d={'width':s.width,'height':s.height,'duration':c.duration/av.time_base,'time_base':str(s.time_base),'frame_count':len(pts)};c.close()
 return d,pts,black
def audio_data(path):
 c=av.open(str(path));s=c.streams.audio[0];h=hashlib.sha256();samples=0;pts=[]
 for f in c.decode(audio=0):
  a=f.to_ndarray();h.update(a.tobytes());samples+=f.samples;pts.append(float(f.pts*f.time_base))
 c.close();return {'decoded_sha256':h.hexdigest(),'decoded_samples':samples,'sample_rate':s.sample_rate,'first_pts':pts[0],'last_pts':pts[-1]}
info,pts,black=video_data(video)
sourcepts=[m['t'] for m in meta['frames']];sourceblack=[m['black'] for m in meta['frames']]
audio=audio_data(video);original_audio=audio_data(source)
result={'output':str(video),'metadata':info,'frame_count_equal':len(pts)==len(sourcepts),'max_frame_pts_error_seconds':max(abs(a-b) for a,b in zip(pts,sourcepts)),'black_frame_mismatches':sum(a!=b for a,b in zip(black,sourceblack)),'audio_identical':audio==original_audio,'output_audio':audio,'source_audio':original_audio,'status':'Final full version' if final else 'Timing preview; unfinished character assets are marked.'}
assert result['frame_count_equal'] and result['max_frame_pts_error_seconds']==0
assert result['black_frame_mismatches']==0 and result['audio_identical']
if final:
 cues=json.loads((R/'outputs/全片复刻_卡点与素材表.json').read_text(encoding='utf-8'))['flash']
 records=json.loads((R/'outputs/全片复刻_逐帧时间表.json').read_text(encoding='utf-8'))
 hashes=[];checks=[]
 for a,b,n,s in cues:
  group=[r for r in records if a<=r['t']<b and not r['black']]
  assert group and all(r['shot']==n for r in group)
  assert group[0]['t']==a
  checks.append({'start':a,'end':b,'asset':n,'lyric':s,'first_visible_pts':group[0]['pts'],'frame_count':len(group)})
  p=R/'outputs/assets/full_pv'/(n+'.png');hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
 assert len(hashes)==len(set(hashes))==21
 jobs=json.loads((R/'work/full_rebuild/image_jobs.json').read_text(encoding='utf-8'))
 assert all(j['status']=='completed' for j in jobs)
 result.update(unique_flash_images=21,flash_timing_checks=checks,remaining_assets=0)
(R/'outputs'/('全片成片_校验结果.json' if final else '全片卡点预览_校验结果.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
