import json,sys,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));sys.path.insert(0,str(P.parents[1]))
from harness import Harness as CloudHarness
from Proposal.security_harness.live_demo.server import Harness as OriginalHarness
import torch
torch.set_num_threads(4)
a=OriginalHarness(); b=CloudHarness()
r=json.loads((P.parent/'security_harness/runs/audio_150_eval/report.json').read_text())
checks=0;max_diff=0.;mismatch=[]
for case in r['results']:
 hist=[{'speaker':t['speaker'],'text':t['transcript']} for t in case['turns'][:-1]]
 for event in case['turns'][-1]['decisions']:
  old=a.local(hist,event['partial']);new=b.local(hist,event['partial']);checks+=1
  if (old['action'],old['source'])!=(new['action'],new['source']):mismatch.append(case['scenario_id'])
  if old['probability'] is not None:max_diff=max(max_diff,abs(old['probability']-new['probability']))
assert not mismatch,mismatch
out={'scenarios':len(r['results']),'partial_checks':checks,'action_or_source_mismatches':len(mismatch),'max_probability_absolute_difference':max_diff,'weights':'Original FP32, no quantization','note':'Offline parity on saved synthetic ASR histories and partials, not deployment latency validation.'}
(P/'model/validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
