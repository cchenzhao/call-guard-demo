"""Replay identical saved streaming-ASR observations through three live-policy ablations."""
import csv, json, time, inspect, textwrap, hashlib, platform, sys
from pathlib import Path
from collections import Counter
import torch
from Proposal.security_harness.live_demo.server import Harness, THRESHOLD, LOW_RISK

P=Path(__file__).resolve().parent
ROOT=P.parent
source=P/'asr_cache.json'
onsets=ROOT/'t1_annotation/password_onset_review_75.csv'
record=json.loads(source.read_text()); cases=record['results']
onset={r['scenario_id']:r for r in csv.DictReader(onsets.open(encoding='utf-8-sig'))}
assert len(cases)==150 and len(onset)==75
assert all(c['status']=='complete' and c['turns'][-1].get('decisions') for c in cases)
torch.set_num_threads(4)
harness=Harness()
# Derive rules directly from the active live policy, avoiding a divergent copy.
code=textwrap.dedent(inspect.getsource(Harness.local))
assert code.count('        history_text =')==0
needle='    history_text = "\\n".join'
assert needle in code
prefix=code[:code.index(needle)]
namespace={'re':__import__('re')}
exec(prefix+'    return {"action": "WATCH", "source": "rules-unresolved"}\n',namespace)
rules=namespace['local']

def mini(history,partial):
 history_text='\n'.join(f'{x["speaker"]}: {x["text"]}' for x in history)
 batch=harness.tokenizer(history_text,'user: '+partial,return_tensors='pt',truncation='only_first',max_length=192)
 with torch.inference_mode(): p=float(torch.softmax(harness.model(**batch).logits,dim=-1)[0,1])
 return {'action':'INTERVENE' if p>=THRESHOLD else 'WAIT' if p<=LOW_RISK else 'WATCH','source':'minilm','probability':p}

def run(case,mode):
 hist=[{'speaker':t['speaker'],'text':t['transcript']} for t in case['turns'][:-1]]
 observed=case['turns'][-1]['decisions']
 trace=[]; previous_finish=0.0
 for event in observed:
  # Simulate serial processing as in receive loop; do not drop or delay ASR events.
  arrival=float(event['asr_received_ms']); start=max(arrival,previous_finish)
  t=time.perf_counter_ns()
  if mode=='rules_only': d=rules(harness,hist,event['partial'])
  elif mode=='minilm_only': d=mini(hist,event['partial'])
  else: d=harness.local(hist,event['partial'])
  elapsed=(time.perf_counter_ns()-t)/1e6
  finish=start+elapsed;previous_finish=finish
  trace.append({'partial':event['partial'],'asr_received_ms':arrival,'start_ms_simulated':round(start,3),'compute_ms_measured':round(elapsed,3),'decision_complete_ms_simulated':round(finish,3),**d})
  if d['action']=='INTERVENE': break
 hit=next((e for e in trace if e['action']=='INTERVENE'),None)
 r=onset.get(case['scenario_id']); t1=float(r['candidate_onset_ms']) if r and r['candidate_onset_ms'] else None
 t3=hit['decision_complete_ms_simulated'] if hit else None
 return {'scenario_id':case['scenario_id'],'target':case['target'],'scheme':mode,'t1_candidate_ms':t1,'t1_source':r['candidate_source'] if r else None,'t3_estimated_ms':t3,'delta_ms':round(t3-t1,3) if t3 is not None and t1 is not None else None,'first_warning_source':hit['source'] if hit else None,'trace':trace}

# Warm all policies before measuring, with an input unrelated to the dataset.
harness.local([{'speaker':'other','text':'How can I help?'}], 'hello there'); mini([], 'hello there'); rules(harness, [], 'hello there')
results=[]
for i,c in enumerate(cases):
 for mode in ['rules_only','minilm_only','cascade']:
  results.append(run(c,mode))
 if (i+1)%25==0: print(f'{i+1}/150 scenarios replayed',flush=True)
summary={}
for mode in ['rules_only','minilm_only','cascade']:
 group=[x for x in results if x['scheme']==mode]; risk=[x for x in group if x['target']=='INTERVENE']; safe=[x for x in group if x['target']=='WAIT']
 valid=[x for x in risk if x['delta_ms'] is not None]
 summary[mode]={'risk_total':len(risk),'risk_warned':sum(x['t3_estimated_ms'] is not None for x in risk),'risk_no_warning':sum(x['t3_estimated_ms'] is None for x in risk),'risk_with_candidate_t1':sum(x['t1_candidate_ms'] is not None for x in risk),'before_onset':sum(x['delta_ms']<0 for x in valid),'at_onset':sum(x['delta_ms']==0 for x in valid),'after_onset':sum(x['delta_ms']>0 for x in valid),'safe_total':len(safe),'safe_warned':sum(x['t3_estimated_ms'] is not None for x in safe),'first_warning_sources':dict(Counter(x['first_warning_source'] for x in group if x['first_warning_source']))}
report={'method':'Offline replay of saved Deepgram ASR partials and received times, history transcripts and 150 audio cases. T3 = saved ASR arrival plus newly measured local inference and simulated serial queue; not a new streaming audio call or directly measured wall-clock T3.','clock':'ASR word timestamps relative to audio start; ASR arrivals relative to local audio-send start. Alignment and send-start offset unverified.','t1':'ASR candidate password value onset; 68 exact first-word matches and 7 post-hoc candidates, none human-verified.','warning':'First INTERVENE only. Negative delta means warning decision before candidate password onset; not necessarily justified or audible.','evaluation':'Exploratory reused synthetic test; no human review of decision correctness, no speaker playback, no recipient, no paid calls. T1 absent for safe cases.','model_threshold':THRESHOLD,'low_threshold':LOW_RISK,'torch_threads':4,'platform':platform.platform(),'input_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'input_onsets_sha256':hashlib.sha256(onsets.read_bytes()).hexdigest(),'results':results,'summary':summary}
(P/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
with (P/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['scenario_id','target','scheme','t1_candidate_ms','t1_source','t3_estimated_ms','delta_ms','first_warning_source'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:x[k] for k in fields} for x in results])
print(json.dumps(summary,indent=2),flush=True)
