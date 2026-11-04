import ast,json,shutil,textwrap,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1];source=P.parent/'security_harness';web=source/'live_demo/web'
for d in ['public','public/images','model','api','scripts']: (P/d).mkdir(exist_ok=True)
for name in ['index.html','style.css','app.js','warning.mp3']:
 if (web/name).exists():shutil.copy2(web/name,P/'public'/name)
# Include all style files referenced by the existing page.
for x in web.glob('*.css'):shutil.copy2(x,P/'public'/x.name)
for src,dest in [('Ninja.jpeg','ninja.jpeg'),('Cat.jpg','cat.jpg')]:shutil.copy2(P.parent/'Image'/src,P/'public/images'/dest)
s=(source/'live_demo/server.py').read_text();tree=ast.parse(s)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Harness')
class_src=ast.get_source_segment(s,cls)
# Keep the rule decisions unchanged, replacing only tokenizer and inference runtime.
a=class_src.index('    def __init__');b=class_src.index('    def local')
class_src=class_src[:a]+'''    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=192, strategy='only_first')
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        raw=b''.join(p.read_bytes() for p in sorted(MODEL_DIR.glob('weights.part*')))
        self.model=ort.InferenceSession(raw,sess_options=options,providers=['CPUExecutionProvider'])
        self.fast=FastPathPolicy(WaitPolicy())

'''+class_src[b:]
a=class_src.index('        batch = self.tokenizer(');b=class_src.index('        if probability >=',a)
class_src=class_src[:a]+'''        enc=self.tokenizer.encode(history_text, 'user: '+partial)
        feeds={'input_ids': np.array([enc.ids],dtype=np.int64),
               'attention_mask': np.array([enc.attention_mask],dtype=np.int64),
               'token_type_ids': np.array([enc.type_ids],dtype=np.int64)}
        logits=self.model.run(None,feeds)[0][0]
        scores=np.exp(logits-np.max(logits));probability=float(scores[1]/scores.sum())
'''+class_src[b:]
class_src=class_src[:class_src.index('    def fallback')]
ps=(source/'policies.py').read_text();pt=ast.parse(ps)
classes='\n\n'.join(ast.get_source_segment(ps,x) for x in pt.body if isinstance(x,ast.ClassDef) and x.name in ['WaitPolicy','FastPathPolicy'])
header='''import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
MODEL_DIR=Path(__file__).resolve().parent/'model'
THRESHOLD=0.9498332142829895
LOW_RISK=0.10
@dataclass
class PolicyReply:
    output: object
    source: str='policy'
'''
(P/'harness.py').write_text(header+'\n'+classes+'\n\n'+class_src)
# Use the same fictional caller prompt, model, streaming protocol, and TTS.
funcs='\n\n'.join(ast.get_source_segment(s,x) for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['agent_messages','agent_text_stream','tts_audio_stream','tls'])
(P/'voice.py').write_text('import os,json,re,ssl,urllib.request\nAGENT_LLM="mistralai/mistral-small-3.2-24b-instruct"\ndef deepgram_key(): return os.environ["DEEPGRAM_API_KEY"]\n\n'+funcs+'\n')
for n in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json']:
 shutil.copy2(source/'classifier/runs/candidate_003/model'/n,P/'model'/n)
print('Created isolated runtime from original harness and voice implementation.')
