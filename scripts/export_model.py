from pathlib import Path
import torch,onnx,hashlib,json
from transformers import AutoModelForSequenceClassification,AutoTokenizer
P=Path(__file__).resolve().parents[1];src=P.parent/'security_harness/classifier/runs/candidate_003/model'
torch.set_num_threads(4)
m=AutoModelForSequenceClassification.from_pretrained(src,attn_implementation='eager').eval();t=AutoTokenizer.from_pretrained(src);batch=t('other: What is your account password?','user: I cannot share it.',return_tensors='pt')
class Wrapper(torch.nn.Module):
 def __init__(self):super().__init__();self.model=m
 def forward(self,input_ids,attention_mask,token_type_ids):return self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).logits
out=Path('/private/tmp/call-guard-fp32.onnx')
torch.onnx.export(Wrapper().eval(),(batch['input_ids'],batch['attention_mask'],batch['token_type_ids']),str(out),input_names=['input_ids','attention_mask','token_type_ids'],output_names=['logits'],dynamic_axes={x:{0:'batch',1:'sequence'} for x in ['input_ids','attention_mask','token_type_ids']},opset_version=17,dynamo=False)
onnx.checker.check_model(str(out));raw=out.read_bytes();part_size=60*1024*1024
for i in range(0,len(raw),part_size):(P/'model'/f'weights.part{i//part_size:02d}').write_bytes(raw[i:i+part_size])
(P/'model/manifest.json').write_text(json.dumps({'format':'ONNX FP32, unquantized','source':'candidate_003','threshold':0.9498332142829895,'onnx_sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'parts':(len(raw)+part_size-1)//part_size},indent=2))
print('Exported original FP32 weights without quantization:',len(raw),'bytes')
