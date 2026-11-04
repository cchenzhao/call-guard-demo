import base64,json,os,time,threading,queue,urllib.request
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI,Request,HTTPException,WebSocket,WebSocketDisconnect
from fastapi.responses import StreamingResponse,JSONResponse
from harness import Harness
from voice import agent_messages,agent_text_stream,tts_audio_stream,tls

app=FastAPI()
@lru_cache(maxsize=1)
def get_harness(): return Harness()
lock=threading.Lock()

@app.websocket('/api/speech')
async def speech(client:WebSocket):
 import asyncio
 from websockets.asyncio.client import connect
 from starlette.concurrency import run_in_threadpool
 origin=client.headers.get('origin')
 host=client.headers.get('x-forwarded-host',client.headers.get('host',''))
 if origin and origin.rstrip('/') not in ['https://'+host,'http://'+host]:
  await client.close(code=1008);return
 await client.accept()
 try:
  config=await asyncio.wait_for(client.receive_json(),10)
  hist=config.get('history',[])
  if not isinstance(hist,list) or len(hist)>40 or any(not isinstance(x,dict) or x.get('speaker') not in ['user','other'] or not isinstance(x.get('text'),str) or len(x['text'])>2000 for x in hist):
   raise ValueError('Invalid history')
  encoding='&encoding=linear16&sample_rate=16000' if config.get('format')=='linear16' else ''
  url='wss://api.deepgram.com/v1/listen?model=nova-3&language=en-US&interim_results=true&punctuate=true&endpointing=300'+encoding
  async with connect(url,additional_headers={'Authorization':'Token '+os.environ['DEEPGRAM_API_KEY']},ssl=tls(),proxy=None,open_timeout=20,max_size=2**20) as upstream:
   await client.send_json({'type':'ready'})
   def decide(text):
    start=time.perf_counter()
    with lock:d=get_harness().local(hist,text)
    return {**d,'decision_ms':(time.perf_counter()-start)*1000}
   async def receive():
    finalized='';latest='';warned=False
    async for raw in upstream:
     event=json.loads(raw)
     if event.get('type')=='Error':raise RuntimeError('Speech recognition failed')
     if event.get('type')!='Results':continue
     text=event['channel']['alternatives'][0].get('transcript','')
     if not text:continue
     latest=(finalized+' '+text).strip()
     if not warned:
      d=await run_in_threadpool(decide,latest)
      await client.send_json({'type':'decision','decision':d});warned=d['action']=='INTERVENE'
     if event.get('is_final'):finalized=latest
    await client.send_json({'type':'done','text':latest})
   receiver=asyncio.create_task(receive())
   try:
    while True:
     message=await asyncio.wait_for(client.receive(),90)
     if message['type']=='websocket.disconnect':break
     if message.get('bytes') is not None:
      if len(message['bytes'])>2**20:raise ValueError('Chunk too large')
      await upstream.send(message['bytes'])
     elif message.get('text') and json.loads(message['text']).get('type')=='finish':
      await upstream.send(json.dumps({'type':'CloseStream'}))
      await asyncio.wait_for(receiver,20);break
   finally:
    if not receiver.done():receiver.cancel()
    await asyncio.gather(receiver,return_exceptions=True)
 except WebSocketDisconnect:pass
 except Exception:
  try:await client.send_json({'type':'error','message':'Speech connection failed. Please restart the call.'})
  except Exception:pass
 finally:
  try:await client.close()
  except Exception:pass

def check_origin(request):
 origin=request.headers.get('origin')
 host=request.headers.get('x-forwarded-host',request.headers.get('host',''))
 if origin and origin.rstrip('/') not in ['https://'+host,'http://'+host]:raise HTTPException(403,'Origin not allowed')

async def body(request):
 check_origin(request)
 raw=await request.body()
 if len(raw)>24000:raise HTTPException(413,'Request too large')
 try:value=json.loads(raw)
 except Exception:raise HTTPException(400,'Invalid JSON')
 hist=value.get('history',[]); text=value.get('text','')
 if not isinstance(hist,list) or len(hist)>40 or not isinstance(text,str) or len(text)>2000:raise HTTPException(400,'Invalid input')
 if any(not isinstance(x,dict) or x.get('speaker') not in ['user','other'] or not isinstance(x.get('text'),str) or len(x['text'])>2000 for x in hist):raise HTTPException(400,'Invalid history')
 return hist,text

@app.get('/api/health')
def health():
 get_harness()
 return {'ready':True,'detector':'original rules + candidate_003 MiniLM FP32 ONNX','speech_configured':bool(os.environ.get('DEEPGRAM_API_KEY')),'agent_configured':bool(os.environ.get('OPENROUTER_API_KEY'))}

@app.post('/api/safety')
async def safety(request:Request):
 hist,text=await body(request)
 def classify():
  start=time.perf_counter()
  with lock:r=get_harness().local(hist,text)
  return {**r,'decision_ms':(time.perf_counter()-start)*1000}
 from starlette.concurrency import run_in_threadpool
 return await run_in_threadpool(classify)

@app.post('/api/speech-token')
async def speech_token(request:Request):
 await body(request)
 if not os.environ.get('DEEPGRAM_API_KEY'):raise HTTPException(503,'Speech is not configured')
 def grant():
  req=urllib.request.Request('https://api.deepgram.com/v1/auth/grant',data=b'{}',headers={'Authorization':'Token '+os.environ['DEEPGRAM_API_KEY'],'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(req,context=tls(),timeout=15) as r:return json.loads(r.read())
 try:
  from starlette.concurrency import run_in_threadpool
  data=await run_in_threadpool(grant)
  return JSONResponse({'access_token':data['access_token'],'expires_in':data.get('expires_in',30)},headers={'Cache-Control':'no-store'})
 except Exception:raise HTTPException(502,'Unable to start speech recognition')

def frame(x):return json.dumps(x)+'\n'
def tts_frames(text):
 try:
  for audio in tts_audio_stream(text):yield frame({'type':'audio','data':base64.b64encode(audio).decode()})
  yield frame({'type':'done'})
 except Exception:yield frame({'type':'error','message':'Speech generation failed'})

@app.post('/api/tts')
async def tts(request:Request):
 _,text=await body(request)
 return StreamingResponse(tts_frames(text),media_type='application/x-ndjson',headers={'Cache-Control':'no-store'})

def reply_frames(hist,text):
 # Stream text and speech concurrently using the same sentence boundaries.
 q=queue.Queue();sentences=queue.Queue();spoken=[]
 def speak():
  try:
   while True:
    txt=sentences.get()
    if txt is None:break
    for audio in tts_audio_stream(txt):q.put({'type':'audio','data':base64.b64encode(audio).decode()})
  except Exception:q.put({'type':'error','message':'Speech generation failed'})
  finally:q.put(None)
 def generate():
  import re
  sentence=''
  try:
   for delta in agent_text_stream(agent_messages(hist,text)):
    spoken.append(delta);sentence+=delta;q.put({'type':'text','delta':delta})
    if re.search(r'[.!?](?:\s|$)',sentence) and len(sentence.strip())>=12:sentences.put(sentence.strip());sentence=''
   if sentence.strip():sentences.put(sentence.strip())
  except Exception:q.put({'type':'error','message':'Caller response failed'})
  finally:sentences.put(None)
 threading.Thread(target=speak,daemon=True).start();threading.Thread(target=generate,daemon=True).start()
 while True:
  try:item=q.get(timeout=55)
  except queue.Empty:yield frame({'type':'error','message':'Response timed out'});break
  if item is None:break
  yield frame(item)
 yield frame({'type':'done','text':''.join(spoken).strip()})

@app.post('/api/agent-reply')
async def reply(request:Request):
 hist,text=await body(request)
 return StreamingResponse(reply_frames(hist,text),media_type='application/x-ndjson',headers={'Cache-Control':'no-store'})
