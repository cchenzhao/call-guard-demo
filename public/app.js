const el = id => document.getElementById(id);
let history = [], finalText = '', turnId = '', holding = false, turnSent = false, intervened = false;
let recorder, stream, socket, busy = false, blocked = false, agentAbort = null;
const warningAudio = new Audio('/warning.mp3');
warningAudio.preload = 'auto';
let agentAudio = null, warningFinished = Promise.resolve(), warningWasPlayed = false;
const decisionLog = [], auditStarted = performance.now();
const ruleSources = new Set(['fast-path', 'context-fast-path', 'refusal', 'context']);
function recordDecision(decision) {
  const rule = ruleSources.has(decision.source);
  const entry = {check: decisionLog.length + 1, turn_id: turnId,
    elapsed_ms: Math.round(performance.now() - auditStarted),
    action: decision.action, source: decision.source,
    layer: rule ? 'fast_path_rules' : decision.source,
    minilm_ran: decision.source === 'minilm',
    model_score: decision.probability ?? null,
    local_decision_ms: decision.decision_ms ?? null, reason: decision.reason};
  decisionLog.push(entry);
  const ruleCount = decisionLog.filter(x => x.layer === 'fast_path_rules').length;
  const modelCount = decisionLog.filter(x => x.minilm_ran).length;
  el('audit-counts').textContent = `Rules ${ruleCount} · MiniLM ${modelCount}`;
  const item = document.createElement('li');
  item.textContent = `#${entry.check} · ${entry.layer} / ${entry.source} → ${entry.action} · ${entry.reason}`
    + (entry.minilm_ran ? ` Score: ${(entry.model_score * 100).toFixed(1)}%.` : ' MiniLM skipped.');
  el('audit-events').appendChild(item);
}
function exportAudit() {
  const blob = new Blob([JSON.stringify({schema: 'call-guard-decisions-v1',
    note: 'Observed cascade decisions only; no counterfactual MiniLM predictions for rule-routed inputs. No transcript or audio included. decision_ms excludes ASR and warning playback.',
    decisions: decisionLog}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = 'call-guard-decisions.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const warningVoice = () => {
  agentAbort?.abort();
  agentAbort = null;
  agentAudio?.pause();
  agentAudio = null;
  warningWasPlayed = true;
  warningAudio.currentTime = 0;
  warningFinished = new Promise(resolve => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      warningAudio.removeEventListener('ended', finish);
      warningAudio.removeEventListener('error', finish);
      resolve();
    };
    warningAudio.addEventListener('ended', finish, {once: true});
    warningAudio.addEventListener('error', finish, {once: true});
    warningAudio.play().catch(() => { warningWasPlayed = false; el('why').textContent = 'Allow audio playback to hear the warning.'; finish(); });
  });
};

function completeIntervention() {
  if (blocked) return;
  blocked = true;
  holding = false;
  busy = false;
  // Surface the warning before stopping media or the socket: cleanup failures
  // must never prevent an intervention from reaching the user.
  el('warning').hidden = false;
  warningVoice();
  agentAbort?.abort();
  agentAbort = null;
  agentAudio?.pause();
  agentAudio = null;
  try { if (recorder?.state === 'recording') recorder.stop(); } catch (_) {}
  try { stream?.getTracks().forEach(track => track.stop()); } catch (_) {}
  try { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type: 'CloseStream'})); } catch (_) {}
  el('mic').disabled = true;
  el('mic-label').textContent = 'Call ended';
  el('start').disabled = true;
  el('call-status').textContent = 'Scam interaction ended.';
}

// Best-effort filtering of the fixed warning if the speakers feed it back into
// the mic. Echo cancellation is not perfect; the user can say the same words.
function removeWarningEcho(text) {
  if (!warningWasPlayed) return text;
  return text.replace(/\bpause\s+before\s+sharing\s*[,.:;!?-]*\s*keep\s+your\s+private\s+account\s+password\s+to\s+yourself\s*[.!?]?/gi, '').trim();
}

async function post(url, body) {
  const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  return response.json();
}
function renderDecision(decision) {
  if (blocked) return;
  recordDecision(decision);
  el('state').textContent = decision.action;
  el('path').textContent = decision.source + (decision.probability === null ? '' : ` · risk ${(decision.probability * 100).toFixed(1)}%`);
  el('why').textContent = decision.reason;
  if (decision.action === 'INTERVENE' && !intervened) {
    intervened = true;
    completeIntervention();
  }
}
async function safety(text, final = false) {
  if (text.trim()) renderDecision(await post('/api/safety', {history, text, final, turn_id: turnId}));
}
function setListening(on) {
  el('mic').classList.toggle('listening', on);
  if (blocked) { el('mic-label').textContent = 'Call ended'; return; }
  el('mic-label').textContent = on ? 'Release to send' : 'Hold to talk';
  el('call-status').textContent = on ? 'Listening securely…' : 'Ready when you are.';
}
async function consumeAudioStream(response, onText) {
  const media = new MediaSource();
  const audio = new Audio(URL.createObjectURL(media));
  agentAudio = audio;
  const queue = [];
  let source, closed = false;
  warningFinished.then(() => {
    if (!blocked && agentAudio === audio) audio.play().catch(() => {});
  });
  media.addEventListener('sourceopen', () => {
    source = media.addSourceBuffer('audio/mpeg'); source.addEventListener('updateend', pump); pump();
  });
  function pump() {
    if (source && !source.updating && queue.length) source.appendBuffer(queue.shift());
    else if (source && !source.updating && closed && media.readyState === 'open') media.endOfStream();
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let pending = '', spoken = '';
  while (true) {
    if (blocked) { reader.cancel().catch(() => {}); break; }
    const {done, value} = await reader.read();
    pending += decoder.decode(value || new Uint8Array(), {stream: !done});
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      const frame = JSON.parse(pending.slice(0, newline)); pending = pending.slice(newline + 1);
      if (frame.type === 'text') { spoken += frame.delta; onText?.(spoken); }
      else if (frame.type === 'audio') { queue.push(Uint8Array.from(atob(frame.data), char => char.charCodeAt(0))); pump(); }
      else if (frame.type === 'done') spoken = frame.text || spoken;
      else if (frame.type === 'error') el('why').textContent = `Agent audio error: ${frame.message}`;
    }
    if (done) break;
  }
  closed = true; pump(); return spoken;
}
async function streamReply() {
  if (blocked) return;
  el('agent').textContent = 'Scam Ninja is responding…';
  el('call-status').textContent = 'Scam Ninja is speaking…';
  agentAbort = new AbortController();
  let response;
  try {
    response = await fetch('/api/agent-reply', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({history, text: finalText}), signal: agentAbort.signal});
  } catch (error) {
    if (error.name === 'AbortError' || blocked) return;
    el('agent').textContent = 'Agent unavailable.';
    return;
  }
  if (!response.ok || !response.body) { el('agent').textContent = 'Agent unavailable.'; return; }
  const spoken = await consumeAudioStream(response, text => { el('agent').textContent = text; });
  if (!blocked && spoken) history.push({speaker: 'user', text: finalText}, {speaker: 'other', text: spoken});
  if (!blocked) el('call-status').textContent = 'Ready when you are.';
}
async function streamTTS(text) {
  if (blocked) return;
  const response = await fetch('/api/tts', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({history: [], text})});
  if (response.ok && response.body) await consumeAudioStream(response);
}
async function finishTurn() {
  setListening(false);
  stream?.getTracks().forEach(track => track.stop());
  el('mic-label').textContent = 'Finishing…';
  el('call-status').textContent = 'Processing your voice…';
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type: 'CloseStream'}));
}
async function beginTurn(event) {
  if (el('mic').disabled || holding || busy || blocked) return;
  event.preventDefault();
  holding = true; busy = true; turnSent = false; intervened = false; warningWasPlayed = false; finalText = ''; turnId = crypto.randomUUID();
  el('mic').setPointerCapture?.(event.pointerId);
  el('mic-label').textContent = 'Connecting…';
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true}});
    if (!holding) { stream.getTracks().forEach(track => track.stop()); busy = false; setListening(false); return; }
    const tokenResponse = await post('/api/speech-token', {});
    if (!tokenResponse.access_token) throw new Error('Speech connection unavailable.');
    socket = new WebSocket('wss://api.deepgram.com/v1/listen?model=nova-3&language=en-US&interim_results=true&punctuate=true&endpointing=300', ['bearer', tokenResponse.access_token]);
    let finalized = '', latest = '', pendingChecks = Promise.resolve(), finished = false;
    const finish = async () => {
      if (finished) return;
      finished = true;
      clearTimeout(flushTimer);
      await pendingChecks;
      if (blocked) { socket.close(); return; }
      finalText = removeWarningEcho(latest);
      try { if (!turnSent && finalText.trim()) { turnSent = true; await streamReply(); } }
      finally { busy = false; setListening(false); socket.close(); }
    };
    let flushTimer;
    socket.onopen = () => {
      if (!holding) { socket.close(); busy = false; setListening(false); return; }
      recorder = new MediaRecorder(stream, {mimeType: 'audio/webm;codecs=opus'});
      recorder.ondataavailable = chunk => { if (chunk.data.size && socket.readyState === WebSocket.OPEN) socket.send(chunk.data); };
      recorder.onstop = () => {
        setListening(false); stream?.getTracks().forEach(track => track.stop());
        if (blocked) { socket.close(); return; }
        el('mic-label').textContent = 'Finishing…';
        if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type:'CloseStream'}));
        flushTimer = setTimeout(finish, 5000);
      };
      recorder.start(100); setListening(true);
    };
    socket.onmessage = message => {
      const frame = JSON.parse(message.data);
      if (frame.type === 'Results') {
        const text = frame.channel?.alternatives?.[0]?.transcript || '';
        if (text) {
          latest = (finalized + ' ' + text).trim();
          const partial = latest;
          pendingChecks = pendingChecks.then(async () => {
            if (!blocked) await safety(partial);
          }).catch(() => { el('why').textContent = 'Detection unavailable. Please restart the call.'; });
          if (frame.is_final) finalized = latest;
        }
      } else if (frame.type === 'Metadata') finish();
      else if (frame.type === 'Error') { el('why').textContent = 'Speech recognition failed.'; socket.close(); }
    };
    socket.onclose = () => {
      holding = false;
      if (recorder?.state === 'recording') recorder.stop();
      stream?.getTracks().forEach(track => track.stop());
      finish();
    };
    socket.onerror = () => { el('why').textContent = 'Speech connection unavailable. Please restart.'; };

  } catch (error) {
    holding = false; busy = false; setListening(false);
    el('why').textContent = 'Microphone unavailable: ' + error.message;
  }
}
function endTurn(event) {
  if (!holding) return;
  event.preventDefault(); holding = false;
  if (recorder?.state === 'recording') recorder.stop();
}
el('start').onclick = () => {
  if (blocked) return;
  const greeting = 'Hello, this is account support. I am calling about a verification issue with your account. Can you help me confirm your details?';
  history = [{speaker: 'other', text: greeting}]; el('agent').textContent = greeting; el('mic').disabled = false; el('call-status').textContent = 'Scam Ninja is speaking…'; streamTTS(greeting);
};
el('mic').addEventListener('pointerdown', beginTurn);
['pointerup', 'pointercancel'].forEach(type => el('mic').addEventListener(type, endTurn));
el('reset-call').onclick = () => window.location.reload();
el('export-audit').onclick = exportAudit;

// Shareable visual-only preview for the project demo; it never opens the mic.
if (new URLSearchParams(window.location.search).get('preview') === 'outcome') completeIntervention();

// Warm the cloud classifier before the user starts a call.
el('start').disabled = true;
el('call-status').textContent = 'Preparing security harness…';
fetch('/api/health').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(status => {
  if (!status.speech_configured || !status.agent_configured) {
    if (!blocked) el('call-status').textContent = 'Security harness ready. Voice service setup is pending.';
    return;
  }
  if (!blocked) { el('start').disabled = false; el('call-status').textContent = 'Ready when you are.'; }
}).catch(() => { el('call-status').textContent = 'Security harness unavailable. Reload to retry.'; });
