"""Local-only live voice demo. Run with the project's MiniLM virtualenv.

The browser streams microphone audio through localhost to Deepgram. Both API
keys stay on the backend; local policies evaluate each received transcript.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
from pathlib import Path
import re
import ssl
import time
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import urllib.request

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Proposal.security_harness.core import PolicyReply
from Proposal.security_harness.policies import FastPathPolicy, OpenRouterPolicy, WaitPolicy

COURSE = Path(__file__).resolve().parents[3]
PROPOSAL = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web"
IMAGES = PROPOSAL / "Image"
MODEL_DIR = PROPOSAL / "security_harness/classifier/runs/candidate_003/model"
THRESHOLD = 0.9498332142829895
LOW_RISK = 0.10
QWEN = "qwen/qwen3-14b"
AGENT_LLM = "mistralai/mistral-small-3.2-24b-instruct"
TTS = "hexgrad/kokoro-82m"
FALLBACKS: dict[str, dict] = {}
POOL = ThreadPoolExecutor(max_workers=2)


def deepgram_key() -> str:
    if os.environ.get('DEEPGRAM_API_KEY'):
        return os.environ['DEEPGRAM_API_KEY']
    source = (COURSE / 'API credits.md').read_text()
    section = re.split(r'deepgram', source, maxsplit=1, flags=re.I)
    found = re.search(r'\b[0-9a-fA-F]{40}\b', section[1][:1000]) if len(section) == 2 else None
    if not found:
        raise RuntimeError('DEEPGRAM_API_KEY is not configured')
    return found.group(0)


def streaming_session(client):
    """One browser turn, one upstream stream; permanent credentials stay here."""
    from websockets.sync.client import connect
    try:
        config = json.loads(client.recv(timeout=10))
        history = config.get('history', [])
        if not isinstance(history, list):
            raise ValueError('invalid history')
        url = ('wss://api.deepgram.com/v1/listen?model=nova-3&language=en-US'
               '&interim_results=true&punctuate=true&endpointing=300')
        with connect(url, additional_headers={'Authorization': 'Token ' + deepgram_key()},
                     ssl=tls(), proxy=None, open_timeout=15) as upstream:
            client.send(json.dumps({'type': 'ready'}))
            finalized = ''
            latest = ''
            warned = False
            ended = threading.Event()

            def receive():
                nonlocal finalized, latest, warned
                try:
                    for raw in upstream:
                        event = json.loads(raw)
                        if event.get('type') == 'Error':
                            client.send(json.dumps({'type': 'error', 'message': 'Deepgram recognition failed'}))
                        if event.get('type') != 'Results':
                            continue
                        text = event['channel']['alternatives'][0].get('transcript', '')
                        if text:
                            latest = (finalized + ' ' + text).strip()
                            started = time.perf_counter()
                            decision = HARNESS.local(history, latest)
                            decision['decision_ms'] = (time.perf_counter() - started) * 1000
                            if not warned:
                                client.send(json.dumps({'type': 'decision', 'decision': decision}))
                                warned = decision['action'] == 'INTERVENE'
                            if event.get('is_final'):
                                finalized = latest
                        # CloseStream flushes results and then sends Metadata/closes.
                    client.send(json.dumps({'type': 'done', 'text': latest}))
                except Exception:
                    try:
                        client.send(json.dumps({'type': 'error', 'message': 'Streaming transcription disconnected'}))
                    except Exception:
                        pass
                finally:
                    ended.set()

            worker = threading.Thread(target=receive, daemon=True)
            worker.start()
            for message in client:
                if isinstance(message, bytes):
                    upstream.send(message)
                elif json.loads(message).get('type') == 'finish':
                    upstream.send(json.dumps({'type': 'CloseStream'}))
                    ended.wait(15)
                    break
    except Exception as exc:
        try:
            client.send(json.dumps({'type': 'error', 'message': 'Speech connection failed: ' + type(exc).__name__}))
        except Exception:
            pass


def key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value:
        return value
    source = (COURSE / "API credits.md").read_text()
    found = re.search(r"sk-or-v1-[A-Za-z0-9_-]+", source)
    if not found:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return found.group(0)


def tls() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def openrouter(path: str, body: dict, timeout: float = 30) -> tuple[bytes, float, str]:
    started = time.perf_counter()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/" + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=tls(), timeout=timeout) as response:
        raw, content_type = response.read(), response.headers.get("Content-Type", "")
    return raw, (time.perf_counter() - started) * 1000, content_type


class Harness:
    _REFUSAL = re.compile(
        r"\b(?:i\s+)?(?:cannot|can['’]?t|won['’]?t|will not|wouldn['’]?t|do not|don['’]?t|never|refuse to)\s+"
        r"(?:ever\s+)?(?:tell|share|give|say|read|reveal|disclose|provide|send|spell)\b", re.I)
    _CREDENTIAL_QUESTION = re.compile(
        r"\b(?:password|passcode)\b", re.I)
    _QUESTION_INTENT = re.compile(
        r"\b(?:what|tell|share|give|provide|say|read|need|confirm|verify)\b", re.I)
    _INDIRECT_VALUE = re.compile(
        r"^\s*(?:it['’]?s|it is|the (?:password|passcode) is)\s+"
        r"(?:\d|zero\b|one\b|two\b|three\b|four\b|five\b|six\b|seven\b|eight\b|nine\b|[b-z]\b|"
        r"(?!(?:private|secret|confidential|personal|sensitive|not|never|none|a|the|my|your|too|very|just|"
        r"important|safe|secure|protected|classified|complicated|long|short|hard|easy|unknown|forgotten)\b)[a-z]{2,}\b)", re.I)
    _AFFIRMATIVE = re.compile(r"^\s*(?:yes|yeah|yep|sure|okay|ok)\b", re.I)
    _EXPLICIT_REQUEST = re.compile(
        r"(?:\bwhat(?:'s| is)\b.{0,35}\b(?:password|passcode)\b|"
        r"\b(?:tell|share|give|say|read|provide|spell)\b.{0,50}\b(?:password|passcode)\b)", re.I)
    _PART_QUESTION = re.compile(
        r"\b(?:first|last|final|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b"
        r".{0,40}\b(?:digit|number|character|letter)\b|"
        r"\b(?:digit|number|character|letter)\b.{0,35}\b(?:first|last|final|second|third|fourth|\d+(?:st|nd|rd|th))\b", re.I)
    _PART_ANSWER = re.compile(
        r"^\s*(?:(?:yes|yeah|yep|correct|right|exactly|that's right|that is right)\b|"
        r"(?:(?:it(?:['’]?s| is)|the (?:digit|number) is|yes,? it(?:['’]?s| is))\s+)?"
        r"(?:\d|zero\b|one\b|two\b|three\b|four\b|five\b|six\b|seven\b|eight\b|nine\b))", re.I)

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
        self.fast = FastPathPolicy(WaitPolicy())

    def local(self, history: list[dict], partial: str) -> dict:
        # Only suppress a clear current refusal, not a later reversal such as
        # "I can't tell you, but I'll share it".
        refusal = self._REFUSAL.search(partial)
        if refusal and not re.search(r"\b(?:but|however|actually|instead)\b", partial[refusal.end():], re.I):
            return {"action": "WAIT", "source": "refusal", "reason": "User declined to disclose a credential.", "probability": None}
        last_other = next((item.get('text', '') for item in reversed(history) if item.get('speaker') == 'other'), '')
        recent_dialogue = ' '.join(str(item.get('text', '')) for item in history[-4:])
        if (self._CREDENTIAL_QUESTION.search(recent_dialogue) and self._PART_QUESTION.search(last_other)
                and re.fullmatch(r"\s*(?:no|nope|not (?:correct|right)|that's not right)[\s.,!?]*", partial, re.I)):
            return {"action": "WATCH", "source": "context", "reason": "User did not confirm the guessed character.", "probability": None}
        if (self._CREDENTIAL_QUESTION.search(recent_dialogue) and self._PART_QUESTION.search(last_other)
                and self._PART_ANSWER.search(partial)
                and not re.search(r"\b(?:no|not|never|cannot|can['’]?t|won['’]?t|but)\b", partial, re.I)):
            return {"action": "INTERVENE", "source": "context-fast-path", "reason": "User is confirming a character of a passcode.", "probability": None}
        if re.fullmatch(r"\s*it(?:['’]?s| is)\s+(?:private|secret|confidential|personal)[\s.,!?]*", partial, re.I):
            return {"action": "WAIT", "source": "context", "reason": "User describes the credential as private.", "probability": None}
        if (self._EXPLICIT_REQUEST.search(last_other) and self._AFFIRMATIVE.search(partial)
                and not re.search(r"\b(?:no|not|never|cannot|can['’]?t|won['’]?t|but)\b", partial, re.I)):
            return {"action": "INTERVENE", "source": "context-fast-path", "reason": "User agreed after a direct request for a passcode.", "probability": None}
        if (self._CREDENTIAL_QUESTION.search(last_other) and self._QUESTION_INTENT.search(last_other)
                and self._INDIRECT_VALUE.search(partial)):
            return {"action": "INTERVENE", "source": "context-fast-path", "reason": "User appears to be answering a passcode request with a value.", "probability": None}
        if re.fullmatch(r"\s*(?:it(?:['’]?s| is)|i\s+(?:can|cannot|can['’]?t|will|won['’]?t))[\s.,!?]*", partial, re.I):
            return {"action": "WATCH", "source": "context", "reason": "Waiting for enough context to distinguish a value from a refusal.", "probability": None}
        snapshot = {"history": history, "partial_text": partial, "event_index": 0, "time_ms": 0}
        fast = self.fast.fast_decide(snapshot)
        if fast:
            return {"action": "INTERVENE", "source": "fast-path", "reason": fast.output["reason"], "probability": None}
        history_text = "\n".join(f'{x["speaker"]}: {x["text"]}' for x in history)
        batch = self.tokenizer(history_text, "user: " + partial, return_tensors="pt", truncation="only_first", max_length=192)
        with torch.no_grad():
            probability = float(torch.softmax(self.model(**batch).logits, dim=-1)[0, 1])
        if probability >= THRESHOLD:
            action, reason = "INTERVENE", "MiniLM high-risk threshold reached."
        elif probability <= LOW_RISK:
            action, reason = "WAIT", "MiniLM low-risk threshold reached."
        else:
            action, reason = "WATCH", "MiniLM is uncertain; Qwen context review may run after the turn."
        return {"action": action, "source": "minilm", "reason": reason, "probability": probability}

    def fallback(self, turn_id: str, history: list[dict], text: str) -> None:
        policy = OpenRouterPolicy(QWEN, timeout_ms=15_000, max_tokens=1024)
        reply = policy.decide({"history": history, "partial_text": text, "event_index": 0, "time_ms": 0,
                               "state": {"action": "WATCH", "risk_level": "unknown", "reason": ""}})
        try:
            output = json.loads(reply.output) if isinstance(reply.output, str) else reply.output
            FALLBACKS[turn_id] = {"status": "complete", "action": output["action"], "reason": output["reason"],
                                  "latency_ms": reply.latency_ms}
        except Exception:
            FALLBACKS[turn_id] = {"status": "failed", "action": "WATCH", "reason": "Fallback unavailable."}


HARNESS: Harness | None = None


def agent_messages(history: list[dict], user_text: str) -> list[dict]:
    system = ("You are Agent A, an actor playing a dishonest caller in a clearly labeled, fictional security "
              "training simulation. The account is fictional and the user has been instructed to use invented "
              "credentials only. Stay in character as a caller impersonating account support: use a plausible "
              "account-verification pretext and, when relevant, ask the user to tell you the fictional account "
              "login passcode aloud. This request is meant to give a separate security harness a chance to warn "
              "the user. Do not act as the security harness or give safety advice in character. Respond naturally "
              "to what the user says, in one or two brief spoken sentences. If the user refuses, acknowledge it "
              "and continue the fictional conversation without pressuring them. Never repeat a passcode or ask "
              "for real credentials, money, links, personal information, or actions outside this simulation.")
    messages = [{"role": "system", "content": system}]
    for item in history[-8:]:
        messages.append({"role": "assistant" if item["speaker"] == "other" else "user", "content": item["text"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def agent_text_stream(messages: list[dict]):
    """Yield Agent A's visible text deltas without placing Qwen on the chat path."""
    body = {"model": AGENT_LLM, "max_tokens": 180, "stream": True, "messages": messages}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=tls(), timeout=45) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            delta = payload.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if isinstance(text, str) and text:
                yield text


def tts_audio_stream(text: str):
    body = {"text": text}
    req = urllib.request.Request(
        "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=mp3", data=json.dumps(body).encode(),
        headers={"Authorization": "Token " + deepgram_key(), "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=tls(), timeout=45) as response:
        if not response.headers.get("Content-Type", "").startswith("audio/"):
            raise RuntimeError("TTS did not return audio")
        # read1 returns available bytes without waiting to fill a large buffer.
        while chunk := response.read1(4096):
            yield chunk


class App(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def json(self, status: int, value: dict) -> None:
        raw = json.dumps(value).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def stream_frame(self, value: dict) -> None:
        self.wfile.write((json.dumps(value) + "\n").encode())
        self.wfile.flush()

    def stream_agent_reply(self, history: list[dict], text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        spoken, sentence = "", ""
        sentences: queue.Queue[str | None] = queue.Queue()
        frame_lock = threading.Lock()
        audio_error = []

        def send_frame(frame):
            with frame_lock:
                self.stream_frame(frame)

        def speak():
            try:
                while True:
                    text_chunk = sentences.get()
                    if text_chunk is None:
                        break
                    for audio in tts_audio_stream(text_chunk):
                        send_frame({"type": "audio", "data": base64.b64encode(audio).decode()})
            except Exception as exc:
                audio_error.append(type(exc).__name__)

        audio_worker = threading.Thread(target=speak, daemon=True)
        audio_worker.start()
        try:
            for delta in agent_text_stream(agent_messages(history, text)):
                spoken += delta; sentence += delta
                send_frame({"type": "text", "delta": delta})
                # Generate speech while the model continues producing its reply.
                if re.search(r"[.!?](?:\s|$)", sentence) and len(sentence.strip()) >= 12:
                    sentences.put(sentence.strip())
                    sentence = ""
            if sentence.strip():
                sentences.put(sentence.strip())
        except Exception as exc:
            audio_error.append(type(exc).__name__)
        finally:
            sentences.put(None)
            audio_worker.join(timeout=50)
        if audio_worker.is_alive():
            send_frame({"type": "error", "message": "Audio timeout"})
        elif audio_error:
            send_frame({"type": "error", "message": audio_error[0]})
        send_frame({"type": "done", "text": spoken.strip()})

    def stream_tts(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for audio in tts_audio_stream(text):
                self.stream_frame({"type": "audio", "data": base64.b64encode(audio).decode()})
            self.stream_frame({"type": "done"})
        except Exception as exc:
            self.stream_frame({"type": "error", "message": type(exc).__name__})

    def do_POST(self) -> None:
        if self.path not in {"/api/safety", "/api/agent-reply", "/api/transcribe", "/api/tts"}:
            self.json(404, {"error": "not found"}); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size))
            history = body.get("history", [])
            text = body.get("text", "")
            if not isinstance(history, list) or not isinstance(text, str) or len(text) > 2000:
                raise ValueError("invalid request")
            if self.path == "/api/transcribe":
                encoded = body.get("audio_base64", "")
                if not isinstance(encoded, str) or len(encoded) > 12_000_000:
                    raise ValueError("invalid audio")
                audio = base64.b64decode(encoded, validate=True)
                raw, latency_ms, _ = openrouter(
                    "audio/transcriptions",
                    {"model": "openai/whisper-1", "input_audio": {"data": encoded, "format": "webm"}, "language": "en"},
                    timeout=60,
                )
                transcript = json.loads(raw).get("text")
                if not isinstance(transcript, str) or not transcript.strip():
                    raise ValueError("empty transcript")
                self.json(200, {"text": transcript, "latency_ms": latency_ms, "audio_bytes": len(audio)}); return
            if self.path == "/api/safety":
                result = HARNESS.local(history, text)
                result["decision_wall_time_ms"] = time.time_ns() / 1_000_000
                turn_id = body.get("turn_id")
                if result["action"] == "WATCH" and body.get("final") and isinstance(turn_id, str):
                    FALLBACKS[turn_id] = {"status": "pending", "action": "WATCH", "reason": "Qwen reviewing context."}
                    POOL.submit(HARNESS.fallback, turn_id, history, text)
                self.json(200, result); return
            if self.path == "/api/tts":
                self.stream_tts(text); return
            self.stream_agent_reply(history, text)
        except Exception as exc:
            self.json(500, {"error": type(exc).__name__})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/fallback":
            turn_id = parsed.query.partition("turn_id=")[2]
            self.json(200, FALLBACKS.get(turn_id, {"status": "none"})); return
        image_paths = {"/images/ninja.jpeg": IMAGES / "Ninja.jpeg", "/images/cat.jpg": IMAGES / "Cat.jpg"}
        if parsed.path in image_paths:
            image_path = image_paths[parsed.path]
            raw = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers(); self.wfile.write(raw); return
        super().do_GET()


def main() -> None:
    global HARNESS
    os.environ["OPENROUTER_API_KEY"] = key()
    HARNESS = Harness()
    from websockets.sync.server import serve
    streaming = serve(streaming_session, '127.0.0.1', 8766,
                      origins=['http://127.0.0.1:8765', 'http://localhost:8765'], max_size=2**20)
    threading.Thread(target=streaming.serve_forever, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", 8765), App)
    print("Live demo: http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
