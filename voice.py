import os,json,re,ssl,urllib.request
AGENT_LLM="mistralai/mistral-small-3.2-24b-instruct"
def deepgram_key(): return os.environ["DEEPGRAM_API_KEY"]

def tls() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

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
