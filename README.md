# Call Guard — Voice Security Harness Demo

A research demo featuring Scam Ninja and Whiskey the Cat. Use invented credentials only.

**Live demo:** https://call-guard-demo.vercel.app/

**Demo video:** https://youtu.be/NzATwahUXQ0

**Research evaluation:** [Reproduce the rules / MiniLM / combined comparison](research/README.md). The repository contains the fixed dataset, checkpoint, cached ASR observations and evaluation code; no API key is needed for that comparison.

**Midway report:** [Call Guard report (PDF)](research/Call_Guard_Midway_Report.pdf).

## Architecture

- Browser microphone → same-origin Vercel WebSocket proxy → Deepgram Nova-3 streaming ASR. The permanent key stays on the server.
- Each nonempty partial + conversation history → server-side refusal/context/fast-path rules, then fine-tuned MiniLM if unresolved.
- Original `candidate_003` checkpoint exported to **FP32 ONNX without quantization**, with the original 0.9498332142829895 warning threshold and 0.10 low threshold.
- `INTERVENE` plays the preloaded warning and ends the simulated call.
- Mistral Small 3.2 24B through OpenRouter generates caller replies; Deepgram Aura-2 Thalia generates speech.

This is a hosting adaptation of the local research prototype. Serverless cold starts, request/network time and per-instance scheduling change live latency. Existing local timing results do not establish cloud performance. Qwen fallback is not active. Neither audio nor conversation transcripts are stored by this application. External providers process content sent to them.

## Deploy on Vercel

1. Import this repository into Vercel.
2. Set `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY` as encrypted environment variables.
3. Deploy with the included `vercel.json`. Python 3.12 is required.

The three `model/weights.part*` files reconstruct the same ONNX model in memory. They keep each Git file below GitHub's 100 MB limit. Do not change individual parts. `manifest.json` records the complete model SHA256.

## Local development

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt uvicorn
# Load the two keys into the environment (never commit them).
uvicorn api.index:app --port 8000
```

Use `vercel dev` for the combined static site and API routing.

`model/validation.json` records parity checks against the original PyTorch harness on saved synthetic voice transcripts. The original checkpoint and research inputs are in `research/`; deployment conversion scripts expect the original course-project layout.

## Endpoints

- `GET /api/health`: warms model, reports readiness and configuration booleans.
- `WebSocket /api/speech`: proxies streaming audio and returns safety decisions / final text; no provider key is sent to the browser.
- `POST /api/safety`: evaluates partial text and history.
- `POST /api/agent-reply`: streams caller text and speech as NDJSON.
- `POST /api/tts`: streams speech as NDJSON.

Provider credentials and keys, original audio recordings, unrelated course files and personal notes are excluded from this repository. The research directory contains synthetic data and text transcripts, not real credentials.
