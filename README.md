# Call Guard — Voice Security Harness Demo

A research demo featuring Scam Ninja and Whiskey the Cat. Use invented credentials only.

## Architecture

- Browser microphone → Deepgram Nova-3 streaming ASR using a short-lived token.
- Each nonempty partial + conversation history → server-side refusal/context/fast-path rules, then fine-tuned MiniLM if unresolved.
- Original `candidate_003` checkpoint exported to **FP32 ONNX without quantization**, with the original 0.9498332142829895 warning threshold and 0.10 low threshold.
- `INTERVENE` plays the preloaded warning and ends the simulated call.
- Mistral Small 3.2 24B through OpenRouter generates caller replies; Deepgram Aura-2 Thalia generates speech.

This is a hosting adaptation of the local research prototype. Serverless cold starts, request/network time and per-instance scheduling change live latency. Existing local timing results do not establish cloud performance. Qwen fallback is not active. Neither audio nor conversation transcripts are stored by this application. External providers process content sent to them.

## Deploy on Vercel

1. Import this repository into Vercel.
2. Set `DEEPGRAM_API_KEY` (Member scope for token grants) and `OPENROUTER_API_KEY` as encrypted environment variables.
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

`model/validation.json` records parity checks against the original PyTorch harness on the saved synthetic voice transcripts. Source model/data are deliberately not copied into this repository; conversion scripts expect the original course-project layout.

## Endpoints

- `GET /api/health`: warms model, reports readiness and configuration booleans.
- `POST /api/speech-token`: mints short-lived Deepgram token; never returns a permanent API key.
- `POST /api/safety`: evaluates partial text and history.
- `POST /api/agent-reply`: streams caller text and speech as NDJSON.
- `POST /api/tts`: streams speech as NDJSON.

Provider credentials and keys, original audio recordings, synthetic datasets, course files and personal notes are excluded from this repository.
