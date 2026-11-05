# Context-Aware Voice Security Agent for Real-Time Credential Disclosure Detection

Can a voice agent recognize when someone is about to disclose a password or passcode and warn them in time? Call Guard explores this question with a context-aware security harness for English voice conversations. It compares detection coverage, false warnings, and decision timing across rules, a fine-tuned MiniLM classifier, and their combination.

**[Live demo](https://call-guard-demo.vercel.app/)** · **[Demo video](https://youtu.be/NzATwahUXQ0)** · **[Midway report](research/Context_Aware_Voice_Security_Agent_Midterm_Report.pdf)** · **[Full reproduction instructions](research/README.md)**

The demo is an illustrative push-to-talk conversation between the fictional caller Scam Ninja and the user, Whiskey the Cat. Use invented credentials only.

## Methodology, environment, and data

The voice environment streams the user's microphone audio to Deepgram Nova-3 for speech recognition. Each available partial transcript is checked alongside the conversation history:

```text
Microphone → streaming ASR → history + partial transcript
        → contextual / fast-path rules → MiniLM if unresolved
        → WAIT / WATCH / INTERVENE → prerecorded warning
```

The harness checks rules first. If a rule returns a decision, MiniLM is skipped; otherwise, the classifier estimates risk. An intervention plays a prerecorded warning and ends the simulated call. The demo's fictional caller uses Mistral Small 3.2 through OpenRouter for replies and Deepgram Aura-2 for speech. Qwen fallback is not active in the live detection flow.

The dataset contains **1,000 synthetic English records in 500 safe/risky families**. It was inspired by source types and coarse context cues from CallCenterEN (300 families), NCSU/FTC robocalls (100), YouTube scam-baiting (43), and HarperValleyBank (57). Mistral generated the synthetic dialogues, fictional credentials, labels, and timestamps. Safe cases include refusal, reset assistance, private typing, tutorial examples, and non-login sharing. Forty flagged risky records were repaired, and approximately 20% of records received author spot checks.

The fixed, family-separated split contains **700 training, 150 validation, and 150 test records**. MiniLM was fine-tuned on text consisting of history and one selected user prefix per training record; it does not process audio directly.

For the voice evaluation, Deepgram Aura-2 generated audio for both sides of the 150 test scenarios. Nova-3 transcribed the focal user turn. All three detection schemes were evaluated on the same cached ASR partials and arrival times, with local computation measured during replay. See the [midway report](research/Context_Aware_Voice_Security_Agent_Midterm_Report.pdf) for the full setup and source references.

## Results and key takeaways

| Detection scheme | Risky scenarios warned | Safe scenarios falsely warned |
|---|---:|---:|
| Rules only | 43/75 | 2/75 |
| MiniLM only | 74/75 | 0/75 |
| Rules + MiniLM | 75/75 | 2/75 |

- **Context matters.** The same credential-related words can appear in refusal, discussion, or actual disclosure. MiniLM improved risk coverage over rules alone on this synthetic set.
- **Earlier is not automatically better.** Detecting a risk is only the first step: the warning must reach the user early enough, while avoiding unnecessary interruptions. Waiting for context may improve a decision but can delay it.
- **Routing affects timing.** The rules-first combination can return `WATCH` and skip MiniLM on an incomplete partial, delaying model review until a later ASR update.

The [timing figure](research/Proposal/security_harness/runs/timeliness_three_way/timeliness_scatter.png) plots the first estimated decision time relative to estimated password onset. Higher and further left indicates more risk cases warned earlier, provided the warnings are justified. The results are exploratory: the data are synthetic and relatively structured, and the graph measures an estimated decision time, not when the user hears a warning. Zero observed false warnings in 75 safe cases does not establish a zero real-world false-warning rate.

## Reproduce the comparison

Follow the [research README](research/README.md) for Python setup, checkpoint restoration, the replay command, and plot generation. The repository includes the fixed data and split, model checkpoint, cached ASR inputs, onset estimates, evaluation code, and saved results. No provider API key or paid API call is needed to replay the comparison or redraw the saved plot.

The workflow was tested in a fresh Python 3.14 virtual environment on macOS. Counts matched the table above; newly measured inference times and individual plotted positions may vary by machine. Replaying inference overwrites the saved results in the research directory, so make a copy first if needed.

## Optional: deploy your own demo

To deploy the voice demo separately, import this repository into Vercel, configure `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY` as encrypted environment variables, and deploy with `vercel.json`. The deployment uses Python 3.12 and paid external voice/LLM services. Never commit your keys.

The three `model/weights.part*` files reconstruct the deployed ONNX checkpoint in memory; `model/manifest.json` records its SHA256. The research evaluation uses a separate PyTorch checkpoint and has its own restoration instructions.

For local API development:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt uvicorn
# Load the two keys into the environment; never commit them.
uvicorn api.index:app --port 8000
```

Use `vercel dev` for the combined static site and API routing. The backend exposes `/api/health`, `/api/speech` (WebSocket), `/api/safety`, `/api/agent-reply`, and `/api/tts`. Provider credentials, original audio recordings, and unrelated personal materials are excluded from this repository.
