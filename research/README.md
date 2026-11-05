# Reproduce the three-way comparison

This directory includes the original PyTorch detector and checkpoint, fixed synthetic dataset and family split, training script, cached streaming-ASR observations, estimated password onsets, and the saved three-way results and figure. The comparison evaluates rules only, MiniLM only, and rules + MiniLM on the same 150 scenarios (75 risky and 75 safe). It does not require provider keys or paid API calls.

Run from this `research/` directory. Use Python 3.14 and install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0 transformers==4.57.6 numpy==2.5.4 matplotlib certifi 'websockets>=14,<16'
python restore_checkpoint.py
python -m Proposal.security_harness.runs.timeliness_three_way.run
python Proposal/security_harness/runs/timeliness_three_way/plot.py
```

The checkpoint is stored in three parts to stay under GitHub's individual file limit. `restore_checkpoint.py` reconstructs it and checks its SHA256. A full rerun takes some time and overwrites the saved `report.json`, `per_case.csv` and plot in `Proposal/security_harness/runs/timeliness_three_way/`; copy those files first if you need the original results. To redraw the saved results without rerunning inference, only run `plot.py`.

Expected counts (risk cases warned / safe cases falsely warned): rules **43/75 / 2/75**, MiniLM **74/75 / 0/75**, rules + MiniLM **75/75 / 2/75**. The first warning decisions are compared on saved ASR partials. T3 combines saved ASR arrival times with newly measured local computation and simulated queuing, so timings vary between machines. T1 is an estimated, unverified password onset. This is an exploratory synthetic-voice comparison, not a measurement of when the user actually heard the warning.

The cached ASR inputs have been reduced to the history, partials, arrival times and labels needed by the evaluation; audio file paths were removed. `Proposal/data/password_v2/` holds the original repaired dataset and family split. The optional training script is `Proposal/security_harness/classifier/train.py`: run `python Proposal/security_harness/classifier/train.py --help` for arguments, choose a **new** output directory, and provide the repaired data and matching manifest. Training may download the base model and can yield different results; it is not needed to reproduce the saved checkpoint evaluation.
