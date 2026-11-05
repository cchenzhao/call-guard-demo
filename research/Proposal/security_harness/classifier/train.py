"""Small, reproducible candidate experiment; test split is a separate command."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/password_v2/synthetic_1000_candidates.jsonl"
MANIFEST = ROOT / "data/password_v2/split_manifest.json"
MODEL = "microsoft/MiniLM-L12-H384-uncased"
SEED = 20261101


def read_data(data_path: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != manifest["data_sha256"]:
        raise ValueError("Dataset no longer matches locked split manifest")
    assignments = manifest["assignments"]
    result = {x: [] for x in ("train", "validation", "test")}
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    quarantined = {}
    for row in rows:
        family = row["source_family_id"]
        history = row["observations"]["history"]
        if any(not isinstance(t, dict) or not isinstance(t.get("speaker"), str) or not isinstance(t.get("text"), str) for t in history):
            quarantined.setdefault(family, set()).add("malformed_history")
        ann = row["annotations"]
        span = ann.get("secret_span")
        if ann["target_action"] == "INTERVENE":
            matches = [event for event in row["observations"]["events"]
                       if event["event_index"] == ann["earliest_justified_event_index"]]
            if not matches or not matches[0]["partial_text"].strip():
                quarantined.setdefault(family, set()).add("empty_labeled_positive_prefix")
        if span and span.get("value"):
            secret = span["value"].casefold()
            if any(isinstance(t, dict) and secret in str(t.get("text", "")).casefold() for t in history):
                quarantined.setdefault(family, set()).add("secret_already_in_history")
    for row in rows:
        if row["source_family_id"] not in quarantined:
            result[assignments[row["source_family_id"]]].append(row)
    quarantine_report = [{"family_id": f, "split": assignments[f], "reasons": sorted(reasons)}
                         for f, reasons in sorted(quarantined.items())]
    return result, digest, hashlib.sha256(manifest_path.read_bytes()).hexdigest(), quarantine_report


def visible_input(row, event):
    # Text only; future events, annotations, provenance, IDs and timestamps are excluded.
    history = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in row["observations"]["history"])
    return history, f'{event["speaker"]}: {event["partial_text"]}'


def training_example(row):
    ann = row["annotations"]
    events = row["observations"]["events"]
    if ann["target_action"] == "INTERVENE":
        index = ann["earliest_justified_event_index"]
        event = next(e for e in events if e["event_index"] == index)
        label = 1
    else:
        event, label = events[-1], 0
    return (*visible_input(row, event), label)


class Examples(Dataset):
    def __init__(self, examples, tokenizer):
        history, partial, self.labels = zip(*examples)
        self.tokens = tokenizer(list(history), list(partial), padding=True, truncation="only_first", max_length=192, return_tensors="pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def predict(model, tokenizer, examples, device):
    loader = DataLoader(Examples(examples, tokenizer), batch_size=16)
    probs, labels, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            probs.extend(torch.softmax(output.logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            losses.append((output.loss.item(), len(batch["labels"])))
    return probs, labels, sum(l * n for l, n in losses) / len(labels)


def metrics(probs, labels, threshold):
    tp = sum(p >= threshold and y == 1 for p, y in zip(probs, labels))
    fp = sum(p >= threshold and y == 0 for p, y in zip(probs, labels))
    pos, neg = sum(labels), len(labels) - sum(labels)
    return {"tp": tp, "fn": pos - tp, "fp": fp, "tn": neg - fp,
            "recall": tp / pos, "false_warning_rate": fp / neg,
            "precision": tp / (tp + fp) if tp + fp else None}


def choose_threshold(probs, labels):
    # Validation-only: maximize recall under 10% false warnings; ties favor fewer warnings.
    candidates = sorted(set(probs + [0.0, 1.000001]))
    choices = [(metrics(probs, labels, t), t) for t in candidates]
    allowed = [(m, t) for m, t in choices if m["false_warning_rate"] <= 0.10]
    return max(allowed, key=lambda mt: (mt[0]["recall"], -mt[0]["false_warning_rate"], mt[1]))[1]


def replay(model, tokenizer, rows, threshold, device):
    examples, ranges = [], []
    for row in rows:
        start = len(examples)
        for event in row["observations"]["events"]:
            examples.append((*visible_input(row, event), int(row["annotations"]["target_action"] == "INTERVENE")))
        ranges.append((start, len(examples)))
    probs, _, _ = predict(model, tokenizer, examples, device)
    risky = [0, 0, 0, 0]  # total, warned, pre-first-secret, protected characters seen
    safe_total = safe_warned = before_annotation = 0
    for row, (start, end) in zip(rows, ranges):
        ann = row["annotations"]
        events = row["observations"]["events"]
        hits = [i for i, p in enumerate(probs[start:end]) if p >= threshold]
        if ann["target_action"] == "WAIT":
            safe_total += 1
            safe_warned += bool(hits)
            continue
        risky[0] += 1
        if not hits:
            # By the end of replay all annotated secret characters are visible.
            risky[3] += len(ann["secret_span"]["value"])
            continue
        event = events[hits[0]]
        risky[1] += 1
        before_annotation += event["event_index"] < ann["earliest_justified_event_index"]
        risky[2] += event["event_index"] < ann["first_secret_event_index"]
        span = ann["secret_span"]
        risky[3] += max(0, min(len(span["value"]), len(event["partial_text"]) - span["start_char"]))
    return {"risky_scenarios": risky[0], "risk_warned": risky[1], "warned_before_first_secret_event": risky[2],
            "warnings_before_draft_earliest_justified_event": before_annotation,
            "protected_characters_visible_by_warning_or_end": risky[3],
            "safe_scenarios": safe_total, "safe_scenarios_warned": safe_warned,
            "timing_assumption": "Zero inference and warning execution delay; event-level replay only, not measured realtime latency."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="Candidate JSONL; its digest must match --manifest.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="Frozen family-split manifest for --data.")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data, digest, manifest_digest, quarantine_report = read_data(args.data, args.manifest)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if args.mode == "train":
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
        args.output.mkdir(parents=True, exist_ok=False)
        tokenizer.save_pretrained(args.output / "model")
        train = Examples([training_example(r) for r in data["train"]], tokenizer)
        val = [training_example(r) for r in data["validation"]]
        loader = DataLoader(train, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(SEED))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        history, best = [], float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train(); running_loss = 0.0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(); output = model(**batch); output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                running_loss += output.loss.item() * len(batch["labels"])
            probs, labels, loss = predict(model, tokenizer, val, device)
            history.append({"epoch": epoch, "train_loss": running_loss / len(train), "validation_loss": loss})
            print(json.dumps(history[-1]), flush=True)
            if loss < best:
                best = loss
                model.save_pretrained(args.output / "model")
                best_epoch = epoch
        model = AutoModelForSequenceClassification.from_pretrained(args.output / "model").to(device)
        probs, labels, loss = predict(model, tokenizer, val, device)
        threshold = choose_threshold(probs, labels)
        result = {"model": MODEL, "seed": SEED, "data_file": str(args.data), "manifest_file": str(args.manifest),
                  "dataset_sha256": digest, "manifest_sha256": manifest_digest,
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
                  "device": str(device), "quarantined_families": quarantine_report,
                  "included_scenarios_by_split": {k: len(v) for k, v in data.items()},
                  "epochs": history, "best_epoch_by_validation_loss": best_epoch,
                  "threshold_selection": "Validation max recall subject to false warning rate <= 0.10; ties favor fewer false warnings and higher threshold.",
                  "threshold": threshold, "validation_loss": loss,
                  "validation_classification": metrics(probs, labels, threshold),
                  "validation_replay": replay(model, tokenizer, data["validation"], threshold, device),
                  "label_caveat": "LLM-drafted candidate labels require human review; not deployment performance."}
        (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    else:
        result = json.loads((args.output / "validation.json").read_text())
        if digest != result["dataset_sha256"] or manifest_digest != result["manifest_sha256"]:
            raise ValueError("Data or split changed since training")
        if (args.output / "test.json").exists():
            raise FileExistsError("Test has already been evaluated for this run")
        tokenizer = AutoTokenizer.from_pretrained(args.output / "model")
        model = AutoModelForSequenceClassification.from_pretrained(args.output / "model").to(device)
        probs, labels, loss = predict(model, tokenizer, [training_example(r) for r in data["test"]], device)
        test = {"threshold_frozen_from_validation": result["threshold"], "test_loss": loss,
                "test_classification": metrics(probs, labels, result["threshold"]),
                "test_replay": replay(model, tokenizer, data["test"], result["threshold"], device),
                "label_caveat": "LLM-drafted candidate labels require human review; not deployment performance."}
        (args.output / "test.json").write_text(json.dumps(test, indent=2) + "\n")
        print(json.dumps(test, indent=2), flush=True)


if __name__ == "__main__":
    main()
