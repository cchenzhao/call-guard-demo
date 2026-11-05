"""Reassemble and verify the original PyTorch classifier checkpoint."""
import hashlib
from pathlib import Path

model = Path(__file__).resolve().parent / 'Proposal/security_harness/classifier/runs/candidate_003/model'
target = model / 'model.safetensors'
expected = 'fe2d81a4341fe0496f880aa048406ee7ca9db73ce9bf5626bca88af3870fd746'
parts = sorted(model.glob('model.safetensors.part??'))
if len(parts) != 3:
    raise SystemExit('Expected three checkpoint parts; verify the repository download.')
if target.exists():
    if hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() == expected:
        print('Checkpoint already restored and verified.'); raise SystemExit(0)
    raise SystemExit('A different checkpoint already exists; refusing to overwrite it.')
digest = hashlib.sha256()
with target.open('xb') as out:
    for part in parts:
        with part.open('rb') as src:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk); out.write(chunk)
if digest.hexdigest() != expected:
    raise SystemExit('Checksum mismatch. Remove the incomplete output after inspection and retry.')
print('Checkpoint restored and verified.')
