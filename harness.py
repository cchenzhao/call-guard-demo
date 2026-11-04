import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
MODEL_DIR=Path(__file__).resolve().parent/'model'
THRESHOLD=0.9498332142829895
LOW_RISK=0.10
@dataclass
class PolicyReply:
    output: object
    source: str='policy'

class WaitPolicy:
    """No-op fallback used to measure the local fast path by itself."""
    def __init__(self):
        self.calls = []

    def decide(self, snapshot):
        self.calls.append(snapshot)
        return PolicyReply({'action':'WAIT', 'reason':'No-op fallback for fast-path baseline.'}, source='fast_path_baseline')

class FastPathPolicy:
    """Warn locally for a narrow, explicit password-disclosure statement.

    The matcher deliberately requires both a private-login cue and a current-turn
    disclosure verb. Everything else remains the fallback policy's responsibility.
    """
    _LOGIN = re.compile(r'\b(?:login|sign[ -]?in|account|mailbox|email account|work account)\b', re.I)
    _CREDENTIAL = re.compile(r'\b(?:password|passcode)\b', re.I)
    _DISCLOSURE = re.compile(
        r"\b(?:i(?:'ll| will)|let me)\s+(?:read|tell|give|spell|share|say)\b|"
        r"\b(?:i(?:'ll| will)\s+)?(?:read|tell|give|spell|share|say)\s+(?:you\s+)?(?:my|the)\b|"
        # Covers both “my password is” and indirect references such as
        # “the hidden field reads” after the dialogue has established it is
        # a login credential.
        r"\b(?:my|the)\b.{0,90}\b(?:password|passcode|phrase|value|one|item|field|characters?)\b.{0,70}\b(?:is|are|reads|says)\b|"
        r"\b(?:the\s+)?characters\s+are\b",
        re.I,
    )
    _NEGATION = re.compile(
        r"\b(?:no|not|never|cannot|can't|won't|will not|don't|do not|refuse|private(?:ly)?|haven't|have not)\b",
        re.I,
    )

    def __init__(self, fallback):
        self.fallback = fallback

    @property
    def calls(self):
        """Expose only provider calls; local fast-path decisions are not model calls."""
        return getattr(self.fallback, 'calls', [])

    def fast_decide(self, snapshot):
        """Return an immediate local decision, or None when LLM review is needed."""
        current = snapshot['partial_text']
        history = ' '.join(item['text'] for item in snapshot['history'])
        visible = f'{history} {current}'
        # The disclosure statement itself must be in the arriving user turn.
        # History may establish that an indirect reference is a login credential.
        has_private_credential = bool(self._CREDENTIAL.search(visible) and self._LOGIN.search(visible))
        if (
            has_private_credential
            and self._DISCLOSURE.search(current)
            and not self._NEGATION.search(current)
            and not self._is_scoped_safe(visible)
        ):
            return PolicyReply(
                {'action':'INTERVENE', 'reason':'Fast path: explicit intent to disclose a private login credential.'},
                source='fast_path',
            )
        return None

    @staticmethod
    def _is_scoped_safe(visible):
        """Recognize explicit non-login controls without masking reuse cases."""
        text = visible.lower()
        if 'guest' in text and 'wi-fi' in text:
            # A guest-network value reused for private email is no longer a
            # scoped guest-network example.
            if not re.search(r'\b(?:same value|reus\w*|also signs|private email)\b', text):
                return True
        if re.search(r'\b(?:fake account|made-?up|nonfunctional|sample password)\b', text):
            if not re.search(r'\b(?:real account|current (?:customer )?account|private account)\b', text):
                return True
        if re.search(r'\b(?:fresh|only for this document|isn\'t used by any account)\b', text):
            return True
        return False

    def decide(self, snapshot):
        """The regular scheduler invokes only the fallback policy."""
        reply = self.fallback.decide(snapshot)
        if not isinstance(reply, PolicyReply):
            reply = PolicyReply(reply)
        if reply.source == 'policy':
            reply.source = 'fallback_policy'
        return reply

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

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=192, strategy='only_first')
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        raw=b''.join(p.read_bytes() for p in sorted(MODEL_DIR.glob('weights.part*')))
        self.model=ort.InferenceSession(raw,sess_options=options,providers=['CPUExecutionProvider'])
        self.fast=FastPathPolicy(WaitPolicy())

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
        enc=self.tokenizer.encode(history_text, 'user: '+partial)
        feeds={'input_ids': np.array([enc.ids],dtype=np.int64),
               'attention_mask': np.array([enc.attention_mask],dtype=np.int64),
               'token_type_ids': np.array([enc.type_ids],dtype=np.int64)}
        logits=self.model.run(None,feeds)[0][0]
        scores=np.exp(logits-np.max(logits));probability=float(scores[1]/scores.sum())
        if probability >= THRESHOLD:
            action, reason = "INTERVENE", "MiniLM high-risk threshold reached."
        elif probability <= LOW_RISK:
            action, reason = "WAIT", "MiniLM low-risk threshold reached."
        else:
            action, reason = "WATCH", "MiniLM is uncertain; Qwen context review may run after the turn."
        return {"action": action, "source": "minilm", "reason": reason, "probability": probability}

