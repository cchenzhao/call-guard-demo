"""Warning-only deterministic event replay. Policy and evaluator are separate."""
from __future__ import annotations
import copy
import heapq
import json
import math
import re
from dataclasses import dataclass
from typing import Any

ACTIONS = {'WAIT', 'WATCH', 'INTERVENE'}
WARNING = 'Pause before sharing: keep your private account password to yourself.'

@dataclass
class PolicyReply:
    output: Any = None
    latency_ms: float = 0
    error: str | None = None
    source: str = 'policy'

@dataclass
class ReplayConfig:
    execution_latency_ms: float = 0
    timeout_ms: float = 5000
    max_calls: int = 32
    mode: str = 'stateful'
    trigger_enabled: bool = False

class ScriptedPolicy:
    """Fixed replies for mechanics tests; no gold-derived policy decisions."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
    def decide(self, snapshot):
        self.calls.append(copy.deepcopy(snapshot))
        if not self.replies:
            return PolicyReply(error='script_exhausted')
        value = self.replies.pop(0)
        return value if isinstance(value, PolicyReply) else PolicyReply(value)

def parse_action(output):
    if isinstance(output, str):
        output = json.loads(output)
    if not isinstance(output, dict) or output.get('action') not in ACTIONS:
        raise ValueError('invalid action')
    if not isinstance(output.get('reason'), str):
        raise ValueError('reason must be string')
    output = {key:value for key,value in output.items() if key in {'action','reason'}}
    return output

def validate_observations(observations):
    history = observations.get('history', [])
    events = observations.get('events', [])
    if not isinstance(history, list) or not isinstance(events, list) or not events:
        raise ValueError('history/events must be lists')
    safe_history = []
    for h in history:
        if not isinstance(h.get('text'), str) or h.get('speaker') not in {'user', 'other'}:
            raise ValueError('invalid history')
        safe_history.append({'speaker': h['speaker'], 'text': h['text']})
    previous, last_time, last_index = '', -1, -1
    safe_events = []
    for e in events:
        t, i, text = e.get('time_ms'), e.get('event_index'), e.get('partial_text')
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or t < last_time:
            raise ValueError('event time must be finite nonnegative and ordered')
        if isinstance(i, bool) or not isinstance(i, int) or i != last_index + 1:
            raise ValueError('event indexes must increase')
        if not isinstance(text, str) or not text.startswith(previous) or e.get('speaker') != 'user':
            raise ValueError('events must be cumulative user text')
        safe_events.append({'event_index': i, 'time_ms': t, 'speaker': 'user', 'partial_text': text})
        previous, last_time, last_index = text, t, i
    return safe_history, safe_events

def replay(observations, policy, config=None):
    config = config or ReplayConfig()
    for name in ('execution_latency_ms', 'timeout_ms'):
        val = getattr(config, name)
        if isinstance(val, bool) or not isinstance(val, (int,float)) or not math.isfinite(val) or val < 0:
            raise ValueError(f'invalid {name}')
    if not isinstance(config.max_calls, int) or isinstance(config.max_calls, bool) or config.max_calls < 1:
        raise ValueError('max_calls must be positive integer')
    if config.mode not in {'stateful', 'stateless', 'turn_end', 'keyword'}:
        raise ValueError('invalid mode')
    history, events = validate_observations(observations)
    queue, serial = [], 0
    for e in events:
        heapq.heappush(queue, (e['time_ms'], 0, serial, 'input', e)); serial += 1
    trace, latest, inflight, pending = [], None, False, False
    state = {'action': 'WAIT', 'risk_level': 'unknown', 'reason': ''}
    warned, scheduled, calls = None, False, 0
    errors = 0
    budget_exhausted = False
    def push(time, priority, kind, payload):
        nonlocal serial
        heapq.heappush(queue, (time, priority, serial, kind, payload)); serial += 1
    def snapshot_for(now):
        snapshot = {'history': copy.deepcopy(history), 'partial_text': latest['partial_text'],
                    'event_index': latest['event_index'], 'time_ms': now}
        if config.mode == 'stateful':
            snapshot['state'] = copy.deepcopy(state)
        return snapshot
    def schedule_fast_path(now):
        """Run a synchronous local rule before an in-flight remote request can block it."""
        nonlocal scheduled
        checker = getattr(policy, 'fast_decide', None)
        if scheduled or checker is None:
            return False
        snapshot = snapshot_for(now)
        try:
            reply = checker(copy.deepcopy(snapshot))
            if reply is None:
                return False
            if not isinstance(reply, PolicyReply):
                reply = PolicyReply(reply)
            decision = parse_action(reply.output)
        except (ValueError, TypeError, json.JSONDecodeError):
            trace.append({'type':'fast_path_error', 'time_ms':now, 'event_index':latest['event_index']})
            return False
        if decision['action'] != 'INTERVENE':
            trace.append({'type':'fast_path_noop', 'time_ms':now, 'event_index':latest['event_index'],
                          **decision, 'decision_source':reply.source})
            return False
        state.update(decision)
        scheduled = True
        trace.append({'type':'decision', 'time_ms':now, 'call_id':None,
                      'source_event_index':latest['event_index'], 'stale':False,
                      **decision, 'error':None, 'decision_source':reply.source})
        push(now + config.execution_latency_ms, 2, 'warning', None)
        return True
    def launch(now):
        nonlocal calls, inflight, pending, errors, budget_exhausted
        pending = False
        if scheduled or latest is None:
            return
        if calls >= config.max_calls:
            if not budget_exhausted:
                trace.append({'type':'budget_stop','time_ms':now,'max_calls':config.max_calls})
            budget_exhausted = True
            return
        if config.trigger_enabled and calls > 0:
            evidence = ' '.join(h['text'] for h in history)+' '+latest['partial_text']
            if state['action'] == 'WAIT' and not re.search(r'password|passcode|sign.in|login|secret|hidden field|username', evidence, re.I):
                trace.append({'type':'trigger_skip','time_ms':now,'event_index':latest['event_index']})
                return
        snapshot = snapshot_for(now)
        calls += 1; inflight = True
        trace.append({'type':'policy_start', 'time_ms':now, 'call_id':calls, 'snapshot':copy.deepcopy(snapshot)})
        try:
            reply = policy.decide(copy.deepcopy(snapshot))
            if not isinstance(reply, PolicyReply): reply = PolicyReply(reply)
            if isinstance(reply.latency_ms, bool) or not isinstance(reply.latency_ms,(int,float)) or not math.isfinite(reply.latency_ms) or reply.latency_ms < 0:
                reply = PolicyReply(error='invalid_latency')
        except Exception as exc:
            # Never log exception text: provider errors can contain credentials.
            reply = PolicyReply(error=type(exc).__name__)
        delay = reply.latency_ms
        if delay >= config.timeout_ms:
            delay = config.timeout_ms; reply = PolicyReply(error='timeout')
        push(now + delay, 1, 'complete', (calls, snapshot['event_index'], reply))
    while queue:
        now, _, _, kind, payload = heapq.heappop(queue)
        if kind == 'input':
            previous = latest['partial_text'] if latest else ''
            latest = payload
            trace.append({'type':'input', 'time_ms':now, **payload, 'delta_text':payload['partial_text'][len(previous):]})
            if not scheduled:
                schedule_fast_path(now)
            if not scheduled:
                pending = True
        elif kind == 'complete':
            cid, source_index, reply = payload
            inflight = False
            if scheduled:
                # A local warning may have superseded this remote request. Its result
                # must not overwrite state or be mistaken for the action that warned.
                trace.append({'type':'ignored_policy_completion', 'time_ms':now, 'call_id':cid,
                              'source_event_index':source_index, 'reason':'fast_path_warning_scheduled'})
                continue
            error = reply.error
            try:
                if error: raise ValueError(error)
                decision = parse_action(reply.output)
            except (ValueError, TypeError, json.JSONDecodeError):
                decision = {'action':'WATCH', 'reason':'Policy failed; no warning executed.'}
                error = error or 'invalid_output'
            if error: errors += 1
            state.update(decision)
            trace.append({'type':'decision', 'time_ms':now, 'call_id':cid, 'source_event_index':source_index,
                          'stale':latest['event_index'] != source_index, **decision, 'error':error,
                          'decision_source':reply.source})
            if decision['action'] == 'INTERVENE' and not scheduled:
                scheduled = True
                push(now + config.execution_latency_ms, 2, 'warning', None)
        elif kind == 'warning':
            warned = now
            trace.append({'type':'warning', 'time_ms':now, 'text':WARNING})
        # Process all arrivals/completions at this timestamp before starting next call.
        if queue and queue[0][0] == now:
            continue
        if pending and not inflight and not scheduled:
            if config.mode != 'turn_end' or latest['event_index'] == events[-1]['event_index']:
                launch(now)
    return {'trace':trace, 'warning_onset_ms':warned, 'final_state':state, 'calls':calls,
            'policy_errors':errors, 'final_text':latest['partial_text'] if latest else '',
            'mode':config.mode, 'budget_exhausted':budget_exhausted, 'warning_only':True}

def evaluate(scenario, result):
    """Only this function sees gold. Characters include spaces/punctuation in span."""
    a = scenario['annotations']; events = scenario['observations']['events']
    span, onset = a.get('secret_span'), result['warning_onset_ms']
    risky = span is not None
    disclosed = None
    if risky:
        visible = events[-1]['partial_text'] if onset is None else ''
        if onset is not None:
            for e in events:
                if e['time_ms'] <= onset: visible = e['partial_text']
        disclosed = min(max(len(visible)-span['start_char'],0), span['end_char']-span['start_char'])
    earliest = a.get('earliest_justified_event_index')
    earliest_time = next((e['time_ms'] for e in events if e['event_index']==earliest),None)
    return {'scenario_id':scenario['scenario_id'], 'target_action':a['target_action'], 'warning_onset_ms':onset,
            'protected_chars':span['end_char']-span['start_char'] if risky else None,
            'disclosed_chars':disclosed,
            'before_first_char':bool(risky and onset is not None and disclosed==0),
            'before_complete_disclosure':bool(risky and onset is not None and disclosed < span['end_char']-span['start_char']),
            'false_interruption':bool(not risky and a['target_action'] in {'WAIT','WATCH'} and onset is not None),
            'warning_latency_ms':onset-earliest_time if onset is not None and earliest_time is not None else None,
            'calls':result['calls'], 'policy_errors':result['policy_errors'],
            'budget_exhausted':result.get('budget_exhausted',False),
            'late_warning':bool(risky and onset is not None and disclosed == span['end_char']-span['start_char'])}

def summarize(metrics):
    risky = [m for m in metrics if m['protected_chars'] is not None]
    wait = [m for m in metrics if m['target_action']=='WAIT']
    watch = [m for m in metrics if m['target_action']=='WATCH']
    controls = wait + watch
    def rate(rows, field): return sum(bool(m[field]) for m in rows)/len(rows) if rows else None
    return {'scenarios':len(metrics), 'risky_count':len(risky), 'benign_count':len(wait),
            'wait_count':len(wait),'watch_count':len(watch),'control_count':len(controls),
            'risky_warning_coverage':sum(m['warning_onset_ms'] is not None for m in risky)/len(risky) if risky else None,
            'no_warning_count':sum(m['warning_onset_ms'] is None for m in risky),
            'late_warning_count':sum(m.get('late_warning',False) for m in risky),
            'before_first_char_rate':rate(risky,'before_first_char'),
            'before_complete_disclosure_rate':rate(risky,'before_complete_disclosure'),
            'mean_disclosed_chars':sum(m['disclosed_chars'] for m in risky)/len(risky) if risky else None,
            'disclosed_chars_distribution':[m['disclosed_chars'] for m in risky],
            'false_interruption_rate':rate(wait,'false_interruption'),
            'watch_false_interruption_rate':rate(watch,'false_interruption'),
            'control_false_interruption_rate':rate(controls,'false_interruption'),
            'calls':sum(m['calls'] for m in metrics), 'policy_errors':sum(m['policy_errors'] for m in metrics),
            'budget_exhausted_count':sum(m.get('budget_exhausted',False) for m in metrics)}
