"""Small routing collector and independently testable balance metrics.

Counts use the boolean selection mask, never scores > 0. Population CV.
The supported runner uses B=1, TP=PP=CP=ETP=1, no capacity/drop/padding policy.
Padding here means right-padded input tokens, excluded from content statistics.
"""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np

SCHEMA = 1


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def read_inputs(path, length, vocab_size=None):
    records, seen = [], set()
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            x = json.loads(line)
            ident, domain, ids = x.get('id'), x.get('domain'), x.get('input_ids')
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f'Line {line_no}: missing, invalid, or duplicate id')
            if not isinstance(domain, str) or not domain:
                raise ValueError(f'Line {line_no}: domain must be a nonempty string')
            if not isinstance(ids, list) or not ids or len(ids) > length:
                raise ValueError(f'Line {line_no}: input_ids length must be 1..{length}')
            if any(type(i) is not int or i < 0 or (vocab_size and i >= vocab_size) for i in ids):
                raise ValueError(f'Line {line_no}: invalid vocabulary ID')
            seen.add(ident)
            records.append({'id': ident, 'domain': domain, 'input_ids': ids})
    if not records:
        raise ValueError('Empty evaluation dataset')
    return records


def scheduling(n, world, rank):
    if n < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid scheduling arguments')
    return [i if i < n else -1 for i in range(rank, math.ceil(n/world)*world, world)]


def metrics(counts, gate_mass=None, topk=None):
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1 or c.size < 2 or not np.isfinite(c).all() or (c < 0).any() or c.sum() <= 0:
        raise ValueError('Counts must be a finite, nonnegative vector with positive total')
    p = c/c.sum()
    nonzero = p > 0
    result = dict(assignments=int(round(c.sum())), load_cv=float(c.std()/c.mean()),
                  max_load_ratio=float(c.max()/c.mean()), min_load_ratio=float(c.min()/c.mean()),
                  normalized_load_entropy=float(-np.sum(p[nonzero]*np.log(p[nonzero]))/np.log(c.size)),
                  effective_experts=float(np.exp(-np.sum(p[nonzero]*np.log(p[nonzero])))),
                  unused_experts=int((c == 0).sum()),
                  underused_below_0_1x=int((c < .1*c.mean()).sum()))
    if topk:
        result['valid_tokens'] = float(c.sum()/topk)
    if gate_mass is not None:
        g = np.asarray(gate_mass, dtype=np.float64)
        if g.shape != c.shape or not np.isfinite(g).all() or (g < 0).any() or g.sum() <= 0:
            raise ValueError('Invalid gate-mass vector')
        q = g/g.sum()
        nz = q > 0
        result.update(gate_mass_cv=float(g.std()/g.mean()),
                      normalized_gate_mass_entropy=float(-np.sum(q[nz]*np.log(q[nz]))/np.log(c.size)))
    return result


def js_divergence(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.sum() <= 0 or b.sum() <= 0:
        raise ValueError('Cannot compare empty or differently shaped loads')
    a, b = a/a.sum(), b/b.sum()
    mid = (a+b)/2
    def kl(x):
        nz = x > 0
        return np.sum(x[nz]*np.log(x[nz]/mid[nz]))
    return float((kl(a)+kl(b))/2)  # nats; in [0, ln(2)]


class RouterCollector:
    """Observe module outputs without changing routing decisions or gradients.

    Reductions stay on the input device until finish(), once per forward pass.
    One microbatch = one independently labeled sequence; all layers called once.
    """
    def __init__(self, model, layers, experts, topk, device):
        import torch
        self.torch = torch
        self.layers, self.experts, self.topk = layers, experts, topk
        if not 1 < topk <= experts:
            raise ValueError('This collector requires 1 < topk <= experts')
        self.device = device
        self.handles = []
        self.active = False
        found = {}
        for name, module in model.named_modules():
            match = re.search(r'(?:^|\.)decoder\.layers\.(\d+)\.mlp\.router$', name)
            if match:
                layer = int(match[1])
                if layer in found:
                    raise ValueError(f'Duplicate router layer {layer}')
                found[layer] = (name, module)
        if set(found) != set(range(layers)):
            raise ValueError(f'Expected router layers 0..{layers-1}, found {sorted(found)}')
        self.router_names = [found[l][0] for l in range(layers)]
        for layer, (_, module) in found.items():
            self.handles.append(module.register_forward_hook(self._hook(layer)))

    def begin(self, valid_mask):
        t = self.torch
        if self.active:
            raise RuntimeError('Previous forward was not finished')
        if valid_mask.ndim != 1 or valid_mask.dtype != t.bool:
            raise ValueError('valid_mask must be a flattened bool tensor; B=1 only')
        self.valid = valid_mask.to(self.device)
        shape = (self.layers, self.experts)
        self.counts = t.zeros(shape, dtype=t.int64, device=self.device)
        self.execution_counts = t.zeros_like(self.counts)
        self.gate_mass = t.zeros(shape, dtype=t.float32, device=self.device)
        self.entropy_sum = t.zeros(self.layers, dtype=t.float32, device=self.device)
        self.bad_rows = t.zeros(self.layers, dtype=t.int64, device=self.device)
        self.calls = [0]*self.layers
        self.active = True

    def _hook(self, layer):
        def observe(module, inputs, output):
            t = self.torch
            if not self.active:
                raise RuntimeError('Router ran outside the registered evaluation batch')
            if self.calls[layer]:
                raise RuntimeError(f'Router layer {layer} ran twice; recomputation is unsupported')
            self.calls[layer] += 1
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise ValueError('Expected router output (scores, routing_map[, ...])')
            scores, route = output[:2]
            if not t.is_tensor(route) or route.dtype != t.bool or route.ndim != 2:
                raise ValueError('Expected dense 2D boolean routing_map')
            if tuple(route.shape) != (self.valid.numel(), self.experts) or scores.shape != route.shape:
                raise ValueError(f'Unexpected router shapes: scores={scores.shape}, map={route.shape}')
            with t.no_grad():
                scores = scores.detach().float()
                accepted = route & self.valid[:, None]
                self.counts[layer] = accepted.sum(dim=0, dtype=t.int64)
                self.execution_counts[layer] = route.sum(dim=0, dtype=t.int64)
                weight = t.where(route, scores, t.zeros_like(scores))
                # Ignore padded rows before arithmetic, including any invalid pad scores.
                weight = t.where(self.valid[:, None], weight, t.zeros_like(weight))
                self.gate_mass[layer] = weight.sum(dim=0)
                row_mass = weight.sum(dim=-1)
                normalized = weight/row_mass.clamp_min(1e-30)[:, None]
                entropy = -(normalized*normalized.clamp_min(1e-30).log()).sum(dim=-1)
                self.entropy_sum[layer] = entropy.sum()/math.log(self.topk)
                bad = ((route.sum(dim=-1) != self.topk) |
                       (~t.isfinite(weight).all(dim=-1)) |
                       (weight < 0).any(dim=-1) | (row_mass <= 0)) & self.valid
                self.bad_rows[layer] = bad.sum()
        return observe

    def finish(self):
        if not self.active or any(n != 1 for n in self.calls):
            raise RuntimeError(f'Each router must run once, got {self.calls}')
        bad = self.bad_rows.cpu().numpy()
        if bad.any():
            raise ValueError(f'Invalid scores or non-top-k rows at layers {np.flatnonzero(bad).tolist()}; '
                             'capacity dropping/padding is unsupported')
        result = dict(counts=self.counts.cpu().numpy(), gate_mass=self.gate_mass.cpu().numpy(),
                      execution_counts=self.execution_counts.cpu().numpy(),
                      token_gate_entropy_sum=self.entropy_sum.cpu().numpy())
        self.active = False
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def save_rank(path, rank, expert_rank, indices, batches, layers, experts):
    if not batches:
        raise ValueError('No evaluation rounds')
    np.savez_compressed(path, schema=SCHEMA, rank=rank, expert_rank=expert_rank,
                        sample_indices=np.asarray(indices, dtype=np.int64),
                        **{k: np.stack([b[k] for b in batches]) for k in batches[0]})
