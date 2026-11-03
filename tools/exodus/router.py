"""The routing and distillation arithmetic, checked against the engine that has to agree with it.

`kindi/moe.py` already contains the reference for this repository's expert block, and it renormalises the top-k weights
(`sel / sel.sum()`). The supplied student block does not. Nothing else in the specification changes the routed update, so a
student trained on the supplied block is trained against a different function from the one the CPU engine will run, and
the difference is a scalar that depends on the expert count.

Three claims, three measurements:

1. **Top-k weights that are not renormalised sum to less than one.** Test `renormalisation` measures the sum the supplied
   block feeds to its experts, compares it to the engine's, and reports the effective gain after the specification's
   damping factor.

2. **Per-sequence halting decided by a batch mean is not per-sequence.** `torch.sigmoid(gate).mean() > 0.85` averages over
   the batch, so one confident sequence cannot halt on its own and one unconfident sequence can halt the rest. Test
   `halting` measures both rules on the same gates.

3. **`batchmean` on a 3-D tensor divides by the batch, not by the tokens.** `F.kl_div(..., reduction="batchmean")` divides
   by `input.size(0)`, so on `[B, S, C]` the value is `S` times the per-token mean and any threshold quoted against it
   silently depends on sequence length. Test `loss_scale` repeats one token 64 times and measures the ratio.

On (2), the measurement corrected the guess: the batch mean does not halt too easily, it halts almost never. With four
sequences, three of them certain, the mean is 0.76 and nothing halts; at batch 64 with 90% certain, it fires and stops the
six rows that were still working. A gate that either ignores the loop or mutilates it is not a gate.

    python -m tools.exodus.router
"""
import numpy as np
import torch
import torch.nn.functional as F


# --- routing ---------------------------------------------------------------------------------------------------------

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def topk_supplied(logits, k):
    """What the student block does: take the top k, keep the softmax values as they are."""
    p = softmax(np.asarray(logits, dtype=np.float64))
    idx = np.argsort(-p, axis=-1)[..., :k]
    return idx, np.take_along_axis(p, idx, axis=-1)


def topk_engine(logits, k):
    """What `kindi.moe.softmax_topk` does: the same selection, then renormalised to sum to one."""
    from kindi.moe import softmax_topk
    idx, w = softmax_topk(np.asarray(logits, dtype=np.float64), k)
    return np.asarray(idx), np.asarray(w)


def loop_gain(weight_sum, max_loops):
    """The per-loop residual gain: the specification's damping times the routed weight the block actually applies."""
    return weight_sum * (1.0 / np.sqrt(2.0 * max_loops))


# --- halting ---------------------------------------------------------------------------------------------------------

def halts_batch_mean(gates, threshold=0.85):
    """`torch.sigmoid(self.halt_gate(h)).mean() > 0.85` - one decision for the whole batch, per loop."""
    return bool(torch.sigmoid(torch.as_tensor(gates, dtype=torch.float64)).mean() > threshold)


def halts_per_sequence(gates, threshold=0.85):
    """What a per-sequence halt flag has to look like: one decision per row."""
    return (torch.sigmoid(torch.as_tensor(gates, dtype=torch.float64)) > threshold).numpy()


# --- distillation loss ------------------------------------------------------------------------------------------------

def kl_supplied(student, teacher, temperature=2.0):
    """The specification's loss, verbatim: 3-D inputs, reduction='batchmean'."""
    p_t = F.softmax(teacher / temperature, dim=-1)
    log_p_s = F.log_softmax(student / temperature, dim=-1)
    return float(F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2))


def kl_per_token(student, teacher, temperature=2.0):
    """The same KL normalised by tokens, which is what a threshold can be quoted against."""
    return kl_supplied(student, teacher, temperature) / student.shape[1]


# --- tests ------------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def _logits(experts=8, seed=0, spread=1.5):
    return np.random.RandomState(seed).randn(experts) * spread


def t_renormalisation():
    sums = []
    for seed in range(50):
        _, w = topk_supplied(_logits(seed=seed), 2)
        sums.append(float(w.sum()))
    idx_s, w_s = topk_supplied(_logits(), 2)
    idx_e, w_e = topk_engine(_logits(), 2)
    check("routing: the engine renormalises to 1", abs(float(w_e.sum()) - 1.0) < 1e-12,
          "engine top-2 sum %.12f" % float(w_e.sum()))
    mean_sum = float(np.mean(sums))
    check("routing: the supplied block does not", mean_sum < 0.75,
          "supplied top-2 sum over 50 draws: mean %.3f (min %.3f, max %.3f)"
          % (mean_sum, min(sums), max(sums)))
    check("routing: the selection agrees, only the scale does not", [int(i) for i in idx_s] == [int(i) for i in idx_e],
          "both select %s" % [int(i) for i in idx_e])
    gain = loop_gain(mean_sum, 3)
    check("routing: two attenuations compound in the loop's residual gain", gain < 0.35,
          "damping 0.408 x unnormalised weight sum %.3f = %.3f per loop" % (mean_sum, gain))


def t_halting():
    gates = np.array([3.0, -0.5, -1.0, 0.2])                 # sigmoid -> 0.953, 0.378, 0.269, 0.550
    per = halts_per_sequence(gates)
    batch = halts_batch_mean(gates)
    check("halting: the confident sequence wants to halt on its own", bool(per[0]),
          "per-sequence %s" % per.astype(int).tolist())
    check("halting: the batch mean overrides it", batch is False,
          "batch mean sigmoid %.3f < 0.85, so nothing halts" % float(np.mean(1 / (1 + np.exp(-gates)))))
    three = np.array([9.0, 9.0, 9.0, -3.0])                 # three rows certain, one not
    per3 = halts_per_sequence(three)
    mean3 = float(np.mean(1 / (1 + np.exp(-three))))
    check("halting: three confident rows out of four still do not halt the batch", halts_batch_mean(three) is False,
          "per-sequence %s, mean sigmoid %.3f < 0.85" % (per3.astype(int).tolist(), mean3))
    many = np.array([9.0] * 58 + [-3.0] * 6)                 # 90% confident, at realistic batch size
    per6 = halts_per_sequence(many)
    check("halting: at batch 64 the mean fires, stopping the rows that were not done", halts_batch_mean(many) is True,
          "per-sequence halts %d of 64, mean sigmoid %.3f > 0.85" % (int(per6.sum()),
                                                                    float(np.mean(1 / (1 + np.exp(-many))))))


def t_loss_scale():
    torch.manual_seed(0)
    c = 32
    student = torch.randn(1, c)
    teacher = torch.randn(1, c)
    one = kl_supplied(student.unsqueeze(0), teacher.unsqueeze(0))
    stacked = student.unsqueeze(1).repeat(1, 64, 1)           # the same token 64 times: any difference is the normaliser
    stacked_teacher = teacher.unsqueeze(1).repeat(1, 64, 1)
    sixty_four = kl_supplied(stacked, stacked_teacher)
    ratio = sixty_four / max(one, 1e-300)
    check("loss: batchmean on [B, S, C] scales with sequence length", abs(ratio - 64.0) < 1e-2,
          "the same token repeated 64x gives %.4fx the loss, not 1x (float32 KL accumulation)" % ratio)
    per_tok = kl_per_token(stacked, stacked_teacher)
    check("loss: dividing by S recovers a length-independent number", abs(per_tok - one) < 1e-5 * max(one, 1e-30),
          "per-token %.7f vs single-token %.7f, relative %.1e" % (per_tok, one, abs(per_tok - one) / one))


def selftest():
    del PASS[:]
    for t in (t_renormalisation, t_halting, t_loss_scale):
        t()
    expected = 10
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
