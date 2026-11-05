"""Which experts a prune may not drop, measured by the outlier their own output produces.

`prune` prices an expert by the routing mass it carries: the sum of the renormalised top-k weight it was applied with.
That is the right currency for a byte budget and the wrong one for a *hazard*, because it treats an expert as a small
linear map whose removal costs its share of the output. Some experts are not that.

Su et al. ("Super Experts", ICLR 2026) found that a handful of experts emit rare extreme activations, that those
activations become the model's attention sinks, and that dropping them is catastrophic rather than proportional: on
Qwen3-30B-A3B, 3 experts out of 6,144 (0.05%) sit in layers 1-3, and removing just those three took the average from
70.22 to 55.00, GSM8K from 89.6 to 42.4 and HumanEval to 0.0 - while removing the same *number* of random experts cost
0.20 points. Their statistic is not routing mass and not weight norm: it is the magnitude of what the expert actually
writes into the residual stream, max |down_proj(silu(gate x) * up x)| over the tokens the router sent it.

This module computes that statistic on our own route log and compares it with the plans `prune` produced. The question
it answers is the one a lossless claim depends on: **does a mass-priced prune already keep the outliers, or is it
keeping them by luck?** A plan that drops a flagged expert is not necessarily wrong - the corpus is ours, not theirs -
but it is no longer a plan whose losslessness follows from the routing mass it gave up, and the gate that measured it
was 1,785 tokens wide.

    python -m tools.expert_merge.superex --pack MODEL.gguf --routes routes.npz --layers 0-3 --plan plan112.json
    python -m tools.expert_merge.superex --from-json out.json --plan a.json,b.json     # re-check, no weights read
    python -m tools.expert_merge.superex --selftest

Cost: one dequantisation of three matrices an expert, for the layers asked for. Four layers of this pack is 512
experts and about 0.9 GB of quantised weights, which is a couple of minutes; all 48 layers is ~11 GB and not the point,
because the outliers are reported to live in the first layers and a full sweep only re-measures the same tail.
"""
import json
import sys

import numpy as np

from .moe import silu

GATE, UP, DOWN = "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"


def _name(layer, short):
    """The GGUF name of a layer's expert stack: `blk.12.ffn_down_exps.weight`."""
    return "blk.%d.%s" % (layer, short)


def _span(spec, layers):
    """`0-3,7,40-41` -> the layer ids asked for, in order, without duplicates."""
    if spec is None:
        return list(layers)
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [l for l in out if l < 0 or l >= layers]
    if bad:
        raise SystemExit("layer %d is outside this pack's %d layers" % (bad[0], layers))
    return sorted(set(out))


def profile(pack, routes, layers=None, log=print, progress=8):
    """max |an expert's output| over the tokens that chose it, for every expert in the layers asked for.

    The expert sees the vector the router actually handed it (`routes["xn"]`, ffn_norm(x)), so this is the statistic the
    model runs, not a probe. An expert no token chose has no output and is recorded as 0 - which is a real statement
    (on this corpus it was never used) and the reason a per-corpus profile is not a per-model one.
    """
    xn = routes["xn"]
    ids = routes["route_ids"]
    n_layer = xn.shape[0]
    n_expert = int(ids.max()) + 1
    layers = _span(layers, n_layer) if layers is not None else list(range(n_layer))
    out = np.zeros((n_layer, n_expert), np.float32)
    counts = np.zeros((n_layer, n_expert), np.int32)
    for done, il in enumerate(layers):
        for e in range(n_expert):
            tok = np.nonzero((ids[il] == e).any(axis=1))[0]
            counts[il, e] = len(tok)
            if not len(tok):
                continue
            x = np.asarray(xn[il][tok], np.float32).T                      # [d, tokens]
            g = pack.expert_weight(_name(il, GATE), e) @ x
            u = pack.expert_weight(_name(il, UP), e) @ x
            h = silu(g) * u
            y = pack.expert_weight(_name(il, DOWN), e) @ h
            out[il, e] = float(np.abs(y).max())
        if log and (done + 1) % progress == 0 or done + 1 == len(layers):
            sub = out[il]
            log("  layer %2d: %d experts used, max %.3f (expert %d), median %.3f"
                % (il, int((counts[il] > 0).sum()), float(sub.max()), int(sub.argmax()),
                   float(np.median(sub[sub > 0])) if (sub > 0).any() else 0.0))
    return out, counts, layers


def flagged(out, frac=0.10, ratio=None):
    """The super-expert mask: an expert whose own output is extreme, judged against what its layer normally writes.

    Two thresholds, because one of them is wrong on this pack. The paper states a **global** one (P99.5 and 10% of the
    largest output anywhere). That works on a model whose layers write comparably, and does not work here: the layer
    median rises 150x with depth (0.31 at layer 1, 56.6 at layer 47), so 10% of the global peak flags 23 experts in
    layer 47 that are only 1.7x their own layer's median, and it would bury a layer-1 expert that is 100x its own.

    `ratio` is the threshold that transfers: flag an expert above `ratio` times **its own layer's** median output. On
    this pack the two populations separate cleanly that way - three experts at 525-2,152x their layer, then a cliff to
    43x - and `ratio=100` is a wide margin inside that gap, not a tuned number.
    """
    out = np.asarray(out, dtype=np.float64)
    peak = float(out.max())
    if ratio is None:
        return (out >= frac * peak) & (out > 0), peak
    med = np.zeros(out.shape[0])
    for l in range(out.shape[0]):
        used = out[l][out[l] > 0]
        med[l] = np.median(used) if len(used) else 0.0
    mask = np.zeros(out.shape, bool)
    for l in range(out.shape[0]):
        if med[l] > 0:
            mask[l] = (out[l] >= ratio * med[l]) & (out[l] > 0)
    return mask, peak


def layer_ratio(out, mask):
    """(layer, expert, output / the layer's median) for the flagged experts, largest first."""
    out = np.asarray(out, dtype=np.float64)
    rows = []
    for l, e in zip(*np.nonzero(mask)):
        used = out[l][out[l] > 0]
        med = float(np.median(used)) if len(used) else 0.0
        rows.append((int(l), int(e), float(out[l, e]), float(out[l, e] / med) if med else 0.0))
    return sorted(rows, key=lambda r: -r[3])


def ranks(out):
    """Each expert's rank in its own layer by output, 1 = the largest. Layers, not the pack: the statistic is a
    property of a layer's routing, and a global rank would mostly report which layers fire hardest."""
    order = np.argsort(-np.asarray(out), axis=1)
    r = np.empty_like(order)
    for l in range(order.shape[0]):
        r[l, order[l]] = np.arange(1, order.shape[1] + 1)
    return r


def check_plan(out, mask, keep_per_layer, mass=None, log=print):
    """Does a plan drop a flagged expert, and how much routing mass would that give up?

    Returns the dropped flagged ids as `[(layer, expert), ...]`. A plan is not judged here: it is *reported*, because
    the alternative - a hard refusal - would be a rule borrowed from another corpus. What the caller can act on is the
    list, and the fact that a plan which drops one has lost its "the mass I gave up was 0.1%" argument.
    """
    rows = []
    for il, keep in enumerate(keep_per_layer):
        if il >= out.shape[0]:
            break
        keep = set(int(k) for k in keep)
        for e in np.nonzero(mask[il])[0]:
            if int(e) not in keep:
                rows.append(dict(layer=int(il), expert=int(e), output=float(out[il, e]),
                                 rank=int(ranks(out[il:il + 1])[0, e]),
                                 mass=float(mass[il, e]) if mass is not None else None))
    if log:
        if not rows:
            log("  keeps every flagged expert (%d of them)" % int(mask.sum()))
        else:
            log("  DROPS %d flagged expert(s): %s"
                % (len(rows), ", ".join("L%dE%d (rank %d, output %.2f)"
                                        % (r["layer"], r["expert"], r["rank"], r["output"]) for r in rows)))
    return rows


def plans_of(obj):
    """A prune json, a list of them, or a single plan: whatever it is, yield (name, keep_per_layer)."""
    if isinstance(obj, dict) and "plans" in obj:
        for p in obj["plans"]:
            yield str(p.get("spec") or p.get("method") or "plan"), p["keep_per_layer"]
    elif isinstance(obj, dict) and "keep_per_layer" in obj:
        yield str(obj.get("spec") or "plan"), obj["keep_per_layer"]
    elif isinstance(obj, list):
        for p in obj:
            yield str(p.get("spec") or "plan"), p["keep_per_layer"]
    else:
        raise SystemExit("not a prune artifact: expected `plans`, `keep_per_layer`, or a list of plans")


def main(argv):
    if "--selftest" in argv:
        return selftest()
    args = dict(zip(argv[::2], argv[1::2]))
    if "--from-json" in argv:
        with open(_arg(argv, "--from-json"), encoding="utf-8") as f:
            got = json.load(f)
        out = np.asarray(got["output"], np.float32)
        layers = got["layers"]
    else:
        from .ggufmoe import Moepack
        from .routes import load

        pack = _arg(argv, "--pack")
        routes_path = _arg(argv, "--routes")
        if not (pack and routes_path):
            print(__doc__.strip().split("\n\n")[-1])
            return 2
        p = Moepack(pack)
        got = load(routes_path)
        print("  %s: %d layers, %d experts; %d tokens of routing" % (
            pack.split("/")[-1].split("\\")[-1], p.n_layer, p.n_expert, got["route_ids"].shape[1]))
        out, counts, layers = profile(p, got, _arg(argv, "--layers"))
    frac = float(_arg(argv, "--frac", 0.10))
    ratio = float(_arg(argv, "--ratio", 100.0))
    mask_global, peak = flagged(out, frac)
    mask, _p = flagged(out, frac, ratio=ratio)
    ranks_all = ranks(out)
    print("  peak output %.3f" % peak)
    print("  a global %.0f%%-of-peak cut flags %d expert(s); above %.0fx their own layer's median: %d"
          % (100.0 * frac, int(mask_global.sum()), ratio, int(mask.sum())))
    for l, e, o, x in layer_ratio(out, mask):
        print("    L%02dE%-3d output %9.2f  %.0fx the layer median" % (l, e, o, x))
    late = sorted(layer_ratio(out, mask_global), key=lambda r: -r[3])[:5]
    if late:
        print("  the global cut's own top five sit at %s x their layer median, which is why it is not the one used"
              % ", ".join("%.1f" % r[3] for r in late))
    mass = None
    if "--routes" in argv:
        from .prune import mass_per_expert

        mass, _count = mass_per_expert(got)
    reports = {}
    for spec in [s for s in (_arg(argv, "--plan") or "").split(",") if s.strip()]:
        with open(spec, encoding="utf-8") as f:
            obj = json.load(f)
        for name, keep in plans_of(obj):
            print("\n  %s / %s" % (spec.split("/")[-1].split("\\")[-1], name))
            reports["%s:%s" % (spec, name)] = check_plan(out, mask, keep, mass=mass)
    js = _arg(argv, "--json")
    if js:
        with open(js, "w", encoding="utf-8") as f:
            json.dump(dict(layers=[int(l) for l in layers], output=out.tolist(), counts=counts.tolist(),
                           flagged=[[int(l), int(e)] for l, e in zip(*np.nonzero(mask))],
                           flagged_global=[[int(l), int(e)] for l, e in zip(*np.nonzero(mask_global))],
                           ratio=ratio,
                           rank=ranks_all.tolist(), frac=frac,
                           dropped=reports), f)
        print("\n  wrote %s" % js)
    return 0


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def selftest():
    """A pack with one deliberately extreme expert: the profile must find it, and a plan must be seen to drop it."""
    import os
    import tempfile

    from .mergepack import _synthetic_pack

    done = []

    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))
        done.append(name)

    td = tempfile.mkdtemp(prefix="superex-")
    src = _synthetic_pack(os.path.join(td, "src.gguf"))
    from .ggufmoe import Moepack

    # the synthetic pack is one layer, four experts. Make expert 2 shout: every other expert writes ~1.0, this one ~30x
    p = Moepack(src)
    down = [p.expert_weight(_name(0, DOWN), e) for e in range(p.n_expert)]
    down[2] = down[2] * 30.0
    out = np.zeros((1, p.n_expert), np.float32)
    for e in range(p.n_expert):
        out[0, e] = float(np.abs(down[e]).max())

    mask, peak = flagged(out, 0.10)
    check("superex: the extreme expert is flagged", mask[0, 2] and int(mask.sum()) == 1, str(mask[0].tolist()))
    check("superex: and the peak is the expert we scaled", abs(peak - out[0, 2]) < 1e-6)
    check("superex: the flag is relative, not absolute - a 30x outlier is 3x the next largest",
          out[0, 2] > 3.0 * sorted(out[0])[-2], str(np.round(out[0], 2).tolist()))

    r = ranks(out)
    check("superex: rank is per layer, 1 for the largest output", r[0, 2] == 1 and r[0, 0] == 2, str(r[0].tolist()))

    # the threshold that transfers is relative to the expert's own layer: a late layer that writes 100x more than an
    # early one hides an extreme early expert under a global cut, and flags a whole late layer that is uniform
    two = np.array([[0.1, 0.2, 20.0], [300.0, 320.0, 250.0]], np.float32)
    glob, _p = flagged(two, 0.10)
    rel, _p2 = flagged(two, 0.10, ratio=50.0)
    check("superex: a global cut misses the early outlier once a later layer writes more",
          not glob[0, 2] and int(glob.sum()) == 3, str(glob.tolist()))
    check("superex: and relative to its own layer the outlier is the only expert flagged",
          rel[0, 2] and int(rel.sum()) == 1, str(rel.tolist()))
    check("superex: the layer ratio is what separates 100x from 1.07x, not the absolute output",
          abs(layer_ratio(two, rel)[0][3] - 100.0) < 1e-3, str(layer_ratio(two, rel)))

    from .prune import mass_per_expert

    # six tokens, top-2: expert 2 is chosen once and with a small weight, so it carries less routing mass than every
    # expert a mass-priced prune would keep - which is exactly the case the outlier statistic has to catch
    routes = dict(route_ids=np.array([[[0, 1], [0, 1], [0, 2], [0, 3], [0, 3], [0, 1]]], np.int16),
                  route_wts=np.array([[[.5, .5], [.5, .5], [.9, .1], [.5, .5], [.5, .5], [.5, .5]]], np.float16))
    mass, _count = mass_per_expert(routes, p.n_expert)
    check("superex: routing mass is blind to the outlier - the three ordinary experts carry more of it",
          mass[0, 2] < mass[0, 0] + 1e-6, str(np.round(mass[0], 3).tolist()))

    keep_ok = [0, 1, 2, 3]
    keep_bad = [0, 1, 3]
    check("superex: a plan that keeps the outlier is reported clean",
          check_plan(out, mask, [keep_ok], mass=mass, log=None) == [])
    bad = check_plan(out, mask, [keep_bad], mass=mass, log=None)
    check("superex: a plan that drops it is reported with its rank and its mass",
          len(bad) == 1 and bad[0]["layer"] == 0 and bad[0]["expert"] == 2 and bad[0]["rank"] == 1
          and bad[0]["mass"] == float(mass[0, 2]), str(bad))
    check("superex: and the mass it dropped is smaller than the experts the plan kept, which is the trap",
          bad[0]["mass"] < mass[0, 0])
    check("superex: a prune artifact is read back as one plan a spec",
          [n for n, _ in plans_of({"plans": [{"spec": "keep=3", "keep_per_layer": [[0, 1, 2]]}]})] == ["keep=3"])
    print("%d claims held" % len(done))
    print("checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
