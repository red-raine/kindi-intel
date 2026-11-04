"""Run the permutation test on the experts a real router log groups together.

The grouping comes from the log, the verdict comes from the weights, and the merge cost is measured on the
activations the router actually produced. Three pair sets are tested per layer:

* **routed** - the pairs whose routing profiles are most alike, i.e. the experts this model treats as
  interchangeable on this corpus. This is the set a merge would operate on.
* **weights** - the pair whose flattened weights are most alike. A merge proposal that never looked at routing
  would start here.
* **random** - a control, so "no pair is a permutation" cannot be a property of how the pairs were picked.

Every pair gets: the identity residual, the best-bijection residual, the best *orthogonal* residual (the most any
relabelling can buy), and the fraction of hidden units that have a one-to-one partner at |cos| > 0.99. A planted
permutation of a real expert is carried through the same code as a positive control.
"""
import json
import os
import sys

import numpy as np

from . import permtest as P


def profile(wts, ids, n_expert):
    """[n_expert, tokens]: the routing weight each expert was given on each token"""
    T = wts.shape[0]
    out = np.zeros((n_expert, T), np.float32)
    for k in range(wts.shape[1]):
        np.add.at(out, (ids[:, k], np.arange(T)), wts[:, k].astype(np.float32))
    return out


def cos_matrix(Pf):
    n = np.linalg.norm(Pf, axis=1, keepdims=True)
    U = Pf / np.maximum(n, 1e-30)
    return U @ U.T


def weight_cos(pack, il, experts):
    """cosine between experts' flattened (gate, up) matrices, which is what a merge-by-similarity would rank

    Loaded one expert at a time and kept only for this layer's Gram matrix: 128 experts of 2 x 768 x 2048 floats is
    1.6 GB, and holding that for fourteen layers is how a 32 GB machine starts swapping.

    Also returns which experts decode at all. A part-written file damages a byte range, not a tensor, so the last
    layer before the break can hold a few experts of noise while the rest of the layer is fine - and a NaN weight is
    not something to hand to an assignment solver, it is something to count.
    """
    rows, valid = [], []
    for e in experts:
        E = P.load(pack, il, e)
        ok = bool(np.isfinite(E["gate"]).all() and np.isfinite(E["up"]).all())
        valid.append(ok)
        rows.append(np.concatenate([E["gate"].ravel(), E["up"].ravel()]) if ok else None)
    M = np.stack([r if r is not None else np.zeros_like(rows[0]) for r in rows])
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-30)
    return M @ M.T, np.array(valid)


def pick(pack, il, cm, fire, min_fire, k_routed, k_rand, rng):
    """the routed pairs, the most-alike weight pair and a random control, for one layer

    Only experts that fire and decode are eligible: a pair of noise rows would answer a different question.
    """
    out, skipped = [], []
    used = np.where(fire >= min_fire)[0]
    if used.size < 2:
        return out, skipped
    wc, valid = weight_cos(pack, il, list(used))
    skipped = [int(e) for e, ok in zip(used, valid) if not ok]
    ok = used[valid]
    if ok.size >= 2:
        idx = {int(e): k for k, e in enumerate(used) if valid[k]}
        sub = np.full((ok.size, ok.size), -np.inf)
        for i, a in enumerate(ok):
            for j, b in enumerate(ok):
                if i != j:
                    sub[i, j] = cm[a, b]
        order = np.dstack(np.unravel_index(np.argsort(-sub, axis=None), sub.shape))[0]
        seen, routed = set(), []
        for i, j in order:
            a, b = int(ok[i]), int(ok[j])
            if (b, a) in seen or a == b:
                continue
            seen.add((a, b))
            routed.append((a, b, float(sub[i, j])))
            if len(routed) >= k_routed:
                break
        out += [("routed", a, b, c) for a, b, c in routed]
        # most-alike weights, among the experts that are used at all
        wsub = np.array([[wc[idx[a], idx[b]] if a != b else -np.inf for b in ok] for a in ok])
        i, j = np.unravel_index(np.argmax(wsub), wsub.shape)
        out.append(("weights", int(ok[i]), int(ok[j]), float(wsub[i, j])))
    if ok.size >= 3:
        for _ in range(k_rand):
            a, b = rng.choice(ok, 2, replace=False)
            out.append(("random", int(a), int(b), float(cm[a, b])))
    return out, skipped


def measure_layer(pack, log, il, k_routed, k_rand, rng):
    """every pair this layer contributes: 3 routed, 1 most-alike by weights, k_rand random controls

    Split out of `main` so layers can be measured in parallel: each one is independent, loads only its own
    experts, and a 48-layer sweep at ~2.5 minutes a layer is the difference between a measurement and an
    overnight job.
    """
    prof = profile(log["route_wts"][il], log["route_ids"][il].astype(np.int64), pack.n_expert)
    fire = (prof > 0).sum(axis=1)
    cm = cos_matrix(prof)
    X = log["xn"][il].astype(np.float32)
    pairs, skipped = pick(pack, il, cm, fire, 5, k_routed, k_rand, rng)
    rows = []
    for rank, (cat, ea, eb, c) in enumerate(pairs):
        A, B = P.load(pack, il, ea), P.load(pack, il, eb)
        m = P.classify_pair(A, B, rng=rng)
        # one orthogonal per layer: a 2048 x 2048 SVD costs seconds and the floor it establishes
        # is a property of the layer's expert family, not of the pair
        want_orth = cat == "routed" and rank == 0
        m["r_orth"] = P.rel_error(B, P.orthogonal(B, A)) if want_orth else float("nan")
        cost = [{k: v for k, v in r.items()} for r in P.merge_cost(A, B, X)] if cat == "routed" else []
        rows.append(dict(layer=il, cat=cat, a=ea, b=eb, profile_cos=c,
                         fire_a=int(fire[ea]), fire_b=int(fire[eb]),
                         verdict=m["verdict"], r_id=m["r_id"], r_perm=m["r_perm"],
                         r_orth=m["r_orth"], r_rand=m["r_rand"], gain=m["gain"],
                         row_cos=m["row_cos_gate"], frac99=m["frac_gt_99"],
                         same_partner=m["same_partner"], merge=cost))
    return rows, skipped


_JOB = {}


def _init(pack_path, routes_path):
    """one pack and one log per worker, not per task"""
    from .ggufmoe import Moepack
    _JOB["pack"] = Moepack(pack_path)
    got = np.load(routes_path, allow_pickle=False)
    _JOB["log"] = dict(zip(got.files, [got[k] for k in got.files]))


def _work(args):
    il, k_routed, k_rand, seed = args
    rows, skipped = measure_layer(_JOB["pack"], _JOB["log"], il, k_routed, k_rand,
                                  np.random.default_rng(seed + il))
    return il, rows, skipped


def main(argv):
    from .ggufmoe import Moepack

    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    path = arg("--routes", "routes.npz")
    pack_path = arg("--pack")
    out_json = arg("--out", "merge.json")
    k_routed = int(arg("--routed", 3))
    jobs = int(arg("--jobs", 1))
    got = np.load(path, allow_pickle=False) if path.endswith(".npz") else None
    log = dict(zip(got.files, [got[k] for k in got.files]))
    if pack_path is None:
        meta_path = path[:-4] + ".json"
        pack_path = json.load(open(meta_path, encoding="utf-8"))["pack"]
    pack = Moepack(pack_path)
    pack.require_complete(log=print, allow_partial="--partial" in argv)
    L, T = log["route_ids"].shape[0], log["route_ids"].shape[1]
    L = min(L, int(arg("--layers", L)))
    # --shard i/n: layers i, i+n, i+2n ... . On Windows a process pool spends more time duplicating handles than
    # measuring, so the shards are separate processes over the same json and rsync it through --shard-merge.
    shard = arg("--shard")
    if shard:
        si, sn = [int(x) for x in shard.split("/")]
    rng = np.random.default_rng(11)
    rows, done = [], set()
    if os.path.exists(out_json):
        try:
            prev = json.load(open(out_json, encoding="utf-8"))
            if prev.get("pack") == pack_path:
                rows = prev.get("rows", [])
                done = {r["layer"] for r in rows}
                print("  resuming: %d rows already measured for layers %s" % (len(rows), sorted(done)))
        except Exception:
            rows, done = [], set()
    print("  %d layers x %d tokens, %d experts, top-%d, %d job(s)"
          % (L, T, pack.n_expert, pack.n_expert_used, jobs))
    todo = [il for il in range(L) if il not in done]
    if shard:
        todo = [il for il in todo if il % sn == si]
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_init, initargs=(pack_path, path)) as pool:
            for il, new, skipped in pool.imap_unordered(_work, [(i, k_routed, 3, 11) for i in todo]):
                rows += new
                if skipped:
                    print("    layer %2d: %d experts do not decode and were skipped (%s...)"
                          % (il, len(skipped), skipped[:6]))
                json.dump(dict(rows=rows, pack=pack_path, layers=L, tokens=T),
                          open(out_json, "w", encoding="utf-8"), indent=1)
                print("    layer %2d done" % il)
    else:
        for il in todo:
            new, skipped = measure_layer(pack, log, il, k_routed, 3, rng)
            rows += new
            if skipped:
                print("    layer %2d: %d experts do not decode and were skipped (%s...)"
                      % (il, len(skipped), skipped[:6]))
            json.dump(dict(rows=rows, pack=pack_path, layers=L, tokens=T),
                      open(out_json, "w", encoding="utf-8"), indent=1)
            print("    layer %2d done" % il)
    # positive control: a real expert against a permutation of itself
    ctrl = []
    for e in (0, 17, 63, 100):
        A = P.load(pack, 0, e)
        pi = rng.permutation(A["gate"].shape[0])
        s = rng.choice([-1.0, 1.0], pi.size).astype(np.float32)
        m = P.classify_pair(A, P.permute(A, pi, s), rng=rng)
        X = log["xn"][0].astype(np.float32)
        c = {r["how"]: r for r in P.merge_cost(A, P.permute(A, pi, s), X)}
        ctrl.append(dict(expert=e, verdict=m["verdict"], r_perm=m["r_perm"],
                         merge_raw=c["raw"]["err_a"], merge_aligned=c["aligned"]["err_a"]))
        print("    control expert %3d: %s (residual %.1e, merge raw %.3f -> aligned %.1e)"
              % (e, m["verdict"], m["r_perm"], c["raw"]["err_a"], c["aligned"]["err_a"]))
    json.dump(dict(rows=rows, control=ctrl, pack=pack_path, layers=L, tokens=T),
              open(out_json, "w", encoding="utf-8"), indent=1)
    print("\n  wrote %s" % out_json)
    return 0


def merge_only(json_path, routes_path):
    """recompute only the merge costs, with the shipped merge, from a pair list already chosen

    The verdicts and residuals do not depend on the merge, so this is the cheap half: no bijection search, no SVD,
    just the two merges per routed pair on the captured activations.
    """
    from .ggufmoe import Moepack
    d = json.load(open(json_path, encoding="utf-8"))
    log = np.load(routes_path, allow_pickle=False)
    pack = Moepack(d["pack"])
    pack.require_complete(log=print, allow_partial="--partial" in sys.argv)
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    todo = [i for i, r in enumerate(d["rows"]) if r["cat"] == "routed"]
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_mload, initargs=(json_path, routes_path)) as pool:
            for i, spec, view in pool.imap_unordered(_mwork, todo):
                d["rows"][i]["merge_spec"], d["rows"][i]["merge_view"] = spec, view
                r = d["rows"][i]
                print("  layer %2d  %3d/%3d  cosine %.3f  spec %s  view %s"
                      % (r["layer"], r["a"], r["b"], spec["aligned"]["cosine"],
                         "%.3f" % spec["aligned"]["err_a"], "%.3f" % view["aligned"]["err_a"]))
    else:
        for i in todo:
            r = d["rows"][i]
            A = P.load(pack, r["layer"], r["a"])
            B = P.load(pack, r["layer"], r["b"])
            X = log["xn"][r["layer"]].astype(np.float32)
            r["merge_spec"] = {x["how"]: x for x in P.merge_cost(A, B, X, spec=True)}
            r["merge_view"] = {x["how"]: x for x in P.merge_cost(A, B, X, spec=False)}
    json.dump(d, open(json_path, "w", encoding="utf-8"), indent=1)
    return 0


def _mload(json_path, routes_path):
    from .ggufmoe import Moepack
    _JOB["d"] = json.load(open(json_path, encoding="utf-8"))
    _JOB["pack"] = Moepack(_JOB["d"]["pack"])
    log = np.load(routes_path, allow_pickle=False)
    _JOB["log"] = {k: log[k] for k in log.files}


def _mwork(i):
    r = _JOB["d"]["rows"][i]
    A = P.load(_JOB["pack"], r["layer"], r["a"])
    B = P.load(_JOB["pack"], r["layer"], r["b"])
    X = _JOB["log"]["xn"][r["layer"]].astype(np.float32)
    return (i, {x["how"]: x for x in P.merge_cost(A, B, X, spec=True)},
            {x["how"]: x for x in P.merge_cost(A, B, X, spec=False)})


def combine(paths, out):
    """one json from several `--shard` runs

    A shard that was seeded with a partly finished json (the way to skip the layers another run already measured)
    carries those rows as well, so rows are deduplicated on (layer, category, expert a, expert b): a silent double
    count is a summary that weights one layer twice, and the medians would be the average of a set nobody described.
    """
    rows, pack, layers, tokens, control = [], None, 0, 0, []
    seen, seen_c, dropped = set(), set(), 0
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        for r in d.get("rows", []):
            key = (r["layer"], r["cat"], r["a"], r["b"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key); rows.append(r)
        for c in d.get("control", []):
            if c.get("expert") in seen_c:
                continue
            seen_c.add(c.get("expert")); control.append(c)
        pack = pack or d.get("pack")
        layers = max(layers, d.get("layers", 0))
        tokens = max(tokens, d.get("tokens", 0))
    rows.sort(key=lambda r: (r["layer"], r["cat"] != "routed", r["a"], r["b"]))
    json.dump(dict(rows=rows, control=control, pack=pack, layers=layers, tokens=tokens),
              open(out, "w", encoding="utf-8"), indent=1)
    print("  %d rows, %d control(s), from %d shard(s), %d duplicate row(s) dropped -> %s"
          % (len(rows), len(control), len(paths), dropped, out))
    return 0


def summarise(path):
    """the numbers for the note, read back off the json"""
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    print("%-9s %4s %5s %7s %7s %7s %7s %6s %6s" %
          ("set", "n", "perm", "profile", "r_id", "r_perm", "gain", "|c|>.99", "same"))
    for cat in ("routed", "weights", "random"):
        sub = [r for r in rows if r["cat"] == cat]
        if not sub:
            continue
        q = lambda k: float(np.median([r[k] for r in sub]))
        print("%-9s %4d %5d %7.3f %7.3f %7.3f %7.3f %8.4f %6.3f" %
              (cat, len(sub), sum(r["verdict"] == "permutation" for r in sub),
               q("profile_cos"), q("r_id"), q("r_perm"), q("gain"), q("frac99"), q("same_partner")))
    routed = [r for r in rows if r["cat"] == "routed" and r["merge"]]
    if routed:
        raw = np.median([r["merge"][0]["err_a"] for r in routed])
        al = np.median([r["merge"][1]["err_a"] for r in routed])
        bet = np.median([r["merge"][0]["err_a_vs_b"] for r in routed])
        print("\n  merge on the routed pairs (median over %d): parents differ by %.3f, "
              "raw average misses by %.3f, aligned average by %.3f"
              % (len(routed), bet, raw, al))
    gains = np.array([r["gain"] for r in rows])
    print("  alignment gain over all %d pairs: median %.3f, p90 %.3f, max %.3f"
          % (gains.size, float(np.median(gains)), float(np.percentile(gains, 90)), float(gains.max())))
    print("  verdicts:", {v: sum(r["verdict"] == v for r in rows)
                         for v in ("permutation", "aligned-similar", "merely-similar")})
    for c in d["control"]:
        print("  control: a permutation of expert %d -> %s, merge %.3f raw vs %.1e aligned"
              % (c["expert"], c["verdict"], c["merge_raw"], c["merge_aligned"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarise(sys.argv[2] if len(sys.argv) > 2 else "merge.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "--merge-only":
        raise SystemExit(merge_only(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--combine":
        argv = sys.argv[2:]                            # `--out` names a file that does not exist yet: take it out of the
        if "--out" in argv:                            # inputs before the list is filtered by extension
            k = argv.index("--out")
            out, argv = argv[k + 1], argv[:k] + argv[k + 2:]
        else:
            out = "merge.json"
        raise SystemExit(combine([a for a in argv if a.endswith(".json")], out))
    else:
        raise SystemExit(main(sys.argv[1:]))
