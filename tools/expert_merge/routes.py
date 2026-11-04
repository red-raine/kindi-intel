"""A real router log: every layer, every token, the 128 logits and the eight experts a real MoE picked.

The log is the input to the whole question this package exists for. "Experts that cluster together" has to mean
something the model actually did, not a similarity computed on weights by the same code that then tests it, so the
grouping comes from routing decisions on a real corpus and only the permutation test looks at weights.

What is stored per layer and token: the 128 router logits (fp16 - the router is 1,024 weights a row, so fp16 is
lossless enough to re-rank), the top-8 ids, their renormalised weights, and `x_n` = ffn_norm(x), the vector the
experts actually saw. That last one is what a merge has to be judged on: two experts that agree on paper and differ
on the activations the router sends them are not interchangeable.
"""
import json
import os

import numpy as np

DEFAULT_DOCS = ("docs/CPU-PERF-2026.md", "tools/kl.py", "README.md", "kindi/cpu.py")


def corpus(tok, files, per_doc=96, root="."):
    """the first `per_doc` tokens of each file, as its own prompt"""
    docs = []
    for name in files:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read(per_doc * 12)
        ids = tok.encode(text)[:per_doc]
        if len(ids) >= 8:
            docs.append((name, ids))
    return docs


def run(model, docs, log=print):
    """one prefill per document; the arrays stack as [layer][doc][token]"""
    out = dict(route_logits=[], route_ids=[], route_wts=[], xn=[], doc=[], pos=[])
    for k, (name, ids) in enumerate(docs):
        log("  %s: %d tokens" % (name, len(ids)))
        got = model.forward(ids, capture=True)
        out["route_logits"].append(got["route_logits"])
        out["route_ids"].append(got["route_ids"])
        out["route_wts"].append(got["route_wts"])
        out["xn"].append(got["xn"])
        out["doc"].append(np.full(len(ids), k, np.int16))
        out["pos"].append(np.arange(len(ids), dtype=np.int16))
    return dict(route_logits=np.concatenate(out["route_logits"], axis=1),
                route_ids=np.concatenate(out["route_ids"], axis=1),
                route_wts=np.concatenate(out["route_wts"], axis=1),
                xn=np.concatenate(out["xn"], axis=1),
                doc=np.concatenate(out["doc"]), pos=np.concatenate(out["pos"]))


def save(path, got, meta):
    np.savez_compressed(path, **got)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return path


def load(path):
    z = np.load(path, allow_pickle=False)
    got = {k: z[k] for k in z.files}
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
        got["meta"] = json.load(f)
    return got


def main(argv):
    from .ggufmoe import Moepack
    from .moe import Qwen3Moe, tokenizer

    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    path = arg("--pack")
    if not path:
        print("usage: python -m tools.expert_merge.routes --pack MODEL.gguf [--out routes.npz] [--tokens 96] "
              "[--partial]")
        return 2
    per_doc = int(arg("--tokens", 96))
    out = arg("--out", "routes.npz")
    pack = Moepack(path)
    print("  reading %s" % path)
    tok = tokenizer(pack)
    docs = corpus(tok, arg("--docs", ",").split(",") if "--docs" in argv else DEFAULT_DOCS, per_doc=per_doc)
    model = Qwen3Moe(path, allow_partial="--partial" in argv)
    got = run(model, docs)
    meta = dict(pack=path, layers=model.n_layers, of_layers=pack.n_layer, experts=pack.n_expert,
                active=pack.n_expert_used, docs=[(n, len(i)) for n, i in docs], tokens=int(got["doc"].size),
                note="router log from a real Qwen3-MoE pack; x_n is the vector the experts saw")
    save(out, got, meta)
    rl = got["route_logits"].astype(np.float32)
    share = []
    for L in range(rl.shape[0]):
        p = np.exp(rl[L] - rl[L].max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        top = np.sort(p, axis=1)[:, -pack.n_expert_used:]
        share.append(top.sum(axis=1).mean())
    print("  %d layers, %d tokens, mean top-8 share %.3f (uniform %.3f)"
          % (rl.shape[0], rl.shape[1], float(np.mean(share)), pack.n_expert_used / pack.n_expert))
    print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
