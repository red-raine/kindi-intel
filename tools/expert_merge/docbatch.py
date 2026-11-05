"""Price the gate's forward: documents one at a time against one batched pass, on the real pack.

`docs/EXPERT-PRUNE.md` ends on an arithmetic limit - the 0.02 tolerance needs 918 tokens at 112 experts a layer and
1,668 at 96, against 765 measured - and the reason it was a limit at all is that the forward pays the dequantiser for
a layer's experts once **a document**: a 256-token document is ~17 minutes however many tokens it holds, because the
experts a document's tokens route to already run close to all 128. Batching the corpus into one pass takes that union
once for the whole corpus instead, so the same hours buy several times the tokens.

This tool is the check, not the argument. It runs the same documents both ways on the real weights - `forward` a
document at a time, `forward_docs` once for all of them - and reports three things:

* whether it is the **same computation**: per-document NLL both ways and the largest token-level disagreement. It is
  not expected to be bit for bit (a BLAS GEMV over 256 rows and one over 768 block differently) but it is expected to
  be far below anything the gate is deciding;
* what the batching **costs in memory**: one layer's f32 experts plus the logits of the whole batch, against a forward
  that never holds more than one document's;
* what it **buys in time**: wall clock both ways, and, from that rate, the corpus a gate could afford if the whole
  corpus goes through one pass.

The layers are truncated with `--layers` so the check is minutes rather than an hour; the ratio it reports is a ratio
of the same code, and the projection scales it by 48/n. Run it at the full depth when a corpus size is being claimed.

  python -m tools.expert_merge.docbatch --pack MODEL.gguf --layers 8 --docs 3 --tokens 96
  python -m tools.expert_merge.docbatch --pack MODEL.gguf --layers 48 --docs 12 --tokens 256
"""
import time

import numpy as np

from .moe import Qwen3Moe, tokenizer
from .prune import _per_token, nll_moe, nll_moe_batch
from .routes import corpus

DEFAULT_DOCS = ("docs/CPU-PERF-2026.md", "tools/kl.py", "docs/BENCHMARKS.md", "docs/PACK-ZOO.md",
                "docs/TESTING.md", "docs/RECIPES.md", "tools/nll.py", "kindi/cpu.py", "README.md",
                "docs/EXPERT-MERGE.md", "tools/bitnet_gemm/README.md", "kindi/quant.py")


def _arg(argv, name, default=None, cast=str):
    if name not in argv:
        return default
    return cast(argv[argv.index(name) + 1])


def measure(pack, layers=None, ndocs=3, tokens=96, names=DEFAULT_DOCS, log=print):
    """Both forwards over the same documents: agreement, wall clock, and the corpus the batched rate affords."""
    model = Qwen3Moe(pack, log=lambda *a: None)
    full = model.n_layers
    if layers:
        model.n_layers = min(int(layers), full)               # a prefix is the same code on fewer layers
    tok = tokenizer(model.pack)
    docs = corpus(tok, names, per_doc=int(tokens))[:int(ndocs)]
    if len(docs) < 2:
        raise SystemExit("docbatch: need at least two documents to compare a batch with a serial pass")
    log("  %d documents, %d tokens each, %d of %d layers (%.2f of the depth)"
        % (len(docs), len(docs[0][1]), model.n_layers, full, model.n_layers / float(full)))

    t0 = time.time()
    single = [nll_moe(model, ids) for _, ids in docs]
    t_single = time.time() - t0
    t0 = time.time()
    batch = nll_moe_batch(model, docs)
    t_batch = time.time() - t0

    per_single = np.concatenate([s["per_token"] for s in single])
    d = np.abs(per_single - batch["per_token"])
    n_tok = int(batch["tokens"])
    log("")
    log("  | document | tokens | nll, serial | nll, batched |")
    log("  |---|---:|---:|---:|")
    for (name, ids), s, b in zip(docs, single, batch["per_doc"]):
        log("  | %s | %d | %.6f | %.6f |" % (name, max(1, len(ids) - 1), s["nll"], b))
    log("")
    log("  token-level agreement: max |dnll| %.3g, mean |dnll| %.3g, %d tokens"
        % (float(d.max()), float(d.mean()), n_tok))
    log("  corpus nll: serial %.6f, batched %.6f (difference %+.2e)"
        % (float(per_single.mean()), batch["nll"], batch["nll"] - float(per_single.mean())))

    # what one batched pass holds that a serial pass does not: a layer's experts as f32, the batch's logits
    p = model.pack
    expert_f32 = int(p.n_expert) * (2 * int(p.n_ff_exp) * int(p.n_embd) + int(p.n_embd) * int(p.n_ff_exp)) * 4
    logits = int(n_tok) * int(p.tensor("output.weight").shape[0]) * 4 if not model.truncated else 0
    log("")
    log("  forwards: serial %.1f s for %d documents, batched %.1f s for the same %d tokens -> %.2fx"
        % (t_single, len(docs), t_batch, n_tok, t_single / max(t_batch, 1e-9)))
    per_doc_s = t_single / len(docs)
    log("  a document costs %.1f s at this depth (%.0f s at full depth), so the serial rate is %.0f s a document "
        "whatever its token count" % (per_doc_s, per_doc_s * full / model.n_layers,
                                      per_doc_s * full / model.n_layers))
    scale = full / float(model.n_layers)
    at_depth = t_batch * scale
    log("  the batched pass scales to %.1f s at full depth for %d tokens; at that rate 1,668 tokens (the 96-expert "
        "requirement) is %.1f min and 918 (112) is %.1f min"
        % (at_depth, n_tok, at_depth * 1668.0 / n_tok / 60.0, at_depth * 918.0 / n_tok / 60.0))
    log("  memory the batch adds: %.2f GB of f32 experts for one layer, %.2f GB of batch logits"
        % (expert_f32 / 1e9, logits / 1e9))
    return dict(docs=[(n, len(i)) for n, i in docs], tokens=n_tok, layers=model.n_layers, depth=full,
                t_serial=t_single, t_batch=t_batch, speedup=t_single / max(t_batch, 1e-9),
                max_dnll=float(d.max()), mean_dnll=float(d.mean()),
                nll_serial=float(per_single.mean()), nll_batch=float(batch["nll"]),
                expert_f32_bytes=expert_f32, logit_bytes=logits)


def main(argv):
    if "--selftest" in argv:
        checks = []

        def ok(name, cond, detail=""):
            if not cond:
                raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))
            checks.append(name)

        # --- the one part of batching that is a property and not a measurement: documents stay apart
        rng = np.random.RandomState(5)
        got = _fake_batch(rng, docs=3, tokens=7)
        ok("docbatch: a batch returns each document's own rows, in the order given",
           got["spans"] == [(0, 7), (7, 14), (14, 21)], str(got["spans"]))
        ok("docbatch: per-document attention over a batch is per-document attention run alone",
           np.allclose(got["alone"], got["batched"], atol=1e-6),
           "max |difference| %.2e" % float(np.abs(got["alone"] - got["batched"]).max()))
        ok("docbatch: reversing the documents changes nothing a document sees",
           np.allclose(got["reversed"], got["batched"], atol=1e-6),
           "max |difference| %.2e" % float(np.abs(got["reversed"] - got["batched"]).max()) if
           got["reversed"].shape == got["batched"].shape else "shape %s" % (got["reversed"].shape,))

        # --- the per-token nll of a batch is the per-token nll of its documents, concatenated in order
        fake = _FakeMoE(rng, docs=3, tokens=6, vocab=11)
        r = nll_moe_batch(fake, fake.docs)
        one_by_one = np.concatenate([_per_token(L, ids) for L, (_, ids) in zip(fake.logits, fake.docs)])
        ok("docbatch: a batched nll keeps the documents in the order it was given them",
           abs(r["nll"] - float(one_by_one.mean())) < 1e-12 and r["tokens"] == one_by_one.size)
        ok("docbatch: a document's own nll in a batch is its nll alone",
           np.allclose(r["per_doc"], [_per_token(L, ids).mean() for L, (_, ids) in zip(fake.logits, fake.docs)]))
        ok("docbatch: one token a document contributes no nll, and the batch does not invent any",
           nll_moe_batch(fake, [(n, i[:1]) for n, i in fake.docs])["tokens"] == 0)
        print("%d checks passed" % len(checks))
        return 0

    pack = _arg(argv, "--pack")
    if not pack:
        print(__doc__.strip().split("\n\n")[-1].strip())
        return 2
    measure(pack, layers=_arg(argv, "--layers", None, int), ndocs=_arg(argv, "--docs", 3, int),
            tokens=_arg(argv, "--tokens", 96, int))
    return 0


def _fake_batch(rng, docs, tokens):
    """The batching contract, without an 11 GB pack: `moe.per_doc` over a toy attention.

    The claim that matters here is not a number, it is a property - tokens of document *k* see only document *k* -
    so the test runs the segmented path `forward_docs` uses (`spans_of` + `per_doc`) against the same layer run on
    each document alone. A batched forward that leaked across a boundary would fail this and pass every timing.
    """
    from .moe import per_doc, softmax, spans_of
    H = 2

    def one(x):
        # the shape of computation a layer does: causal attention to itself
        sc = np.einsum("thd,shd->hts", x, x) / np.sqrt(x.shape[-1])
        mask = np.triu(np.ones((x.shape[0], x.shape[0]), dtype=bool), 1)
        return x + np.einsum("hts,shd->thd", softmax(np.where(mask[None], -np.inf, sc)), x)

    xs = [rng.randn(tokens, H, 4).astype(np.float32) for _ in range(docs)]
    alone = np.concatenate([one(x) for x in xs])
    cat = np.concatenate(xs)
    spans = spans_of(xs)
    batched = per_doc(cat.copy(), spans, one)
    order = list(reversed(range(docs)))                       # same documents, another order
    rev = per_doc(np.concatenate([xs[i] for i in order]), spans_of([xs[i] for i in order]), one)
    back = np.concatenate([rev[k * tokens:(k + 1) * tokens] for k in order])   # back in original order
    return dict(spans=spans, alone=alone, batched=batched, reversed=back)


class _FakeMoE(object):
    """A model that only has what `nll_moe_batch` reads: documents in, logits out, one row a token."""

    def __init__(self, rng, docs, tokens, vocab):
        self.docs = [("doc%d" % k, rng.randint(0, vocab, size=tokens)) for k in range(docs)]
        self.logits = [rng.randn(len(ids), vocab).astype(np.float32) for _, ids in self.docs]

    def forward_docs(self, seqs, log=None):
        return dict(per_doc=[dict(logits=L) for L in self.logits])


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
