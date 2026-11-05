"""Merging co-activated expert groups into a real pack, and measuring what the merge costs in NLL.

`cluster_budget.py` groups experts by co-activation to ask how many bytes a paging schedule would fetch. This uses
the same grouping to ask the other question: what happens to the model when each group is *replaced* by one averaged
expert. That answer needs a pack, because NLL has to be computed by a model whose experts really are shared - a mask
that maps two routing slots onto one matrix would leave the router and the renormalisation untouched, which is not
the model a merged pack is.

So this writes a GGUF: every non-expert tensor is copied byte for byte, the three expert stacks are replaced by
`n_groups` merged matrices, the router loses the rows of the experts that were merged away, and `expert_count` says
so. The result is a normal MoE pack that `tools.expert_merge.ggufmoe` reads and `tools.expert_merge.moe` runs, which
is what makes the before/after comparison two models and not two settings.

    python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 64 --out MERGED.gguf
    python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 64 --out M.gguf \
        --gate --corpus docs/BENCHMARKS.md,kindi/tp.py --docs 2 --tokens 48 --json merge.json
    python -m tools.expert_merge.mergepack --selftest

Three decisions the code makes, and why:

**The grouping is one partition for the whole model**, from the layer-averaged co-activation matrix, by average
linkage with a size cap `ceil(experts / groups)`. A per-layer partition would be a different model per layer and
would make "the merge" 48 separate experiments; a single partition is what a deployable pack can be.

**The average is weighted by routing mass, not by a Fisher.** `merge.py` implements the specification's
Fisher-weighted, permutation-aligned fusion, and `docs/EXPERT-MERGE.md` records that its own alignment gate rejects
every real pair (mean aligned cosine 0.081 against a 0.85 gate): the hidden units of two co-activated experts are not
one function relabelled, so permuting them to an anchor aligns noise. The arithmetic here is therefore the plain
weighted mean - the case the specification falls back to - with the weight being how much of the layer's routing
weight each member actually carried in the log. A member nothing routes to contributes nothing; every member
contributes in proportion to the mass it was applied with.

**The router row of a group is the same weighted mean of its members' rows.** That is the choice a merged pack forces:
the new expert has one row, so the logits of the experts it replaces have to be pooled. It changes routing as well as
arithmetic, and the gate sees both together - the note reports that rather than pretending the router is untouched.
"""
import json
import math
import os
import sys

import numpy as np

from tools.bitnet_gemm.cluster_budget import coactivation, merged_mass

from .ggufmoe import Moepack
from .prune import mass_per_expert

EXPERT_STACKS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
ROUTER = "ffn_gate_inp.weight"


# --- grouping -------------------------------------------------------------------------------------------------

def groups(coc, n_groups, log=None):
    """Average-linkage agglomeration on the co-activation matrix, size-capped, deterministic.

    Merging the two groups with the highest mean co-activation between their members, and never letting a group grow
    past `ceil(experts / groups)`, gives a partition whose groups are as co-activated as a greedy rule can make them
    and whose sizes cannot run away (one huge group and 63 singletons would be a legal k-means answer and a useless
    merge). Ties break on the lower member id, so the answer does not depend on dictionary order.
    """
    coc = np.asarray(coc, dtype=np.float64)
    n = coc.shape[0]
    n_groups = int(n_groups)
    if not 1 <= n_groups <= n:
        raise SystemExit("mergepack: %d groups from %d experts is not a partition" % (n_groups, n))
    cap = int(math.ceil(n / float(n_groups)))
    members = [[i] for i in range(n)]
    while len(members) > n_groups:
        best = None
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if len(members[i]) + len(members[j]) > cap:
                    continue
                s = float(coc[np.ix_(members[i], members[j])].mean())
                if best is None or s > best[0] + 1e-12:
                    best = (s, i, j)
        if best is None:                                  # the cap cannot be met; the only way out is to raise it
            cap += 1
            continue
        _s, i, j = best
        members[i] = sorted(members[i] + members[j])
        members.pop(j)
    order = sorted(range(len(members)), key=lambda k: (-len(members[k]), members[k][0]))
    gids = np.zeros(n, dtype=np.int64)
    for new, k in enumerate(order):
        for e in members[k]:
            gids[e] = new
    if log:
        sizes = np.bincount(gids, minlength=n_groups)
        log("  %d experts -> %d groups (sizes %d-%d, cap %d)" % (n, n_groups, sizes.min(), sizes.max(), cap))
    return gids


def merge_matrices(mats, weights):
    """The weighted mean of a group's matrices, with the weights normalised inside the group.

    `mats` are [out, in] arrays (or [in] vectors for a router row); `weights` the routing mass each member carried.
    A group whose members carried no mass at all - possible for experts the log never routed to - falls back to a
    plain mean, because the alternative is dividing by zero and dropping the expert entirely.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    w = (w / s) if s > 0 else np.full(w.size, 1.0 / w.size)
    out = np.zeros_like(mats[0], dtype=np.float64)
    for wi, m in zip(w, mats):
        out += wi * np.asarray(m, dtype=np.float64)
    return out.astype(np.float32)


# --- writing the pack ------------------------------------------------------------------------------------------

def _value_type(v):
    from gguf import GGUFValueType

    if isinstance(v, bool):
        return GGUFValueType.BOOL, None
    if isinstance(v, str):
        return GGUFValueType.STRING, None
    if isinstance(v, float):
        return GGUFValueType.FLOAT32, None
    if isinstance(v, int):
        return (GGUFValueType.UINT32, None) if v >= 0 else (GGUFValueType.INT32, None)
    if isinstance(v, (list, tuple)):
        vals = list(v)
        if vals and all(isinstance(x, str) for x in vals):
            return GGUFValueType.ARRAY, GGUFValueType.STRING
        if vals and all(isinstance(x, bool) for x in vals):
            return GGUFValueType.ARRAY, GGUFValueType.BOOL
        if vals and all(isinstance(x, float) for x in vals):
            return GGUFValueType.ARRAY, GGUFValueType.FLOAT32
        neg = any(isinstance(x, int) and x < 0 for x in vals)
        return GGUFValueType.ARRAY, (GGUFValueType.INT32 if neg else GGUFValueType.UINT32)
    raise SystemExit("mergepack: cannot write a metadata value of type %s" % type(v).__name__)


def writable(ty):
    """Can this build of `gguf` actually *write* that type?

    `GGML_QUANT_SIZES` and the decode path cover every type a real pack contains; `quants.quantize` does not - this
    build raises `NotImplementedError` for every K-quant. A pack that must be re-quantised therefore cannot be written
    in the type its source used, which is why the merge runs at one working type for both arms (`--expert-type`)
    instead of at the source's: a merged arm written more precisely than its control would look better than it is.
    """
    from gguf import GGMLQuantizationType as T
    from gguf import quants as Q

    t = T(int(ty))
    if t in (T.F32, T.F16, T.BF16):
        return True
    try:
        Q.quantize(np.zeros((2, 256), dtype=np.float32), t)
    except NotImplementedError:
        return False
    return True


def _add(writer, name, ne, ty, flat):
    """One tensor, quantised to the type it had in the source.

    `ne` is GGUF's own order (fastest-first) and `flat` is the data in that order - which is the same bytes as the
    NumPy array's C order for the reversed shape. `writer.add_tensor` takes the *NumPy* shape and reverses it when
    it writes the header, so the shape handed over is `reversed(ne)`; passing `ne` here transposes every tensor, and a
    transposed weight still loads and still predicts something (which is why the selftest reads the file back).
    """
    from gguf import GGMLQuantizationType as T
    from gguf import quants as GQ

    t = T(int(ty))
    if t == T.F32:
        arr = np.ascontiguousarray(flat, dtype=np.float32).ravel()
    elif t == T.F16:
        arr = np.ascontiguousarray(flat, dtype=np.float16).ravel()
    else:
        cols = int(ne[0])
        q = GQ.quantize(np.ascontiguousarray(flat, dtype=np.float32).reshape(-1, cols), t)
        arr = np.ascontiguousarray(q).view(np.uint8).ravel()
    writer.add_tensor(name, arr, raw_shape=tuple(int(d) for d in reversed(ne)), raw_dtype=t)


def write_pack(src, dst, gids, mass, expert_type=None, used=None, keep_router=False, log=print, progress=None):
    """Write `dst` as `src` with each group of experts replaced by one mass-weighted average.

    Non-expert tensors are copied as bytes, not re-quantised: a copy that dequantised and re-quantised the attention
    would put the source's own quantisation error into the comparison. Everything that changes is in the expert
    stacks, the router, and `expert_count`.

    `keep_router` splits the merge into its two levers. The default pools each group's router rows into one, which
    makes a pack that is a normal MoE of `n_groups` experts - and changes routing as well as arithmetic. With it, the
    router is the source's own (128 rows, same bytes, same quantisation), the stacks stay merged, and the pack records
    the group map so a reader can route over 128 rows and pool the result onto 64 stacks. The difference between the
    two packs is then the router and nothing else, which is what `docs/EXPERT-MERGE.md` needed to attribute the merge
    cost instead of quoting one number for two changes.
    """
    from gguf import GGUFWriter

    pack = Moepack(src)
    g = pack.g
    gids = np.asarray(gids, dtype=np.int64)
    n_groups = int(gids.max()) + 1
    n_expert = pack.n_expert
    if gids.size != n_expert:
        raise SystemExit("mergepack: %d group ids for %d experts" % (gids.size, n_expert))
    members = [np.nonzero(gids == k)[0] for k in range(n_groups)]

    et = int(expert_type) if expert_type is not None else None
    if et is not None and not writable(et):
        raise SystemExit("mergepack: this gguf cannot write type %d; pick one it can (e.g. Q8_0, Q5_0, Q4_0, F16)" % et)
    kv = dict(g.kv)
    kv["%s.expert_count" % pack.arch] = n_groups
    if keep_router:
        # the router keeps the source's rows, so the file states two counts: `expert_count` is what the stacks hold,
        # and these say how many routing slots there are and which stack each one feeds. It is not a pack llama.cpp
        # can serve - it is a measurement arm - and `moe.py` refuses it unless it reads all three keys.
        kv["kindi.router_rows"] = int(n_expert)
        kv["kindi.router_used"] = int(pack.n_expert_used)
        kv["kindi.expert_map"] = [int(x) for x in gids]
    if used is not None:
        # how many experts a token reads is a routing decision, so a pack can carry fewer groups than the source's k
        # "same fraction" arm; it is written here rather than patched afterwards when the merge is being built anyway
        if not 1 <= int(used) <= n_groups:
            raise SystemExit("mergepack: --used %d of %d groups is not a routing breadth this pack can have"
                             % (int(used), n_groups))
        kv["%s.expert_used_count" % pack.arch] = int(used)
    writer = GGUFWriter(dst, pack.arch, use_temp_file=True)   # this writes general.architecture itself
    for k in sorted(kv):
        if k == "general.architecture":
            continue
        vtype, sub = _value_type(kv[k])
        writer.add_key_value(k, kv[k], vtype, sub)

    copied = 0
    for name, (ne, ty, off) in sorted(pack.info.items(), key=lambda kv2: kv2[1][2]):
        layer = None
        if name.startswith("blk."):
            layer = int(name.split(".")[1])
        short = ".".join(name.split(".")[2:]) if layer is not None else name
        if short in EXPERT_STACKS and layer is not None:
            n_in, n_out, n_exp = ne
            if n_exp != n_expert:
                raise SystemExit("mergepack: %s stacks %d experts, the metadata says %d" % (name, n_exp, n_expert))
            merged = np.zeros((n_groups, n_out, n_in), dtype=np.float32)
            m = mass[layer] if layer < mass.shape[0] else np.zeros(n_expert)
            for k, mem in enumerate(members):
                mats = [pack.expert_weight(name, int(e)) for e in mem]
                merged[k] = merge_matrices(mats, m[mem])
            _add(writer, name, (n_in, n_out, n_groups), et if et is not None else ty, merged.ravel())
        elif short == ROUTER and layer is not None and keep_router:
            # byte for byte: the rows are the source's decisions, and re-quantising them would be a second change
            span = pack._span(ty, int(np.prod(ne)))
            start = pack.g.data_start + off
            raw = np.ascontiguousarray(np.asarray(pack._raw[start:start + span]))
            writer.add_tensor(name, raw, raw_shape=tuple(int(d) for d in reversed(ne)), raw_dtype=from_type(ty))
        elif short == ROUTER and layer is not None:
            n_in, n_exp = ne
            rows = [pack.tensor(name)[int(e)] for e in range(n_exp)]       # [d_model] each
            merged = np.stack([merge_matrices([rows[e] for e in mem], mass[layer][mem]) for mem in members])
            _add(writer, name, (n_in, n_groups), et if et is not None else ty, merged.ravel())
        else:
            span = pack._span(ty, int(np.prod(ne)))
            start = g.data_start + off
            raw = np.ascontiguousarray(np.asarray(pack._raw[start:start + span]))
            writer.add_tensor(name, raw, raw_shape=tuple(int(d) for d in reversed(ne)), raw_dtype=from_type(ty))
            copied += int(span)
        if progress and layer is not None and short == EXPERT_STACKS[-1]:
            progress(layer)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dict(path=dst, groups=n_groups, bytes=int(os.path.getsize(dst)),
                router_rows=int(pack.n_expert) if keep_router else n_groups,
                source_bytes=int(os.path.getsize(src)), copied_bytes=int(copied))


def splice_router(merged, router_from, dst, gids, mass=None, log=print):
    """The merged pack with the source's router spliced in: every tensor copied as bytes, only the router replaced.

    This is the other half of the merge, and it is deliberately not a second merge. Re-merging would dequantise and
    re-quantise 9.77 GB of expert stacks that the first merge already wrote, and a stack that differs from its
    counterpart by one quantisation step makes the two arms differ in more than the thing being measured. Copying the
    bytes makes the stacks bit-identical, so the only difference between this pack and the matched-fraction arm is
    which router picks the stacks a token reads.

    `gids` is the map the merge used. It is checked against the pack it is applied to rather than trusted: a pool of
    each group's router rows is what the merged pack holds, so if the map explains that pack the pooled row and the
    group's mean agree to the pack's own quantisation error, and if it does not, this refuses instead of writing a
    pack whose map is for a different merge.
    """
    from gguf import GGUFWriter

    m, s = Moepack(merged), Moepack(router_from)
    gids = np.asarray(gids, dtype=np.int64)
    if gids.size != s.n_expert:
        raise SystemExit("mergepack: %d map entries for a %d-expert source" % (gids.size, s.n_expert))
    if m.n_expert != int(gids.max()) + 1:
        raise SystemExit("mergepack: %s has %d experts but the map names %d groups"
                         % (os.path.basename(merged), m.n_expert, int(gids.max()) + 1))
    if m.n_layer != s.n_layer:
        raise SystemExit("mergepack: %s and %s do not have the same layer count" % (merged, router_from))

    name0 = "blk.0." + ROUTER
    have = m.tensor(name0)                                  # [groups, d_model]
    rows = s.tensor(name0)                                  # [experts, d_model]
    w = np.ones(s.n_expert, np.float32) if mass is None else np.asarray(mass[0], np.float32)

    def worst_map(g):
        """the worst group's cosine between the pooled row the pack holds and the pool this map predicts"""
        out = []
        for k in range(m.n_expert):
            mem = np.nonzero(g == k)[0]
            if not mem.size:
                return -1.0
            ww = w[mem]
            want = (rows[mem] * ww[:, None]).sum(0) / max(float(ww.sum()), 1e-9)
            a, b = have[k], want
            out.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)))
        return float(min(out))

    # A pooled row is compared with the pool the map predicts, so the number is limited by the pack's own
    # quantisation: a Q4_0 router lands near 0.99 and a wrong map near 0.55. The check is therefore a comparison
    # with a shuffled map rather than an absolute threshold, which is the only version that cannot be passed by a
    # pack whose router says nothing.
    worst = worst_map(gids)
    shuffled = worst_map(np.random.RandomState(0).permutation(gids))
    if worst < 0.95 or worst - shuffled < 0.2:
        raise SystemExit("mergepack: the map does not explain %s's router (worst group cosine %.3f, a shuffled map "
                         "scores %.3f on layer 0): this routes file is not the one the pack was merged with"
                         % (os.path.basename(merged), worst, shuffled))
    log("  the map explains %s's router: worst group cosine %.4f on layer 0 (a shuffled map scores %.4f)"
        % (os.path.basename(merged), worst, shuffled))

    kv = dict(m.g.kv)
    kv["kindi.router_rows"] = int(s.n_expert)
    kv["kindi.router_used"] = int(s.n_expert_used)
    kv["kindi.expert_map"] = [int(x) for x in gids]
    writer = GGUFWriter(dst, m.arch, use_temp_file=True)
    for k in sorted(kv):
        if k == "general.architecture":
            continue
        vtype, sub = _value_type(kv[k])
        writer.add_key_value(k, kv[k], vtype, sub)
    for name, (ne, ty, off) in sorted(m.info.items(), key=lambda kv2: kv2[1][2]):
        short = ".".join(name.split(".")[2:]) if name.startswith("blk.") else name
        if short == ROUTER:
            sne, sty, soff = s.info[name]
            span = s._span(sty, int(np.prod(sne)))
            start = s.g.data_start + soff
            raw = np.ascontiguousarray(np.asarray(s._raw[start:start + span]))
            writer.add_tensor(name, raw, raw_shape=tuple(int(d) for d in reversed(sne)), raw_dtype=from_type(sty))
        else:
            span = m._span(ty, int(np.prod(ne)))
            start = m.g.data_start + off
            raw = np.ascontiguousarray(np.asarray(m._raw[start:start + span]))
            writer.add_tensor(name, raw, raw_shape=tuple(int(d) for d in reversed(ne)), raw_dtype=from_type(ty))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dict(path=dst, bytes=int(os.path.getsize(dst)), merged_bytes=int(os.path.getsize(merged)),
                groups=m.n_expert, router_rows=int(s.n_expert), router_used=int(s.n_expert_used),
                map_cosine=worst, shuffle_cosine=shuffled)


def from_type(ty):
    from gguf import GGMLQuantizationType as T

    return T(int(ty))


# --- the comparison --------------------------------------------------------------------------------------------

def compare_arms(base_pack, arms, docs, max_delta=0.02, block=8, log=print):
    """One baseline walk, one walk a pack, one verdict each.

    The baseline is loaded and scored once, which is what makes several arms affordable: the Q4_0 control is 17 GB and
    each merged arm 8.9 GB, while the scored tokens are the cheap part. Every walk sees the same tokenisation and the
    same positions, so an arm's difference from the baseline is comparable with another arm's - and each arm is also
    scored against every other arm, because "the merge costs 1.4 nats" is two changes added together and the arms
    exist to separate them.

    The corpus goes through one batched forward a pack (`forward_docs`): the dequantiser is the cost of a document, so
    a document at a time would make a corpus of seven unaffordable for arms this large.
    """
    from .moe import Qwen3Moe
    from .prune import gate, nll_moe_batch

    b = Qwen3Moe(base_pack, log=lambda *a: None)
    rb = nll_moe_batch(b, docs)
    baseline = rb["per_token"]
    b_docs = [(n, v, max(1, len(ids) - 1)) for (n, ids), v in zip(docs, rb["per_doc"])]
    for name, nll, ntok in b_docs:
        log("    baseline %-24s %8d tokens  nll %.4f" % (name, ntok, nll))
    del b

    out, per_token = [], {}
    for path in arms:
        arm = Qwen3Moe(path, log=lambda *a: None)
        ra = nll_moe_batch(arm, docs)
        del arm
        after = ra["per_token"]
        per_token[path] = after
        a_docs = [(n, v, max(1, len(ids) - 1)) for (n, ids), v in zip(docs, ra["per_doc"])]
        for name, nll, ntok in a_docs:
            log("    %-20s %-24s %8d tokens  nll %.4f" % (os.path.basename(path)[:20], name, ntok, nll))
        v = gate(dict(per_token=baseline), [dict(per_token=after)], max_delta=max_delta, block=block)[0]
        out.append(dict(pack=path, baseline_nll=float(baseline.mean()), merged_nll=float(after.mean()),
                        tokens=int(baseline.size), baseline_docs=b_docs, merged_docs=a_docs, gate=v, vs={}))

    # each arm against each other arm: the same paired tokens, so this is the difference between two models and not
    # between two comparisons against the control
    for i, a in enumerate(out):
        for b2 in out[i + 1:]:
            v = gate(dict(per_token=per_token[a["pack"]]), [dict(per_token=per_token[b2["pack"]])],
                     max_delta=max_delta, block=block)[0]
            a["vs"][b2["pack"]] = v
            b2["vs"][a["pack"]] = dict(v, delta=-v["delta"], ci_lo=-v["ci_hi"], ci_hi=-v["ci_lo"],
                                       passed=-v["delta"] <= max_delta, marginal=False)
    return out


def compare(src, merged, docs, max_delta=0.02, block=8, log=print):
    """NLL of both packs on the same tokens, paired token by token, with a moving-block interval.

    One process, two models, the same `forward`: the source is the baseline and the merged pack is the arm, and the
    difference is indexed by position so a document cannot contribute to one side only.
    """
    return compare_arms(src, [merged], docs, max_delta=max_delta, block=block, log=log)[0]


# --- routing breadth --------------------------------------------------------------------------------------------

_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _kv_uint32_offset(path, key):
    """The byte offset of a uint32 metadata value, read from the header alone - so it can be rewritten in place.

    A GGUF key/value is a length-prefixed key, a type, then the value; only a scalar value can be rewritten without
    shifting everything after it, which is why this insists on type 4 and refuses a string or an array.
    """
    import struct

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("mergepack: %s is not a GGUF file" % path)
        ver = struct.unpack("<I", f.read(4))[0]
        if ver not in (2, 3):
            raise SystemExit("mergepack: %s is GGUF v%d, and this reader knows v2 and v3" % (path, ver))
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            n = struct.unpack("<Q", f.read(8))[0]
            k = f.read(n).decode("utf-8", errors="replace")
            ty = struct.unpack("<I", f.read(4))[0]
            pos = f.tell()
            if k == key:
                if ty != 4:
                    raise SystemExit("mergepack: %s holds %r as metadata type %d, not a uint32" % (path, key, ty))
                return pos
            _skip_value(f, ty, k, path)
    raise SystemExit("mergepack: %s has no %r in its header, so there is no routing breadth to rewrite" % (path, key))


def _skip_value(f, ty, key, path):
    import struct

    if ty == 8:                                              # string
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if ty == 9:                                              # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
            return
        size = _SCALAR_BYTES.get(et)
        if size is None:
            raise SystemExit("mergepack: %s: array %r has element type %d, which this reader cannot skip"
                             % (path, key, et))
        f.seek(size * n, 1)
        return
    size = _SCALAR_BYTES.get(ty)
    if size is None:
        raise SystemExit("mergepack: %s: %r has metadata type %d, which this reader cannot skip" % (path, key, ty))
    f.seek(size, 1)


def retopk(src, dst, used, log=print):
    """The same pack at a different routing breadth: every byte copied, `expert_used_count` rewritten.

    k is a header field, not a weight. Changing it changes how many of the (already merged) experts a token reads and
    renormalises over, and nothing else in the file moves - so a paired walk of the copy against its source prices
    routing breadth with the merge held fixed. That is the arm the k/N asymmetry needs: 8 of 128 experts and 4 of 64
    are the same fraction of the pool.
    """
    import shutil
    import struct

    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("mergepack: --retopk writes a new pack; it does not patch its source in place")
    pack = Moepack(src)
    n, used = pack.n_expert, int(used)
    if not 1 <= used <= n:
        raise SystemExit("mergepack: %d of %d experts is not a routing breadth this pack can have" % (used, n))
    key = "%s.expert_used_count" % pack.arch
    off = _kv_uint32_offset(src, key)
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(off)
        f.write(struct.pack("<I", used))
    out = Moepack(dst)
    if (out.n_expert, out.n_expert_used) != (n, used):
        raise SystemExit("mergepack: %s reads back as %d of %d experts, not %d of %d"
                         % (dst, out.n_expert_used, out.n_expert, used, n))
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit("mergepack: %s is %d bytes where its source is %d, so a copy went wrong"
                         % (dst, os.path.getsize(dst), os.path.getsize(src)))
    if not out.payload_report()["ok"]:
        raise SystemExit("mergepack: %s does not decode after the rewrite" % dst)
    log("  %s: %d experts, %d used -> %d used; %d bytes, every tensor still decodes"
        % (dst, n, pack.n_expert_used, used, os.path.getsize(dst)))
    return dict(path=dst, source=src, experts=n, used_before=pack.n_expert_used, used_after=used,
                bytes=int(os.path.getsize(dst)), offset=int(off))


# --- the checks ------------------------------------------------------------------------------------------------

def _synthetic_pack(path, n_expert=4, d=32, ff=16, head=2, hkv=1, hd=16, vocab=64, seed=0):
    """A one-layer qwen3moe GGUF small enough to check byte for byte, written with the same writer the merge uses."""
    from gguf import GGUFWriter

    rs = np.random.RandomState(seed)
    kv = {"general.architecture": "qwen3moe", "general.name": "mergepack-selftest",
          "qwen3moe.block_count": 1, "qwen3moe.embedding_length": d, "qwen3moe.feed_forward_length": 2 * ff,
          "qwen3moe.attention.head_count": head, "qwen3moe.attention.head_count_kv": hkv,
          "qwen3moe.attention.key_length": hd, "qwen3moe.attention.value_length": hd,
          "qwen3moe.expert_count": n_expert, "qwen3moe.expert_used_count": min(2, n_expert),
          "qwen3moe.expert_feed_forward_length": ff, "qwen3moe.rope.freq_base": 10000.0,
          "qwen3moe.attention.layer_norm_rms_epsilon": 1e-6, "tokenizer.ggml.model": "gpt2"}
    w = GGUFWriter(path, "qwen3moe", use_temp_file=True)
    for k in sorted(kv):
        if k == "general.architecture":
            continue
        vtype, sub = _value_type(kv[k])
        w.add_key_value(k, kv[k], vtype, sub)
    n = lambda *shape: (rs.randn(*shape) * 0.05).astype(np.float32)
    tensors = [("token_embd.weight", n(vocab, d), (d, vocab)),
               ("output_norm.weight", (np.ones(d) * 0.5).astype(np.float32), (d,)),
               ("output.weight", n(vocab, d), (d, vocab))]
    s = "blk.0."
    tensors += [(s + "attn_q.weight", n(head * hd, d), (d, head * hd)),
                (s + "attn_k.weight", n(hkv * hd, d), (d, hkv * hd)),
                (s + "attn_v.weight", n(hkv * hd, d), (d, hkv * hd)),
                (s + "attn_output.weight", n(d, head * hd), (head * hd, d)),
                (s + "attn_q_norm.weight", (np.ones(hd) * 0.4).astype(np.float32), (hd,)),
                (s + "attn_k_norm.weight", (np.ones(hd) * 0.4).astype(np.float32), (hd,)),
                (s + "attn_norm.weight", (np.ones(d) * 0.6).astype(np.float32), (d,)),
                (s + "ffn_norm.weight", (np.ones(d) * 0.6).astype(np.float32), (d,)),
                (s + "ffn_gate_inp.weight", n(n_expert, d), (d, n_expert)),
                (s + "ffn_gate_exps.weight", n(n_expert, ff, d), (d, ff, n_expert)),
                (s + "ffn_up_exps.weight", n(n_expert, ff, d), (d, ff, n_expert)),
                (s + "ffn_down_exps.weight", n(n_expert, d, ff), (ff, d, n_expert))]
    for name, arr, ne in tensors:
        w.add_tensor(name, np.ascontiguousarray(arr).ravel(), raw_shape=tuple(reversed(ne)),
                     raw_dtype=from_type(0))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def selftest():
    import shutil
    import tempfile

    done = []

    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))
        done.append(name)

    def refuses(name, fn, exc=SystemExit):
        """a check that is a refusal: the call must raise, and raising is the claim"""
        try:
            fn()
        except exc:
            done.append(name)
        else:
            raise AssertionError(name)

    # --- the grouping
    coc = np.zeros((4, 4))
    for a, b in ((0, 1), (2, 3)):
        coc[a, b] = coc[b, a] = 9.0
        coc[a, a] = coc[b, b] = 9.0
    coc += np.eye(4)
    g = groups(coc, 2, log=None)
    check("mergepack: average linkage finds the two co-activated pairs",
          sorted(np.nonzero(g == g[0])[0].tolist()) == [0, 1] or sorted(np.nonzero(g == g[0])[0].tolist()) == [1, 0],
          str(g))
    check("mergepack: and they are the ones that co-fire", {tuple(sorted(np.nonzero(g == k)[0])) for k in set(g.tolist())}
          == {(0, 1), (2, 3)}, str(g))
    check("mergepack: every group is used and every expert is in one",
          len(set(g.tolist())) == 2 and np.bincount(g).sum() == 4, str(g))
    sprawling = groups(coc, 3, log=None)
    sizes = np.bincount(sprawling)
    check("mergepack: a size cap keeps the groups even at a group count that does not divide", sizes.max() <= 2,
          str(sizes))
    refuses("mergepack: more groups than experts is refused", lambda: groups(coc, 5, log=None))

    # --- the fusion arithmetic
    a, b = np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32)
    check("mergepack: one member is itself", np.allclose(merge_matrices([a], [5.0]), a))
    check("mergepack: zero weights fall back to a plain mean", np.allclose(merge_matrices([a, b], [0.0, 0.0]), 0.5))
    check("mergepack: mass weights decide the mean", np.allclose(merge_matrices([a, b], [3.0, 1.0]), 0.25))
    check("mergepack: the result is f32, not f64", merge_matrices([a, b], [1.0, 1.0]).dtype == np.float32)

    # --- which types this build can actually write, since that decides what a matched pair of arms can be
    from gguf import GGMLQuantizationType as T

    check("mergepack: the types a matched pair of arms can use are writable",
          writable(int(T.Q4_0)) and writable(int(T.F16)) and writable(int(T.F32)))
    check("mergepack: and the writer's answer is the quantiser's, not a guess",
          writable(int(T.Q8_0)) == True, "Q8_0 must round-trip through quantize/dequantize")

    # --- a real file through the real writer, read back by the real reader
    td = tempfile.mkdtemp(prefix="mergepack-")
    try:
        src = _synthetic_pack(os.path.join(td, "src.gguf"))
        mass = np.array([[4.0, 2.0, 1.0, 1.0]])            # one layer, four experts
        gids = np.array([0, 0, 1, 1])
        dst = os.path.join(td, "merged.gguf")
        rep = write_pack(src, dst, gids, mass, log=lambda *a: None)
        check("mergepack: the written pack has one expert a group", rep["groups"] == 2, str(rep))
        m = Moepack(dst)
        check("mergepack: and its metadata says so", m.n_expert == 2, str(m.n_expert))
        check("mergepack: the rest of the metadata survived", m.n_embd == 32 and m.n_ff_exp == 16 and m.n_layer == 1)
        check("mergepack: every tensor decodes at its new shape",
              m.payload_report()["ok"] and m.expert_weight("blk.0.ffn_gate_exps.weight", 1).shape == (16, 32))
        src_pack = Moepack(src)
        w0 = src_pack.expert_weight("blk.0.ffn_gate_exps.weight", 0)
        w1 = src_pack.expert_weight("blk.0.ffn_gate_exps.weight", 1)
        got = m.expert_weight("blk.0.ffn_gate_exps.weight", 0)
        want = (4.0 * w0 + 2.0 * w1) / 6.0
        check("mergepack: group 0 is the mass-weighted mean of its members", np.allclose(got, want, atol=1e-6),
              "max diff %.3g" % float(np.abs(got - want).max()))
        other = m.expert_weight("blk.0.ffn_gate_exps.weight", 1)
        check("mergepack: and group 1 is a different matrix", not np.allclose(got, other))
        check("mergepack: the file is smaller than its source", rep["bytes"] < rep["source_bytes"])

        # the merged pack is a model: it loads and its forward produces finite logits
        from .moe import Qwen3Moe

        ids = [3, 5, 7, 11]
        before = Qwen3Moe(src, log=lambda *a: None).forward(ids)["logits"]
        after = Qwen3Moe(dst, log=lambda *a: None).forward(ids)["logits"]
        check("mergepack: the source forward is finite", np.isfinite(before).all())
        check("mergepack: the merged forward is finite too", np.isfinite(after).all())
        check("mergepack: and it is not the same model any more", not np.allclose(before, after, atol=1e-4))

        # a kept-router pack: the source's rows, the merged stacks, and a map between them
        kept = os.path.join(td, "merged-router.gguf")
        kr = write_pack(src, kept, gids, mass, used=1, keep_router=True, log=lambda *a: None)
        mk = Moepack(kept)
        src_r = Moepack(src).tensor("blk.0.ffn_gate_inp.weight")
        got_r = mk.tensor("blk.0.ffn_gate_inp.weight")
        check("mergepack: --keep-router keeps the source's router rows, not a pooled version",
              got_r.shape == src_r.shape and np.array_equal(got_r, src_r), "%s vs %s" % (got_r.shape, src_r.shape))
        check("mergepack: the router it scores has more slots than the stacks it owns",
              mk.router_rows == 4 and mk.n_expert == 2 and kr["router_rows"] == 4,
              "rows %d stacks %d" % (mk.router_rows, mk.n_expert))
        check("mergepack: the map says which stack each slot feeds",
              mk.router_map is not None and mk.router_map.tolist() == [0, 0, 1, 1], str(mk.router_map))
        check("mergepack: and the stacks are still the merged ones",
              np.allclose(mk.expert_weight("blk.0.ffn_gate_exps.weight", 0),
                          m.expert_weight("blk.0.ffn_gate_exps.weight", 0), atol=1e-6))
        mk_ids = [3, 5, 7, 11]
        kept_logits = Qwen3Moe(kept, log=lambda *a: None).forward(mk_ids)["logits"]
        check("mergepack: a kept-router pack forwards, and is neither the source nor the pooled pack",
              np.isfinite(kept_logits).all() and not np.allclose(kept_logits, before, atol=1e-4)
              and not np.allclose(kept_logits, after, atol=1e-4))
        refuses("mergepack: a per-slot prune mask on a kept-router pack is refused",
                lambda: Qwen3Moe(kept, log=lambda *a: None).set_prune([None] * 1), ValueError)

        # the pooling the reader runs: two slots in one group add their weight onto one stack, and a token that
        # chose k slots ending up on fewer stacks reads only the stacks it has
        from .moe import pool_to_stacks

        lg = np.array([[3.0, 2.0, 1.0, 0.0]], dtype=np.float32)         # slot 0 > 1 > 2 > 3
        top, w = pool_to_stacks(lg, np.array([0, 0, 1, 1]), router_used=4, stacks_used=1, n_stacks=2)
        check("mergepack: pooling a token's slots sums their weight on one stack",
              top.tolist() == [[0]] and abs(float(w.sum()) - 1.0) < 1e-6, "%s %s" % (top.tolist(), w.tolist()))
        two = np.array([[0.05, 0.05, 9.0, 9.0]], dtype=np.float32)
        top2, w2 = pool_to_stacks(two, np.array([0, 0, 1, 1]), router_used=4, stacks_used=2, n_stacks=2)
        check("mergepack: and keeps the stacks with the most weight, in the order the reader can apply",
              sorted(top2[0].tolist()) == [0, 1] and abs(float(w2.sum()) - 1.0) < 1e-6)
        top3, w3 = pool_to_stacks(two, np.array([0, 0, 1, 1]), router_used=4, stacks_used=1, n_stacks=2)
        check("mergepack: a stack that two of four slots chose can win a top-1 over one they did not",
              top3[0].tolist() == [1], str(top3.tolist()))

        # the same arm, written the cheap way: the merged pack's bytes with the source's router spliced in
        sp = os.path.join(td, "spliced.gguf")
        sr = splice_router(dst, src, sp, gids, mass=mass, log=lambda *a: None)
        ms = Moepack(sp)
        check("mergepack: the splice keeps every merged stack bit for bit",
              all(np.array_equal(ms.expert_weight("blk.0.%s" % s, 0), m.expert_weight("blk.0.%s" % s, 0))
                  for s in EXPERT_STACKS))
        check("mergepack: and takes the router from the source", np.array_equal(
              ms.tensor("blk.0." + ROUTER), src_pack.tensor("blk.0." + ROUTER)))
        check("mergepack: a spliced pack states both counts and its map",
              ms.n_expert == 2 and ms.router_rows == 4 and ms.router_map.tolist() == [0, 0, 1, 1]
              and sr["map_cosine"] > 0.99, str(sr))
        refuses("mergepack: a map that does not explain the pack's router is refused, not written",
                lambda: splice_router(dst, src, os.path.join(td, "bad.gguf"), np.array([0, 0, 0, 1]), mass=mass,
                                      log=lambda *a: None))

        # routing breadth is a header field: the same weights at another k, with nothing else moving
        k1 = os.path.join(td, "merged-k1.gguf")
        rr = retopk(dst, k1, 1)
        m1 = Moepack(k1)
        with open(dst, "rb") as f:
            raw = f.read()
        with open(k1, "rb") as f:
            raw2 = f.read()
        moved = [i for i in range(len(raw)) if raw[i] != raw2[i]]
        check("mergepack: --retopk moves one four-byte field and nothing else",
              m1.n_expert == 2 and m1.n_expert_used == 1 and rr["used_before"] == 2 and len(raw) == len(raw2)
              and moved and min(moved) >= rr["offset"] and max(moved) < rr["offset"] + 4,
              "%d bytes differ" % len(moved))
        check("mergepack: and the copy still decodes and runs at its new breadth",
              m1.payload_report()["ok"]
              and np.isfinite(Qwen3Moe(k1, log=lambda *a: None).forward(ids)["logits"]).all())
        try:
            retopk(dst, os.path.join(td, "too-wide.gguf"), 3)
            refused = ""
        except SystemExit as e:
            refused = str(e)
        check("mergepack: a breadth wider than the expert count is refused", "3 of 2 experts" in refused, refused[:60])

        # the gate over existing packs: one baseline walk, one a pack, a verdict each
        reps = compare_arms(src, [src], [("synthetic", ids)], log=lambda *a: None)
        check("mergepack: --compare walks the baseline and every arm, and a pack against itself is zero",
              len(reps) == 1 and reps[0]["gate"]["delta"] == 0.0 and reps[0]["tokens"] > 0, str(reps[0]["gate"]))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("%d claims held" % len(done))
    return 0


# --- the CLI ---------------------------------------------------------------------------------------------------

def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv):
    if "--selftest" in argv:
        selftest()
        print("checks passed")
        return 0
    if "--retopk" in argv:
        src = _arg(argv, "--retopk")
        dst, used = _arg(argv, "--out"), _arg(argv, "--used")
        if not (src and dst and used):
            print("usage: python -m tools.expert_merge.mergepack --retopk PACK.gguf --used 4 --out PACK-k4.gguf")
            return 2
        retopk(src, dst, used)
        return 0

    if "--compare" in argv:
        base = _arg(argv, "--compare")
        arms = [p for p in (_arg(argv, "--against") or "").split(",") if p.strip()]
        names = [n for n in (_arg(argv, "--corpus") or "").split(",") if n.strip()]
        if not (base and arms and names):
            print("usage: python -m tools.expert_merge.mergepack --compare BASE.gguf --against A.gguf[,B.gguf] "
                  "--corpus A,B --docs 2 --tokens 48 --json out.json")
            return 2
        from .moe import tokenizer
        from .routes import corpus

        docs = corpus(tokenizer(Moepack(base)), names, per_doc=int(_arg(argv, "--tokens", 48)))[:int(_arg(argv, "--docs", 2))]
        if not docs:
            raise SystemExit("mergepack --compare: no corpus to gate on")
        print("  gate corpus: %s" % ", ".join("%s(%d tokens)" % (n, len(i)) for n, i in docs))
        reps = compare_arms(base, arms, docs, max_delta=float(_arg(argv, "--max-delta", 0.02)))
        for r in reps:
            v = r["gate"]
            print("  %s: nll %.4f -> %.4f (%+.4f, 95%% CI %+.4f..%+.4f) over %d tokens, tolerance %+.3f: %s"
                  % (os.path.basename(r["pack"]), r["baseline_nll"], r["merged_nll"], v["delta"], v["ci_lo"],
                     v["ci_hi"], r["tokens"], v["max_delta"], "ACCEPTED" if v["passed"] else "REJECTED"))
        print("")
        print("  arm against arm (paired tokens, positive = the first arm is worse):")
        for r in reps:
            for other, v in sorted(r["vs"].items()):
                print("    %-24s vs %-24s %+8.4f  %+.4f..%+.4f  over %d tokens%s"
                      % (os.path.basename(r["pack"])[:24], os.path.basename(other)[:24], v["delta"], v["ci_lo"],
                         v["ci_hi"], v["tokens"], "  (resolved)" if v["ci_lo"] > 0 or v["ci_hi"] < 0 else ""))
        js = _arg(argv, "--json")
        if js:
            with open(js, "w", encoding="utf-8") as f:
                json.dump(dict(baseline=base, corpus=names, per_doc=int(_arg(argv, "--tokens", 48)), arms=reps),
                          f, indent=1, sort_keys=True)
            print("  wrote %s" % js)
        return 0

    src = _arg(argv, "--pack")
    routes_path = _arg(argv, "--routes")
    out = _arg(argv, "--out")
    router_from = _arg(argv, "--router-from")
    if router_from and src and routes_path and out:
        # the other half of the merge: this pack's stacks, the source pack's router, and the map between them
        from .routes import load

        got = load(routes_path)
        meta = got.get("meta", {})
        n_expert = int(meta.get("experts", got["route_ids"].max() + 1))
        mass, _count = mass_per_expert(got, n_expert)
        n_groups = int(_arg(argv, "--groups", Moepack(src).n_expert))
        coc = coactivation(got["route_ids"], got["route_wts"], np.arange(got["route_wts"].shape[1]), n_expert)
        gids = groups(coc + coc.T, n_groups)
        rep = splice_router(src, router_from, out, gids, mass=mass)
        print("  wrote %s: %.2f GB, %d stacks, the source's %d router rows, %d used"
              % (out, rep["bytes"] / 1e9, rep["groups"], rep["router_rows"], rep["router_used"]))
        for a in ("--used", "--expert-type"):
            if a in argv:
                print("  note: %s does not apply to --router-from (nothing is re-merged and no breadth is written)" % a)
        if _arg(argv, "--json"):
            with open(_arg(argv, "--json"), "w", encoding="utf-8") as f:
                json.dump(dict(merged=src, router_from=router_from, out=out, groups=n_groups,
                               gids=[int(x) for x in gids], splice=rep), f, indent=1, sort_keys=True)
        return 0
    if not (src and routes_path and out):
        print("usage: python -m tools.expert_merge.mergepack --pack SRC.gguf --routes ROUTES.npz --groups 64 "
              "--out MERGED.gguf [--used N] [--keep-router] [--gate --corpus A,B --docs 2 --tokens 48 "
              "--json out.json]\n"
              "       python -m tools.expert_merge.mergepack --pack MERGED.gguf --router-from SRC.gguf --routes ROUTES.npz "
              "--out MERGED-ROUTER.gguf [--groups 64] [--json out.json]\n"
              "       python -m tools.expert_merge.mergepack --retopk PACK.gguf --used 4 --out PACK-k4.gguf\n"
              "       python -m tools.expert_merge.mergepack --compare BASE.gguf --against A.gguf[,B.gguf] "
              "--corpus A,B --docs 2 --tokens 48 --json out.json [--selftest]")
        return 2
    from .routes import corpus, load

    got = load(routes_path)
    meta = got.get("meta", {})
    n_expert = int(meta.get("experts", got["route_ids"].max() + 1))
    mass, _count = mass_per_expert(got, n_expert)
    n_groups = int(_arg(argv, "--groups", 64))
    print("  %s: %d layers, %d experts, %d tokens logged" % (os.path.basename(src), mass.shape[0], n_expert,
                                                             got["route_wts"].shape[1]))
    coc = coactivation(got["route_ids"], got["route_wts"], np.arange(got["route_wts"].shape[1]), n_expert)
    gids = groups(coc + coc.T, n_groups)
    exposed = merged_mass(gids[got["route_ids"].astype(np.int64)], got["route_wts"].astype(np.float32))
    print("  a token's two slots land in one group %.1f%% of the time (the share the merge averages)"
          % (100.0 * exposed))

    et_name = _arg(argv, "--expert-type")
    et = None
    if et_name:
        from gguf import GGMLQuantizationType as T

        et = int(T[et_name])
    keep_router = "--keep-router" in argv
    rep = write_pack(src, out, gids, mass, expert_type=et, used=_arg(argv, "--used"), keep_router=keep_router,
                     progress=lambda il: il % 8 == 7 and print("    merged through layer %d" % il))
    if keep_router:
        print("  router: the source's %d rows kept byte for byte over %d merged stacks (kindi.expert_map)" %
              (rep["router_rows"], n_groups))
    if et_name:
        print("  expert stacks and router written as %s (the source's K-quants cannot be re-quantised by this build)"
              % et_name)
    print("  wrote %s: %.2f GB from %.2f GB (%.0f%%), %d experts -> %d groups"
          % (out, rep["bytes"] / 1e9, rep["source_bytes"] / 1e9,
             100.0 * rep["bytes"] / rep["source_bytes"], n_expert, n_groups))

    payload = dict(source=src, merged=out, groups=n_groups, gids=[int(x) for x in gids],
                   merged_mass=exposed, write=rep, expert_type=et_name or "as the source",
                   used=int(_arg(argv, "--used")) if _arg(argv, "--used") else None)
    if "--gate" in argv:
        from .moe import tokenizer

        pack = Moepack(src)
        tok = tokenizer(pack)
        names = [n for n in (_arg(argv, "--corpus") or "").split(",") if n.strip()]
        if not names:
            names = [d[0] for d in meta.get("docs", [])]
            print("  gate corpus: the log's own documents (in-sample)")
        docs = corpus(tok, names, per_doc=int(_arg(argv, "--tokens", 48)))[:int(_arg(argv, "--docs", 2))]
        if not docs:
            raise SystemExit("mergepack: no corpus to gate on")
        print("  gate corpus: %s" % ", ".join("%s(%d tokens)" % (n, len(i)) for n, i in docs))
        base = _arg(argv, "--baseline", src)
        if base != src:
            print("  baseline pack: %s (not the source: the two arms share one quantisation)" % base)
        c = compare(base, out, docs, max_delta=float(_arg(argv, "--max-delta", 0.02)))
        payload["gate"] = c
        v = c["gate"]
        print("  nll %.4f -> %.4f (%+.4f, 95%% CI %+.4f..%+.4f) over %d tokens, tolerance %+.3f: %s"
              % (c["baseline_nll"], c["merged_nll"], v["delta"], v["ci_lo"], v["ci_hi"], v["tokens"],
                 v["max_delta"], "ACCEPTED" if v["passed"] else "REJECTED"))
    js = _arg(argv, "--json")
    if js:
        with open(js, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        print("  wrote %s" % js)
    gated = payload.get("gate")
    return 0 if (not gated or gated["gate"]["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
