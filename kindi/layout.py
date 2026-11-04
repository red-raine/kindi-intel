"""Load-time weight layouts (host side, before upload).

repack(): every ternary tensor from the pack layout (bytes [slot][row], scales [phase][row]) to the GEMV layout:
  weights: 4 rows interleaved per 16 bytes, [R/16][C/512][4][32][4 rows][4 slots] - one 128-bit load gives a lane
           one 16-column word of 4 rows, and a warp's loads for one group and phase are contiguous;
  scales:  per (16-row group, 512-column phase), the 64 fp16 values the 32 lanes of a warp need, in lane order, so
           a warp reads them with one coalesced 128-byte load (lane l holds rows base(l), base(l)+1 of its group,
           base(l) = (l&1)*8 + (l>>1&1)*4 + (l>>2&1)*2, for the 128-column group of lane l / 8).
reorder_head(): optional, for the fast head: the head/embedding rows in cluster order, each cluster padded to whole
  16-row groups (padding rows: zero weights, token id -1)."""
import numpy as np

from . import packio


def _base_row(lane):
    return (lane & 1) * 8 + ((lane >> 1) & 1) * 4 + ((lane >> 2) & 1) * 2


def scales16(Sm):
    """Sm: [R, P] uint16 (fp16 bits) -> the GEMV scale layout, flat [R/16 * NPH * 32 * 2]."""
    R, P = Sm.shape
    NPH = P // 4
    lanes = np.arange(32)
    rows = np.array([_base_row(l) for l in lanes])
    grp = lanes >> 3
    g16 = np.arange(R // 16)[:, None, None, None]
    ph = np.arange(NPH)[None, :, None, None]
    r = g16 * 16 + rows[None, None, :, None] + np.arange(2)[None, None, None, :]
    p = ph * 4 + grp[None, None, :, None]
    return np.ascontiguousarray(Sm[r, p]).reshape(-1)


def repack(bi, bc, dp):
    """In place on the host arrays. Returns {name: (rows, ne0)}."""
    done = {}
    for name, rec in bi.items():
        R, C = rec["rows"], rec["ne0"]
        o, d = rec["bc_off"], packio.block_of(rec)
        if R % 16 or C % 512 or o % 16 or d % 2:
            raise ValueError("tensor %s (%d x %d) does not fit the GEMV layout" % (name, R, C))
        ns, P = C // 4, C // 128
        w = bc[o:o + ns * R].reshape(ns, R).T.reshape(R // 4, 4, ns // 4, 4).transpose(0, 2, 1, 3)
        # [16-row group][512-column phase][4-row chunk][lane][16 bytes]: the 2 KB a warp reads for one group and
        # phase are contiguous (DRAM page hits; +6% GEMV bandwidth on the K1200)
        w = w.reshape(R // 16, 4, C // 512, 32, 4, 4).transpose(0, 2, 1, 3, 4, 5)
        bc[o:o + ns * R] = np.ascontiguousarray(w).reshape(-1)
        dp[d:d + P * R] = scales16(np.ascontiguousarray(dp[d:d + P * R].reshape(P, R).T))
        done[name] = (R, C)
    return done


def reorder_head(bi, bc, dp, clusters, name="token_embd.weight"):
    """Before repack(): rebuild the head tensor's region (the tied token_embd, or a separate output.weight) in cluster
    order, padded, and shift the offsets of the tensors stored after it. clusters: dict with 'assign' [R] and
    'centroids' [K, C]. Returns (bc, dp, meta)."""
    assign = np.asarray(clusters["assign"]).astype(np.int64)
    cent = np.asarray(clusters["centroids"]).astype(np.float32)
    rec = bi[name]
    R, C = rec["rows"], rec["ne0"]
    ns, P = C // 4, C // 128
    o, d = rec["bc_off"], packio.block_of(rec)
    if len(assign) != R or cent.shape[1] != C:
        raise ValueError("cluster file does not match this pack")
    K = cent.shape[0]
    order = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=K)
    padded = (counts + 15) // 16 * 16
    Rp = int(padded.sum())
    starts = np.concatenate([[0], np.cumsum(padded)[:-1]])
    src_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    permtok = np.full(Rp, -1, dtype=np.int32)
    for k in range(K):
        permtok[starts[k]:starts[k] + counts[k]] = order[src_start[k]:src_start[k] + counts[k]]
    real = permtok >= 0
    permpos = np.empty(R, dtype=np.int32)
    permpos[permtok[real]] = np.nonzero(real)[0]
    nb = np.full((ns, Rp), 40, dtype=np.uint8)            # 40 = four zero trits
    nsc = np.zeros((P, Rp), dtype=np.uint16)
    nb[:, real] = bc[o:o + ns * R].reshape(ns, R)[:, permtok[real]]
    nsc[:, real] = dp[d:d + P * R].reshape(P, R)[:, permtok[real]]
    dbc, ddp = (Rp - R) * ns, (Rp - R) * P
    bc = np.concatenate([bc[:o], nb.reshape(-1), bc[o + ns * R:]])
    dp = np.concatenate([dp[:d], nsc.reshape(-1), dp[d + P * R:]])
    for other, r in bi.items():
        if other != name and r["bc_off"] > o:
            r["bc_off"] += dbc
            if "dblk" in r:                     # a v2 record: offsets in blocks, one byte a block on disk
                r["dblk"] += ddp
            else:
                r["dpm_off"] += ddp * 2
    rec["rows"] = Rp
    meta = {"R": R, "Rp": Rp, "K": K, "cent": cent, "permtok": permtok, "permpos": permpos,
            "gstart": (starts // 16).astype(np.int32), "gcount": (padded // 16).astype(np.int32)}
    return bc, dp, meta
