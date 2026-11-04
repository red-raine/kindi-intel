/* kcpu.c - Kindi CPU backend (Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only): one token of a Ternary-Bonsai-2 (Qwen3.5 hybrid GDN/attention) model in C.
   The whole forward pass runs inside one OpenMP parallel region (barriers between the stages), so there is no
   Python or thread start-up cost per operation. Arithmetic plan as kindi/engine.py: fp64 norm sums, Hadamard folds
   with the pack's sign vectors, 81-entry fp32 table GEMV for every ternary matrix, fp16 K/V cache. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif

/* the entry points ctypes calls: MSVC exports nothing without this, gcc exports everything */
#if defined(_WIN32)
#define KCAPI __declspec(dllexport)
#else
#define KCAPI
#endif

typedef struct { const uint8_t* W; const float* S; int R, C; } Mat;   /* W [R/8][C/128][64] u32 pair codes, S [R/8][C/128][8] */

/* Mixture-of-experts FFN: a router over NEXP experts, the NACT highest with their softmax weights, and 3 matrices per
   expert (gate, up, down) in the same ternary form as a dense FFN. nexp == 0 means the layer is dense and the fields
   are unused, so a dense pack pays nothing for this. `staged` is set when the expert matrices point into a buffer that
   a fetch hook refills, which is how an expert can live outside resident memory. */
typedef struct {
  Mat router;
  int nexp, nact, staged;
  const Mat* experts;      /* 3 per expert: gate [E x D], up [E x D], down [D x E] */
  const float* sF;         /* the fold signs of the expert width E (the dense sF is NFF wide) */
} Moe;

typedef struct {
  int is_attn;
  const float *attn_norm, *post_norm;
  Moe moe;
  /* GDN */
  Mat qkv, gate, ssm_out;
  const float *beta_w, *alpha_w, *dt_b, *ssm_a, *conv_w, *ssm_norm;
  float *M, *conv;                       /* state [NV][128][128], conv [3][CONV_CH] */
  /* attention */
  Mat q, k, v, o;
  const float *q_norm, *k_norm;
  uint16_t *Kc, *Vc;                     /* [ctx][NHKV][HD] fp16 */
  /* quantised K/V cache (kc_kv_quant): 2-bit trits with one scale per (position, head), and q8_0's 32-value int8
     blocks with an fp16 scale. Allocated by the host, which also owns their lifetime; unused when the mode is 0. */
  uint8_t *Kt, *Vt;                      /* [ctx][NHKV][HD/4] four 2-bit codes per byte */
  uint16_t *Ks, *Vs;                     /* [ctx][NHKV] ternary scale, fp16 */
  int8_t *Kq, *Vq;                       /* [ctx][NHKV][HD] q8_0 codes */
  uint16_t *Kqs, *Vqs;                   /* [ctx][NHKV][HD/32] q8_0 block scales, fp16 */
  uint8_t *K4, *V4;                      /* [ctx][NHKV][HD/2] 4-bit signed codes, two a byte */
  uint16_t *K4s, *V4s;                   /* [ctx][NHKV] 4-bit scale, fp16 */
  uint8_t *K2, *V2;                      /* [ctx][NHKV][HD/4] four 2-bit codes per byte, levels {-3,-1,1,3} */
  uint16_t *K2s, *V2s;                   /* [ctx][NHKV] 2-bit scale, fp16 */
  uint8_t *K2b, *V2b;                    /* the same code with one scale per 32 values instead of per head */
  uint16_t *K2bs, *V2bs;                 /* [ctx][NHKV][HD/32] 2-bit block scales, fp16 */
  uint8_t *K5, *V5;                      /* [ctx][NHKV][HD/2] two 4-bit codes per byte, the 4-level code twice */
  uint16_t *K5s, *V5s;                   /* [ctx][NHKV] 4-bit-as-two-passes scale, fp16 */
  /* FFN */
  Mat ffg, ffu, ffd;
} Layer;

typedef struct {
  int D, NL, NFF, NH, NHKV, HD, NK, HK, NV, CONV_CH, HB, rope_half, ctx, vocab;
  Layer* L;
  Mat emb, head;
  const float *out_norm, *sD, *sV, *sA, *sF, *cos_t, *sin_t;
  /* scratch */
  float *h, *xn, *xp, *y, *qkv, *zv, *cs, *qk, *gout, *outn, *m, *qn, *kr, *vv, *qr, *kro, *sc, *att, *ffg, *ffu,
        *T, *T2, *logits, *gb, *gg;   /* T2: the second token's pair table (kc_step2 only) */
  signed char *Q; float *QS;             /* int8 prototype only (KINDI_CPU_Q8=1): quantised input and group scales */
  double* red;                           /* per-thread partial sums */
  /* expert routing scratch, and the hook a paged-expert load uses (NULL when every expert is resident) */
  float *pr, *ex, *psel;
  int* sel;
  void (*moe_fetch)(void*, int, int);
  void* moe_ctx;
  int kvq;                               /* K/V cache format: 0 fp16, 1 ternary, 2 q8_0, 3 4-bit, 4 2-bit,
                                            5 4 bits as two 4-level passes, 6 2 bits with per-32 block scales
                                            (kc_kv_quant) */
  int kvq_k, kvq_v;                      /* the two sides separately (kc_kv_quant2): K and V need not match */
} Model;

/* ------------------------------------------------------------------------------------------- helpers */
/* stage timers (thread 0, between synchronised points): 0 embed 1 norm+fold 2 gemv 3 gdn-other 4 attn-other 5 ffn-other 6 head */
double kc_times[8];
static double t_last; static int t_cur;
#define TS(cat) do { if(omp_get_thread_num() == 0){ const double _t = omp_get_wtime(); kc_times[t_cur] += _t - t_last; t_last = _t; t_cur = (cat); } } while(0)
static inline void span(long long n, int* a, int* b){
  const int nt = omp_get_num_threads(), id = omp_get_thread_num();
  *a = (int)(n * id / nt); *b = (int)(n * (id + 1) / nt);
}
/* gcc inlines _cvtsh_ss/_cvtss_sh here; MSVC links them as functions that its linker cannot find, so the Windows
   build uses the integer form below (the K/V cache is a storage format, not an arithmetic one, so this is a size and
   speed question, not a correctness one) */
#if defined(__F16C__) && !defined(_MSC_VER)
static inline float h2f(uint16_t h){ return _cvtsh_ss(h); }
static inline uint16_t f2h(float f){ return _cvtss_sh(f, 0); }
#else
/* no F16C: the same conversion in integer arithmetic (round to nearest even), so the K/V path is not a reason to
   require AVX2 - the fp16 cache is a storage format here, not an arithmetic one */
static inline float h2f(uint16_t h){
  const int e = (h >> 10) & 31, m = h & 1023;
  const uint32_t s = (uint32_t)(h >> 15) << 31;
  if(e == 0){ if(!m) return (s ? -0.0f : 0.0f); float v = m * (1.0f / 16777216.0f); return s ? -v : v; }
  if(e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
  { union { uint32_t u; float f; } u; u.u = s | ((uint32_t)(e - 15 + 127) << 23) | ((uint32_t)m << 13); return u.f; }
}
static inline uint16_t f2h(float f){
  union { float f; uint32_t u; } v; v.f = f;
  const uint32_t u = v.u, s = (u >> 16) & 0x8000;
  const int e = (int)((u >> 23) & 255) - 127;
  if(e > 15) return (uint16_t)(s | 0x7C00);
  if(e < -24) return (uint16_t)s;
  if(e < -14){
    const uint32_t m = ((u & 0x7FFFFF) | 0x800000) >> (uint32_t)(-e - 1);
    return (uint16_t)(s | ((m + 0x1000) >> 13));
  }
  return (uint16_t)(s | ((uint32_t)(e + 15) << 10) | ((u >> 13) & 1023));
}
#endif
static inline float silu(float x){ return x / (1.0f + expf(-x)); }
static inline float sigm(float x){ return 1.0f / (1.0f + expf(-x)); }

static void fwht_block(float* x, int hb){
  for(int h = 1; h < hb; h <<= 1)
    for(int i = 0; i < hb; i += 2 * h)
      for(int j = i; j < i + h; ++j){ const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
  const float s = 1.0f / sqrtf((float)hb);
  for(int j = 0; j < hb; ++j) x[j] *= s;
}

/* out = fold(v) = FWHT_blocks(sg * v); in parallel over blocks (must be called by all threads; ends in a barrier) */
static void fold(const Model* m, const float* v, const float* sg, float* out, int n){
  int b;                       /* declared outside the for: MSVC's /openmp needs C89 loop form */
  #pragma omp for schedule(static)
  for(b = 0; b < n / m->HB; ++b){
    float* o = out + (size_t)b * m->HB;
    for(int j = 0; j < m->HB; ++j) o[j] = sg[b * m->HB + j] * v[b * m->HB + j];
    fwht_block(o, m->HB);
  }
}

/* ------------------------------------------------------------------------------------------- ternary GEMV */
/* Weights as pair codes, rows in SIMD lanes: per (8-row block, 128-column group) 64 uint32; word p holds the 8 rows'
   4-bit codes for columns 2p, 2p+1 (0..7 = which of the 8 nonzero (w0, w1) pairs, 8 = both zero; 2 bits per weight).
   Per call, a table of the 8 nonzero pair sums for every column pair (m->T, C/2 x 8 floats); then one vpermps per
   8 rows x 2 columns. Per-group sums are scaled (the 8 rows' scales are contiguous) into a vector accumulator, so a
   row's result needs no horizontal sum. Every pair sum is formed once in fp32; no activation quantisation. */
KCAPI void kc_init(void){}
static void pair_tables(const float* x, float* T, int p0, int p1){
  for(int p = p0; p < p1; ++p){
    const float x0 = x[2 * p], x1 = x[2 * p + 1];
    float* t = T + (size_t)p * 8;
    t[0] = -x0 - x1; t[1] = -x1; t[2] = x0 - x1; t[3] = -x0;
    t[4] = x0; t[5] = -x0 + x1; t[6] = x1; t[7] = x0 + x1;
  }
}
/* --------------------------------------------------------------------------------- kernel tiers
   Three implementations of the same GEMV over the same pack, and the run picks the fastest one that reproduces the
   other bit for bit. Tier 0 is scalar C with no intrinsics at all, tier 1 is the AVX2 pair-table kernel, tier 2 is
   the int8-activation proto (not exact, so it is never chosen by the gate). The scalar tier is written to the same
   operation order as the vector one - the same table entries, the same even/odd partial sums, the same fused
   multiply-add - which is what makes an exact comparison possible instead of an approximate one. */
#define KCT_SCALAR 1
#define KCT_AVX2 2
#define KCT_Q8 4
static int g_tier = 0;                 /* 0 = the fastest tier this build has */

KCAPI int kc_tier_mask(void){
  int m = KCT_SCALAR;
#if defined(__AVX2__)
  m |= KCT_AVX2 | KCT_Q8;
#endif
  return m;
}
KCAPI void kc_set_tier(int t){ g_tier = t; }
KCAPI int kc_get_tier(void){ return g_tier; }

/* the small-margin guard (KINDI_CPU_Q8GUARD=<gap>): at a decision whose top-1/top-2 gap is under the measured logit
   drift, the int8 argmax is a coin toss decided by quantisation noise, so the head is recomputed with the production
   fp32 kernel from the same folded input and the decision taken from those logits instead. The head is the only place
   a decision can be re-derived without replaying the state, but it is not a cheap GEMV: it is 508.6 M of the model's
   1880 M parameters, and one fallback measured +10.0 ms on an 87.4 ms int8 token. So the guard is affordable at the 1%
   trigger rate it actually fires at and not at a high one, and it does not recover the fp32 decisions either - see the
   measurements in docs/CPU-PERF-2026.md. Off at 0 (the default); g_force_fp32 is the tier override the fallback uses. */
static float g_guard_eps = -1.0f;            /* <0 = not read from the environment yet */
static long long g_guard_seen, g_guard_hits;
static int g_force_fp32;
static float guard_eps(void){
  if(g_guard_eps < 0.0f){
    const char* e = getenv("KINDI_CPU_Q8GUARD");
    g_guard_eps = e ? (float)atof(e) : 0.0f;
    if(g_guard_eps < 0.0f) g_guard_eps = 0.0f;
  }
  return g_guard_eps;
}
/* decisions examined and decisions that fell back, so a sweep can report what the guard cost */
KCAPI void kc_guard_stats(long long* seen, long long* hits){ *seen = g_guard_seen; *hits = g_guard_hits; }
KCAPI void kc_guard_reset(void){ g_guard_seen = 0; g_guard_hits = 0; }

/* the exact tail (KINDI_CPU_Q8TAIL=<k>): the last k blocks and the head are computed with the fp32 kernel while the
   rest of the stack stays int8, so the head sees a stream that has only been drifted by the layers before the tail.
   The guard above showed a head-only fallback cannot recover a decision; this asks how much of the stack has to be
   exact before the decision is exact again. k = 0 (the default) is the plain int8 path and k = NL is plain fp32, which
   makes that end a control: it must reproduce the fp32 arm token for token (checked, 3/3 prompts). */
static int g_tail = -1;
static int q8_tail(void){
  if(g_tail < 0){
    const char* e = getenv("KINDI_CPU_Q8TAIL");
    g_tail = e ? atoi(e) : 0;
    if(g_tail < 0) g_tail = 0;
  }
  return g_tail;
}

static int tier(void){
  if(g_force_fp32) return (kc_tier_mask() & KCT_AVX2) ? KCT_AVX2 : KCT_SCALAR;
  if(g_tier) return g_tier;
  { const char* e = getenv("KINDI_CPU_Q8");
    if(e && e[0] == '1' && (kc_tier_mask() & KCT_Q8)) return (g_tier = KCT_Q8); }
  return (g_tier = (kc_tier_mask() & KCT_AVX2) ? KCT_AVX2 : KCT_SCALAR);
}

#if defined(__AVX2__)
/* Two row blocks walk the group list together and share every table load: the table part of a group is read once
   for 16 rows instead of once for 8, which halves the table traffic, and the two independent accumulator chains give
   the scheduler work to overlap with the vpermps latency. The arithmetic per row is unchanged, so the result is the
   same bits as the one-block-at-a-time form. Table reads are unaligned loads so that nothing depends on where the
   table buffer happens to sit, and the second block is only touched when it is inside this thread's range. */
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const __m256i eight = _mm256_set1_epi32(8);          /* the code of a column pair that is 0 * 0 */
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; rb += 2){
    const int two = (rb + 1 < b1);
    const uint32_t* pw0 = P + ((size_t)rb * G) * 64;
    const uint32_t* pw1 = pw0 + (two ? (size_t)G * 64 : 0);
    const float* S0 = A->S + (size_t)rb * G * 8;
    const float* S1 = two ? S0 + (size_t)G * 8 : S0;
    __m256 X0 = _mm256_setzero_ps(), X1 = X0;
    for(int g = 0; g < G; ++g){
      const uint32_t* q0 = pw0 + (size_t)g * 64;
      const uint32_t* q1 = pw1 + (size_t)g * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      /* the weight stream is the one DRAM stream this loop has: pull the group two ahead into L1 while this one is
         being consumed, so a cache miss lands on a line that is already in flight */
      if(g + 4 < G){
        for(int k = 0; k < 64; k += 16) _mm_prefetch((const char*)(q0 + 256 + k), _MM_HINT_T0);
        if(two) for(int k = 0; k < 64; k += 16) _mm_prefetch((const char*)(q1 + 256 + k), _MM_HINT_T0);
      }
      __m256 p0a = _mm256_setzero_ps(), p0b = p0a, p1a = p0a, p1b = p0a;
      for(int p = 0; p < 64; p += 2){
        const __m256 t0 = _mm256_loadu_ps(pt + 8 * p);
        const __m256 t1 = _mm256_loadu_ps(pt + 8 * p + 8);
        const __m256i c0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q0[p]), sh), f);
        const __m256i c1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q0[p + 1]), sh), f);
        const __m256 v0 = _mm256_permutevar8x32_ps(t0, c0);
        const __m256 v1 = _mm256_permutevar8x32_ps(t1, c1);
        const __m256 z0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(c0, eight));
        const __m256 z1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(c1, eight));
        p0a = _mm256_add_ps(p0a, _mm256_andnot_ps(z0, v0));
        p0b = _mm256_add_ps(p0b, _mm256_andnot_ps(z1, v1));
        if(two){
          const __m256i e0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q1[p]), sh), f);
          const __m256i e1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q1[p + 1]), sh), f);
          const __m256 u0 = _mm256_permutevar8x32_ps(t0, e0);
          const __m256 u1 = _mm256_permutevar8x32_ps(t1, e1);
          const __m256 w0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(e0, eight));
          const __m256 w1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(e1, eight));
          p1a = _mm256_add_ps(p1a, _mm256_andnot_ps(w0, u0));
          p1b = _mm256_add_ps(p1b, _mm256_andnot_ps(w1, u1));
        }
      }
      X0 = _mm256_fmadd_ps(_mm256_add_ps(p0a, p0b), _mm256_loadu_ps(S0 + (size_t)g * 8), X0);
      if(two) X1 = _mm256_fmadd_ps(_mm256_add_ps(p1a, p1b), _mm256_loadu_ps(S1 + (size_t)g * 8), X1);
    }
    _mm256_storeu_ps(y + rb * 8, X0);
    if(two) _mm256_storeu_ps(y + (rb + 1) * 8, X1);
  }
}

/* ------------------------------------------------------------------- int8 activation quantisation (prototype)
   Off by default; KINDI_CPU_Q8=1 routes every gemv through the int8 form below. This is NOT the production path: it
   exists to answer one question, whether the numerics an AVX-VNNI kernel would produce can still be token-exact.

   Three quantisation schemes live here, selectable so they can be compared on the same host:
     - KINDI_CPU_Q8G=<2|4|8|16|32|64|128>  the group width of the fine-grained form (default 128 columns). A group
       has one scale, so 32 means one scale per 32 columns instead of one per 128.
     - KINDI_CPU_Q8C=1  a static per-channel input scale, calibrated once (kc_cal_begin..kc_cal_end) and then applied
       on the ordinary fp32 path: the input is rounded to its channel's grid and handed to the production GEMV, so the
       weight kernel is unchanged and only the activation precision is coarser than fp32.
     - KINDI_CPU_Q8GUARD=<gap>  the small-margin guard: a decision whose top-1/top-2 gap is below <gap> is recomputed
       with the fp32 head from the same folded input and decided there instead of on the int8 logits.
   The weight scale stays one per 128-column block in every scheme: it is part of the pack.

   The arithmetic it computes is exactly what vpdpbusd would compute from this same pack: the folded input is
   quantised to int8 per 128-column group, each weight is used as its signed ternary value (-1/0/1), the products are
   summed in int32, and the group total is rescaled by (activation scale * row scale). vpdpbusd is an integer
   multiply-accumulate and vpmaddubsw+vpmaddwd+vpaddd produce the identical integers, so the VALUE of the dot product
   does not depend on which integer instructions perform it. That is why the numerics can be settled without SDE:
   SDE emulates the instruction, it does not change the result of one. Only the instruction count needs VNNI. */
static int q8_on(void){
  static int v = -1;
  if(v < 0){ const char* e = getenv("KINDI_CPU_Q8"); v = (e && e[0] == '1') ? 1 : 0; }
  return v;
}
/* the activation group width (KINDI_CPU_Q8G): must divide the 128-column weight block and be at least 2 */
static int q8_group(void){
  static int v = 0;
  if(!v){
    const char* e = getenv("KINDI_CPU_Q8G");
    v = e ? atoi(e) : 128;
    if(v != 2 && v != 4 && v != 8 && v != 16 && v != 32 && v != 64 && v != 128) v = 128;
  }
  return v;
}
static int q8_pc(void){
  static int v = -1;
  if(v < 0){ const char* e = getenv("KINDI_CPU_Q8C"); v = (e && e[0] == '1') ? 1 : 0; }
  return v;
}

/* the gemv call sites, which with the width form the calibration key below */
#define KCAL_HEAD 0
#define KCAL_QKV 1
#define KCAL_SSM 2
#define KCAL_ATTQKV 3
#define KCAL_ATTN_O 4
#define KCAL_MOE_UP 5
#define KCAL_MOE_ROUTER 6
#define KCAL_MOE_DOWN 7
#define KCAL_DENSE_UP 8
#define KCAL_DENSE_DOWN 9
#define KCAL_MAX 32

/* static per-channel scales for KINDI_CPU_Q8C: a calibration pass (kc_cal_begin, tokens, kc_cal_end) records the
   largest |x| each input column ever had; 127/max is then the column's step size. The key is the call site *and* the
   width, not the width alone: five sites feed the 2048-wide products (GDN qkv/gate, ssm_out, attention qkv, attention o,
   dense up) and their inputs are folded with different sign vectors, so their calibrated maxima span 5.07 to 9.08 and a
   table shared between them would size the smallest input's grid by the largest's. */
static float* g_cal[KCAL_MAX];
static int g_cal_site[KCAL_MAX];
static int g_cal_c[KCAL_MAX];
static int g_cal_n;
static int g_cal_on;
static float* g_dq;                     /* the dequantised copy the per-channel mode feeds the production GEMV */
static int g_dq_cap;

static float* cal_find(int site, int C){
  for(int i = 0; i < g_cal_n; ++i) if(g_cal_site[i] == site && g_cal_c[i] == C) return g_cal[i];
  return NULL;
}
static float* cal_slot(int site, int C, int create){
  float* s = cal_find(site, C);
  if(s || !create || g_cal_n >= KCAL_MAX) return s;
  s = (float*)calloc((size_t)C, sizeof(float));
  if(!s) return NULL;
  g_cal[g_cal_n] = s; g_cal_site[g_cal_n] = site; g_cal_c[g_cal_n] = C; ++g_cal_n;
  return s;
}
static void cal_accum(int site, int C, const float* x){
  float* s = cal_slot(site, C, 1);
  if(!s) return;
  for(int j = 0; j < C; ++j){ const float av = fabsf(x[j]); if(av > s[j]) s[j] = av; }
}
static void dq_ensure(int C){
  if(C > g_dq_cap){
    free(g_dq);
    g_dq = (float*)malloc((size_t)C * sizeof(float));
    g_dq_cap = g_dq ? C : 0;
  }
}
KCAPI void kc_cal_begin(void){
  for(int i = 0; i < g_cal_n; ++i) memset(g_cal[i], 0, (size_t)g_cal_c[i] * sizeof(float));
  g_cal_on = 1;
}
KCAPI int kc_cal_end(void){ g_cal_on = 0; return g_cal_n; }
/* the recorded (site, width) pairs, so a caller can save and load them */
KCAPI int kc_cal_keys(int* sites, int* widths, int max){
  const int n = g_cal_n < max ? g_cal_n : max;
  for(int i = 0; i < n; ++i){ sites[i] = g_cal_site[i]; widths[i] = g_cal_c[i]; }
  return g_cal_n;
}
KCAPI void kc_cal_get(int site, int C, float* out){
  const float* s = cal_find(site, C);
  for(int j = 0; j < C; ++j) out[j] = s ? s[j] : 0.0f;
}
KCAPI void kc_cal_put(int site, int C, const float* in){
  float* s = cal_slot(site, C, 1);
  if(!s) return;
  for(int j = 0; j < C; ++j) s[j] = in[j];
}
/* round every column to its own grid and write the product back as fp32: the production GEMV then sees a value it can
   use unchanged, so this mode measures the quantisation alone */
static void quant_cols(const float* x, float* out, int C, const float* s){
  if(!s){ for(int j = 0; j < C; ++j) out[j] = x[j]; return; }
  int j;
  #pragma omp for schedule(static)
  for(j = 0; j < C; ++j){
    const float mx = s[j];
    const float st = (mx > 0.0f ? mx : 1.0f) / 127.0f;
    long v = lrintf(x[j] / st);
    if(v > 127) v = 127; else if(v < -127) v = -127;
    out[j] = (float)v * st;
  }
}
/* symmetric int8 per GS-column group: scale = max|x|/127 so the largest magnitude lands on +-127 */
static void quant_x(const float* x, signed char* Q, float* QS, int C, int GS){
  int g;
  #pragma omp for schedule(static)
  for(g = 0; g < C / GS; ++g){
    const float* xg = x + (size_t)g * GS;
    float mx = 0.0f;
    for(int i = 0; i < GS; ++i){ const float av = fabsf(xg[i]); if(av > mx) mx = av; }
    const float s = mx > 0.0f ? mx / 127.0f : 1.0f;
    QS[g] = s;
    for(int i = 0; i < GS; ++i){
      long v = lrintf(xg[i] / s);
      if(v > 127) v = 127; else if(v < -127) v = -127;
      Q[(size_t)g * GS + i] = (signed char)v;
    }
  }
}
/* rows8 with int8 activations: same pack, same row-block walk, same per-group scaling, but the pair sum is an integer
   {w0,w1}.{xq[2p],xq[2p+1]} accumulated in int32 instead of an fp32 table lookup. The code->weight maps are the same
   pair_tables decomposition, held as int32 lanes so vpermd can select them: code c -> (T0[c], T1[c]). */
static void rows8_q8(const Mat* A, float* y, const signed char* Q, const float* QS, int b0, int b1, int GS){
  const int G = A->C / 128;                                  /* one row-scale slot per 128 columns */
  const int NG = A->C / GS;                                  /* activation groups, one scale each */
  const int NB = 128 / GS;                                   /* groups per 128-column block */
  const int WPB = GS / 2;                                    /* u32 words per group */
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const __m256i eight = _mm256_set1_epi32(8);                 /* 0 * 0: both weights of the pair are zero */
  const __m256i T0 = _mm256_setr_epi32(-1, 0, 1, -1, 1, -1, 0, 1);
  const __m256i T1 = _mm256_setr_epi32(-1, -1, -1, 0, 0, 1, 1, 1);
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; rb += 2){
    const int two = (rb + 1 < b1);
    const uint32_t* pw0 = P + ((size_t)rb * G) * 64;
    const uint32_t* pw1 = pw0 + (two ? (size_t)G * 64 : 0);
    const float* S0 = A->S + (size_t)rb * G * 8;
    const float* S1 = two ? S0 + (size_t)G * 8 : S0;
    __m256 X0 = _mm256_setzero_ps(), X1 = X0;
    for(int ng = 0; ng < NG; ++ng){
      const int g = ng / NB, sub = (ng % NB) * WPB;
      const uint32_t* q0 = pw0 + (size_t)g * 64 + sub;
      const uint32_t* q1 = pw1 + (size_t)g * 64 + sub;
      const signed char* xq = Q + (size_t)ng * GS;
      __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
      for(int p = 0; p < WPB; ++p){                           /* one word = 8 rows x 2 columns */
        const __m256i a0 = _mm256_set1_epi32((int)xq[2 * p]);
        const __m256i a1 = _mm256_set1_epi32((int)xq[2 * p + 1]);
        const __m256i c0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q0[p]), sh), f);
        const __m256i z0 = _mm256_cmpeq_epi32(c0, eight);
        const __m256i w0 = _mm256_andnot_si256(z0, _mm256_permutevar8x32_epi32(T0, c0));
        const __m256i w1 = _mm256_andnot_si256(z0, _mm256_permutevar8x32_epi32(T1, c0));
        acc0 = _mm256_add_epi32(acc0, _mm256_add_epi32(_mm256_mullo_epi32(w0, a0), _mm256_mullo_epi32(w1, a1)));
        if(two){
          const __m256i e0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q1[p]), sh), f);
          const __m256i z1 = _mm256_cmpeq_epi32(e0, eight);
          const __m256i u0 = _mm256_andnot_si256(z1, _mm256_permutevar8x32_epi32(T0, e0));
          const __m256i u1 = _mm256_andnot_si256(z1, _mm256_permutevar8x32_epi32(T1, e0));
          acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(_mm256_mullo_epi32(u0, a0), _mm256_mullo_epi32(u1, a1)));
        }
      }
      const __m256 qs = _mm256_set1_ps(QS[ng]);
      X0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0), _mm256_mul_ps(qs, _mm256_loadu_ps(S0 + (size_t)g * 8)), X0);
      if(two) X1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1), _mm256_mul_ps(qs, _mm256_loadu_ps(S1 + (size_t)g * 8)), X1);
    }
    _mm256_storeu_ps(y + rb * 8, X0);
    if(two) _mm256_storeu_ps(y + (rb + 1) * 8, X1);
  }
}

#endif /* __AVX2__ */

/* ---------------------------------------- tier 0: the same arithmetic in scalar C, no intrinsics anywhere
   Every float operation below happens in the order the vector kernel performs it: the table is built by pair_tables
   (not duplicated), each row keeps its own accumulator, the even and odd column pairs sum separately and are added
   together before the scale, and the scale multiply is fused (fmaf) because the vector path uses vfmadd. A plain
   multiply-then-add would round twice and give a different last bit, which is exactly the difference the gate looks
   for. */
static void rows8_ref(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; rb += 2){
    const int two = (rb + 1 < b1);
    const uint32_t* pw0 = P + ((size_t)rb * G) * 64;
    const uint32_t* pw1 = pw0 + (two ? (size_t)G * 64 : 0);
    const float* S0 = A->S + (size_t)rb * G * 8;
    const float* S1 = two ? S0 + (size_t)G * 8 : S0;
    float X0[8], X1[8];
    for(int l = 0; l < 8; ++l){ X0[l] = 0.0f; X1[l] = 0.0f; }
    for(int g = 0; g < G; ++g){
      const uint32_t* q0 = pw0 + (size_t)g * 64;
      const uint32_t* q1 = pw1 + (size_t)g * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      for(int l = 0; l < 8; ++l){
        float p0a = 0.0f, p0b = 0.0f, p1a = 0.0f, p1b = 0.0f;
        for(int p = 0; p < 64; p += 2){
          const int c0 = (int)((q0[p] >> (4 * l)) & 15);
          const int c1 = (int)((q0[p + 1] >> (4 * l)) & 15);
          p0a += (c0 == 8) ? 0.0f : pt[8 * p + c0];
          p0b += (c1 == 8) ? 0.0f : pt[8 * p + 8 + c1];
          if(two){
            const int e0 = (int)((q1[p] >> (4 * l)) & 15);
            const int e1 = (int)((q1[p + 1] >> (4 * l)) & 15);
            p1a += (e0 == 8) ? 0.0f : pt[8 * p + e0];
            p1b += (e1 == 8) ? 0.0f : pt[8 * p + 8 + e1];
          }
        }
        X0[l] = fmaf(p0a + p0b, S0[(size_t)g * 8 + l], X0[l]);
        if(two) X1[l] = fmaf(p1a + p1b, S1[(size_t)g * 8 + l], X1[l]);
      }
    }
    for(int l = 0; l < 8; ++l){
      y[rb * 8 + l] = X0[l];
      if(two) y[(rb + 1) * 8 + l] = X1[l];
    }
  }
}

/* rows8_b2's arithmetic in scalar C: one walk over the codes for both tokens, each token's chains kept apart */
static void rows8_b2_ref(const Mat* A, float* y0, float* y1, const float* T, const float* U, int b0, int b1){
  const int G = A->C / 128;
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; rb += 2){
    const int two = (rb + 1 < b1);
    const uint32_t* pw0 = P + ((size_t)rb * G) * 64;
    const uint32_t* pw1 = pw0 + (two ? (size_t)G * 64 : 0);
    const float* S0 = A->S + (size_t)rb * G * 8;
    const float* S1 = two ? S0 + (size_t)G * 8 : S0;
    float A0[8], B0[8], A1[8], B1[8];
    for(int l = 0; l < 8; ++l){ A0[l] = 0.0f; B0[l] = 0.0f; A1[l] = 0.0f; B1[l] = 0.0f; }
    for(int g = 0; g < G; ++g){
      const uint32_t* q0 = pw0 + (size_t)g * 64;
      const uint32_t* q1 = pw1 + (size_t)g * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      const float* pu = U + (size_t)g * 64 * 8;
      for(int l = 0; l < 8; ++l){
        float p0a = 0.0f, p0b = 0.0f, p1a = 0.0f, p1b = 0.0f, r0a = 0.0f, r0b = 0.0f, r1a = 0.0f, r1b = 0.0f;
        for(int p = 0; p < 64; p += 2){
          const int c0 = (int)((q0[p] >> (4 * l)) & 15);
          const int c1 = (int)((q0[p + 1] >> (4 * l)) & 15);
          const float t0 = (c0 == 8) ? 0.0f : pt[8 * p + c0];
          const float t1 = (c1 == 8) ? 0.0f : pt[8 * p + 8 + c1];
          p0a += t0; p0b += t1;
          r0a += (c0 == 8) ? 0.0f : pu[8 * p + c0];
          r0b += (c1 == 8) ? 0.0f : pu[8 * p + 8 + c1];
          if(two){
            const int e0 = (int)((q1[p] >> (4 * l)) & 15);
            const int e1 = (int)((q1[p + 1] >> (4 * l)) & 15);
            p1a += (e0 == 8) ? 0.0f : pt[8 * p + e0];
            p1b += (e1 == 8) ? 0.0f : pt[8 * p + 8 + e1];
            r1a += (e0 == 8) ? 0.0f : pu[8 * p + e0];
            r1b += (e1 == 8) ? 0.0f : pu[8 * p + 8 + e1];
          }
        }
        const float s0 = S0[(size_t)g * 8 + l], s1 = S1[(size_t)g * 8 + l];
        A0[l] = fmaf(p0a + p0b, s0, A0[l]);
        B0[l] = fmaf(r0a + r0b, s0, B0[l]);
        if(two){ A1[l] = fmaf(p1a + p1b, s1, A1[l]); B1[l] = fmaf(r1a + r1b, s1, B1[l]); }
      }
    }
    for(int l = 0; l < 8; ++l){
      y0[rb * 8 + l] = A0[l]; y1[rb * 8 + l] = B0[l];
      if(two){ y0[(rb + 1) * 8 + l] = A1[l]; y1[(rb + 1) * 8 + l] = B1[l]; }
    }
  }
}

/* inside the parallel region: up to 3 matrices with the same input; the threads build the pair tables (or quantise
   the input, on the int8 prototype path), then split the concatenated rows in 8-row blocks; ends in a barrier.
   `site` names the call site (KCAL_*) for the per-channel calibration: two sites can share a width and not a scale. */
static void gemv(Model* m, int n, const Mat* const* A, float* const* y, const float* x, int site){
  const int prev = t_cur; TS(2);
  int a, b;
  const int t = tier();
  const int q8 = (t == KCT_Q8);
  float* cs = cal_find(site, A[0]->C);
  const int pc = !q8 && q8_pc() && cs != NULL;   /* per-channel static scales, production kernels */
  const float* xx = x;
  if(g_cal_on){
    #pragma omp single
    cal_accum(site, A[0]->C, x);         /* the input as the multiplier sees it, before any rounding */
  }
  if(pc){
    #pragma omp single
    dq_ensure(A[0]->C);
    quant_cols(x, g_dq, A[0]->C, cs);
    xx = g_dq;
  }
  if(q8){
#if defined(__AVX2__)
    quant_x(xx, m->Q, m->QS, A[0]->C, q8_group());
#else
    const int dx = A[0]->C; for(int i = 0; i < dx; ++i) m->Q[i] = 0;  /* unreachable: no int8 tier in a scalar build */
#endif
  }
  else { span(A[0]->C / 2, &a, &b); pair_tables(xx, m->T, a, b); }
  #pragma omp barrier
  long long Ut = 0;
  for(int i = 0; i < n; ++i) Ut += A[i]->R / 8;
  int a0, a1;
  span(Ut, &a0, &a1);
  int base = 0;
  for(int i = 0; i < n; ++i){
    const int U = A[i]->R / 8;
    const int u0 = a0 > base ? a0 - base : 0, u1 = (a1 - base) < U ? (a1 - base) : U;
    base += U;
    if(u1 > u0){
      if(q8){
#if defined(__AVX2__)
        rows8_q8(A[i], y[i], m->Q, m->QS, u0, u1, q8_group());
#else
        rows8_ref(A[i], y[i], m->T, u0, u1);        /* no int8 path in a scalar build */
#endif
      } else if(t == KCT_SCALAR){
        rows8_ref(A[i], y[i], m->T, u0, u1);
      } else {
#if defined(__AVX2__)
        rows8(A[i], y[i], m->T, u0, u1);
#else
        rows8_ref(A[i], y[i], m->T, u0, u1);
#endif
      }
    }
  }
  #pragma omp barrier
  TS(prev);
}
static void gemv1(Model* m, const Mat* A, float* y, const float* x, int site){
  const Mat* a[1] = {A}; float* yy[1] = {y}; gemv(m, 1, a, yy, x, site);
}

/* rows8 for two tokens: one pass over the weights serves both, because the decode (vpbroadcastd, vpsrlvd, vpand) and
   the zero test (vpcmpeqd) depend only on the codes. Each token keeps its own accumulator chain and its own add order
   over g and p, so every output is the same bits as two rows8 passes - verified over all 151 matrices of the pack. */
#if defined(__AVX2__)
static void rows8_b2(const Mat* A, float* y0, float* y1, const float* T, const float* U, int b0, int b1){
  const int G = A->C / 128;
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const __m256i eight = _mm256_set1_epi32(8);
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; rb += 2){
    const int two = (rb + 1 < b1);
    const uint32_t* pw0 = P + ((size_t)rb * G) * 64;
    const uint32_t* pw1 = pw0 + (two ? (size_t)G * 64 : 0);
    const float* S0 = A->S + (size_t)rb * G * 8;
    const float* S1 = two ? S0 + (size_t)G * 8 : S0;
    __m256 A0 = _mm256_setzero_ps(), A1 = A0, B0 = A0, B1 = A0;
    for(int g = 0; g < G; ++g){
      const uint32_t* q0 = pw0 + (size_t)g * 64;
      const uint32_t* q1 = pw1 + (size_t)g * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      const float* pu = U + (size_t)g * 64 * 8;
      if(g + 4 < G){
        for(int k = 0; k < 64; k += 16) _mm_prefetch((const char*)(q0 + 256 + k), _MM_HINT_T0);
        if(two) for(int k = 0; k < 64; k += 16) _mm_prefetch((const char*)(q1 + 256 + k), _MM_HINT_T0);
      }
      __m256 p0a = _mm256_setzero_ps(), p0b = p0a, p1a = p0a, p1b = p0a;
      __m256 r0a = p0a, r0b = p0a, r1a = p0a, r1b = p0a;
      for(int p = 0; p < 64; p += 2){
        const __m256 t0 = _mm256_loadu_ps(pt + 8 * p);
        const __m256 t1 = _mm256_loadu_ps(pt + 8 * p + 8);
        const __m256 u0 = _mm256_loadu_ps(pu + 8 * p);
        const __m256 u1 = _mm256_loadu_ps(pu + 8 * p + 8);
        const __m256i c0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q0[p]), sh), f);
        const __m256i c1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q0[p + 1]), sh), f);
        const __m256 z0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(c0, eight));
        const __m256 z1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(c1, eight));
        p0a = _mm256_add_ps(p0a, _mm256_andnot_ps(z0, _mm256_permutevar8x32_ps(t0, c0)));
        p0b = _mm256_add_ps(p0b, _mm256_andnot_ps(z1, _mm256_permutevar8x32_ps(t1, c1)));
        r0a = _mm256_add_ps(r0a, _mm256_andnot_ps(z0, _mm256_permutevar8x32_ps(u0, c0)));
        r0b = _mm256_add_ps(r0b, _mm256_andnot_ps(z1, _mm256_permutevar8x32_ps(u1, c1)));
        if(two){
          const __m256i e0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q1[p]), sh), f);
          const __m256i e1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q1[p + 1]), sh), f);
          const __m256 w0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(e0, eight));
          const __m256 w1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(e1, eight));
          p1a = _mm256_add_ps(p1a, _mm256_andnot_ps(w0, _mm256_permutevar8x32_ps(t0, e0)));
          p1b = _mm256_add_ps(p1b, _mm256_andnot_ps(w1, _mm256_permutevar8x32_ps(t1, e1)));
          r1a = _mm256_add_ps(r1a, _mm256_andnot_ps(w0, _mm256_permutevar8x32_ps(u0, e0)));
          r1b = _mm256_add_ps(r1b, _mm256_andnot_ps(w1, _mm256_permutevar8x32_ps(u1, e1)));
        }
      }
      const __m256 s0 = _mm256_loadu_ps(S0 + (size_t)g * 8);
      A0 = _mm256_fmadd_ps(_mm256_add_ps(p0a, p0b), s0, A0);
      B0 = _mm256_fmadd_ps(_mm256_add_ps(r0a, r0b), s0, B0);
      if(two){
        const __m256 s1 = _mm256_loadu_ps(S1 + (size_t)g * 8);
        A1 = _mm256_fmadd_ps(_mm256_add_ps(p1a, p1b), s1, A1);
        B1 = _mm256_fmadd_ps(_mm256_add_ps(r1a, r1b), s1, B1);
      }
    }
    _mm256_storeu_ps(y0 + rb * 8, A0);
    _mm256_storeu_ps(y1 + rb * 8, B0);
    if(two){
      _mm256_storeu_ps(y0 + (rb + 1) * 8, A1);
      _mm256_storeu_ps(y1 + (rb + 1) * 8, B1);
    }
  }
}

#endif /* __AVX2__ */

/* inside the parallel region: n matrices, one weight stream for two tokens; ends in a barrier */
static void gemv2(Model* m, int n, const Mat* const* A, float* const* yA, float* const* yB, const float* xA,
                  const float* xB){
  const int prev = t_cur; TS(2);
  int a, b;
  span(A[0]->C / 2, &a, &b);
  pair_tables(xA, m->T, a, b);
  pair_tables(xB, m->T2, a, b);
  #pragma omp barrier
  long long Ut = 0;
  for(int i = 0; i < n; ++i) Ut += A[i]->R / 8;
  int a0, a1;
  span(Ut, &a0, &a1);
  int base = 0;
  for(int i = 0; i < n; ++i){
    const int U = A[i]->R / 8;
    const int u0 = a0 > base ? a0 - base : 0, u1 = (a1 - base) < U ? (a1 - base) : U;
    base += U;
    if(u1 > u0){
      if(tier() == KCT_SCALAR) rows8_b2_ref(A[i], yA[i], yB[i], m->T, m->T2, u0, u1);
      else {
#if defined(__AVX2__)
        rows8_b2(A[i], yA[i], yB[i], m->T, m->T2, u0, u1);
#else
        rows8_b2_ref(A[i], yA[i], yB[i], m->T, m->T2, u0, u1);
#endif
      }
    }
  }
  #pragma omp barrier
  TS(prev);
}

/* ----------------------------------------------------------- the vector LUT: one lookup for N tokens
   `rows8_b2` spends one vpermps per token per pair position, so a lookup returns 8 rows x 2 columns = 16 weights no
   matter how many tokens are in flight. The form below makes the table entry itself carry the whole token batch:

       TT[p][c] = (int16: token A's sum for pair code c) | (int16: token B's sum) << 16

   Eight codes of int16 pairs is eight int32 - exactly one vpermd source - so one lookup fetches 8 rows x 2 columns x 2
   tokens = 32 weights, twice the fp32 form. At N = 4 two tables (A,B) and (C,D) are permuted per pair position, so 64
   weights over 2 lookups: still 32 a lookup, and the codes are decoded once for four tokens instead of once per token.

   This is an int8 kernel, not an exact one: an entry is an int16 partial sum of int8 activations, so the values differ
   from the fp32 tables the production kernel builds - the same trade rows8_q8 makes. int16 is enough for one 128-column
   group (64 pairs of two int8 values, each pair at most 254), and the group's activation scale multiplies in when the
   sum is widened to int32 and fp32.

   The table is built per group into 2 KiB of stack and consumed by every row block before the next group replaces it.
   That is the tiling the Vec-LUT paper (arXiv 2512.06443) reaches from the other end - its entry is an N-element int16
   token vector selected by a scalar index, this one is a per-row int16 vector selected by vpermd, and both win by
   making a lookup's output wider than one token. It costs a read and a write of y per group, and reads the table once
   per row block.

   Measurement surface only: kc_vlut_gemv and kc_b2_gemv are exports for tools/vlut. Neither is wired into the forward
   pass, and the forward pass does not change. */
KCAPI int kc_vlut_ok(void){
#if defined(__AVX2__)
  return 1;
#else
  return 0;
#endif
}

/* int8 activations in the vector LUT's form: one fp32 scale per 128-column group, the granularity of the row scale, so
   a group is where the int16 sum ends and the fp32 multiply starts. Same arithmetic as quant_x, written serially: this
   entry is called outside a parallel region and quant_x's `#pragma omp for` is only well defined inside one. */
KCAPI void kc_q8_quant(const float* x, int C, signed char* Q, float* QS){
  const int GS = 128;
  for(int g = 0; g < C / GS; ++g){
    const float* xg = x + (size_t)g * GS;
    float mx = 0.0f;
    for(int i = 0; i < GS; ++i){ const float av = fabsf(xg[i]); if(av > mx) mx = av; }
    const float s = mx > 0.0f ? mx / 127.0f : 1.0f;
    QS[g] = s;
    for(int i = 0; i < GS; ++i){
      long v = lrintf(xg[i] / s);
      if(v > 127) v = 127; else if(v < -127) v = -127;
      Q[(size_t)g * GS + i] = (signed char)v;
    }
  }
}

/* the production two-token kernel, single-threaded over every row block, so the comparison is the same arithmetic on
   the same pack with none of the thread pool. gemv2 builds the pair tables and then calls rows8_b2, and both steps are
   inside this call, exactly as the vector LUT's table build is inside its own. */
KCAPI void kc_b2_gemv(const Mat* A, int nt, const float* const* X, float* const* Y, float* T, float* U){
  const int P = A->C / 2;
  for(int t = 0; t + 1 < nt; t += 2){
    pair_tables(X[t], T, 0, P);
    pair_tables(X[t + 1], U, 0, P);
#if defined(__AVX2__)
    rows8_b2(A, Y[t], Y[t + 1], T, U, 0, A->R / 8);
#else
    rows8_b2_ref(A, Y[t], Y[t + 1], T, U, 0, A->R / 8);
#endif
  }
}

#if defined(__AVX2__)
/* the eight pair codes' two weight signs as int16 lanes: code c of the pack is the pair (even column, odd column), the
   same map pair_tables writes t[0..7] from */
static const short VLUT_EVEN[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
static const short VLUT_ODD[8]  = {-1, -1, -1, 0, 0, 1, 1, 1};

/* one group's table for one token pair, 64 pairs x 8 codes of (A,B) int16. `p` is the pair position inside the group
   and `c` the code, so the layout is TT[8*p + c] and a vpermd over it is indexed by the eight rows' codes. */
static void vlut_table(const signed char* xA, const signed char* xB, int32_t* TT){
  const __m128i ve = _mm_loadu_si128((const __m128i*)VLUT_EVEN);
  const __m128i vo = _mm_loadu_si128((const __m128i*)VLUT_ODD);
  for(int p = 0; p < 64; ++p){
    const __m128i ea = _mm_add_epi16(_mm_mullo_epi16(ve, _mm_set1_epi16((short)xA[2 * p])),
                                     _mm_mullo_epi16(vo, _mm_set1_epi16((short)xA[2 * p + 1])));
    const __m128i eb = _mm_add_epi16(_mm_mullo_epi16(ve, _mm_set1_epi16((short)xB[2 * p])),
                                     _mm_mullo_epi16(vo, _mm_set1_epi16((short)xB[2 * p + 1])));
    _mm_storeu_si128((__m128i*)(TT + 8 * p), _mm_unpacklo_epi16(ea, eb));         /* codes 0..3, A in the low half */
    _mm_storeu_si128((__m128i*)(TT + 8 * p + 4), _mm_unpackhi_epi16(ea, eb));     /* codes 4..7 */
  }
}

/* the two int16 halves of an accumulator lane as sign-extended int32: token A in the low half, B in the high */
#define VLUT_LO(v) _mm256_srai_epi32(_mm256_slli_epi32((v), 16), 16)
#define VLUT_HI(v) _mm256_srai_epi32((v), 16)

static void vlut2(const Mat* A, const signed char* xA, const signed char* xB, const float* qsA, const float* qsB,
                  float* yA, float* yB){
  const int G = A->C / 128, U = A->R / 8;
  const uint32_t* P = (const uint32_t*)A->W;
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const __m256i eight = _mm256_set1_epi32(8);
  int32_t TT[64 * 8];
  for(int g = 0; g < G; ++g){
    vlut_table(xA + (size_t)g * 128, xB + (size_t)g * 128, TT);
    const float qa = qsA[g], qb = qsB[g];
    for(int u = 0; u < U; ++u){
      const uint32_t* q = P + ((size_t)u * G + g) * 64;
      if(g + 4 < G) for(int k = 0; k < 64; k += 16) _mm_prefetch((const char*)(q + 256 + k), _MM_HINT_T0);
      __m256i acc = _mm256_setzero_si256();
      for(int p = 0; p < 64; ++p){
        const __m256i c = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q[p]), sh), f);
        const __m256i z = _mm256_cmpeq_epi32(c, eight);              /* code 8 is the pair 0*0: no contribution */
        const __m256i v = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i*)(TT + 8 * p)), c);
        acc = _mm256_add_epi16(acc, _mm256_andnot_si256(z, v));
      }
      const __m256 s = _mm256_loadu_ps(A->S + ((size_t)u * G + g) * 8);
      __m256 X0 = g ? _mm256_loadu_ps(yA + u * 8) : _mm256_setzero_ps();
      __m256 X1 = g ? _mm256_loadu_ps(yB + u * 8) : _mm256_setzero_ps();
      X0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(VLUT_LO(acc)), _mm256_mul_ps(_mm256_set1_ps(qa), s), X0);
      X1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(VLUT_HI(acc)), _mm256_mul_ps(_mm256_set1_ps(qb), s), X1);
      _mm256_storeu_ps(yA + u * 8, X0);
      _mm256_storeu_ps(yB + u * 8, X1);
    }
  }
}

static void vlut4(const Mat* A, const signed char* const* X, const float* const* QS, float* const* Y){
  const int G = A->C / 128, U = A->R / 8;
  const uint32_t* P = (const uint32_t*)A->W;
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const __m256i eight = _mm256_set1_epi32(8);
  int32_t T0[64 * 8], T1[64 * 8];
  for(int g = 0; g < G; ++g){
    vlut_table(X[0] + (size_t)g * 128, X[1] + (size_t)g * 128, T0);
    vlut_table(X[2] + (size_t)g * 128, X[3] + (size_t)g * 128, T1);
    const __m256 q0 = _mm256_set1_ps(QS[0][g]), q1 = _mm256_set1_ps(QS[1][g]);
    const __m256 q2 = _mm256_set1_ps(QS[2][g]), q3 = _mm256_set1_ps(QS[3][g]);
    for(int u = 0; u < U; ++u){
      const uint32_t* q = P + ((size_t)u * G + g) * 64;
      if(g + 4 < G) for(int k = 0; k < 64; k += 16) _mm_prefetch((const char*)(q + 256 + k), _MM_HINT_T0);
      __m256i ab = _mm256_setzero_si256(), cd = _mm256_setzero_si256();
      for(int p = 0; p < 64; ++p){
        const __m256i c = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)q[p]), sh), f);
        const __m256i z = _mm256_cmpeq_epi32(c, eight);
        const __m256i v0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i*)(T0 + 8 * p)), c);
        const __m256i v1 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i*)(T1 + 8 * p)), c);
        ab = _mm256_add_epi16(ab, _mm256_andnot_si256(z, v0));
        cd = _mm256_add_epi16(cd, _mm256_andnot_si256(z, v1));
      }
      const __m256 s = _mm256_loadu_ps(A->S + ((size_t)u * G + g) * 8);
      __m256 Y0 = g ? _mm256_loadu_ps(Y[0] + u * 8) : _mm256_setzero_ps();
      __m256 Y1 = g ? _mm256_loadu_ps(Y[1] + u * 8) : _mm256_setzero_ps();
      __m256 Y2 = g ? _mm256_loadu_ps(Y[2] + u * 8) : _mm256_setzero_ps();
      __m256 Y3 = g ? _mm256_loadu_ps(Y[3] + u * 8) : _mm256_setzero_ps();
      Y0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(VLUT_LO(ab)), _mm256_mul_ps(q0, s), Y0);
      Y1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(VLUT_HI(ab)), _mm256_mul_ps(q1, s), Y1);
      Y2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(VLUT_LO(cd)), _mm256_mul_ps(q2, s), Y2);
      Y3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(VLUT_HI(cd)), _mm256_mul_ps(q3, s), Y3);
      _mm256_storeu_ps(Y[0] + u * 8, Y0);
      _mm256_storeu_ps(Y[1] + u * 8, Y1);
      _mm256_storeu_ps(Y[2] + u * 8, Y2);
      _mm256_storeu_ps(Y[3] + u * 8, Y3);
    }
  }
}
#endif /* __AVX2__ */

KCAPI int kc_vlut_gemv(const Mat* A, int nt, const signed char* const* X, const float* const* QS, float* const* Y){
#if defined(__AVX2__)
  if(A->C % 128 || A->R % 8) return -2;
  if(nt == 2){ vlut2(A, X[0], X[1], QS[0], QS[1], Y[0], Y[1]); return 0; }
  if(nt == 4){ vlut4(A, X, QS, Y); return 0; }
  (void)Y;
  return -1;
#else
  (void)A; (void)nt; (void)X; (void)QS; (void)Y;
  return -1;
#endif
}

/* --------------------------------------------------------------------------------- activation dump
   Diagnostic, off unless KINDI_CPU_DUMP names a file: the FFN input of every layer is appended once per layer
   per token as 2*D floats - the normalised residual m->xn, then its fold m->xp, the exact vector the ternary GEMV
   consumes. The layer loop in kc_step runs in order for every thread, so the rows of one token are its layers in
   order and a host that steps tokens one at a time gets token-major, layer-minor rows. KINDI_CPU_DUMP_MAX bounds the
   row count (default one million). With the variable unset this is a NULL FILE* test in one place and nothing else:
   no scratch, no arithmetic, no branch in any kernel. */
static FILE* g_dump_f = NULL;
static long g_dump_rows = 0, g_dump_max = 0;
static void dump_open(void){
  const char* p = getenv("KINDI_CPU_DUMP");
  if(g_dump_f || !p || !p[0]) return;
  g_dump_f = fopen(p, "wb");
  { const char* mx = getenv("KINDI_CPU_DUMP_MAX"); g_dump_max = mx ? atol(mx) : 1000000L; }
}
static void dump_row(Model* m){
  #pragma omp barrier
  if(omp_get_thread_num() == 0 && g_dump_rows < g_dump_max){
    fwrite(m->xn, sizeof(float), (size_t)m->D, g_dump_f);
    fwrite(m->xp, sizeof(float), (size_t)m->D, g_dump_f);
    ++g_dump_rows;
  }
  #pragma omp barrier
}

/* ------------------------------------------------------------------------------------------- norms */
/* residual add (if add) + RMS norm + fold of m->h into xn / xp; all threads; ends in a barrier */
static void rmsfold(Model* m, const float* w, int add){
  const int D = m->D;
  const int prev = t_cur; TS(1);
  #pragma omp single
  {
    double acc = 0.0;
    for(int e = 0; e < D; ++e){
      const float xf = add ? m->h[e] + m->y[e] : m->h[e];
      m->h[e] = xf;
      acc += (double)xf * (double)xf;
    }
    const float s = 1.0f / sqrtf((float)(acc / D) + 1e-6f);
    for(int e = 0; e < D; ++e) m->xn[e] = (m->h[e] * s) * w[e];
  }
  fold(m, m->xn, m->sD, m->xp, D);
  TS(prev);
}

static void rms_rows(const float* x, const float* w, float* out, int rows, int n){
  for(int r = 0; r < rows; ++r){
    const float* xr = x + (size_t)r * n;
    double acc = 0.0;
    for(int e = 0; e < n; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
    for(int e = 0; e < n; ++e) out[(size_t)r * n + e] = (xr[e] * sc) * w[e];
  }
}

/* ------------------------------------------------------------------------------------------- blocks */
/* the part of the GDN block between its two products: gates, conv, delta rule, gated output and fold. Split out so a
   two-token pass can batch both products and still step the recurrence in token order. */
static void gdn_mid(Model* m, Layer* L){
  const int D = m->D, NV = m->NV, NK = m->NK, HK = m->HK, CC = m->CONV_CH;
  int h, c, r, e;
  /* gates (per v head) and the causal conv (per channel) */
  #pragma omp for schedule(static) nowait
  for(h = 0; h < NV; ++h){
    float ab = 0.f, aa = 0.f;
    for(int i = 0; i < D; ++i){ ab += L->beta_w[(size_t)h * D + i] * m->xn[i]; aa += L->alpha_w[(size_t)h * D + i] * m->xn[i]; }
    m->gb[h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    m->gg[h] = sp * L->ssm_a[h];
  }
  #pragma omp for schedule(static)
  for(c = 0; c < CC; ++c){
    float* cv = L->conv;
    const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = m->qkv[c];
    const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
    cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
    m->cs[c] = silu(o);
  }
  /* L2 norm of the q and k heads */
  #pragma omp for schedule(static)
  for(r = 0; r < 2 * NK; ++r){
    const float* xr = m->cs + (size_t)r * HK;
    double acc = 0.0;
    for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    for(int e = 0; e < HK; ++e) m->qk[(size_t)r * HK + e] = xr[e] * sc;
  }
  /* delta rule per v head, then the per-head RMS norm */
  const float scale = 1.0f / sqrtf((float)HK);
  #pragma omp for schedule(static)
  for(h = 0; h < NV; ++h){
    const int kh = h % NK;
    const float* ks = m->qk + (size_t)(NK + kh) * HK;
    const float* qs = m->qk + (size_t)kh * HK;
    const float* v = m->cs + 2 * NK * HK + (size_t)h * HK;
    const float e = expf(m->gg[h]);
    float* Mh = L->M + (size_t)h * HK * HK;
    float sk[128], o[128], dd[128];
    for(int j = 0; j < HK; ++j) sk[j] = 0.f;
    for(int i = 0; i < HK; ++i){ const float ki = ks[i]; const float* mr = Mh + i * HK; for(int j = 0; j < HK; ++j) sk[j] += (mr[j] * e) * ki; }
    for(int j = 0; j < HK; ++j){ dd[j] = (v[j] - sk[j]) * m->gb[h]; o[j] = 0.f; }
    for(int i = 0; i < HK; ++i){
      const float ki = ks[i], qi = qs[i];
      float* mr = Mh + i * HK;
      for(int j = 0; j < HK; ++j){ const float mm = mr[j] * e + ki * dd[j]; mr[j] = mm; o[j] += mm * qi; }
    }
    for(int j = 0; j < HK; ++j) m->gout[(size_t)h * HK + j] = o[j] * scale;
    rms_rows(m->gout + (size_t)h * HK, L->ssm_norm, m->outn + (size_t)h * HK, 1, HK);
  }
  /* gated output in the grouped v-head order, then the fold */
  const int rep = NV / NK;
  #pragma omp for schedule(static)
  for(e = 0; e < NV * HK; ++e){
    int src = e;
    if(rep > 1){ const int hd = e % HK, grp = e / HK; src = hd + HK * (grp / rep + NK * (grp % rep)); }
    m->m[e] = silu(m->zv[src]) * m->outn[src];
  }
  fold(m, m->m, m->sV, m->xp, NV * HK);
}

static void gdn(Model* m, Layer* L, int add){
  TS(3);
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[2] = {&L->qkv, &L->gate}; float* yy[2] = {m->qkv, m->zv}; gemv(m, 2, a, yy, m->xp, KCAL_QKV); }
  gdn_mid(m, L);
  gemv1(m, &L->ssm_out, m->y, m->xp, KCAL_SSM);
}

/* ------------------------------------------------------------------------------------ quantised K/V cache
   The K/V cache is a storage format, not an arithmetic one: whenever a path needs a head's 256 values it materialises
   them into fp32 and runs the same dot products, so the quantisation changes the values and nothing else. Ternary
   codes are 0 = -1, 1 = 0, 2 = +1 (3 is unused and reads as 0), four to a byte. The scale is the least-squares
   optimum for the code it induces - start at absmean, assign, then alpha = <v,c>/<c,c> - because the value that
   matters is the reconstruction error of the head, not an extremum. q8_0 is ggml's layout: 32 signed bytes with an
   fp16 scale of amax/127, which is also a per-32-block round trip. */
static const float KT[4] = {-1.0f, 0.0f, 1.0f, 0.0f};

static float tern_pack(const float* v, uint8_t* out, int n){
  int c[512];
  float a = 0.f;
  for(int j = 0; j < n; ++j) a += fabsf(v[j]);
  a /= (float)n;
  if(!(a > 0.f)) a = 1e-8f;
  /* assign at |v| > a/2, then alpha = <v,c>/<c,c>, until alpha stops moving. Three steps of this from mean|v|
     stopped 1.4% short of the grid optimum on real cache values (relL2 0.4921 against 0.4853 per head), and the
     cost is a few compares per cache write, so it runs to convergence instead of to a fixed count. */
  for(int it = 0; it < 12; ++it){
    float num = 0.f; int cn = 0;
    for(int j = 0; j < n; ++j){
      const float h = 0.5f * a;
      const int cj = v[j] > h ? 1 : (v[j] < -h ? -1 : 0);
      c[j] = cj; num += v[j] * (float)cj; cn += cj ? 1 : 0;
    }
    const float na = (cn && num > 0.f) ? num / (float)cn : a;
    const int settled = it >= 2 && fabsf(na - a) <= 1e-3f * a;
    a = na;
    if(!(a > 0.f)) a = 1e-8f;
    if(settled) break;
  }
  /* the codes that go with the scale returned, not the ones from the iteration before it */
  for(int j = 0; j < n; ++j) c[j] = v[j] > 0.5f * a ? 1 : (v[j] < -0.5f * a ? -1 : 0);
  for(int j = 0; j < n; j += 4)
    out[j >> 2] = (uint8_t)((c[j] + 1) | ((c[j + 1] + 1) << 2) | ((c[j + 2] + 1) << 4) | ((c[j + 3] + 1) << 6));
  return a;
}
static void tern_unpack(const uint8_t* p, float sc, float* out, int n){
  for(int j = 0; j < n; ++j) out[j] = sc * KT[(p[j >> 2] >> (2 * (j & 3))) & 3];
}
/* 2-bit uniform: four levels {-3,-1,+1,+3}*a, four codes a byte, one scale per (position, head). Ternary spends the
   same two bits on three levels and drives everything inside its dead zone to zero; on 372 real cache vectors the
   4-level alphabet reconstructs 16% closer on K and 8% closer on V, so this is what "2 bits" should mean.

   The scale is chosen on the objective, not by the path: the assignment map is piecewise constant in a, so iterating
   from one start is bistable - a one-ulp difference in the input can settle in a different basin, and on a
   position-0 head the two basins measured 0.63 and 0.79 relL2. Four starts (three fractions of amax and the mean) are
   each iterated to a fixed point and the candidate with the lowest squared error wins; that costs ~10k compares per
   cache write, against ~10^8 multiply-adds per token. */
static const float LT[4] = {-3.0f, -1.0f, 1.0f, 3.0f};
static float q2_pack(const float* v, uint8_t* out, int n){
  int bc[512], c[512];
  float am = 0.f, mn = 0.f;
  for(int j = 0; j < n; ++j){ const float f = fabsf(v[j]); if(f > am) am = f; mn += f; }
  const float starts[4] = {am / 6.0f, am / 9.0f, am / 12.0f, 0.5f * mn / (float)n};
  float ba = 1e-8f, best = -1.f;
  for(int s = 0; s < 4; ++s){
    float a = starts[s] > 0.f ? starts[s] : 1e-8f;
    for(int it = 0; it < 10; ++it){
      float num = 0.f, den = 0.f;
      const float b = 2.0f * a;            /* decisions at 0 and +-2a: the midpoints between the levels */
      for(int j = 0; j < n; ++j){
        const float x = v[j];
        const int k = x >= b ? 3 : (x > 0.f ? 2 : (x > -b ? 1 : 0));
        num += x * LT[k]; den += LT[k] * LT[k];
      }
      const float na = den > 0.f ? num / den : a;
      if(!(na > 0.f)) break;
      a = na;
    }
    /* the codes that go with the settled scale, and that scale's error */
    const float b = 2.0f * a;
    float sse = 0.f;
    for(int j = 0; j < n; ++j){
      const float x = v[j];
      const int k = x >= b ? 3 : (x > 0.f ? 2 : (x > -b ? 1 : 0));
      const float d = x - a * LT[k];
      c[j] = k; sse += d * d;
    }
    if(best < 0.f || sse < best){ best = sse; ba = a; for(int j = 0; j < n; ++j) bc[j] = c[j]; }
  }
  for(int j = 0; j < n; j += 4)
    out[j >> 2] = (uint8_t)(bc[j] | (bc[j + 1] << 2) | (bc[j + 2] << 4) | (bc[j + 3] << 6));
  return ba;
}
static void q2_unpack(const uint8_t* p, float sc, float* out, int n){
  for(int j = 0; j < n; ++j) out[j] = sc * LT[(p[j >> 2] >> (2 * (j & 3))) & 3];
}
/* 2 bits with a scale per 32-value block instead of per head: 8 bytes of codes and one fp16 scale a block, so 2.5 bits
   a value against 2.0625 for mode 4. The question it answers is whether the four-level alphabet is the limit on the
   values or the single scale is. Same four-start fit as q2_pack, run per block; a block whose scale came out as zero
   keeps a zero scale and all-zero codes, which reconstructs exactly. */
static void q2b_pack(const float* v, uint8_t* out, uint16_t* sc, int n){
  for(int b = 0; b < n / 32; ++b){
    const float* x = v + b * 32;
    float am = 0.f, mn = 0.f;
    for(int j = 0; j < 32; ++j){ const float f = fabsf(x[j]); if(f > am) am = f; mn += f; }
    const float starts[4] = {am / 6.0f, am / 9.0f, am / 12.0f, 0.5f * mn / 32.0f};
    float ba = 1e-8f, best = -1.f;
    int bc[32], c[32];
    for(int s = 0; s < 4; ++s){
      float a = starts[s] > 0.f ? starts[s] : 1e-8f;
      for(int it = 0; it < 10; ++it){
        float num = 0.f, den = 0.f;
        const float bb = 2.0f * a;
        for(int j = 0; j < 32; ++j){
          const float y = x[j];
          const int k = y >= bb ? 3 : (y > 0.f ? 2 : (y > -bb ? 1 : 0));
          num += y * LT[k]; den += LT[k] * LT[k];
        }
        const float na = den > 0.f ? num / den : a;
        if(!(na > 0.f)) break;
        a = na;
      }
      const float bb = 2.0f * a;
      float sse = 0.f;
      for(int j = 0; j < 32; ++j){
        const float y = x[j];
        const int k = y >= bb ? 3 : (y > 0.f ? 2 : (y > -bb ? 1 : 0));
        const float d = y - a * LT[k];
        c[j] = k; sse += d * d;
      }
      if(best < 0.f || sse < best){ best = sse; ba = a; for(int j = 0; j < 32; ++j) bc[j] = c[j]; }
    }
    uint8_t* o = out + b * 8;
    for(int j = 0; j < 32; j += 4)
      o[j >> 2] = (uint8_t)(bc[j] | (bc[j + 1] << 2) | (bc[j + 2] << 4) | (bc[j + 3] << 6));
    sc[b] = f2h(ba);
  }
}
static void q2b_unpack(const uint8_t* p, const uint16_t* sc, float* out, int n){
  for(int b = 0; b < n / 32; ++b){
    const float s = h2f(sc[b]);
    const uint8_t* o = p + b * 8;
    for(int j = 0; j < 32; ++j) out[b * 32 + j] = s * LT[(o[j >> 2] >> (2 * (j & 3))) & 3];
  }
}
static void q8_pack(const float* v, int8_t* out, uint16_t* sc, int n){
  for(int b = 0; b < n / 32; ++b){
    float am = 0.f;
    for(int j = 0; j < 32; ++j){ const float f = fabsf(v[b * 32 + j]); if(f > am) am = f; }
    const float d = am / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.f;
    sc[b] = f2h(d);
    for(int j = 0; j < 32; ++j){
      float q = nearbyintf(v[b * 32 + j] * id);
      if(q > 127.f) q = 127.f; else if(q < -127.f) q = -127.f;
      out[b * 32 + j] = (int8_t)q;
    }
  }
}
/* 4-bit uniform: two signed nibbles a byte, one scale per head, 16 levels from -8 to 7. This is the rung the BitNet
   papers call free, so the scale is worth fitting: amax/7 (what this shipped with) leaves 16% of the reconstruction
   error on the table against the least-squares scale, because a packed head's maximum is a poor estimate of the step
   the rest of the vector wants. Same four starts and same error test as q2_pack. */
static float q4_pack(const float* v, uint8_t* out, int n){
  int bc[512], c[512];
  float am = 0.f, mn = 0.f;
  for(int j = 0; j < n; ++j){ const float f = fabsf(v[j]); if(f > am) am = f; mn += f; }
  const float starts[4] = {am / 7.0f, am / 9.0f, am / 12.0f, 0.5f * mn / (float)n};
  float ba = 1e-8f, best = -1.f;
  for(int s = 0; s < 4; ++s){
    float a = starts[s] > 0.f ? starts[s] : 1e-8f;
    for(int it = 0; it < 10; ++it){
      float num = 0.f, den = 0.f;
      const float ia = 1.0f / a;
      for(int j = 0; j < n; ++j){
        float q = nearbyintf(v[j] * ia);
        if(q > 7.f) q = 7.f; else if(q < -8.f) q = -8.f;
        num += v[j] * q; den += q * q;
      }
      const float na = den > 0.f ? num / den : a;
      if(!(na > 0.f)) break;
      a = na;
    }
    const float ia = 1.0f / a;
    float sse = 0.f;
    for(int j = 0; j < n; ++j){
      float q = nearbyintf(v[j] * ia);
      if(q > 7.f) q = 7.f; else if(q < -8.f) q = -8.f;
      const int qi = (int)q;
      const float d = v[j] - a * q;
      c[j] = qi; sse += d * d;
    }
    if(best < 0.f || sse < best){ best = sse; ba = a; for(int j = 0; j < n; ++j) bc[j] = c[j]; }
  }
  for(int j = 0; j < n; j += 2)
    out[j >> 1] = (uint8_t)((bc[j] & 15) | ((bc[j + 1] & 15) << 4));
  return ba;
}
/* 4 bits as two 4-level passes: v = a*(L[k1] + 0.25*L[k2]) with L = {-3,-1,1,3}, so the 2-bit alphabet is the codec
   and the second pass is a residue on it. The 16 levels come out even (0.25a apart from -3.75a to 3.75a) and one of
   the two codes can be read as a 4-entry table, but the level set is not what pays: the fitted scale is. Same start
   set and error test as q2_pack, and the same buffer size as mode 3 - four bits a value either way. */
static const uint8_t Q5C[16] = {          /* sorted level -8..7 -> packed code (k1 | k2 << 2) */
  0, 4, 8, 12,        /* L1=-3: -3.75 -3.25 -2.75 -2.25 */
  1, 5, 9, 13,        /* L1=-1: -1.75 -1.25 -0.75 -0.25 */
  2, 6, 10, 14,       /* L1=+1: +0.25 +0.75 +1.25 +1.75 */
  3, 7, 11, 15};      /* L1=+3: +2.25 +2.75 +3.25 +3.75 */
static float q5_pack(const float* v, uint8_t* out, int n){
  int bc[512], c[512];
  float am = 0.f, mn = 0.f;
  for(int j = 0; j < n; ++j){ const float f = fabsf(v[j]); if(f > am) am = f; mn += f; }
  const float starts[4] = {am / 3.75f, am / 5.0f, am / 7.0f, 0.35f * mn / (float)n};
  float ba = 1e-8f, best = -1.f;
  for(int s = 0; s < 4; ++s){
    float a = starts[s] > 0.f ? starts[s] : 1e-8f;
    for(int it = 0; it < 10; ++it){
      float num = 0.f, den = 0.f;
      const float ia = 1.0f / a;
      for(int j = 0; j < n; ++j){
        int m = (int)lrintf(2.0f * v[j] * ia - 0.5f);       /* levels at (2m+1)/4 of a */
        if(m > 7) m = 7; else if(m < -8) m = -8;
        const float lv = 0.25f * (float)(2 * m + 1);
        num += v[j] * lv; den += lv * lv;
      }
      const float na = den > 0.f ? num / den : a;
      if(!(na > 0.f)) break;
      a = na;
    }
    const float ia = 1.0f / a;
    float sse = 0.f;
    for(int j = 0; j < n; ++j){
      int m = (int)lrintf(2.0f * v[j] * ia - 0.5f);
      if(m > 7) m = 7; else if(m < -8) m = -8;
      const float d = v[j] - a * 0.25f * (float)(2 * m + 1);
      c[j] = m; sse += d * d;
    }
    if(best < 0.f || sse < best){ best = sse; ba = a; for(int j = 0; j < n; ++j) bc[j] = c[j]; }
  }
  for(int j = 0; j < n; j += 2)
    out[j >> 1] = (uint8_t)(Q5C[bc[j] + 8] | (Q5C[bc[j + 1] + 8] << 4));
  return ba;
}
static const float LT5[16] = {            /* indexed by the packed code, not by the sorted level */
  -3.75f, -1.75f, 0.25f, 2.25f,
  -3.25f, -1.25f, 0.75f, 2.75f,
  -2.75f, -0.75f, 1.25f, 3.25f,
  -2.25f, -0.25f, 1.75f, 3.75f};
static void q5_unpack(const uint8_t* p, float sc, float* out, int n){
  for(int j = 0; j < n; ++j) out[j] = sc * LT5[(p[j >> 1] >> (4 * (j & 1))) & 15];
}
static void q4_unpack(const uint8_t* p, float sc, float* out, int n){
  for(int j = 0; j < n; ++j){
    const int q = (p[j >> 1] >> (4 * (j & 1))) & 15;
    out[j] = sc * (float)(q < 8 ? q : q - 16);
  }
}
static void q8_unpack(const int8_t* p, const uint16_t* sc, float* out, int n){
  for(int b = 0; b < n / 32; ++b){
    const float d = h2f(sc[b]);
    for(int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float)p[b * 32 + j];
  }
}

/* --------------------------------------------------------------------------------- V capture
   Diagnostic, off unless KINDI_CPU_VCAP names a file: every V vector as the cache is about to quantise it - the fp32
   `m->vv` of each (layer, position, head), before any codec touches it. This is the input the K/V-cache question is
   asked about, and it is the one thing a synthetic pack cannot supply: `kindi.synthpack`'s weights are random, so a
   codec measured on it is measured on the wrong distribution.

   The value does not depend on KINDI_CPU_KVQ, so one capture serves every format: the same file is the input to `q2`
   against `q2b`, to q8_0, and to anything added later. Records are self-describing (layer, position, head) because the
   writes happen inside a parallel loop and their order is the scheduler's, not the model's. Writing is one `critical`
   around an HD-float record per (position, head) per attention layer, so the hook costs one branch anywhere but there
   and nothing at all when the variable is unset. KINDI_CPU_VCAP_MAX bounds the record count (default 65536, ~134 MB of
   fp32 at head_dim 512; the reader takes a prefix without complaint). */
#define VCAP_MAGIC 0x4B564331          /* "KVC1" */
typedef struct { int32_t magic, nl, natt, nhkv, hd; } VcapHead;     /* natt: the attention layers, for a reader's check */
static FILE* g_vcap_f = NULL;
static long long g_vcap_n = 0, g_vcap_max = 0;

/* Arm a capture explicitly. The environment variable below is the shell's convenience, but it is not enough on the
   MSVC build: that DLL links the CRT statically (/MT), so its getenv reads a private copy of the environment taken at
   load time, and a host that sets os.environ *after* loading the library - which is what a test does - changes nothing
   it can see. An explicit entry point is the same code on every build and does not depend on load order. */
KCAPI int kc_vcap_begin(Model* m, const char* path, long long limit){
  if(g_vcap_f || !path || !path[0]) return -1;
  g_vcap_f = fopen(path, "wb");
  if(!g_vcap_f) return -2;
  g_vcap_n = 0;
  g_vcap_max = limit > 0 ? limit : 65536;
  int natt = 0;
  for(int il = 0; il < m->NL; ++il) natt += m->L[il].is_attn ? 1 : 0;
  const VcapHead h = {VCAP_MAGIC, m->NL, natt, m->NHKV, m->HD};
  if(fwrite(&h, sizeof h, 1, g_vcap_f) != 1){ fclose(g_vcap_f); g_vcap_f = NULL; return -3; }
  return 0;
}
static void vcap_open(Model* m){
  const char* p = getenv("KINDI_CPU_VCAP");
  if(!g_vcap_f && p && p[0]){
    const char* mx = getenv("KINDI_CPU_VCAP_MAX");
    kc_vcap_begin(m, p, mx ? atol(mx) : 0);
  }
}
static void vcap_vec(int il, int pos, int kv, const float* v, int hd){
  if(!g_vcap_f || g_vcap_n >= g_vcap_max) return;
  #pragma omp critical(kc_vcap)
  {
    if(g_vcap_n < g_vcap_max){
      const int32_t k[4] = {il, pos, kv, hd};
      fwrite(k, sizeof k, 1, g_vcap_f);
      fwrite(v, sizeof(float), (size_t)hd, g_vcap_f);
      ++g_vcap_n;
    }
  }
}
KCAPI void kc_vcap_close(void){ if(g_vcap_f){ fclose(g_vcap_f); g_vcap_f = NULL; } }
KCAPI long long kc_vcap_count(void){ return g_vcap_n; }

/* the part of the attention block between its two products: RoPE, the K/V cache, scores, softmax, the weighted V */
static void attn_mid(Model* m, Layer* L, int pos, int il){
  const int NH = m->NH, NKV = m->NHKV, HD = m->HD, rh = m->rope_half, G = NH / NKV, n = pos + 1;
  const float* cs = m->cos_t + (size_t)pos * rh;
  const float* sn = m->sin_t + (size_t)pos * rh;
  int r, t, hh, u;
  #pragma omp for schedule(static)
  for(r = 0; r < NH + NKV; ++r){
    float tmp[512];
    float* dst;
    if(r < NH){ rms_rows(m->qn + (size_t)r * 2 * HD, L->q_norm, tmp, 1, HD); dst = m->qr + (size_t)r * HD; }
    else { rms_rows(m->kr + (size_t)(r - NH) * HD, L->k_norm, tmp, 1, HD); dst = m->kro + (size_t)(r - NH) * HD; }
    for(int j = 0; j < rh; ++j){
      const float a = tmp[j], b = tmp[j + rh];
      dst[j] = a * cs[j] - b * sn[j];
      dst[j + rh] = a * sn[j] + b * cs[j];
    }
    for(int j = 2 * rh; j < HD; ++j) dst[j] = tmp[j];
    if(r >= NH){
      const int kv = r - NH;
      const float* vv = m->vv + (size_t)kv * HD;
      const size_t ix = (size_t)pos * NKV + kv;
      vcap_vec(il, pos, kv, vv, HD);
      if(m->kvq_k == 1) L->Ks[ix] = f2h(tern_pack(dst, L->Kt + ix * (HD / 4), HD));
      else if(m->kvq_k == 6) q2b_pack(dst, L->K2b + ix * (HD / 4), L->K2bs + ix * (HD / 32), HD);
      else if(m->kvq_k == 2) q8_pack(dst, L->Kq + ix * HD, L->Kqs + ix * (HD / 32), HD);
      else if(m->kvq_k == 3) L->K4s[ix] = f2h(q4_pack(dst, L->K4 + ix * (HD / 2), HD));
      else if(m->kvq_k == 4) L->K2s[ix] = f2h(q2_pack(dst, L->K2 + ix * (HD / 4), HD));
      else if(m->kvq_k == 5) L->K5s[ix] = f2h(q5_pack(dst, L->K5 + ix * (HD / 2), HD));
      else { uint16_t* K = L->Kc + ix * HD; for(int j = 0; j < HD; ++j) K[j] = f2h(dst[j]); }
      if(m->kvq_v == 1) L->Vs[ix] = f2h(tern_pack(vv, L->Vt + ix * (HD / 4), HD));
      else if(m->kvq_v == 6) q2b_pack(vv, L->V2b + ix * (HD / 4), L->V2bs + ix * (HD / 32), HD);
      else if(m->kvq_v == 2) q8_pack(vv, L->Vq + ix * HD, L->Vqs + ix * (HD / 32), HD);
      else if(m->kvq_v == 3) L->V4s[ix] = f2h(q4_pack(vv, L->V4 + ix * (HD / 2), HD));
      else if(m->kvq_v == 4) L->V2s[ix] = f2h(q2_pack(vv, L->V2 + ix * (HD / 4), HD));
      else if(m->kvq_v == 5) L->V5s[ix] = f2h(q5_pack(vv, L->V5 + ix * (HD / 2), HD));
      else { uint16_t* V = L->Vc + ix * HD; for(int j = 0; j < HD; ++j) V[j] = f2h(vv[j]); }
    }
  }
  /* scores sc[head][t] */
  #pragma omp for schedule(static)
  for(t = 0; t < n; ++t){
    for(int kv = 0; kv < NKV; ++kv){
      const size_t ix = (size_t)t * NKV + kv;
      float kf[512];
      if(m->kvq_k == 1) tern_unpack(L->Kt + ix * (HD / 4), h2f(L->Ks[ix]), kf, HD);
      else if(m->kvq_k == 6) q2b_unpack(L->K2b + ix * (HD / 4), L->K2bs + ix * (HD / 32), kf, HD);
      else if(m->kvq_k == 2) q8_unpack(L->Kq + ix * HD, L->Kqs + ix * (HD / 32), kf, HD);
      else if(m->kvq_k == 3) q4_unpack(L->K4 + ix * (HD / 2), h2f(L->K4s[ix]), kf, HD);
      else if(m->kvq_k == 4) q2_unpack(L->K2 + ix * (HD / 4), h2f(L->K2s[ix]), kf, HD);
      else if(m->kvq_k == 5) q5_unpack(L->K5 + ix * (HD / 2), h2f(L->K5s[ix]), kf, HD);
      else { const uint16_t* K = L->Kc + ix * HD; for(int j = 0; j < HD; ++j) kf[j] = h2f(K[j]); }
      for(int g = 0; g < G; ++g){
        const float* q = m->qr + (size_t)(kv * G + g) * HD;
        float a = 0.f;
        for(int j = 0; j < HD; ++j) a += kf[j] * q[j];
        m->sc[(size_t)(kv * G + g) * m->ctx + t] = a * 0.0625f;
      }
    }
  }
  /* softmax per head */
  #pragma omp for schedule(static)
  for(hh = 0; hh < NH; ++hh){
    float* s = m->sc + (size_t)hh * m->ctx;
    float mx = -1e30f;
    for(int t = 0; t < n; ++t) mx = fmaxf(mx, s[t]);
    float tot = 0.f;
    for(int t = 0; t < n; ++t) tot += expf(s[t] - mx);
    for(int t = 0; t < n; ++t) s[t] = expf(s[t] - mx) / tot;
  }
  /* weighted V, gated; parallel over (head, 32-dim slice) */
  #pragma omp for schedule(static)
  for(u = 0; u < NH * (HD / 32); ++u){
    const int hh = u / (HD / 32), d0 = (u % (HD / 32)) * 32, kv = hh / G;
    const float* p = m->sc + (size_t)hh * m->ctx;
    float acc[32];
    for(int j = 0; j < 32; ++j) acc[j] = 0.f;
    for(int t = 0; t < n; ++t){
      const size_t ix = (size_t)t * NKV + kv;
      const float pt = p[t];
      if(m->kvq_v){
        float vf[32];
        if(m->kvq_v == 1) tern_unpack(L->Vt + ix * (HD / 4) + (d0 >> 2), h2f(L->Vs[ix]), vf, 32);
        else if(m->kvq_v == 6) q2b_unpack(L->V2b + ix * (HD / 4) + (d0 >> 2), L->V2bs + ix * (HD / 32) + (d0 >> 5), vf, 32);
        else if(m->kvq_v == 2) q8_unpack(L->Vq + ix * HD + d0, L->Vqs + ix * (HD / 32) + (d0 >> 5), vf, 32);
        else if(m->kvq_v == 3) q4_unpack(L->V4 + ix * (HD / 2) + (d0 >> 1), h2f(L->V4s[ix]), vf, 32);
        else if(m->kvq_v == 4) q2_unpack(L->V2 + ix * (HD / 4) + (d0 >> 2), h2f(L->V2s[ix]), vf, 32);
        else q5_unpack(L->V5 + ix * (HD / 2) + (d0 >> 1), h2f(L->V5s[ix]), vf, 32);
        for(int j = 0; j < 32; ++j) acc[j] += pt * vf[j];
      } else {
        const uint16_t* V = L->Vc + ix * HD + d0;
        for(int j = 0; j < 32; ++j) acc[j] += pt * h2f(V[j]);
      }
    }
    for(int j = 0; j < 32; ++j) m->att[(size_t)hh * HD + d0 + j] = acc[j] * sigm(m->qn[(size_t)hh * 2 * HD + HD + d0 + j]);
  }
  fold(m, m->att, m->sA, m->xp, NH * HD);
}

static void attn(Model* m, Layer* L, int pos, int add, int il){
  TS(4);
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[3] = {&L->q, &L->k, &L->v}; float* yy[3] = {m->qn, m->kr, m->vv}; gemv(m, 3, a, yy, m->xp, KCAL_ATTQKV); }
  attn_mid(m, L, pos, il);
  gemv1(m, &L->o, m->y, m->xp, KCAL_ATTN_O);
}

/* one expert's output into m->ex: gate and up on the folded input, silu(gate) * up, fold, down. The expert's three
   matrices are Mat pointers like any other, so a staging buffer and a resident expert are the same code. */
static void moe_expert(Model* m, Layer* L, int e){
  int i;
  const Mat* a[2] = {&L->moe.experts[3 * e], &L->moe.experts[3 * e + 1]};
  float* yy[2] = {m->ffg, m->ffu};
  const int n = L->moe.experts[3 * e].R;              /* the expert's own width, which need not be NFF */
  gemv(m, 2, a, yy, m->xp, KCAL_MOE_UP);
  #pragma omp for schedule(static)
  for(i = 0; i < n; ++i) m->m[i] = silu(m->ffg[i]) * m->ffu[i];
  fold(m, m->m, L->moe.sF, m->xp, n);
  gemv1(m, &L->moe.experts[3 * e + 2], m->ex, m->xp, KCAL_MOE_DOWN);
}

/* the MoE FFN: route, take the nact highest, renormalise, and sum the experts' outputs weighted by their probability.
   The fetch hook runs before each expert's products, so an expert that is not resident can be brought in by the host
   (that is the whole difference between a paged and a resident expert from the kernel's point of view). */
static void moe_ffn(Model* m, Layer* L, int il){
  int i, j;
  gemv1(m, &L->moe.router, m->pr, m->xp, KCAL_MOE_ROUTER);
  #pragma omp single
  {
    const int N = L->moe.nexp, K = L->moe.nact;
    float mx = m->pr[0];
    double tot = 0.0;
    for(i = 1; i < N; ++i) if(m->pr[i] > mx) mx = m->pr[i];
    for(i = 0; i < N; ++i){ m->pr[i] = expf(m->pr[i] - mx); tot += m->pr[i]; }
    for(i = 0; i < N; ++i) m->pr[i] = (float)(m->pr[i] / tot);
    for(j = 0; j < K; ++j){
      int bi = 0; float bv = -1.0f;
      for(i = 0; i < N; ++i){
        int taken = 0, q;
        for(q = 0; q < j; ++q) if(m->sel[q] == i) taken = 1;
        if(!taken && m->pr[i] > bv){ bv = m->pr[i]; bi = i; }
      }
      m->sel[j] = bi; m->psel[j] = bv;
    }
    { float sp = 0.0f;
      for(j = 0; j < K; ++j) sp += m->psel[j];
      if(sp <= 0.0f) sp = 1.0f;
      for(j = 0; j < K; ++j) m->psel[j] /= sp; }
    for(i = 0; i < m->D; ++i) m->y[i] = 0.0f;
  }
  #pragma omp barrier
  for(j = 0; j < L->moe.nact; ++j){
    const int ex = m->sel[j];
    if(m->moe_fetch){
      m->moe_fetch(m->moe_ctx, il, ex);
      #pragma omp barrier
    }
    /* moe_expert overwrites xp with its own folded intermediate, so each expert starts from a fresh fold of the
       block's input; one FWHT per expert is nothing against that expert's products */
    fold(m, m->xn, m->sD, m->xp, m->D);
    moe_expert(m, L, ex);
    #pragma omp for schedule(static)
    for(i = 0; i < m->D; ++i) m->y[i] += m->psel[j] * m->ex[i];
  }
}

static void ffn(Model* m, Layer* L, int il){
  int e;
  TS(5);
  rmsfold(m, L->post_norm, 1);
  if(g_dump_f) dump_row(m);
  if(L->moe.nexp){ moe_ffn(m, L, il); return; }
  { const Mat* a[2] = {&L->ffg, &L->ffu}; float* yy[2] = {m->ffg, m->ffu}; gemv(m, 2, a, yy, m->xp, KCAL_DENSE_UP); }
  #pragma omp for schedule(static)
  for(e = 0; e < m->NFF; ++e) m->m[e] = silu(m->ffg[e]) * m->ffu[e];
  fold(m, m->m, m->sF, m->xp, m->NFF);
  gemv1(m, &L->ffd, m->y, m->xp, KCAL_DENSE_DOWN);
}

static void embed(Model* m, int tok){
  const int C = m->emb.C, R = m->emb.R;
  int g, bl;
  #pragma omp for schedule(static)
  for(g = 0; g < C / 128; ++g){
    static const signed char W0[9] = {-1, 0, 1, -1, 1, -1, 0, 1, 0}, W1[9] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
    const uint32_t* pw = (const uint32_t*)m->emb.W + ((size_t)(tok / 8) * (C / 128) + g) * 64;
    const float d = m->emb.S[((size_t)(tok / 8) * (C / 128) + g) * 8 + (tok & 7)];
    for(int p = 0; p < 64; ++p){
      const int code = (pw[p] >> (4 * (tok & 7))) & 15;
      m->m[g * 128 + 2 * p] = (float)W0[code] * d;
      m->m[g * 128 + 2 * p + 1] = (float)W1[code] * d;
    }
  }
  #pragma omp for schedule(static)
  for(bl = 0; bl < m->D / m->HB; ++bl){
    float* o = m->m + (size_t)bl * m->HB;
    fwht_block(o, m->HB);
    for(int j = 0; j < m->HB; ++j) m->h[bl * m->HB + j] = m->sD[bl * m->HB + j] * o[j];
  }
}

/* one token; head != 0: also the output head (logits in m->logits); returns the greedy token or -1 */
KCAPI int kc_step(Model* m, int tok, int pos, int head){
  int best = -1;
  int redo = 0;              /* shared: the guard's verdict for this decision */
  dump_open();
  vcap_open(m);
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    embed(m, tok);
    const int tail = q8_tail();
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      /* every thread stores the same value for the same layer, and the barrier inside the layer's first gemv orders the
         store before any thread reads the tier, so no single-thread flag protocol is needed here */
      if(tail) g_force_fp32 = (il >= m->NL - tail);
      if(L->is_attn) attn(m, L, pos, il > 0, il); else gdn(m, L, il > 0);
      ffn(m, L, il);
    }
    if(head){
      TS(6);
      rmsfold(m, m->out_norm, 1);
      if(tail){
        #pragma omp single
        { g_force_fp32 = 1; }              /* the head is part of the exact tail */
      }
      gemv1(m, &m->head, m->logits, m->xp, KCAL_HEAD);
      if(tail){
        #pragma omp single
        { g_force_fp32 = 0; }
      }
      if(tier() == KCT_Q8 && q8_tail() == 0 && guard_eps() > 0.0f){
        #pragma omp single
        {
          float v1 = m->logits[0], v2 = -3.4e38f;     /* top-1 and top-2 of the int8 logits */
          for(int i = 1; i < m->head.R; ++i){
            const float v = m->logits[i];
            if(v > v1){ v2 = v1; v1 = v; } else if(v > v2) v2 = v;
          }
          ++g_guard_seen;
          if((v1 - v2) < guard_eps()){ ++g_guard_hits; redo = 1; }
        }
        if(redo){
          #pragma omp single
          { g_force_fp32 = 1; }              /* implicit barrier: every thread sees it before its gemv entry */
          gemv1(m, &m->head, m->logits, m->xp, KCAL_HEAD);      /* fp32 head, same folded input */
          #pragma omp single
          { g_force_fp32 = 0; }
        }
      }
      #pragma omp single
      {
        int bi = 0; float bv = m->logits[0];
        for(int i = 1; i < m->head.R; ++i) if(m->logits[i] > bv){ bv = m->logits[i]; bi = i; }
        best = bi;
      }
    } else {
      #pragma omp single
      { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }   /* keep h = the residual after the last block */
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return best;
}

/* Two tokens at pos and pos+1 in one pass. Every product is one weight stream for both tokens (rows8_b2); the
   per-token parts run in token order on the shared state, so token 1's recurrence and attention see what token 0
   produced. b is a second Model over the same weights and the same Layer state with its own scratch. The result is
   bit-identical to kc_step(m, t0, pos) followed by kc_step(m, t1, pos + 1). out[0..1] get the greedy tokens when
   head != 0 (b's logits are in b->logits). */
KCAPI int kc_step2(Model* m, Model* b, int t0, int t1, int pos, int head, int* out, float* snap){
  vcap_open(m);
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    int e;                     /* private to each thread, and the induction variable of the two loops below */
    embed(m, t0);
    embed(b, t1);
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      const int add = il > 0;
      if(L->is_attn){
        rmsfold(m, L->attn_norm, add);
        rmsfold(b, L->attn_norm, add);
        {
          const Mat* a[3] = {&L->q, &L->k, &L->v};
          float* ya[3] = {m->qn, m->kr, m->vv}; float* yb[3] = {b->qn, b->kr, b->vv};
          gemv2(m, 3, a, ya, yb, m->xp, b->xp);
        }
        attn_mid(m, L, pos, il);
        attn_mid(b, L, pos + 1, il);
      } else {
        rmsfold(m, L->attn_norm, add);
        rmsfold(b, L->attn_norm, add);
        {
          const Mat* a[2] = {&L->qkv, &L->gate};
          float* ya[2] = {m->qkv, m->zv}; float* yb[2] = {b->qkv, b->zv};
          gemv2(m, 2, a, ya, yb, m->xp, b->xp);
        }
        gdn_mid(m, L);
        if(snap){
          /* The state after token 0 (all layers), so a rejected draft rolls back without replaying token 0. One
             thread only, and the region's implicit barrier keeps token 1 out of L->M and L->conv until it is
             copied: gdn_mid(b, ...) rewrites conv almost at once, so an unsynchronised copy loses it. */
          #pragma omp single
          {
            const size_t sz = (size_t)m->NV * m->HK * m->HK, cz = (size_t)3 * m->CONV_CH;
            float* o = snap + (size_t)il * (sz + cz);
            memcpy(o, L->M, sz * sizeof(float));
            memcpy(o + sz, L->conv, cz * sizeof(float));
          }
        }
        gdn_mid(b, L);
      }
      {
        const Mat* a[1] = {L->is_attn ? &L->o : &L->ssm_out};
        float* ya[1] = {m->y}; float* yb[1] = {b->y};
        gemv2(m, 1, a, ya, yb, m->xp, b->xp);
      }
      /* FFN: the dense path batches both tokens over one weight stream. A MoE layer cannot: the two tokens may route
         to different experts, so each token runs its own experts in turn. Correct, but it gives up the two-token gain
         on those layers, which `verify-tier` reports by comparing the two paths. */
      if(L->moe.nexp){
        ffn(m, L, il);
        ffn(b, L, il);
      } else {
        rmsfold(m, L->post_norm, 1);
        rmsfold(b, L->post_norm, 1);
        {
          const Mat* a[2] = {&L->ffg, &L->ffu};
          float* ya[2] = {m->ffg, m->ffu}; float* yb[2] = {b->ffg, b->ffu};
          gemv2(m, 2, a, ya, yb, m->xp, b->xp);
        }
        #pragma omp for schedule(static)
        for(e = 0; e < m->NFF; ++e) m->m[e] = silu(m->ffg[e]) * m->ffu[e];
        #pragma omp for schedule(static)
        for(e = 0; e < m->NFF; ++e) b->m[e] = silu(b->ffg[e]) * b->ffu[e];
        fold(m, m->m, m->sF, m->xp, m->NFF);
        fold(b, b->m, b->sF, b->xp, m->NFF);
        {
          const Mat* a[1] = {&L->ffd};
          float* ya[1] = {m->y}; float* yb[1] = {b->y};
          gemv2(m, 1, a, ya, yb, m->xp, b->xp);
        }
      }
    }
    if(head){
      TS(6);
      rmsfold(m, m->out_norm, 1);
      rmsfold(b, b->out_norm, 1);
      {
        const Mat* a[1] = {&m->head};
        float* ya[1] = {m->logits}; float* yb[1] = {b->logits};
        gemv2(m, 1, a, ya, yb, m->xp, b->xp);
      }
      #pragma omp single
      {
        const int R = m->head.R;
        int i0 = 0, i1 = 0;
        float v0 = m->logits[0], v1 = b->logits[0];
        for(int i = 1; i < R; ++i){
          if(m->logits[i] > v0){ v0 = m->logits[i]; i0 = i; }
          if(b->logits[i] > v1){ v1 = b->logits[i]; i1 = i; }
        }
        out[0] = i0; out[1] = i1;
      }
    } else {
      #pragma omp single
      {
        for(int e = 0; e < m->D; ++e){ m->h[e] += m->y[e]; b->h[e] += b->y[e]; }
      }
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

/* put the recurrent state back to what kc_step2 snapshotted (the state after its first token) */
KCAPI void kc_state_load(Model* m, const float* snap){
  for(int il = 0; il < m->NL; ++il){
    Layer* L = m->L + il;
    if(L->is_attn) continue;
    const size_t sz = (size_t)m->NV * m->HK * m->HK, cz = (size_t)3 * m->CONV_CH;
    const float* o = snap + (size_t)il * (sz + cz);
    memcpy(L->M, o, sz * sizeof(float));
    memcpy(L->conv, o + sz, cz * sizeof(float));
  }
}

/* Run one layer's FFN block on a given input, in place of a whole token: the test surface for the MoE path, since a
   dense forward pass cannot isolate the routing arithmetic. in is D floats; the residual input y is taken as zero, so
   the block's output is what ffn() leaves in m->y. */
KCAPI void kc_ffn_probe(Model* m, int il, const float* in, float* out){
  int i;
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for(i = 0; i < m->D; ++i){ m->h[i] = in[i]; m->y[i] = 0.0f; }
    #pragma omp barrier
    ffn(m, m->L + il, il);
    #pragma omp for schedule(static)
    for(i = 0; i < m->D; ++i) out[i] = m->y[i];
  }
}

/* select the K/V cache format: 0 fp16 (default), 1 ternary, 2 q8_0, 3 4-bit, 4 2-bit four-level, 5 4 bits as two
   4-level passes. The arrays are allocated by the host, so this only records the choice and refuses a mode whose
   arrays were not handed over */
static int kvq_ok(const Model* m, int mode){
  /* layer 0 is a GDN layer and holds no K/V cache, so look for the first layer that does */
  int il = 0;
  while(il < m->NL && !(m->L + il)->Kc) ++il;
  if(mode < 0 || mode > 6 || il == m->NL) return 0;
  const Layer* A = m->L + il;
  if(mode == 1 && !A->Ks) return 0;
  if(mode == 2 && !A->Kqs) return 0;
  if(mode == 3 && !A->K4s) return 0;
  if(mode == 4 && !A->K2s) return 0;
  if(mode == 5 && !A->K5s) return 0;
  if(mode == 6 && !A->K2bs) return 0;
  return 1;
}

KCAPI int kc_kv_quant(Model* m, int mode){
  if(!kvq_ok(m, mode)) return m->kvq;
  m->kvq = m->kvq_k = m->kvq_v = mode;
  return m->kvq;
}

/* the same choice per side, so K can be q8_0 while V is ternary: returns the packed choice k*8+v, which equals the
   request on success and the previous setting on refusal.

   The multiplier is 8, not 4: the formats now run 0..6, and k*4+v stopped being injective once v could reach 4 (K=q8_0
   with V=4-bit gave 8, the same code as K=4-bit with V=fp16), so a refused setting could have matched a wanted one. */
KCAPI int kc_kv_quant2(Model* m, int k, int v){
  if(!kvq_ok(m, k) || !kvq_ok(m, v)) return m->kvq_k * 8 + m->kvq_v;
  m->kvq_k = k; m->kvq_v = v; m->kvq = k;          /* m->kvq stays the K side, which is what callers quote */
  return m->kvq_k * 8 + m->kvq_v;
}

KCAPI void kc_dump_close(void){ if(g_dump_f){ fclose(g_dump_f); g_dump_f = NULL; } }

KCAPI int kc_sizeof_model(void){ return (int)sizeof(Model); }
KCAPI int kc_sizeof_layer(void){ return (int)sizeof(Layer); }

/* per-stage seconds since the last kc_reset_times (kc_times: 0 embed 1 norm+fold 2 gemv 3 gdn-other
   4 attn-other 5 ffn-other 6 head), for the CPU profiler. */
KCAPI void kc_get_times(double* out){ for(int i = 0; i < 8; ++i) out[i] = kc_times[i]; }
KCAPI void kc_reset_times(void){ for(int i = 0; i < 8; ++i) kc_times[i] = 0.0; }
