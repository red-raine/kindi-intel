// T-MAC-style ternary GEMV on AVX2: the LUT idea from the design note, with the arithmetic it needs.
//
// The note builds each 16-entry table in int8. A two-weight table holds v0 + v1 where each term is up to 127 in
// magnitude, so entries reach +/-254 and `static_cast<int8_t>` wraps: with int8 activations, 4.5% of activation pairs
// produce a table with at least one wrong entry, and the note's own `max_diff < 1e-3` assertion cannot pass.
//
// This version keeps int16 tables, split into two byte planes so one `vpshufb` still fetches each byte plane. That is
// two shuffles per table instead of one - the honest price of int8 activations - and the int16 result is widened to
// int32 only every `FLUSH` groups, which is safe because |entry| <= 254 and 254 * 64 = 16256 < 32767.
//
// Layout: a (K,) int8, w_packed (K/4, N) uint8 - one byte per output neuron per group of four weights, so a warp's
// load is contiguous. That transpose is the caller's job (`pack_for_tmac` in the harness).
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int FLUSH = 32;              // groups in int16 before widening: two nibbles per group, 508 * 32 < 32767

inline __m128i table_lo(const int8_t* a) {
    // one byte plane (low) of the int16 table: entry e = v0 + v1, v0, v1 in {0, +-a0}, {0, +-a1}
    alignas(16) int8_t lut[16];
    for (int e = 0; e < 16; ++e) {
        int w0 = e & 3, w1 = (e >> 2) & 3;
        int v0 = (w0 == 1) ? a[0] : ((w0 == 2) ? -a[0] : 0);
        int v1 = (w1 == 1) ? a[1] : ((w1 == 2) ? -a[1] : 0);
        lut[e] = static_cast<int8_t>((v0 + v1) & 0xFF);
    }
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut));
}

inline __m128i table_hi(const int8_t* a) {
    alignas(16) int8_t lut[16];
    for (int e = 0; e < 16; ++e) {
        int w0 = e & 3, w1 = (e >> 2) & 3;
        int v0 = (w0 == 1) ? a[0] : ((w0 == 2) ? -a[0] : 0);
        int v1 = (w1 == 1) ? a[1] : ((w1 == 2) ? -a[1] : 0);
        lut[e] = static_cast<int8_t>(((v0 + v1) >> 8) & 0xFF);
    }
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut));
}

// One 32-neuron block of the GEMV: each byte holds four weight codes, low nibble = weights 0,1 of the group, high
// nibble = 2,3. A lookup returns 32 int16 values = 64 bytes = two registers, which is why it is a pair.
// The plan planes are 32 bytes each and are loaded unaligned, so no allocator has to promise 32-byte alignment.
inline void block_dot(const uint8_t* w, const int8_t* pl, const int8_t* ph, const int8_t* ql, const int8_t* qh,
                      __m256i low_mask, __m256i& out0, __m256i& out1) {
    __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    __m256i n_lo = _mm256_and_si256(bytes, low_mask);
    __m256i n_hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);

    // int16 table lookup as two byte planes: value = zext(lo plane) | (zext(hi plane) << 8)
    auto lookup = [&](const int8_t* plane_lo, const int8_t* plane_hi, __m256i idx, __m256i& r0, __m256i& r1) {
        __m256i sl = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(plane_lo)), idx);
        __m256i sh = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(plane_hi)), idx);
        __m128i sl_lo = _mm256_castsi256_si128(sl), sl_hi = _mm256_extracti128_si256(sl, 1);
        __m128i sh_lo = _mm256_castsi256_si128(sh), sh_hi = _mm256_extracti128_si256(sh, 1);
        r0 = _mm256_or_si256(_mm256_cvtepu8_epi16(sl_lo), _mm256_slli_epi16(_mm256_cvtepu8_epi16(sh_lo), 8));
        r1 = _mm256_or_si256(_mm256_cvtepu8_epi16(sl_hi), _mm256_slli_epi16(_mm256_cvtepu8_epi16(sh_hi), 8));
    };
    __m256i a0, a1, b0, b1;
    lookup(pl, ph, n_lo, a0, a1);
    lookup(ql, qh, n_hi, b0, b1);
    out0 = _mm256_add_epi16(a0, b0);
    out1 = _mm256_add_epi16(a1, b1);
}

}  // namespace

torch::Tensor tmac_gemv_avx2(torch::Tensor a_quant, torch::Tensor w_packed, double scale_a,
                             torch::Tensor scale_w) {
    TORCH_CHECK(a_quant.is_contiguous() && a_quant.scalar_type() == torch::kInt8, "a_quant must be contiguous int8");
    TORCH_CHECK(w_packed.is_contiguous() && w_packed.scalar_type() == torch::kUInt8, "w_packed must be contiguous uint8");
    TORCH_CHECK(a_quant.dim() == 1, "GEMV takes a (K,) activation vector");

    const int64_t K = a_quant.size(0);
    const int64_t g = w_packed.size(0);
    const int64_t N = w_packed.size(1);
    TORCH_CHECK(K % 4 == 0 && g == K / 4, "w_packed must be (K // 4, N)");
    TORCH_CHECK(N % 32 == 0, "N must be a multiple of 32");

    const int8_t* a = a_quant.data_ptr<int8_t>();
    const uint8_t* w = w_packed.data_ptr<uint8_t>();
    const float* sw = scale_w.data_ptr<float>();

    // one low/high byte-plane pair per group of four activations (two tables: weights 0,1 and 2,3),
    // each plan replicated into both 128-bit lanes so one vpshufb serves all 32 bytes
    std::vector<int8_t> pl(32 * g), ph(32 * g), ql(32 * g), qh(32 * g);
    for (int64_t i = 0; i < g; ++i) {
        auto put = [&](std::vector<int8_t>& dst, __m128i t) {
            __m256i wide = _mm256_broadcastsi128_si256(t);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[32 * i]), wide);
        };
        put(pl, table_lo(a + 4 * i));
        put(ph, table_hi(a + 4 * i));
        put(ql, table_lo(a + 4 * i + 2));
        put(qh, table_hi(a + 4 * i + 2));
    }

    auto out = torch::empty({N}, torch::dtype(torch::kFloat32));
    float* o = out.data_ptr<float>();
    __m256i low_mask = _mm256_set1_epi8(0x0F);

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n += 32) {
        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
        __m256i part0 = _mm256_setzero_si256(), part1 = _mm256_setzero_si256();
        int inpart = 0;

        for (int64_t i = 0; i < g; ++i) {
            __m256i d0, d1;
            block_dot(w + i * N + n, &pl[32 * i], &ph[32 * i], &ql[32 * i], &qh[32 * i], low_mask, d0, d1);
            part0 = _mm256_add_epi16(part0, d0);
            part1 = _mm256_add_epi16(part1, d1);
            if (++inpart == FLUSH || i + 1 == g) {
                // each int16 register covers 16 neurons: its two 128-bit halves widen to 8 int32 each
                acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(part0)));
                acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(part0, 1)));
                acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(part1)));
                acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(part1, 1)));
                part0 = _mm256_setzero_si256();
                part1 = _mm256_setzero_si256();
                inpart = 0;
            }
        }
        alignas(32) int32_t buf[32];
        _mm256_store_si256(reinterpret_cast<__m256i*>(buf), acc0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(buf + 8), acc1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(buf + 16), acc2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(buf + 24), acc3);
        for (int k = 0; k < 32; ++k) o[n + k] = static_cast<float>(buf[k]) * static_cast<float>(scale_a) * sw[n + k];
    }
    return out;
}

// Cache-blocked variant: N tiles of `n_blk` outputs (0 = one tile over all N) and K tiles of `k_grp_blk` groups
// (0 = all of K). The tables are rebuilt per N tile, because a table is only reusable by the neurons in the block
// that built it - which is exactly the cost the tiling has to earn back.
torch::Tensor tmac_gemv_avx2_tiled(torch::Tensor a_quant, torch::Tensor w_packed, double scale_a,
                                   torch::Tensor scale_w, int64_t n_blk, int64_t k_grp_blk) {
    TORCH_CHECK(a_quant.is_contiguous() && a_quant.scalar_type() == torch::kInt8, "a_quant must be contiguous int8");
    TORCH_CHECK(w_packed.is_contiguous() && w_packed.scalar_type() == torch::kUInt8, "w_packed must be contiguous uint8");
    const int64_t K = a_quant.size(0), g = w_packed.size(0), N = w_packed.size(1);
    TORCH_CHECK(K % 4 == 0 && g == K / 4, "w_packed must be (K // 4, N)");
    TORCH_CHECK(N % 32 == 0, "N must be a multiple of 32");
    if (n_blk <= 0 || n_blk > N) n_blk = N;
    n_blk = (n_blk / 32) * 32;
    if (k_grp_blk <= 0 || k_grp_blk > g) k_grp_blk = g;

    const int8_t* a = a_quant.data_ptr<int8_t>();
    const uint8_t* w = w_packed.data_ptr<uint8_t>();
    const float* sw = scale_w.data_ptr<float>();
    auto out = torch::empty({N}, torch::dtype(torch::kFloat32));
    float* o = out.data_ptr<float>();
    __m256i low_mask = _mm256_set1_epi8(0x0F);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t t = 0; t < N; t += n_blk) {
        const int64_t n_end = std::min(t + n_blk, N);
        std::vector<int8_t> pl(32 * g), ph(32 * g), ql(32 * g), qh(32 * g);
        std::vector<__m256i> acc(4 * ((n_end - t) / 32 + 1));
        for (auto& x : acc) x = _mm256_setzero_si256();

        for (int64_t k0 = 0; k0 < g; k0 += k_grp_blk) {
            const int64_t k1 = std::min(k0 + k_grp_blk, g);
            for (int64_t i = k0; i < k1; ++i) {                 // tables for this K block only
                auto put = [&](std::vector<int8_t>& dst, __m128i tb) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[32 * i]),
                                        _mm256_broadcastsi128_si256(tb));
                };
                put(pl, table_lo(a + 4 * i));
                put(ph, table_hi(a + 4 * i));
                put(ql, table_lo(a + 4 * i + 2));
                put(qh, table_hi(a + 4 * i + 2));
            }
            int64_t blk = 0;
            for (int64_t n = t; n < n_end; n += 32, ++blk) {
                __m256i acc0 = acc[4 * blk], acc1 = acc[4 * blk + 1];
                __m256i acc2 = acc[4 * blk + 2], acc3 = acc[4 * blk + 3];
                __m256i part0 = _mm256_setzero_si256(), part1 = _mm256_setzero_si256();
                int inpart = 0;
                for (int64_t i = k0; i < k1; ++i) {
                    __m256i d0, d1;
                    block_dot(w + i * N + n, &pl[32 * i], &ph[32 * i], &ql[32 * i], &qh[32 * i], low_mask, d0, d1);
                    part0 = _mm256_add_epi16(part0, d0);
                    part1 = _mm256_add_epi16(part1, d1);
                    if (++inpart == FLUSH || i + 1 == k1) {
                        acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(part0)));
                        acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(part0, 1)));
                        acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(part1)));
                        acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(part1, 1)));
                        part0 = _mm256_setzero_si256();
                        part1 = _mm256_setzero_si256();
                        inpart = 0;
                    }
                }
                acc[4 * blk] = acc0; acc[4 * blk + 1] = acc1;
                acc[4 * blk + 2] = acc2; acc[4 * blk + 3] = acc3;
            }
        }

        for (int64_t n = t, blk = 0; n < n_end; n += 32, ++blk) {
            alignas(32) int32_t buf[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf), acc[4 * blk]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf + 8), acc[4 * blk + 1]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf + 16), acc[4 * blk + 2]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf + 24), acc[4 * blk + 3]);
            for (int k = 0; k < 32; ++k)
                o[n + k] = static_cast<float>(buf[k]) * static_cast<float>(scale_a) * sw[n + k];
        }
    }
    return out;
}

void set_threads(int n) { omp_set_num_threads(n); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tmac_gemv_avx2", &tmac_gemv_avx2, "ternary GEMV, LUT lookup on AVX2 (int16 tables)");
    m.def("tmac_gemv_avx2_tiled", &tmac_gemv_avx2_tiled, "same GEMV, N and K cache blocking");
    m.def("set_threads", &set_threads, "OpenMP thread count for the GEMV");
}
