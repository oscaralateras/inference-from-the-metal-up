/*
 * T3 — Memory hierarchy microbenchmark. (a) BANDWIDTH, (b) TILING, (c) MEMORY-vs-COMPUTE.
 *
 * Goal: measure how fast this machine can *stream* data as the working set grows — and watch
 * the speed step DOWN each time the data outgrows a cache tier (L1 -> L2 -> L3 -> DRAM).
 *
 * How: for each working-set size, read (sum) the whole array many times and compute GB/s =
 * bytes-read / time. Two design choices make this a clean BANDWIDTH test (not a latency test):
 *   1. SEQUENTIAL access. We read a[0], a[1], a[2]... in order. That's prefetcher-friendly and is
 *      the same sequential streaming pattern weights follow during inference — the realistic number.
 *   2. SEVERAL independent accumulators. A single running sum is a dependent chain (each add
 *      waits on the last), so it would be limited by the *adder's* latency, not memory. Summing
 *      into 8 independent accumulators (the T2 ILP trick) hides that latency, so MEMORY is the
 *      bottleneck — which is what we want to measure.
 * Small array -> lives in fast L1 -> very high GB/s. Big array -> DRAM -> low GB/s. The curve of
 * GB/s vs size is a staircase whose steps land on the real cache boundaries.
 *
 * Experiment (b): TILING MATMUL. The same matrix multiply C = A·B done three ways — naive `ijk`,
 * loop-reordered `ikj`, and cache-blocked (tiled) — measuring GFLOP/s. Identical arithmetic; the
 * only difference is how well each keeps data in cache. It shows how restructuring a computation to
 * reuse cache turns a memory-starved loop into a fast one (the principle behind Flash-Attention).
 *
 * Experiment (c): MEMORY-BOUND vs COMPUTE-BOUND. A fixed weight matrix W times a BATCH of B activation
 * vectors. B=1 is decode (matrix×vector): W is streamed from DRAM and each weight used once → memory-
 * bound, math units starved. Larger B is batched decode (matrix×matrix) — the same GEMM shape prefill
 * gets naturally from a long prompt: each weight is reused across all B columns → compute-bound, math
 * units saturated. GFLOP/s climbing as B grows is the empirical reason batching raises decode
 * THROUGHPUT — reusing each DRAM weight-read across B tokens. It's a throughput/utilisation win, not a
 * per-token latency win — the payoff of the whole artefact.
 *
 * Output: CSV on stdout (redirected to results/memory.csv); human summary on stderr.
 * NOTE: canonical numbers must come from Linux x86 — cache sizes + bandwidth differ on ARM.
 */

#define _POSIX_C_SOURCE 199309L  /* expose clock_gettime under -std=c11 on glibc (see T2) */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>  /* sysconf() — reads the CPU's real cache sizes on Linux */

/* ---- experiment parameters ---- */
#define MIN_BYTES    (1u << 12)   /* smallest working set: 4 KB   (sits inside L1). */
#define MAX_BYTES    (1u << 29)   /* largest working set:  512 MB (deep into DRAM). */
#define TARGET_BYTES (1ull << 30) /* process ~1 GiB per measurement, so every size does the same
                                   * total work and the timed interval is large enough. */
#define TRIALS       5            /* independent timings; report the MEDIAN. */

/* qsort comparator for doubles (median of the trial timings). */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Monotonic wall-clock in nanoseconds (never runs backwards — right for durations). */
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Read the whole array once, returning its sum.
 *
 * `noinline` keeps the machine code identical across calls. Returning the sum makes the read
 * observable, so the optimiser can't delete it. The 8 independent accumulators break the
 * add-after-add dependency so the loop runs at MEMORY speed, not adder speed. */
__attribute__((noinline))
static uint64_t read_sum(const uint64_t *a, size_t n) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 += a[i + 0]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
        s4 += a[i + 4]; s5 += a[i + 5]; s6 += a[i + 6]; s7 += a[i + 7];
    }
    for (; i < n; i++) s0 += a[i];  /* tail, if n isn't a multiple of 8 */
    return (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7);
}

/* Measure streaming bandwidth (GB/s) for one working-set size. */
static double bandwidth_gb_s(const uint64_t *a, size_t n, size_t bytes, uint64_t *sink) {
    /* Repeat the read enough times to touch ~TARGET_BYTES in total (>=1 pass). */
    long passes = (long)(TARGET_BYTES / bytes);
    if (passes < 1) passes = 1;

    *sink = read_sum(a, n);  /* warm-up: prime caches + clock the CPU up, untimed */

    double per_trial[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        double t0 = now_ns();
        volatile uint64_t acc = 0;
        for (long p = 0; p < passes; p++) acc += read_sum(a, n);
        double t1 = now_ns();
        *sink = acc;
        /* total bytes read / nanoseconds = bytes/ns = GB/s (1 byte/ns == 1 GB/s). */
        per_trial[t] = (double)bytes * (double)passes / (t1 - t0);
    }
    qsort(per_trial, TRIALS, sizeof(double), cmp_double);
    return per_trial[TRIALS / 2];  /* median (TRIALS is odd) */
}

/* Emit the machine's real cache sizes as CSV rows so plot.py can mark the L1/L2/L3 boundaries.
 * sysconf's cache queries are a Linux/glibc extension → a no-op on macOS (authoring only). */
static void print_cache_sizes(void) {
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    const struct { const char *name; int query; } levels[] = {
        {"L1", _SC_LEVEL1_DCACHE_SIZE},
        {"L2", _SC_LEVEL2_CACHE_SIZE},
        {"L3", _SC_LEVEL3_CACHE_SIZE},
    };
    for (size_t k = 0; k < sizeof(levels) / sizeof(levels[0]); k++) {
        long sz = sysconf(levels[k].query);
        if (sz > 0) printf("cache,%s,%ld,size_bytes,%ld\n", levels[k].name, sz, sz);
    }
#endif
}

/* Experiment (a): sweep the working set from 4 KB to 512 MB and print bandwidth per size. */
static void run_bandwidth(void) {
    fprintf(stderr, "bandwidth hierarchy (sequential read, 8 independent accumulators):\n");
    for (size_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes <<= 1) {
        size_t n = bytes / sizeof(uint64_t);
        uint64_t *a = malloc(bytes);
        if (!a) { perror("malloc"); exit(1); }
        for (size_t i = 0; i < n; i++) a[i] = i;  /* fill so every byte is real, touched data */

        uint64_t sink;
        double gbps = bandwidth_gb_s(a, n, bytes, &sink);

        printf("bandwidth,read,%zu,gb_per_s,%.2f\n", bytes, gbps);
        fprintf(stderr, "  %7zu KB : %8.1f GB/s%s\n",
                bytes / 1024, gbps, sink == 0xdeadbeef ? " " : "");  /* sink keeps the read alive */

        free(a);
    }
}

/* ===========================================================================
 * EXPERIMENT (b): TILING MATMUL — the same C = A·B three ways, GFLOP/s each.
 * ===========================================================================*/

#define BS        64   /* block (tile) size for the cache-blocked version. */
#define TRIALS_MM 3    /* matmul runs are long + stable, so a small median suffices. */

/* General matmul shapes: A is (a_rows × a_cols), B is (b_rows × b_cols), and the result C is
 * (a_rows × b_cols). The product is only defined when the INNER dimensions match — A's columns
 * equal B's rows. This checks that and fails fast on a mismatch rather than computing garbage.
 * The caller allocates C as a_rows × b_cols. */
static void require_matmul_dims(int a_cols, int b_rows) {
    if (a_cols != b_rows) {
        fprintf(stderr, "FATAL: matmul inner dimensions don't match "
                        "(A has %d columns, B has %d rows)\n", a_cols, b_rows);
        exit(1);
    }
}

/* naive `ijk`: the textbook triple loop. B is walked DOWN its columns (stride N = b_cols), so it
 * thrashes the cache — the slow baseline. */
__attribute__((noinline))
static void matmul_naive(const float *A, int a_rows, int a_cols,
                         const float *B, int b_rows, int b_cols, float *C) {
    require_matmul_dims(a_cols, b_rows);
    const int M = a_rows, K = a_cols, N = b_cols;  /* C is M×N; K is the shared inner dimension */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

/* loop-reordered `ikj`: the inner loop now streams A, B and C rows SEQUENTIALLY (stride 1) —
 * cache-friendly and it vectorises. We zero C first because this version accumulates into it. */
__attribute__((noinline))
static void matmul_ikj(const float *A, int a_rows, int a_cols,
                       const float *B, int b_rows, int b_cols, float *C) {
    require_matmul_dims(a_cols, b_rows);
    const int M = a_rows, K = a_cols, N = b_cols;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            const float *brow = &B[k * N];
            float *crow = &C[i * N];
            for (int j = 0; j < N; j++)
                crow[j] += a * brow[j];        /* sequential -> streams + vectorises */
        }
    }
}

/* cache-blocked (tiled): the same `ikj` maths, but done in BS×BS tiles so a tile of A, B and C
 * stays hot in cache and is fully reused before we move on. */
__attribute__((noinline))
static void matmul_blocked(const float *A, int a_rows, int a_cols,
                           const float *B, int b_rows, int b_cols, float *C) {
    require_matmul_dims(a_cols, b_rows);
    const int M = a_rows, K = a_cols, N = b_cols;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
    for (int ii = 0; ii < M; ii += BS)
        for (int kk = 0; kk < K; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
                for (int i = ii; i < ii + BS && i < M; i++)
                    for (int k = kk; k < kk + BS && k < K; k++) {
                        float a = A[i * K + k];
                        int jmax = (jj + BS < N) ? jj + BS : N;
                        for (int j = jj; j < jmax; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
}

/* Time one matmul variant, returning median GFLOP/s. A matmul does 2·M·K·N floating-point ops
 * (M·N output cells, each a K-long dot product = K multiply-adds = 2K flops). */
static double time_matmul(void (*fn)(const float *, int, int, const float *, int, int, float *),
                          const float *A, int a_rows, int a_cols,
                          const float *B, int b_rows, int b_cols, float *C) {
    fn(A, a_rows, a_cols, B, b_rows, b_cols, C);  /* warm-up */
    double per[TRIALS_MM];
    for (int t = 0; t < TRIALS_MM; t++) {
        double t0 = now_ns();
        fn(A, a_rows, a_cols, B, b_rows, b_cols, C);
        double t1 = now_ns();
        double flops = 2.0 * (double)a_rows * (double)a_cols * (double)b_cols;
        per[t] = flops / (t1 - t0);   /* flops / ns = GFLOP/s */
    }
    qsort(per, TRIALS_MM, sizeof(double), cmp_double);
    return per[TRIALS_MM / 2];
}

/* Experiment (b): compare the methods on SQUARE matrices (the cache lesson is clearest square —
 * the functions themselves handle any rectangular shape). Sizes span from L1/L2-resident up to
 * 4096×4096 (64 MB/matrix), which OUTGROWS the L3 — the regime where cache-blocking finally pays
 * off, because ikj's re-reads spill to DRAM while the tiled version keeps its tiles hot. `naive`
 * is O(N^3) with a cache-thrashing constant, so it's only timed up to 1024 (above that it takes
 * minutes and adds nothing — its collapse is already clear). */
static void run_tiling(void) {
    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int NAIVE_MAX_N = 1024;  /* skip naive above this: too slow to be worth timing. */
    const struct {
        const char *name;
        void (*fn)(const float *, int, int, const float *, int, int, float *);
    } methods[] = {
        {"naive", matmul_naive},
        {"ikj", matmul_ikj},
        {"blocked", matmul_blocked},
    };

    fprintf(stderr, "\ntiling matmul (GFLOP/s, higher is better):\n");
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int N = sizes[s];
        size_t elems = (size_t)N * (size_t)N;
        float *A = malloc(elems * sizeof(float));
        float *B = malloc(elems * sizeof(float));
        float *C = malloc(elems * sizeof(float));
        if (!A || !B || !C) { perror("malloc"); exit(1); }
        for (size_t i = 0; i < elems; i++) {
            A[i] = (float)(i % 97) * 0.01f;
            B[i] = (float)(i % 89) * 0.01f;
        }
        for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
            if (methods[m].fn == matmul_naive && N > NAIVE_MAX_N) {
                fprintf(stderr, "  N=%4d  %-8s : skipped (O(N^3), too slow to time)\n",
                        N, methods[m].name);
                continue;  /* naive is measured only where it finishes in seconds */
            }
            /* square case: A is N×N, B is N×N, so C is N×N */
            double gf = time_matmul(methods[m].fn, A, N, N, B, N, N, C);
            double csum = 0.0;                 /* sum C so its writes can't be optimised away */
            for (size_t i = 0; i < elems; i++) csum += C[i];
            printf("tiling,%s,%d,gflop_per_s,%.2f\n", methods[m].name, N, gf);
            fprintf(stderr, "  N=%4d  %-8s : %7.2f GFLOP/s  (csum=%.3e)\n",
                    N, methods[m].name, gf, csum);
        }
        free(A); free(B); free(C);
    }
}

/* ===========================================================================
 * EXPERIMENT (c): MEMORY-BOUND vs COMPUTE-BOUND — GEMV (decode) vs GEMM (batched).
 * ===========================================================================*/

#define CROSS_DIM 4096   /* weight matrix W is CROSS_DIM × CROSS_DIM = 64 MB — bigger than any cache,
                          * so W is always streamed from DRAM (that's what makes B=1 memory-bound). */

/* Fixed weight matrix W (M×K) times a batch of B activation vectors X (K×B) → C (M×B).
 *   B = 1   : matrix × VECTOR  = decode.          W streamed once, each weight used once → MEMORY-bound.
 *   B large : matrix × MATRIX  = batched decode.  Each weight reused across all B columns → COMPUTE-bound.
 * Sweep B and watch GFLOP/s climb from starved (low) to saturated (plateau). Uses the cache-friendly
 * ikj kernel so the ONLY thing changing with B is the arithmetic intensity (work done per byte of W). */
static void run_crossover(void) {
    const int batches[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const int M = CROSS_DIM, K = CROSS_DIM;

    float *W = malloc((size_t)M * (size_t)K * sizeof(float));
    if (!W) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < (size_t)M * (size_t)K; i++) W[i] = (float)(i % 97) * 0.01f;

    fprintf(stderr, "\nmemory-bound vs compute-bound (W is %dx%d, GFLOP/s vs batch):\n", M, K);
    for (size_t b = 0; b < sizeof(batches) / sizeof(batches[0]); b++) {
        int B = batches[b];
        float *X = malloc((size_t)K * (size_t)B * sizeof(float));  /* activations: K×B */
        float *C = malloc((size_t)M * (size_t)B * sizeof(float));  /* output: M×B */
        if (!X || !C) { perror("malloc"); exit(1); }
        for (size_t i = 0; i < (size_t)K * (size_t)B; i++) X[i] = (float)(i % 89) * 0.01f;

        /* W is M×K, X is K×B, so C is M×B. */
        double gf = time_matmul(matmul_ikj, W, M, K, X, K, B, C);
        double csum = 0.0;                       /* sum C so its writes can't be optimised away */
        for (size_t i = 0; i < (size_t)M * (size_t)B; i++) csum += C[i];
        /* Wall-time of one batched step = per-request decode latency (a user waits the whole step).
         * Derived from gf (which is flops/ns): ns = flops/gf. It RISES with B — batching isn't free. */
        double step_ms = (2.0 * (double)M * (double)K * (double)B) / gf / 1e6;
        printf("crossover,batched,%d,gflop_per_s,%.2f\n", B, gf);
        printf("crossover,latency,%d,ms_per_step,%.3f\n", B, step_ms);
        fprintf(stderr, "  batch=%4d : %7.2f GFLOP/s  %8.3f ms/step  (csum=%.3e)\n",
                B, gf, step_ms, csum);
        free(X); free(C);
    }
    free(W);
}

int main(void) {
    printf("experiment,variant,size,metric,value\n");
    print_cache_sizes();  /* cache,L1/L2/L3 rows first, so plot.py can annotate */
    run_bandwidth();      /* experiment (a) */
    run_tiling();         /* experiment (b) */
    run_crossover();      /* experiment (c) */
    return 0;
}
