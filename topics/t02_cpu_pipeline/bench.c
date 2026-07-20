/*
 * T2 — CPU pipeline microbenchmark: (a) BRANCH MISPREDICTION, (b) ILP, (c) SIMD.
 *
 * The whole point: run the *exact same work* over the *exact same data*, changing
 * only whether the data is SORTED. Sorting doesn't change the arithmetic — it
 * changes how PREDICTABLE a branch is. A modern CPU guesses which way each `if`
 * will go and speculatively runs ahead; a wrong guess ("misprediction") means it
 * throws away that work and restarts the pipeline (~10-20 cycles wasted). When the
 * data is sorted, the branch goes false-false-false-...-true-true-true, so the
 * predictor is right almost every time. When it's shuffled, the branch is a
 * coin-flip and the predictor is wrong ~50% of the time. The time difference IS
 * the cost of misprediction.
 *
 * Experiment (b): INSTRUCTION-LEVEL PARALLELISM (ILP). Same trick, different pipeline
 * feature. We do the SAME number of multiply-adds two ways: as one long DEPENDENT
 * chain (each step needs the previous result, so the CPU must run them one-at-a-time —
 * "latency-bound") vs as several INDEPENDENT chains (the CPU overlaps them, running
 * many at once — "throughput-bound"). Same arithmetic, but the independent version is
 * far faster. That gap is the cost of a data dependency — and it's exactly why serial,
 * one-token-at-a-time decode is slow even on hardware built for massive throughput.
 *
 * Experiment (c): SIMD (Single Instruction, Multiple Data). One more pipeline feature:
 * a vector unit applies ONE instruction to several array elements at once. We run the
 * same element-wise maths two ways — one element at a time (scalar) vs several per
 * instruction (auto-vectorised) — and measure the width the hardware buys us. This is
 * the "wide" in the wide, throughput-optimised hardware that makes matrix math fast.
 *
 * Output: CSV on stdout (redirected to results/pipeline.csv by the Makefile);
 *         a human-readable summary on stderr (still shows in the terminal).
 *
 * Build/run:  make run          (portable: uses clock_gettime, no rdtsc)
 * NOTE: canonical numbers must come from a Linux x86 box — Apple Silicon/ARM
 *       mutes this effect and has a different branch predictor.
 */

/* Ask glibc to expose POSIX clock_gettime / CLOCK_MONOTONIC. Needed on Linux because
 * -std=c11 is strict ISO C, which hides POSIX extensions unless we request them. Must be
 * defined before any header is included. (macOS exposes them regardless, so this is a
 * no-op there.) */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---- experiment parameters ------------------------------------------------ */
#define N         (1 << 15)  /* 32768 elements. Small enough to sit in fast cache,
                              * so we measure the BRANCH, not memory latency (that's
                              * a separate artefact, T3). */
#define THRESHOLD 128        /* values are 0..255, so ~half clear the bar -> when the
                              * data is shuffled the branch is a genuine coin-flip. */
#define REPEATS   1000       /* re-run the sum loop this many times per timing, so the
                              * measured interval is large vs the clock's resolution. */
#define TRIALS    15         /* independent timings; we report the MEDIAN (robust to
                              * the odd OS hiccup — one slow trial can't skew it). */
#define SEED      1u         /* fixed seed => the array is identical every run. */

/* Compiler barrier — the standard microbenchmark trick to stop the optimiser cheating.
 * `asm volatile("" ::: "memory")` emits ZERO instructions, but tells the compiler "memory
 * may have changed here", so it must reload inputs and actually re-run the work each pass
 * instead of noticing our loop computes the same value every time and running it just once
 * (which would give a meaningless ~0 ns). Cf. Google Benchmark's DoNotOptimize/ClobberMemory. */
static inline void clobber(void) {
    __asm__ volatile("" : : : "memory");
}

/* qsort comparator: ascending integer order. The (x>y)-(x<y) form returns -1/0/1
 * without the overflow risk of a naive `x - y`. */
static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* The hot loop under test.
 *
 * `__attribute__((noinline))` stops the compiler from inlining + re-optimising this
 * differently at each call site, so the sorted and unsorted runs execute IDENTICAL
 * machine code — only the data differs. Returning `sum` makes the result observable,
 * which forbids the optimiser from deleting the loop as dead code. This return value
 * is our "sink".
 *
 * The `if` on the next-to-last line is THE branch this whole artefact is about — so we
 * must stop the compiler from ERASING it. At -O2 on GCC/x86 it will happily "if-convert"
 * this into a branchless conditional-move (cmov): compute both outcomes, select one, no
 * jump. Then there is no branch to mispredict and sorted == unsorted (exactly the null
 * result we first saw on x86). The clobber() barrier inside the taken path fixes this:
 * a side-effecting barrier can't be run unconditionally, so the compiler is forced to
 * keep a real, data-dependent branch (and can't vectorise the loop either). It emits no
 * instructions, so the runtime cost we measure is genuinely the branch's. */
__attribute__((noinline))
static long long sum_above_threshold(const int *data, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] >= THRESHOLD) {   /* <-- predictable when sorted, coin-flip when not */
            sum += data[i];
            clobber();                /* keep this a REAL branch (block cmov / vectorisation) */
        }
    }
    return sum;
}

/* Monotonic wall-clock in nanoseconds. CLOCK_MONOTONIC never jumps backwards
 * (unlike wall-clock time), which is what you want for measuring durations. */
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Time the hot loop over `data`, returning the MEDIAN nanoseconds-per-element.
 * `*checksum_out` receives the loop's sum, printed later so the compiler can't
 * elide the work and so we can sanity-check that sorted and unsorted agree. */
static double time_loop(const int *data, long long *checksum_out) {
    double per_elem[TRIALS];

    /* Warm-up: touch the data and prime the caches + branch predictor once,
     * untimed, so the first timed trial isn't paying start-up costs. */
    volatile long long warm = sum_above_threshold(data, N);
    (void)warm;

    for (int t = 0; t < TRIALS; t++) {
        volatile long long sink = 0;           /* volatile => the accumulation can't be optimised away */
        double t0 = now_ns();
        for (int r = 0; r < REPEATS; r++) {
            sink += sum_above_threshold(data, N);
            clobber();   /* treat `data` as possibly-changed -> force a real recompute each pass */
        }
        double t1 = now_ns();
        per_elem[t] = (t1 - t0) / ((double)REPEATS * N);   /* ns per element processed */
        *checksum_out = sink;                  /* same value every repeat/trial */
    }

    qsort(per_elem, TRIALS, sizeof(double), cmp_double);
    return per_elem[TRIALS / 2];               /* median (TRIALS is odd) */
}

/* ===========================================================================
 * EXPERIMENT (b): INSTRUCTION-LEVEL PARALLELISM (ILP)
 * ===========================================================================*/

/* ---- ILP parameters ---- */
#define ITERS 100000000L  /* 100 million multiply-adds total in EACH variant — so both
                           * do identical work. One call is already long enough to time
                           * accurately, so (unlike the branch loop) we don't need an
                           * inner REPEATS loop here. */
#define LANES 8           /* how many independent chains the throughput variant runs.
                           * A single multiply-add takes ~4-5 cycles to finish, but the
                           * CPU can start a new one every ~1 cycle — so it needs several
                           * independent chains "in flight" to hide that latency. 8 is
                           * comfortably enough to expose the gap. */

/* The unit of work in both variants is one "fused multiply-add" (FMA): acc = acc*a + b.
 * We use doubles (decimals) with |a| < 1 so the running value stays bounded — it can't
 * blow up to infinity or collapse to zero, which would make the timing meaningless. */

/* DEPENDENT chain — ONE accumulator, fully serial.
 *
 * Each iteration reads the `acc` that the previous iteration just wrote, so the CPU
 * literally cannot begin FMA #2 until FMA #1 has finished. Every step pays the full
 * ~4-5 cycle LATENCY of a multiply-add. This is the latency-bound case. */
__attribute__((noinline))
static double dependent_chain(long iters, double a, double b) {
    double acc = 1.0;
    for (long i = 0; i < iters; i++) {
        acc = acc * a + b;   /* <-- serial dependency: this step waits for the last acc */
    }
    return acc;
}

/* INDEPENDENT chains — LANES separate accumulators that never touch each other.
 *
 * The SAME total number of FMAs, but now acc[0]'s update doesn't depend on acc[1]'s,
 * so the CPU's out-of-order engine can keep many of them running at once. It stops
 * waiting on latency and instead runs at full THROUGHPUT — much faster, despite doing
 * identical arithmetic. (The compiler may even auto-vectorise the independent lanes
 * into SIMD, which only widens the gap — that's the throughput story, reinforced.) */
__attribute__((noinline))
static double independent_chains(long iters, double a, double b) {
    double acc[LANES];
    for (int k = 0; k < LANES; k++) {
        acc[k] = 1.0;
    }
    /* Each outer pass issues LANES independent FMAs; running iters/LANES passes keeps
     * the TOTAL FMA count equal to the dependent version's `iters` (fair comparison). */
    for (long i = 0; i < iters / LANES; i++) {
        for (int k = 0; k < LANES; k++) {
            acc[k] = acc[k] * a + b;   /* LANES of these, with NO dependency between them */
        }
    }
    /* Combine the lanes once at the very end. This little bit IS dependent, but it runs
     * once — not inside the hot loop — so it doesn't affect the measurement. */
    double sum = 0.0;
    for (int k = 0; k < LANES; k++) {
        sum += acc[k];
    }
    return sum;
}

/* Time one chain variant, returning the MEDIAN nanoseconds-per-multiply-add.
 *
 * The first parameter is a FUNCTION POINTER — a variable that holds "which function to
 * run". `double (*fn)(long, double, double)` reads as: "fn points to a function that
 * takes (long, double, double) and returns a double." Both dependent_chain and
 * independent_chains have exactly that shape, so this one helper can time either — we
 * just pass in the function we want. Inside, `fn(iters, a, b)` calls whichever it was
 * handed, the same way we'd call the function by name. */
static double time_chain(double (*fn)(long, double, double),
                         long iters, long long *checksum_out) {
    /* `volatile` forces a and b to be re-read on every call, so the compiler can't prove
     * the calls are identical and collapse 15 trials into one precomputed result (which
     * would read as ~0 ns). |a| < 1 also keeps acc bounded (no inf / nan). */
    volatile double a = 0.9999999, b = 1.0;
    double per_op[TRIALS];

    volatile double warm = fn(iters, a, b);   /* untimed warm-up, same reasoning as before */
    (void)warm;

    for (int t = 0; t < TRIALS; t++) {
        double t0 = now_ns();
        volatile double sink = fn(iters, a, b);   /* volatile => result is kept, loop can't be deleted */
        double t1 = now_ns();
        per_op[t] = (t1 - t0) / (double)iters;    /* ns per multiply-add */
        *checksum_out = (long long)sink;          /* observable value (sanity marker) */
    }

    qsort(per_op, TRIALS, sizeof(double), cmp_double);
    return per_op[TRIALS / 2];   /* median (TRIALS is odd) */
}


/* ===========================================================================
 * EXPERIMENT (c): SIMD  (scalar vs auto-vectorised)
 * ===========================================================================*/

/* ---- SIMD parameters ---- */
#define M           4096      /* elements per array — 16 KB of float, sits in L1 cache
                               * so we measure the VECTOR WIDTH, not memory bandwidth. */
#define VEC_REPEATS 100000L   /* repeat the element loop this many times so the total is
                               * long enough to time. The repeat index `r` also nudges
                               * the maths each pass, so the compiler can't compute it
                               * once and skip the rest. */

/* Both kernels compute the SAME thing: out[i] = x[i]*(a+r) + y[i], for every element,
 * VEC_REPEATS times. Across elements the work is independent, so a vector unit can do
 * several elements per instruction. The ONLY difference between the two functions below
 * is whether we let the compiler USE those vector instructions.
 *
 * `restrict` is a promise to the compiler that x, y and out don't overlap in memory.
 * Without it, the compiler must assume writing out[i] might change x[] or y[], which
 * forces it to process elements one-by-one — i.e. it blocks vectorisation. */

/* SCALAR baseline — vectorisation switched OFF: one element at a time.
 *   GCC   : the optimize("no-tree-vectorize") attribute disables it for the function.
 *   Clang : the #pragma right before the loop disables it for that loop.
 * (The #if picks the right mechanism for whichever compiler is building — GCC on the
 *  Linux box, Clang on the Mac.) */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-vectorize")))
#else
__attribute__((noinline))
#endif
static float saxpy_scalar(const float *restrict x, const float *restrict y,
                          float *restrict out, float a) {
    for (long r = 0; r < VEC_REPEATS; r++) {
#if defined(__clang__)
        _Pragma("clang loop vectorize(disable)")
#endif
        for (int i = 0; i < M; i++) {
            out[i] = x[i] * (a + (float)r) + y[i];
        }
    }
    float checksum = 0.0f;                       /* observable result -> loop can't be deleted */
    for (int i = 0; i < M; i++) checksum += out[i];
    return checksum;
}

/* VECTORISED — identical code, vectorisation ALLOWED: several elements per instruction.
 *   GCC   : optimize("O3") guarantees the vectoriser runs even though we build at -O2.
 *   Clang : already auto-vectorises this at -O2, so no extra hint needed. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("O3")))
#else
__attribute__((noinline))
#endif
static float saxpy_vector(const float *restrict x, const float *restrict y,
                          float *restrict out, float a) {
    for (long r = 0; r < VEC_REPEATS; r++) {
        for (int i = 0; i < M; i++) {
            out[i] = x[i] * (a + (float)r) + y[i];
        }
    }
    float checksum = 0.0f;
    for (int i = 0; i < M; i++) checksum += out[i];
    return checksum;
}

/* Time one saxpy variant, returning the MEDIAN nanoseconds-per-element. Same shape as
 * time_chain: a function pointer lets this one helper measure either kernel. */
static double time_saxpy(float (*fn)(const float *, const float *, float *, float),
                         const float *x, const float *y, float *out,
                         long long *checksum_out) {
    const float a = 1.0f;
    double per_elem[TRIALS];

    volatile float warm = fn(x, y, out, a);   /* untimed warm-up */
    (void)warm;

    for (int t = 0; t < TRIALS; t++) {
        double t0 = now_ns();
        volatile float sink = fn(x, y, out, a);
        double t1 = now_ns();
        per_elem[t] = (t1 - t0) / ((double)M * VEC_REPEATS);   /* ns per element */
        *checksum_out = (long long)sink;
        clobber();   /* stop the compiler reusing one trial's result for the others */
    }

    qsort(per_elem, TRIALS, sizeof(double), cmp_double);
    return per_elem[TRIALS / 2];   /* median (TRIALS is odd) */
}


int main(void) {
    /* ===================== experiment (a): BRANCH ===================== */
    int *data = malloc((size_t)N * sizeof(int));
    if (!data) { perror("malloc"); return 1; }

    /* Fill with reproducible pseudo-random values in [0, 256). */
    srand(SEED);
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 256;
    }

    /* variant 1: UNSORTED (unpredictable branch) */
    long long checksum_unsorted = 0;
    double ns_unsorted = time_loop(data, &checksum_unsorted);

    /* variant 2: SORTED (predictable branch). Sorting only reorders the SAME values,
     * so the sum is unchanged -> checksum_sorted must equal checksum_unsorted. Only
     * branch predictability changed. */
    qsort(data, N, sizeof(int), cmp_int);
    long long checksum_sorted = 0;
    double ns_sorted = time_loop(data, &checksum_sorted);

    /* ===================== experiment (b): ILP ======================= */
    /* No array needed — this measures a property of the pipeline, not memory. Same
     * work both ways; only the data dependency between operations changes. */
    long long checksum_dependent = 0, checksum_independent = 0;
    double ns_dependent   = time_chain(dependent_chain,    ITERS, &checksum_dependent);
    double ns_independent = time_chain(independent_chains, ITERS, &checksum_independent);

    /* ===================== experiment (c): SIMD ====================== */
    /* Three small float arrays; x and y are inputs, out is where results are written. */
    float *x   = malloc((size_t)M * sizeof(float));
    float *y   = malloc((size_t)M * sizeof(float));
    float *out = malloc((size_t)M * sizeof(float));
    if (!x || !y || !out) { perror("malloc"); return 1; }
    for (int i = 0; i < M; i++) {
        x[i] = (float)i / M;   /* deterministic values in [0, 1) — no rng needed */
        y[i] = 1.0f;
    }
    long long checksum_scalar = 0, checksum_vector = 0;
    double ns_scalar = time_saxpy(saxpy_scalar, x, y, out, &checksum_scalar);
    double ns_vector = time_saxpy(saxpy_vector, x, y, out, &checksum_vector);

    /* ===================== CSV to stdout (plot.py reads this) ========= */
    printf("experiment,variant,n,ns_per_elem,checksum\n");
    printf("branch,unsorted,%d,%.4f,%lld\n",   N,     ns_unsorted,    checksum_unsorted);
    printf("branch,sorted,%d,%.4f,%lld\n",     N,     ns_sorted,      checksum_sorted);
    printf("ilp,dependent,%ld,%.4f,%lld\n",    ITERS, ns_dependent,   checksum_dependent);
    printf("ilp,independent,%ld,%.4f,%lld\n",  ITERS, ns_independent, checksum_independent);
    printf("simd,scalar,%d,%.4f,%lld\n",       M,     ns_scalar,      checksum_scalar);
    printf("simd,vector,%d,%.4f,%lld\n",       M,     ns_vector,      checksum_vector);

    /* ===================== human summary to stderr =================== */
    fprintf(stderr, "\nbranch misprediction (ns per element):\n");
    fprintf(stderr, "  unsorted : %.3f ns/elem\n", ns_unsorted);
    fprintf(stderr, "  sorted   : %.3f ns/elem\n", ns_sorted);
    fprintf(stderr, "  speedup  : %.2fx faster when sorted\n", ns_unsorted / ns_sorted);
    fprintf(stderr, "  checksum : %lld (must match: %s)\n",
            checksum_sorted, checksum_sorted == checksum_unsorted ? "OK" : "MISMATCH!");

    fprintf(stderr, "\ninstruction-level parallelism (ns per multiply-add):\n");
    fprintf(stderr, "  dependent  (1 chain)    : %.3f ns/op\n", ns_dependent);
    fprintf(stderr, "  independent (%d chains)  : %.3f ns/op\n", LANES, ns_independent);
    fprintf(stderr, "  speedup                 : %.2fx faster when independent\n",
            ns_dependent / ns_independent);

    fprintf(stderr, "\nSIMD (ns per element):\n");
    fprintf(stderr, "  scalar     : %.4f ns/elem\n", ns_scalar);
    fprintf(stderr, "  vectorised : %.4f ns/elem\n", ns_vector);
    fprintf(stderr, "  speedup    : %.2fx faster when vectorised\n", ns_scalar / ns_vector);

    free(data);
    free(x);
    free(y);
    free(out);
    return 0;
}
