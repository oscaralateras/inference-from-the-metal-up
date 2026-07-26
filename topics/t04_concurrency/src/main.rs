//! T4 — Concurrency & synchronization microbenchmark (Rust, std-only).
//!
//! What caps the throughput of a CONCURRENT INFERENCE SERVER — the coordination taxes on the shared
//! state a token-serving loop lives on. Four experiments, each a real serving component:
//!   (a) FALSE SHARING — per-worker token counters packed vs padded: the invisible cache-line tax.
//!   (b) THE RACE      — the shared "tokens served" metric: unsynchronised loses counts (atomicity).
//!   (c) CONTENTION    — claiming the next token to serve: mutex vs atomic vs sharded, vs threads.
//!   (d) THE SCHEDULER — a shared request queue vs sharded per-worker queues: dispatch throughput.
//!
//! Emits CSV on stdout (redirected to results/concurrency.csv); a human summary on stderr.
//! Canonical numbers must come from a MULTICORE Linux x86 box — false sharing and contention only
//! appear with several real cores, and the 64-byte cache-line assumption below is x86 (Apple
//! Silicon uses 128-byte lines, which would change the padding).

use std::collections::VecDeque;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

/// An x86 cache line is 64 bytes. Forcing a counter to 64-byte alignment puts it alone on its own
/// line, so a neighbouring thread's writes can't drag it into a coherence tug-of-war.
#[repr(align(64))]
struct CachePadded(AtomicU64);

/// Wall-clock seconds to run `f` (which does the parallel work).
fn elapsed_secs(f: impl FnOnce()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

/// 1, 2, 4, … up to `max`, plus `max` itself if it isn't a power of two. The thread-count sweep.
fn thread_counts(max: usize) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut t = 1;
    while t < max {
        counts.push(t);
        t *= 2;
    }
    counts.push(max);
    counts.dedup();
    counts
}

// ===========================================================================
// (a) FALSE SHARING — per-worker token counters, packed vs padded.
// ===========================================================================

/// Hammer ONE atomic counter `iters` times with a relaxed increment.
///
/// `Relaxed` means no cross-thread ordering is requested — so the *only* thing that can differ in
/// cost between two memory layouts is the cache-line coherence traffic. That isolates false sharing:
/// same instruction, same work, the layout is the sole variable.
fn hammer(counter: &AtomicU64, iters: u64) {
    for _ in 0..iters {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Spawn one thread per counter, each hammering its own counter `iters` times, and return the
/// wall-clock cost as **nanoseconds per increment**. The caller owns the memory LAYOUT behind the
/// `&AtomicU64`s — this function is blind to whether they share a cache line or not.
fn time_hammer(counters: &[&AtomicU64], iters: u64) -> f64 {
    let secs = elapsed_secs(|| {
        thread::scope(|scope| {
            for &counter in counters {
                // `scope` guarantees every thread joins before it returns, so the borrows are safe —
                // Rust's "fearless concurrency": the compiler proves no thread outlives the data.
                scope.spawn(move || hammer(counter, iters));
            }
        });
    });
    let total_ops = counters.len() as u64 * iters;
    secs * 1e9 / total_ops as f64
}

/// Experiment (a): the SAME work — `n` threads, each doing `iters` private increments — laid out two
/// ways. Serving tie: this is a **per-worker token counter** (each request-worker tallies the tokens
/// it has generated). Real engines keep these hot per-worker stats; packed adjacently they false-
/// share and silently tax serving throughput — the fix (pad to a line) is a real serving optimisation.
///   * `adjacent`: `n` counters packed in a `Vec` (8 × 8-byte counters per 64-byte line) → threads
///     on the same line fight over it: every write invalidates the neighbours' cached copy, so the
///     line ping-pongs between cores. Nothing is logically shared, yet the hardware treats it as if
///     it were — **false sharing**.
///   * `padded`: each counter forced onto its own 64-byte line → no coherence contention.
fn run_false_sharing(n_threads: usize, iters: u64) {
    let adjacent: Vec<AtomicU64> = (0..n_threads).map(|_| AtomicU64::new(0)).collect();
    let adj_refs: Vec<&AtomicU64> = adjacent.iter().collect();
    let adjacent_ns = time_hammer(&adj_refs, iters);

    let padded: Vec<CachePadded> = (0..n_threads)
        .map(|_| CachePadded(AtomicU64::new(0)))
        .collect();
    let pad_refs: Vec<&AtomicU64> = padded.iter().map(|c| &c.0).collect();
    let padded_ns = time_hammer(&pad_refs, iters);

    println!("false_sharing,adjacent,{n_threads},ns_per_op,{adjacent_ns:.3}");
    println!("false_sharing,padded,{n_threads},ns_per_op,{padded_ns:.3}");
    eprintln!("(a) false sharing ({n_threads} workers, {iters} token-increments each):");
    eprintln!("    adjacent (shared line) : {adjacent_ns:.3} ns/op");
    eprintln!("    padded   (own line)    : {padded_ns:.3} ns/op");
    eprintln!(
        "    slowdown from false sharing: {:.2}x",
        adjacent_ns / padded_ns
    );
}

// ===========================================================================
// (b) THE RACE — the shared "tokens served" throughput metric.
// ===========================================================================

/// Experiment (b): `n` workers each bump ONE shared counter `iters` times. Serving tie: the server's
/// global tokens-served metric.
///   * `racy`  : a non-atomic read-modify-write (separate `load` then `store`). Two workers can read
///     the same value and both write back the same +1 — updates are LOST. Each op is atomic so this
///     is well-defined (not UB), but it's logically a race, so the total comes out too low.
///   * `atomic`: one `fetch_add` — indivisible, so no update is ever lost. The total is exact.
///
/// (A true `total += 1` on shared memory is a data race — *undefined behaviour* that Rust rejects at
/// compile time. We use two separate atomic ops to demonstrate lost updates safely.)
fn run_race(n_threads: usize, iters: u64) {
    let expected = n_threads as u64 * iters;

    let racy = AtomicU64::new(0);
    thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| {
                for _ in 0..iters {
                    let v = racy.load(Ordering::Relaxed);
                    racy.store(v + 1, Ordering::Relaxed);
                }
            });
        }
    });
    let racy_final = racy.load(Ordering::Relaxed);
    let racy_lost = 100.0 * expected.saturating_sub(racy_final) as f64 / expected as f64;

    let atomic = AtomicU64::new(0);
    thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| {
                for _ in 0..iters {
                    atomic.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    let atomic_final = atomic.load(Ordering::Relaxed);
    let atomic_lost = 100.0 * expected.saturating_sub(atomic_final) as f64 / expected as f64;

    println!("race,racy,{n_threads},lost_pct,{racy_lost:.2}");
    println!("race,atomic,{n_threads},lost_pct,{atomic_lost:.2}");
    eprintln!("\n(b) the race ({n_threads} workers x {iters}, expected {expected}):");
    eprintln!("    racy   (load+store): counted {racy_final} -> {racy_lost:.1}% of updates LOST");
    eprintln!("    atomic (fetch_add) : counted {atomic_final} -> {atomic_lost:.1}% lost");
}

// ===========================================================================
// (c) CONTENTION — token dispatch: mutex vs atomic vs sharded, vs thread count.
// ===========================================================================

/// Experiment (c): `n` workers each claim the next token to serve `ops` times, three ways, swept over
/// thread count. Serving tie: how workers grab the next unit of work.
///   * `mutex`  : one `Mutex<u64>` — every claim locks, so claims serialise (Amdahl's serial part).
///   * `atomic` : one shared `AtomicU64::fetch_add` — lock-free, but the single hot cache line still
///     bounces between cores.
///   * `sharded`: each worker on its OWN padded atomic (same op as `atomic`, but no sharing).
///
/// `sharded` vs `atomic` isolates *pure contention* (same instruction, only sharing differs).
/// Metric: throughput in millions of dispatches/sec.
fn run_contention(max_threads: usize, ops: u64) {
    eprintln!("\n(c) contention - token dispatch (throughput Mops/s, higher is better):");
    for &t in &thread_counts(max_threads) {
        let total = t as u64 * ops;

        let mutex = Mutex::new(0u64);
        let mutex_secs = elapsed_secs(|| {
            thread::scope(|scope| {
                for _ in 0..t {
                    scope.spawn(|| {
                        for _ in 0..ops {
                            *mutex.lock().unwrap() += 1;
                        }
                    });
                }
            });
        });

        let atomic = AtomicU64::new(0);
        let atomic_secs = elapsed_secs(|| {
            thread::scope(|scope| {
                for _ in 0..t {
                    scope.spawn(|| {
                        for _ in 0..ops {
                            atomic.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
        });

        let shards: Vec<CachePadded> = (0..t).map(|_| CachePadded(AtomicU64::new(0))).collect();
        let sharded_secs = elapsed_secs(|| {
            thread::scope(|scope| {
                for shard in &shards {
                    scope.spawn(move || hammer(&shard.0, ops));
                }
            });
        });

        let mutex_mops = total as f64 / mutex_secs / 1e6;
        let atomic_mops = total as f64 / atomic_secs / 1e6;
        let sharded_mops = total as f64 / sharded_secs / 1e6;
        println!("contention,mutex,{t},mops_per_s,{mutex_mops:.1}");
        println!("contention,atomic,{t},mops_per_s,{atomic_mops:.1}");
        println!("contention,sharded,{t},mops_per_s,{sharded_mops:.1}");
        eprintln!(
            "    {t:2} workers: mutex {mutex_mops:8.1} | atomic {atomic_mops:8.1} | sharded {sharded_mops:8.1}"
        );
    }
}

// ===========================================================================
// (d) THE SCHEDULER — request dispatch: one locked queue vs sharded per-worker ranges.
// ===========================================================================

/// Experiment (d): drain a fixed batch of `work` request-items with `n` workers, swept over thread
/// count. Serving tie: the continuous-batching dispatcher.
///   * `global_lock`: one `Mutex<VecDeque<usize>>` — every worker pops the next request under the one
///     lock, so the scheduler itself serialises and becomes the bottleneck.
///   * `sharded`    : the batch is partitioned into per-worker ranges up front — no shared queue, so
///     dispatch scales with workers.
///
/// Metric: dispatch throughput in millions of items/sec.
fn run_scheduler(max_threads: usize, work: usize) {
    eprintln!("\n(d) scheduler - request dispatch (throughput Mops/s, higher is better):");
    for &t in &thread_counts(max_threads) {
        let queue: Mutex<VecDeque<usize>> = Mutex::new((0..work).collect());
        let global_secs = elapsed_secs(|| {
            thread::scope(|scope| {
                for _ in 0..t {
                    scope.spawn(|| {
                        let mut sum = 0usize;
                        loop {
                            // Lock only to pop: the guard drops at the `;`, before we use `item`.
                            // (A `while let` would hold the lock across the loop body in ed. 2021.)
                            let item = queue.lock().unwrap().pop_front();
                            match item {
                                Some(x) => sum += x,
                                None => break,
                            }
                        }
                        black_box(sum);
                    });
                }
            });
        });

        let sharded_secs = elapsed_secs(|| {
            thread::scope(|scope| {
                for w in 0..t {
                    let lo = work * w / t;
                    let hi = work * (w + 1) / t;
                    scope.spawn(move || {
                        let mut sum = 0usize;
                        for item in lo..hi {
                            sum += black_box(item);
                        }
                        black_box(sum);
                    });
                }
            });
        });

        let global_mops = work as f64 / global_secs / 1e6;
        let sharded_mops = work as f64 / sharded_secs / 1e6;
        println!("scheduler,global_lock,{t},mops_per_s,{global_mops:.1}");
        println!("scheduler,sharded,{t},mops_per_s,{sharded_mops:.1}");
        eprintln!("    {t:2} workers: global_lock {global_mops:8.1} | sharded {sharded_mops:8.1}");
    }
}

fn main() {
    let n = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("experiment,variant,threads,metric,value");
    run_false_sharing(n, 50_000_000); // (a)
    run_race(n, 10_000_000); // (b)
    run_contention(n, 5_000_000); // (c)
    run_scheduler(n, 20_000_000); // (d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correctness invariant: atomic `fetch_add` never loses an update, so the counters must reach
    /// the exact expected total. (Experiment (b) shows what happens when you drop the atomicity.)
    #[test]
    fn every_atomic_increment_lands() {
        let n_threads = 4usize;
        let iters = 100_000u64;
        let counters: Vec<AtomicU64> = (0..n_threads).map(|_| AtomicU64::new(0)).collect();
        let refs: Vec<&AtomicU64> = counters.iter().collect();
        let _ = time_hammer(&refs, iters);
        let total: u64 = counters.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        assert_eq!(total, n_threads as u64 * iters);
    }

    #[test]
    fn thread_counts_are_powers_of_two_plus_max() {
        assert_eq!(thread_counts(8), vec![1, 2, 4, 8]);
        assert_eq!(thread_counts(10), vec![1, 2, 4, 8, 10]);
        assert_eq!(thread_counts(1), vec![1]);
    }
}
