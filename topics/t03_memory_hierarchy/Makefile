# T3 — Memory hierarchy microbenchmark
#
# Authoring is portable (clock_gettime); CANONICAL numbers must be measured on a
# Linux x86 box — cache sizes, prefetcher behaviour, and memory bandwidth differ on
# Apple Silicon. On the Mac you get a working build to debug; on Linux x86 you get
# the numbers (and the machine's real cache sizes) that go in the lab note.
#
#   make run    -> build + run, writing results/memory.csv
#   make clean  -> remove the binary

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
# On some older glibc, clock_gettime needs -lrt; modern Ubuntu (glibc >= 2.17) does not.
LDLIBS  ?=

bench: bench.c
	$(CC) $(CFLAGS) -o bench bench.c $(LDLIBS)

.PHONY: run clean
run: bench | results
	./bench > results/memory.csv
	@echo "wrote results/memory.csv"

results:
	mkdir -p results

clean:
	rm -f bench
