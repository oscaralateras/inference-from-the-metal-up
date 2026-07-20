# T2 — CPU pipeline microbenchmark
#
# Authoring is portable (clock_gettime); CANONICAL numbers must be measured on a
# Linux x86 box — Apple Silicon/ARM mutes the branch-misprediction effect and has
# no rdtsc. On the Mac you get a working build to debug; on Linux x86 you get the
# numbers that go in the lab note.
#
#   make run    -> build + run, writing results/pipeline.csv
#   make clean  -> remove the binary

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
# On some older glibc, clock_gettime needs -lrt; modern Ubuntu (glibc >= 2.17) does not.
LDLIBS  ?=

bench: bench.c
	$(CC) $(CFLAGS) -o bench bench.c $(LDLIBS)

.PHONY: run clean
run: bench | results
	./bench > results/pipeline.csv
	@echo "wrote results/pipeline.csv"

results:
	mkdir -p results

clean:
	rm -f bench
