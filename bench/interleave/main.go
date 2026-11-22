// Command interleave measures an erasure-coding encode (k data shards, m parity
// shards: k*m multiply-accumulates over each shard) with gf256 against the
// reference klauspost/reedsolomon, single core, in interleaved rounds so that
// drift in machine load hits both sides alike. It refuses to run when the
// 1-minute load average exceeds -maxload.
//
// ISA-L is the preferred reference but is not installed on the measuring
// hosts; klauspost/reedsolomon (GFNI/AVX2/AVX-512/NEON assembly) stands in.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"slices"
	"time"

	"github.com/go-simd/gf256"
	"github.com/klauspost/reedsolomon"
)

func main() {
	k := flag.Int("k", 10, "data shards")
	m := flag.Int("m", 4, "parity shards")
	size := flag.Int("size", 1<<20, "shard size in bytes")
	rounds := flag.Int("rounds", 15, "interleaved rounds")
	reps := flag.Int("reps", 20, "encodes per round")
	load := flag.Float64("load", -1, "1-minute load average at start (the caller reads uptime); refused above -maxload")
	maxLoad := flag.Float64("maxload", 8, "refuse to measure above this load")
	flag.Parse()
	if *load < 0 {
		fmt.Fprintln(os.Stderr, "pass -load <1-min load average> (read it from uptime)")
		os.Exit(2)
	}
	if *load > *maxLoad {
		fmt.Fprintf(os.Stderr, "load %.1f > %.1f: defer the measurement\n", *load, *maxLoad)
		os.Exit(3)
	}
	runtime.GOMAXPROCS(1)

	enc, err := reedsolomon.New(*k, *m, reedsolomon.WithMaxGoroutines(1))
	if err != nil {
		panic(err)
	}
	shards := make([][]byte, *k+*m)
	for i := range shards {
		shards[i] = make([]byte, *size)
	}
	for i := 0; i < *k; i++ {
		for j := range shards[i] {
			shards[i][j] = byte(i*131 + j*7)
		}
	}
	// The coefficient matrix: any fixed non-trivial constants do for a throughput
	// measurement of the kernel, the work per byte is identical.
	coef := make([]byte, *k**m)
	for i := range coef {
		coef[i] = byte(2 + i*5)
	}

	ours := func() {
		for p := 0; p < *m; p++ {
			dst := shards[*k+p]
			gf256.MulSlice(dst, shards[0], coef[p*(*k)])
			for d := 1; d < *k; d++ {
				gf256.MulAddSlice(dst, shards[d], coef[p*(*k)+d])
			}
		}
	}
	ref := func() {
		if err := enc.Encode(shards); err != nil {
			panic(err)
		}
	}
	timeit := func(f func()) float64 {
		f() // warm
		t := time.Now()
		for i := 0; i < *reps; i++ {
			f()
		}
		sec := time.Since(t).Seconds() / float64(*reps)
		return float64(*k**size) / sec / 1e9 // GB/s of data shards encoded
	}
	var a, b []float64
	for r := 0; r < *rounds; r++ {
		if r%2 == 0 {
			a = append(a, timeit(ours))
			b = append(b, timeit(ref))
		} else {
			b = append(b, timeit(ref))
			a = append(a, timeit(ours))
		}
	}
	med := func(x []float64) float64 { slices.Sort(x); return x[len(x)/2] }
	fmt.Printf("%s/%s tier=%s k=%d m=%d shard=%d rounds=%d load=%.1f\n",
		runtime.GOOS, runtime.GOARCH, gf256.Tier(), *k, *m, *size, *rounds, *load)
	fmt.Printf("gf256 (k*m MulAddSlice)      median %.2f GB/s\n", med(a))
	fmt.Printf("klauspost/reedsolomon Encode median %.2f GB/s\n", med(b))
}
