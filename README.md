<p align="center"><img src="https://raw.githubusercontent.com/go-simd/brand/main/social/go-simd.png" alt="go-simd/gf256" width="720"></p>

# gf256

[![ci](https://github.com/go-simd/gf256/actions/workflows/ci.yml/badge.svg)](https://github.com/go-simd/gf256/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-simd/gf256.svg)](https://pkg.go.dev/github.com/go-simd/gf256)

SIMD **GF(2^8) multiply and multiply-accumulate by a constant** over byte slices,
the two inner loops of a Reed-Solomon erasure encoder, in pure Go
(`CGO_ENABLED=0`, assembly generated with [go-asmgen](https://github.com/go-asmgen/asmgen)).

```go
import "github.com/go-simd/gf256"

gf256.MulSlice(dst, src, c)    // dst[i]  = c * src[i]
gf256.MulAddSlice(dst, src, c) // dst[i] ^= c * src[i]
gf256.Mul(a, b)                // scalar product
gf256.Tier()                   // "gfni+avx2", "avx2", "ssse3", "neon" or "generic"
```

The field is the one used by ISA-L, klauspost/reedsolomon and Backblaze:
polynomial `x^8+x^4+x^3+x^2+1` (0x11D), generator 2. Lengths are `min(len(dst),
len(src))`; `dst` and `src` must be identical or disjoint.

## Kernels

| tier        | CPU            | method                                                                 | block |
| ----------- | -------------- | ---------------------------------------------------------------------- | ----: |
| `gfni+avx2` | amd64, GFNI    | one `VGF2P8AFFINEQB` with the 8x8 bit matrix of `c` (broadcast)        |  32 B |
| `avx2`      | amd64          | `VPSHUFB` on the low and high nibble tables of `c`, XOR                |  32 B |
| `ssse3`     | amd64          | `PSHUFB`, same tables                                                  |  16 B |
| `neon`      | arm64          | `TBL`, same tables                                                     |  16 B |
| `generic`   | everything else| 64 KiB product table, one lookup per byte                              |   1 B |

GFNI is detected with its own CPUID probe (leaf 7, ECX bit 8), because
`golang.org/x/sys/cpu` records it only on AVX-512 parts and would miss AVX2-only
CPUs. Tails shorter than a block go through the table loop.

Hosted CI runners (AVX2, no GFNI) and QEMU 10.2 `-cpu max` cannot execute the
GFNI kernel, so it has **not run on real hardware yet**. It is therefore offered
only if, at start-up, it reproduces the product table for all 255 non-trivial
constants; otherwise `Tier()` reports `avx2`. On a GFNI host,
`GF256_WANT_TIER=gfni+avx2 go test ./...` fails unless the tier is present.

GF arithmetic is exact, so the contract is equality: every tier is asserted
byte-for-byte against a table-free shift-and-add reference for all 256 constants
and lengths around every block boundary, plus fuzzing, in-place and short-slice
cases, and guard-page tests (operands flush against `PROT_NONE` pages on both
sides, lengths 0..200) that fault on a one-byte over-read or over-write. The
matrix fed to `VGF2P8AFFINEQB` is proven against the Intel SDM definition in pure
Go, so that test runs on every host.

## Sources

- Intel SDM, `GF2P8AFFINEQB`/`VGF2P8AFFINEQB` (read through the
  [x86 reference mirror](https://www.felixcloutier.com/x86/gf2p8affineqb)): matrix
  in the second source, output bit *i* is the parity of matrix byte *7-i* AND the
  data byte; requires AVX+GFNI for the 256-bit VEX form.
- [ISA-L](https://github.com/intel/isa-l) `erasure_code/ec_base.c`/`ec_base.h`:
  the 32-byte split table (16 low-nibble products, 16 high-nibble products), the
  0x11D reduction, and `gf_table_gfni`, the per-constant GFNI matrices.
- [klauspost/reedsolomon](https://github.com/klauspost/reedsolomon): the
  performance reference below (ISA-L is not installed on the measuring hosts).

## Performance

Reference: klauspost/reedsolomon `Encode`, single goroutine, 10 data + 4 parity
shards (40 multiply-accumulates per shard column). Source:
[`bench/interleave`](bench/interleave), interleaved rounds, medians.

Not yet measured: the only hosts available when v0.1.0 was cut were at 1-minute
load 21-40 (threshold 8), and no host with GFNI was reachable, so no number is
claimed. Run `go run . -load <1-minute load average>` in `bench/interleave` on a
quiet host (it refuses above load 8). Open item until a table lands here.

## Layout

`gf256.go` (field, tables, dispatch), `gf256_{amd64,arm64,noasm}.go`,
generated `gf256_{amd64,arm64}.s`; regenerate with `cd gen && go run .`
(CI fails if the committed assembly is stale).

BSD-3-Clause.
