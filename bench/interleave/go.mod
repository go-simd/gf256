module github.com/go-simd/gf256/bench/interleave

go 1.27.1

require (
	github.com/go-simd/gf256 v0.0.0
	github.com/klauspost/reedsolomon v1.13.3
)

require (
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	golang.org/x/sys v0.50.0 // indirect
)

replace github.com/go-simd/gf256 => ../..
