package gf256

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
)

// allTiers is every runnable tier plus the generic loop.
func allTiers() []tier { return append(append([]tier{}, tiers...), tier{name: "generic"}) }

func withTier(t testing.TB, ti tier) {
	old := active
	active = ti
	t.Cleanup(func() { active = old })
}

func refMul(dst, src []byte, c byte) {
	for i := range src {
		dst[i] = mulRef(c, src[i])
	}
}

func refMulAdd(dst, src []byte, c byte) {
	for i := range src {
		dst[i] ^= mulRef(c, src[i])
	}
}

func TestFieldAxioms(t *testing.T) {
	// x^8 = x^4+x^3+x^2+1: 2*128 = 0x1d; 2 generates the group of order 255.
	if got := Mul(2, 128); got != 0x1d {
		t.Fatalf("2*128 = %#x, want 0x1d", got)
	}
	seen := map[byte]bool{}
	x := byte(1)
	for i := 0; i < 255; i++ {
		seen[x] = true
		x = Mul(x, 2)
	}
	if len(seen) != 255 || x != 1 {
		t.Fatalf("2 does not generate GF(256)*: %d elements, back at %#x", len(seen), x)
	}
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b += 7 {
			if Mul(byte(a), byte(b)) != Mul(byte(b), byte(a)) {
				t.Fatalf("Mul not commutative at %d,%d", a, b)
			}
			for c := 0; c < 256; c += 31 {
				l := Mul(byte(a), byte(b)^byte(c))
				r := Mul(byte(a), byte(b)) ^ Mul(byte(a), byte(c))
				if l != r {
					t.Fatalf("not distributive at %d,%d,%d", a, b, c)
				}
			}
		}
	}
}

func TestTierName(t *testing.T) {
	if Tier() == "" {
		t.Fatal("empty tier name")
	}
	t.Logf("tiers: %d SIMD, active %s", len(tiers), Tier())
}

// Every constant, every tier, lengths around every block boundary.
func TestAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	lens := []int{0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 127, 128, 129, 255, 1000}
	for _, ti := range allTiers() {
		t.Run(ti.name, func(t *testing.T) {
			withTier(t, ti)
			for c := 0; c < 256; c++ {
				for _, n := range lens {
					src := make([]byte, n)
					seed := make([]byte, n)
					for i := range src {
						src[i], seed[i] = byte(rng.Uint32()), byte(rng.Uint32())
					}
					want := make([]byte, n)
					got := make([]byte, n)
					refMul(want, src, byte(c))
					MulSlice(got, src, byte(c))
					if !bytes.Equal(got, want) {
						t.Fatalf("MulSlice c=%d n=%d\n got %x\nwant %x", c, n, got, want)
					}
					want = append(want[:0], seed...)
					got = append(got[:0], seed...)
					refMulAdd(want, src, byte(c))
					MulAddSlice(got, src, byte(c))
					if !bytes.Equal(got, want) {
						t.Fatalf("MulAddSlice c=%d n=%d\n got %x\nwant %x", c, n, got, want)
					}
				}
			}
		})
	}
}

func TestMismatchedLengthsAndInPlace(t *testing.T) {
	for _, ti := range allTiers() {
		t.Run(ti.name, func(t *testing.T) {
			withTier(t, ti)
			src := make([]byte, 100)
			for i := range src {
				src[i] = byte(i * 7)
			}
			// dst shorter: only len(dst) bytes written.
			dst := make([]byte, 40)
			MulSlice(dst, src, 29)
			want := make([]byte, 40)
			refMul(want, src[:40], 29)
			if !bytes.Equal(dst, want) {
				t.Fatal("short dst wrong")
			}
			// src shorter: the tail of dst is untouched.
			dst = bytes.Repeat([]byte{0xaa}, 100)
			MulAddSlice(dst, src[:37], 3)
			want = bytes.Repeat([]byte{0xaa}, 100)
			refMulAdd(want, src[:37], 3)
			if !bytes.Equal(dst, want) {
				t.Fatal("short src wrong")
			}
			// in place.
			buf := append([]byte{}, src...)
			MulSlice(buf, buf, 0x8e)
			want = make([]byte, 100)
			refMul(want, src, 0x8e)
			if !bytes.Equal(buf, want) {
				t.Fatal("in place MulSlice wrong")
			}
			buf = append([]byte{}, src...)
			MulAddSlice(buf, buf, 5) // x ^ 5x = 4x
			refMul(want, src, 4)
			if !bytes.Equal(buf, want) {
				t.Fatal("in place MulAddSlice wrong")
			}
			MulSlice(nil, nil, 9)
			MulAddSlice(nil, src, 9)
		})
	}
}

func FuzzMulAdd(f *testing.F) {
	f.Add([]byte("hello, erasure coding"), byte(0x1d), []byte{1, 2, 3})
	f.Add(make([]byte, 200), byte(255), make([]byte, 200))
	f.Add([]byte{}, byte(2), []byte{})
	f.Fuzz(func(t *testing.T, src []byte, c byte, seed []byte) {
		for _, ti := range allTiers() {
			withTier(t, ti)
			dst := make([]byte, len(src))
			copy(dst, seed)
			want := append([]byte{}, dst...)
			refMulAdd(want, src, c)
			MulAddSlice(dst, src, c)
			if !bytes.Equal(dst, want) {
				t.Fatalf("%s MulAddSlice c=%d len=%d", ti.name, c, len(src))
			}
			refMul(want, src, c)
			MulSlice(dst, src, c)
			if !bytes.Equal(dst, want) {
				t.Fatalf("%s MulSlice c=%d len=%d", ti.name, c, len(src))
			}
		}
	})
}

func BenchmarkMulAdd(b *testing.B) {
	for _, ti := range allTiers() {
		for _, n := range []int{256, 4096, 65536, 1 << 20} {
			b.Run(fmt.Sprintf("%s/%d", ti.name, n), func(b *testing.B) {
				withTier(b, ti)
				src, dst := make([]byte, n), make([]byte, n)
				for i := range src {
					src[i] = byte(i * 31)
				}
				b.SetBytes(int64(n))
				for b.Loop() {
					MulAddSlice(dst, src, 0x57)
				}
			})
		}
	}
}

func BenchmarkMul(b *testing.B) {
	for _, ti := range allTiers() {
		for _, n := range []int{4096, 1 << 20} {
			b.Run(fmt.Sprintf("%s/%d", ti.name, n), func(b *testing.B) {
				withTier(b, ti)
				src, dst := make([]byte, n), make([]byte, n)
				b.SetBytes(int64(n))
				for b.Loop() {
					MulSlice(dst, src, 0x57)
				}
			})
		}
	}
}

// affineByte is the pseudocode of Intel SDM VGF2P8AFFINEQB for imm8 = 0: output
// bit i is the parity of matrix byte 7-i ANDed with the data byte.
func affineByte(m uint64, x byte) byte {
	var out byte
	for i := 0; i < 8; i++ {
		row := byte(m >> (8 * (7 - i)))
		v := row & x
		v ^= v >> 4
		v ^= v >> 2
		v ^= v >> 1
		out |= (v & 1) << i
	}
	return out
}

// The matrix table feeds an instruction this host may lack, so its meaning is
// proven here in pure Go against the SDM definition, for every c and x.
func TestMatrixMatchesSDM(t *testing.T) {
	for c := 0; c < 256; c++ {
		for x := 0; x < 256; x++ {
			if got, want := affineByte(mat[c], byte(x)), mulRef(byte(c), byte(x)); got != want {
				t.Fatalf("matrix of %#x on %#x: %#x, want %#x", c, x, got, want)
			}
		}
	}
}

// GF256_WANT_TIER lets a CI lane that knows its CPU assert the tier exists
// (a tier silently missing would leave its kernel untested).
func TestWantTier(t *testing.T) {
	want := os.Getenv("GF256_WANT_TIER")
	if want == "" {
		t.Skip("GF256_WANT_TIER not set")
	}
	for _, ti := range tiers {
		if ti.name == want {
			return
		}
	}
	t.Fatalf("tier %q not available; have %v", want, tiers)
}
