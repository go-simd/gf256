// Package gf256 multiplies byte slices by a constant in GF(2^8), the field of
// Reed-Solomon erasure codes.
//
// The field is the one ISA-L, klauspost/reedsolomon and Backblaze use: the
// polynomial x^8+x^4+x^3+x^2+1 (0x11D), addition is XOR. MulSlice sets
// dst = c*src and MulAddSlice accumulates dst ^= c*src, the two inner loops of
// an erasure encoder.
//
// Kernels, fastest available first (see Tier):
//
//   - amd64 "gfni+avx2": one VGF2P8AFFINEQB per 32 bytes with the 8x8 bit
//     matrix of c (Intel SDM, VGF2P8AFFINEQB).
//   - amd64 "avx2": two VPSHUFB nibble lookups per 32 bytes.
//   - amd64 "ssse3": two PSHUFB nibble lookups per 16 bytes.
//   - arm64 "neon": two TBL nibble lookups per 16 bytes.
//   - "generic": a 64 KiB product table, one lookup per byte.
//
// GF arithmetic is exact: every tier returns the same bytes as the scalar
// reference, which the tests assert for all 256 constants.
package gf256

// poly is the reduction polynomial 0x11D without its x^8 term.
const poly = 0x1d

// mulRef is the equality oracle: shift-and-add multiplication, no tables.
func mulRef(a, b byte) byte {
	var p byte
	for ; b != 0; b >>= 1 {
		if b&1 != 0 {
			p ^= a
		}
		hi := a & 0x80
		a <<= 1
		if hi != 0 {
			a ^= poly
		}
	}
	return p
}

var (
	// mulTab[c][x] = c*x.
	mulTab [256][256]byte
	// nib[c] is the split table of c: nib[c][i] = c*i for i < 16, then
	// nib[c][16+i] = c*(i<<4). c*x = nib[c][x&15] ^ nib[c][16+x>>4].
	nib [256][32]byte
	// mat[c] is the VGF2P8AFFINEQB matrix of x -> c*x: byte 7-i of the qword is
	// the row producing output bit i, whose bit k is bit i of c*2^k.
	mat [256]uint64
)

func init() {
	for c := 0; c < 256; c++ {
		for x := 0; x < 256; x++ {
			mulTab[c][x] = mulRef(byte(c), byte(x))
		}
		for i := 0; i < 16; i++ {
			nib[c][i] = mulTab[c][i]
			nib[c][16+i] = mulTab[c][i<<4]
		}
		var m uint64
		for i := 0; i < 8; i++ {
			var row uint64
			for k := 0; k < 8; k++ {
				row |= uint64(mulTab[c][1<<k]>>i&1) << k
			}
			m |= row << (8 * (7 - i))
		}
		mat[c] = m
	}
}

// Mul returns a*b in GF(2^8).
func Mul(a, b byte) byte { return mulTab[a][b] }

// kernel processes a prefix of dst and src (equal lengths) and returns how many
// bytes it handled; the generic loop finishes the rest.
type kernel func(dst, src []byte, c byte) int

type tier struct {
	name        string
	mul, mulAdd kernel
}

var (
	// tiers are the SIMD tiers this CPU can run, fastest first (archTiers).
	tiers  = archTiers()
	active = pick(tiers)
)

func pick(ts []tier) tier {
	if len(ts) > 0 {
		return ts[0]
	}
	return tier{name: "generic"}
}

// Tier names the kernel MulSlice and MulAddSlice use on this CPU.
func Tier() string { return active.name }

// MulSlice sets dst[i] = c*src[i] for i < min(len(dst), len(src)). dst and src
// must be identical or disjoint.
func MulSlice(dst, src []byte, c byte) {
	n := min(len(dst), len(src))
	dst, src = dst[:n], src[:n]
	switch c {
	case 0:
		clear(dst)
		return
	case 1:
		copy(dst, src)
		return
	}
	done := 0
	if active.mul != nil {
		done = active.mul(dst, src, c)
	}
	row := &mulTab[c]
	for i := done; i < n; i++ {
		dst[i] = row[src[i]]
	}
}

// MulAddSlice sets dst[i] ^= c*src[i] for i < min(len(dst), len(src)). dst and
// src must be identical or disjoint.
func MulAddSlice(dst, src []byte, c byte) {
	n := min(len(dst), len(src))
	dst, src = dst[:n], src[:n]
	if c == 0 {
		return
	}
	done := 0
	if active.mulAdd != nil {
		done = active.mulAdd(dst, src, c)
	}
	row := &mulTab[c]
	for i := done; i < n; i++ {
		dst[i] ^= row[src[i]]
	}
}
