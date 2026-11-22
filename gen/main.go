// Command gen writes ../gf256_amd64.s and ../gf256_arm64.s with go-asmgen: the
// region kernels of package gf256.
//
// Every kernel has the shape  k(dst, src []byte, p, n)  and processes the first
// n bytes of both slices (n is a multiple of the kernel's block, checked by the
// Go wrapper). Per constant c the wrapper passes p:
//
//   - GFNI+AVX2 (mulGFNI, mulAddGFNI): p = the 8x8 GF(2) matrix of x -> c*x as
//     one uint64, broadcast to every qword; one VGF2P8AFFINEQB per 32 bytes.
//     Intel SDM, VGF2P8AFFINEQB: dst.bit[i] = parity(matrix.byte[7-i] & x) ^ imm8.bit[i],
//     matrix in the second source (memory) operand, data in the first.
//   - AVX2 / SSSE3 / NEON: p = pointer to the 32-byte split table, the 16
//     products c*n for the low nibble then the 16 products c*(n<<4) for the
//     high nibble (ISA-L gf_vect_mul_init layout); c*x = lo[x&15] ^ hi[x>>4],
//     two byte shuffles (VPSHUFB / PSHUFB / VTBL) per vector.
//
// cpuidGFNI reports CPUID.(EAX=7,ECX=0):ECX bit 8, which x/sys/cpu only records
// when AVX-512 is present (HasAVX512GFNI) and so misses AVX2-only parts.
//
// Run: cd gen && go run .
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func sig(p abi.Arg) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), p, abi.Scalar("n", abi.Int64)},
		nil,
	)
}

func ymm(i int) string { return fmt.Sprintf("Y%d", i) }

// genGFNI emits the GFNI+AVX2 kernel: 64 bytes per iteration, then one 32-byte
// block when n%64 == 32.
func genGFNI(f *emit.File, name string, muladd bool) {
	b := amd64.NewFunc(name, sig(abi.Scalar("m", abi.Int64)), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").
		LoadArg("m", "AX").LoadArg("n", "CX").
		Raw("MOVQ AX, X0").
		Raw("VPBROADCASTQ X0, Y0").
		Raw("CMPQ CX, $64").
		Raw("JLO l32").
		Label("l64").
		Raw("VMOVDQU (SI), Y1").
		Raw("VMOVDQU 32(SI), Y2").
		Raw("VGF2P8AFFINEQB $0, Y0, Y1, Y1").
		Raw("VGF2P8AFFINEQB $0, Y0, Y2, Y2")
	if muladd {
		b.Raw("VPXOR (DI), Y1, Y1").Raw("VPXOR 32(DI), Y2, Y2")
	}
	b.Raw("VMOVDQU Y1, (DI)").
		Raw("VMOVDQU Y2, 32(DI)").
		Raw("ADDQ $64, SI").Raw("ADDQ $64, DI").
		Raw("SUBQ $64, CX").
		Raw("CMPQ CX, $64").
		Raw("JHS l64").
		Label("l32").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Raw("VMOVDQU (SI), Y1").
		Raw("VGF2P8AFFINEQB $0, Y0, Y1, Y1")
	if muladd {
		b.Raw("VPXOR (DI), Y1, Y1")
	}
	b.Raw("VMOVDQU Y1, (DI)").
		Label("done").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// avx2Block emits one 32-byte block through registers Y(r), Y(r+1): load, split
// nibbles, two VPSHUFB (tables Y6 lo, Y7 hi; mask Y8), XOR, store.
func avx2Block(b *amd64.Builder, off int, r int, muladd bool) {
	d, h := ymm(r), ymm(r+1)
	b.Raw("VMOVDQU %d(SI), %s", off, d).
		Raw("VPSRLQ $4, %s, %s", d, h).
		Raw("VPAND Y8, %s, %s", d, d).
		Raw("VPAND Y8, %s, %s", h, h).
		Raw("VPSHUFB %s, Y6, %s", d, d).
		Raw("VPSHUFB %s, Y7, %s", h, h).
		Raw("VPXOR %s, %s, %s", d, h, d)
	if muladd {
		b.Raw("VPXOR %d(DI), %s, %s", off, d, d)
	}
	b.Raw("VMOVDQU %s, %d(DI)", d, off)
}

func genAVX2(f *emit.File, name string, muladd bool) {
	b := amd64.NewFunc(name, sig(abi.Scalar("tbl", abi.Ptr)), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").
		LoadArg("tbl", "DX").LoadArg("n", "CX").
		Raw("VBROADCASTI128 (DX), Y6").
		Raw("VBROADCASTI128 16(DX), Y7").
		Raw("MOVQ $0x0f0f0f0f0f0f0f0f, AX").
		Raw("MOVQ AX, X8").
		Raw("VPBROADCASTQ X8, Y8").
		Raw("CMPQ CX, $64").
		Raw("JLO l32").
		Label("l64")
	avx2Block(b, 0, 0, muladd)
	avx2Block(b, 32, 2, muladd)
	b.Raw("ADDQ $64, SI").Raw("ADDQ $64, DI").
		Raw("SUBQ $64, CX").
		Raw("CMPQ CX, $64").
		Raw("JHS l64").
		Label("l32").
		Raw("TESTQ CX, CX").
		Raw("JZ done")
	avx2Block(b, 0, 0, muladd)
	b.Label("done").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

func genSSSE3(f *emit.File, name string, muladd bool) {
	b := amd64.NewFunc(name, sig(abi.Scalar("tbl", abi.Ptr)), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").
		LoadArg("tbl", "DX").LoadArg("n", "CX").
		Raw("MOVOU (DX), X6").
		Raw("MOVOU 16(DX), X7").
		Raw("MOVQ $0x0f0f0f0f0f0f0f0f, AX").
		Raw("MOVQ AX, X8").
		Raw("PUNPCKLQDQ X8, X8").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Label("loop").
		Raw("MOVOU (SI), X0").
		Raw("MOVO X0, X1").
		Raw("PSRLQ $4, X1").
		Raw("PAND X8, X0").
		Raw("PAND X8, X1").
		Raw("MOVO X6, X2").
		Raw("PSHUFB X0, X2").
		Raw("MOVO X7, X3").
		Raw("PSHUFB X1, X3").
		Raw("PXOR X3, X2")
	if muladd {
		b.Raw("MOVOU (DI), X4").Raw("PXOR X4, X2")
	}
	b.Raw("MOVOU X2, (DI)").
		Raw("ADDQ $16, SI").Raw("ADDQ $16, DI").
		Raw("SUBQ $16, CX").
		Raw("JNZ loop").
		Label("done").
		Ret()
	f.Add(b.Func())
}

func genCPUID(f *emit.File) {
	b := amd64.NewFunc("cpuidGFNI", abi.LayoutArgs(nil, []abi.Arg{abi.Scalar("ok", abi.Int64)}), 0)
	b.Raw("MOVL $7, AX").
		Raw("XORL CX, CX").
		Raw("CPUID").
		Raw("SHRL $8, CX").
		Raw("ANDL $1, CX").
		Raw("MOVQ CX, R8").
		StoreRet("R8", "ok").
		Ret()
	f.Add(b.Func())
}

// genNEON emits the NEON kernel: 64 bytes per iteration, then 16-byte blocks.
func genNEON(f *emit.File, name string, muladd bool) {
	b := arm64.NewFunc(name, sig(abi.Scalar("tbl", abi.Ptr)), 0)
	b.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").
		LoadArg("tbl", "R4").LoadArg("n", "R2").
		Raw("VLD1 (R4), [V28.B16, V29.B16]").
		Raw("VMOVI $15, V30.B16").
		Raw("MOVD R0, R5"). // R5 = dst read pointer for muladd
		Raw("CMP $64, R2").
		Raw("BLO l16").
		Label("l64").
		Raw("VLD1.P 64(R1), [V0.B16, V1.B16, V2.B16, V3.B16]")
	for j := 0; j < 4; j++ {
		neonLane(b, j)
	}
	if muladd {
		b.Raw("VLD1 (R5), [V12.B16, V13.B16, V14.B16, V15.B16]")
		for j := 0; j < 4; j++ {
			b.Raw("VEOR V%d.B16, V%d.B16, V%d.B16", 12+j, 4+j, 4+j)
		}
	}
	b.Raw("VST1.P [V4.B16, V5.B16, V6.B16, V7.B16], 64(R0)").
		Raw("ADD $64, R5").
		Raw("SUB $64, R2").
		Raw("CMP $64, R2").
		Raw("BHS l64").
		Label("l16").
		Raw("CBZ R2, done").
		Raw("VLD1.P 16(R1), [V0.B16]")
	neonLane(b, 0)
	if muladd {
		b.Raw("VLD1 (R5), [V12.B16]").Raw("VEOR V12.B16, V4.B16, V4.B16")
	}
	b.Raw("VST1.P [V4.B16], 16(R0)").
		Raw("ADD $16, R5").
		Raw("SUB $16, R2").
		Raw("CBNZ R2, l16").
		Label("done").
		Ret()
	f.Add(b.Func())
}

// neonLane computes V(4+j) = lo[V(j)&15] ^ hi[V(j)>>4]; V(8+j) is scratch.
func neonLane(b *arm64.Builder, j int) {
	b.Raw("VAND V30.B16, V%d.B16, V%d.B16", j, 4+j).
		Raw("VUSHR $4, V%d.B16, V%d.B16", j, 8+j).
		Raw("VTBL V%d.B16, [V28.B16], V%d.B16", 4+j, 4+j).
		Raw("VTBL V%d.B16, [V29.B16], V%d.B16", 8+j, 8+j).
		Raw("VEOR V%d.B16, V%d.B16, V%d.B16", 8+j, 4+j, 4+j)
}

func write(name, s string) {
	if err := os.WriteFile(name, []byte(s), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

func main() {
	fa := emit.NewFile("amd64")
	genGFNI(fa, "mulGFNI", false)
	genGFNI(fa, "mulAddGFNI", true)
	genAVX2(fa, "mulAVX2", false)
	genAVX2(fa, "mulAddAVX2", true)
	genSSSE3(fa, "mulSSSE3", false)
	genSSSE3(fa, "mulAddSSSE3", true)
	genCPUID(fa)
	write("../gf256_amd64.s", fa.String())

	fr := emit.NewFile("arm64")
	genNEON(fr, "mulNEON", false)
	genNEON(fr, "mulAddNEON", true)
	write("../gf256_arm64.s", fr.String())
}
