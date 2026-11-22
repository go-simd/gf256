//go:build !amd64 && !arm64

package gf256

func archTiers() []tier { return nil }
