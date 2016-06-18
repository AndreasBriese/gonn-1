package main

import (
	"testing"
)

func BenchmarkSLP(b *testing.B) {
	output = false

	X := getX()
	Y := getY()
	if len(X) == 0 {
		panic("Expected len(X) > 0")
	}
	alpha := 0.1
	numFeatures := len(X[0])
	numSamples := len(X)
	numEpochs := 100
	weights := randomWeights(numFeatures)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		train(numEpochs, numFeatures, numSamples, alpha, weights, Y, X)
	}

	test(numSamples, weights, Y, X)
}
