package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
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

func TestOR(t *testing.T) {
	X := [][]float64{
		[]float64{1, 1, 1},
		[]float64{1, 1, 0},
		[]float64{1, 0, 1},
		[]float64{1, 0, 0},
	}
	Y := []float64{
		1,
		1,
		1,
		0,
	}

	alpha := 0.1
	numFeatures := len(X[0])
	numSamples := len(X)
	numEpochs := 100
	weights := randomWeights(numFeatures)

	train(numEpochs, numFeatures, numSamples, alpha, weights, Y, X)

	// Verify
	for p := 0; p < numSamples; p++ {
		activation := dotProduct(weights, X[p])
		target := Y[p]
		assert.True(t, equals(activation, target), "Prediction for sample %v failed (got %v expected %v)", X[p], activation, target)
	}
}

func TestAND(t *testing.T) {
	X := [][]float64{
		[]float64{1, 1, 1},
		[]float64{1, 1, 0},
		[]float64{1, 0, 1},
		[]float64{1, 0, 0},
	}
	Y := []float64{
		1,
		0,
		0,
		0,
	}

	alpha := 0.1
	numFeatures := len(X[0])
	numSamples := len(X)
	numEpochs := 100
	weights := randomWeights(numFeatures)

	train(numEpochs, numFeatures, numSamples, alpha, weights, Y, X)

	// Verify
	for p := 0; p < numSamples; p++ {
		activation := dotProduct(weights, X[p])
		target := Y[p]
		assert.True(t, equals(activation, target), "Prediction for sample %v failed (got %v expected %v)", X[p], activation, target)
	}
}
