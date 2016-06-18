package main

import (
	"fmt"
	"math/rand"
)

func getX() [][]float64 {
	return [][]float64{
		[]float64{1, 1, 1},
		[]float64{1, 0, 1},
		[]float64{1, 0, 0},
		[]float64{1, 1, 0},
	}
}

func getY() []float64 {
	return []float64{
		1,
		0,
		0,
		0,
	}
}

func randomWeights(amount int) []float64 {
	weights := make([]float64, amount)
	for i := 0; i < len(weights); i++ {
		weights[i] = rand.Float64()
	}
	return weights
}

func dotProduct(a, b []float64) (sum float64) {
	if len(a) != len(b) {
		panic("Lists a and b should be of the same length")
	}
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return
}

func mulMatrix(output []float64, a float64, b []float64) {
	if len(output) != len(b) {
		panic("Output and b should be of the same length")
	}

	for i := 0; i < len(output); i++ {
		output[i] = b[i] * a
	}
}

func addMatrix(original, b []float64) {
	if len(original) != len(b) {
		panic("Length of original and b should be the same")
	}
	for i := 0; i < len(original); i++ {
		original[i] += b[i]
	}
}

func train(numEpochs, numFeatures, numSamples int, alpha float64, weights, Y []float64, X [][]float64) {
	tempWeights := make([]float64, numFeatures)

	for e := 0; e < numEpochs; e++ {
		for p := 0; p < numSamples; p++ {
			activation := dotProduct(weights, X[p])
			target := Y[p]
			errorTerm := target - activation
			mulMatrix(tempWeights, errorTerm*alpha, X[p])

			addMatrix(weights, tempWeights)
		}
	}
}

func test(numSamples int, weights, Y []float64, X [][]float64) {
	for p := 0; p < numSamples; p++ {
		activation := dotProduct(weights, X[p])
		target := Y[p]
		if output {
			fmt.Printf("For index %d\tpredicted %.2f\tactual %.2f\ttherefore %v\n", p, activation, target, equals(activation, target))
		}
	}
}

func equals(predicted, actual float64) bool {
	bothOne := predicted > 0.5 && actual > 0.5
	bothZero := predicted < 0.5 && actual < 0.5
	return bothOne || bothZero
}

var output = true

func main() {
	X := getX()
	Y := getY()

	alpha := 0.1
	numFeatures := len(X[0])
	numSamples := len(X)
	numEpochs := 100
	weights := randomWeights(numFeatures)

	train(numEpochs, numFeatures, numSamples, alpha, weights, Y, X)
	test(numSamples, weights, Y, X)
}
