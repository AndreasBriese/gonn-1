package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"strconv"
)

func randomWeights(x, y int) [][]float64 {
	weights := make([][]float64, x)
	halfmax := math.Sqrt(6) / math.Sqrt(float64(x+y))

	for i := 0; i < len(weights); i++ {
		weights[i] = make([]float64, y)
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = (rand.Float64() - 0.5) * halfmax
		}
	}
	return weights
}

func loadData(url string) [][]float64 {
	f, err := os.Open(url)
	if err != nil {
		panic(err)
	}

	r := csv.NewReader(f)
	recs, err := r.ReadAll()
	if err != nil {
		panic(err)
	}

	rows := make([][]float64, len(recs))
	for i := 0; i < len(recs); i++ {
		rows[i] = make([]float64, len(recs[i]))
		for j := 0; j < len(recs[i]); j++ {
			rows[i][j], err = strconv.ParseFloat(recs[i][j], 32)
			if err != nil {
				panic(err)
			}
		}
	}
	return rows
}

func sigm(input float64) (sigmoid, gradient float64) {
	sigmoid = 1 / (1 + math.Exp(-input))
	gradient = sigmoid * (1 - sigmoid)
	return
}

func errFunc(target, actual float64) float64 {
	diff := target - actual
	return 0.5 * math.Pow(diff, 2) // TODO: maybe diff * diff is faster?
}

func dotProduct(a, b []float64) (sum float64) {
	if len(a) != len(b) {
		panic(fmt.Errorf("Lists a and b should be of the same length; instead got %d and %d", len(a), len(b)))
	}
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return
}

func dotProductTranspose(a []float64, b [][]float64, c int) (sum float64) {
	if len(a) != len(b) {
		panic(fmt.Errorf("Lists a and b should be of the same length; instead got %d and %d", len(a), len(b)))
	}
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i][c]
	}
	return
}

func mulMatrix(output []float64, a float64, b []float64) {
	if len(output) != len(b) {
		panic(fmt.Errorf("Output and b should be of the same length; instead got %d and %d", len(output), len(b)))
	}

	for i := 0; i < len(output); i++ {
		output[i] = b[i] * a
	}
}

func subMatrix(original, b []float64) {
	if len(original) != len(b) {
		panic("Length of original and b should be the same")
	}
	for i := 0; i < len(original); i++ {
		original[i] -= b[i]
	}
}

func equals(predicted, actual float64) bool {
	bothOne := predicted > 0.5 && actual > 0.5
	bothZero := predicted < 0.5 && actual < 0.5
	return bothOne || bothZero
}

var output = true

func main() {
	f, err := os.Create("cpu.pprof")
	if err != nil {
		log.Fatal(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	XTest := loadData("data/X_Test.csv")         // = pixels  * digits
	TTest := loadData("data/T_Test.csv")         // = classes * digits
	XTraining := loadData("data/X_Training.csv") // = pixels  * digits
	TTraining := loadData("data/T_Training.csv") // = classes * digits
	eta := 0.001

	numEpochs := 1000
	numSamples := len(XTraining)
	numFeatures := len(XTraining[0])

	numInput := numFeatures        // We have 400 pixels (20x20)
	numHidden := 25                // TODO - this can be whatever we want it to be
	numOutput := len(TTraining[0]) // We have 10 times 0 or 1

	W1 := randomWeights(numHidden, numInput)
	W2 := randomWeights(numOutput, numHidden)

	layer2Inputs := make([]float64, numHidden)
	layer2Outputs := make([]float64, len(layer2Inputs))
	layer2Gradients := make([]float64, len(layer2Inputs))
	layer2Delta := make([]float64, len(layer2Inputs))

	layer3Inputs := make([]float64, numOutput)
	layer3Outputs := make([]float64, len(layer3Inputs))
	layer3Gradients := make([]float64, len(layer3Inputs))
	layer3Delta := make([]float64, len(layer3Inputs))

	var errTotal float64
	tempWeights1 := make([]float64, len(W1[0]))
	tempWeights2 := make([]float64, len(W2[0]))

	// Train
	for e := 0; e < numEpochs; e++ {
		for s := 0; s < numSamples; s++ {
			// Compute output of layer 2 (hidden)
			for i := 0; i < len(layer2Inputs); i++ {
				layer2Inputs[i] = dotProduct(W1[i], XTraining[s])
				layer2Outputs[i], layer2Gradients[i] = sigm(layer2Inputs[i]) // all pixels multiplied by all the appropriate weights
			}

			// Compute output of layer 3 (output)
			for i := 0; i < len(layer3Inputs); i++ {
				layer3Inputs[i] = dotProduct(W2[i], layer2Outputs)
				layer3Outputs[i], layer3Gradients[i] = sigm(layer3Inputs[i]) // all pixels multiplied by all the appropriate weights

				// Compute error of layer 3 (output)
				errTotal += errFunc(layer3Outputs[i], TTraining[s][i])
				layer3Delta[i] = (layer3Outputs[i] - TTraining[s][i]) * layer3Gradients[i]
			}

			// Back-prop error of layer 2 (hidden)
			for i := 0; i < len(layer2Inputs); i++ {
				sum := dotProductTranspose(layer3Delta, W2, i)
				layer2Delta[i] = sum * layer2Gradients[i]
			}

			// Update the weights of layer 1->2
			for i := 0; i < len(layer2Inputs); i++ {
				mulMatrix(tempWeights1, eta*layer2Delta[i], XTraining[s])
				subMatrix(W1[i], tempWeights1)
			}

			// Update the weights of layer 2->3
			for i := 0; i < len(layer3Inputs); i++ {
				mulMatrix(tempWeights2, eta*layer3Delta[i], layer2Outputs)
				subMatrix(W2[i], tempWeights2)
			}
		}
	}

	// Test
	var correct int

	for s := 0; s < len(XTest); s++ {
		// Compute output of layer 2 (hidden)
		for i := 0; i < len(layer2Inputs); i++ {
			layer2Inputs[i] = dotProduct(W1[i], XTest[s])
			layer2Outputs[i], _ = sigm(layer2Inputs[i]) // all pixels multiplied by all the appropriate weights
		}

		// Compute output of layer 3 (output)
		c := true
		for i := 0; i < len(layer3Inputs); i++ {
			layer3Inputs[i] = dotProduct(W2[i], layer2Outputs)
			layer3Outputs[i], _ = sigm(layer3Inputs[i]) // all pixels multiplied by all the appropriate weights

			if !equals(layer3Outputs[i], TTest[s][i]) {
				c = false
			}
		}

		// If any of the output-nodes did not match, c is false
		if c {
			correct++
		}
	}

	if output {
		fmt.Printf("Classification rate was %.6f out of %d test samples", (float64(correct) / float64(len(TTest))), len(TTest))
	}
}
