// File:  NeuralNetwork.java
// Name:  Mo K. Eid (mohamedkeid@gmail.com)
// Desc:  A customizable neural network class built from scratch that makes use of batch gradient descent

import java.util.Random;

class NeuralNetwork {
    // Instance variables related to the structure of the neural network
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int numClassifiers;

    // Matrix containing the weight vectors between each layer
    private double[][] weightsForAllLayers;

    // Instance variables related to optimization
    private double learningRate;
    private double regularizationRate;

    public NeuralNetwork(int paramInputSize, int[] paramHiddenLayerSizes, int paramNumClassifiers) {
        inputSize = paramInputSize;
        hiddenLayerSizes = paramHiddenLayerSizes;
        numClassifiers = paramNumClassifiers;
        weightsForAllLayers = generateRandomWeights();
        learningRate = 0.1;
        regularizationRate = 0.3;
    }

    // Infer a function from labeled training data
    public void train(int[][] trainingExamples, int[] trainingActual, int numIterations) {
        int i = 0;
        while(i < numIterations) {
            System.out.println("Training Iteration: " + (i + 1) + " of " + numIterations);
            backPropagation(trainingExamples, trainingActual);
            i++;
        }
    }

    // Make prediction based on its training
    public double[][] predict(int[] input) {
        double[][] predictions = new double[this.hiddenLayerSizes.length + 1][];

        for(int activationLayerIndex = 0; activationLayerIndex < predictions.length; activationLayerIndex++) {
            boolean onOutputLayer = (activationLayerIndex == predictions.length - 1);
            boolean previousIsInput = (activationLayerIndex == 0);

            // Compute layer sizes (the + 1 is for the bias node)
            int previousLayerSize = previousIsInput ? inputSize : (hiddenLayerSizes[activationLayerIndex - 1] + 1);
            int currentLayerSize = onOutputLayer ? numClassifiers : (hiddenLayerSizes[activationLayerIndex]);
            boolean shouldAddBias = !onOutputLayer;
            predictions[activationLayerIndex] = new double[currentLayerSize + (shouldAddBias ? 1 : 0)];

            // Compute activations for each node in the current layer
            for(int activationNodeIndex = 0; activationNodeIndex < currentLayerSize; activationNodeIndex++) {
                // Sum the connections between the appropriate nodes in the current layer and the previous layer
                double sum = 0.0;
                for(int previousNodeIndex = 0; previousNodeIndex < previousLayerSize; previousNodeIndex++) {
                    int weightIndex = (previousLayerSize * activationNodeIndex) + previousNodeIndex;
                    double weightVal = weightsForAllLayers[activationLayerIndex][weightIndex];
                    double previousNodeVal = previousIsInput ? input[previousNodeIndex] : predictions[activationLayerIndex - 1][previousNodeIndex];
                    sum += weightVal * previousNodeVal;
                }

                // Activate the summed value and assign it in the predictions matrix/initW
                double activation = activate(sum);
                predictions[activationLayerIndex][activationNodeIndex] = activation;
            }

            // Add bias if needed
            if(shouldAddBias) {
                int biasIndex = hiddenLayerSizes[activationLayerIndex];
                predictions[activationLayerIndex][biasIndex] = 1;
            }
        }

        return predictions;
    }

    // Return the rate the neural network predicts its own labeled training examples correctly
    public double checkAccuracy(int[][] inputSet, int[] actualSet) {
        double numCorrect = 0.0;

        for(int inputIndex = 1; inputIndex < inputSet.length - 1; inputIndex++) {
            int[] trainingExample = inputSet[inputIndex];
            double[][] predictions = predict(trainingExample);
            double[] predictedOutput = predictions[predictions.length - 1];
            int predictedClass = translatePrediction(predictedOutput);
            numCorrect += (predictedClass == actualSet[inputIndex]) ? 1 : 0;
        }

        return numCorrect / inputSet.length;
    }

    // Backward propagate errors in batch mode and update weights given labeled training data
    private void backPropagation( int[][] trainingExamples, int[] trainingActual) {
        // Initialize the delta accumulator
        double[][] gradient = new double[hiddenLayerSizes.length + 1][];

        // Learn from each training example
        for(int trainingExampleIndex = 0; trainingExampleIndex < trainingExamples.length - 1; trainingExampleIndex++) {
            double[][] predictions = predict(trainingExamples[trainingExampleIndex]);
            double[][] deltas = new double[weightsForAllLayers.length][];

            // Iterate through each weights layer
            for(int deltaLayerIndex = deltas.length - 1; deltaLayerIndex >= 0; deltaLayerIndex--) {
                boolean onOutputLayer = (deltaLayerIndex == deltas.length - 1);
                int currentLayerSize = onOutputLayer ? numClassifiers : hiddenLayerSizes[deltaLayerIndex] + 1;
                deltas[deltaLayerIndex] = new double[currentLayerSize];

                // Initialize the gradients for this layer if null
                if(gradient[deltaLayerIndex] == null)
                    gradient[deltaLayerIndex] = new double[weightsForAllLayers[deltaLayerIndex].length];

                // Calculate output deltas
                if(onOutputLayer) {
                    // Encode the class into a vector
                    int[] encodedActual = oneHotEncode(trainingActual[trainingExampleIndex]);

                    for(int nodeIndex = 0; nodeIndex < numClassifiers; nodeIndex++)
                        deltas[deltaLayerIndex][nodeIndex] = predictions[predictions.length - 1][nodeIndex] - encodedActual[nodeIndex];
                }

                // Calculate hidden layer deltas
                else {
                    boolean nextLayerIsOutput = (deltaLayerIndex == deltas.length - 2);
                    int nextLayerSize = nextLayerIsOutput ? numClassifiers : hiddenLayerSizes[deltaLayerIndex + 1];

                    // Iterate through each activation node
                    for(int nodeIndex = 0; nodeIndex < currentLayerSize; nodeIndex++) {
                        double delta = 0.0;

                        // Iterate through each of the next layer's activation nodes
                        for(int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                            int weightIndex = (currentLayerSize * nextNodeIndex) + nodeIndex;
                            double weightVal = weightsForAllLayers[deltaLayerIndex + 1][weightIndex];
                            double activationPrime = activatePrime(predictions[deltaLayerIndex + 1][nextNodeIndex]);
                            delta += weightVal * deltas[deltaLayerIndex + 1][nextNodeIndex] * activationPrime;
                        }

                        deltas[deltaLayerIndex][nodeIndex] = delta;
                    }
                }
            }

            // Accumulate deltas
            for(int deltaLayerIndex = deltas.length - 1; deltaLayerIndex >= 0; deltaLayerIndex--) {
                // Get parameters on the current layer
                boolean onOutputLayer = (deltaLayerIndex == deltas.length - 1);
                int currentLayerSize = onOutputLayer ? numClassifiers : hiddenLayerSizes[deltaLayerIndex];

                // Get parameters on the previous layer
                boolean previousIsInput = (deltaLayerIndex == 0);
                int previousLayerSize = previousIsInput ? inputSize : hiddenLayerSizes[deltaLayerIndex - 1] + 1;

                // Use each weight connection to calculate the gradient
                for(int previousNodeIndex = 0; previousNodeIndex < previousLayerSize; previousNodeIndex++) {
                    for(int currentNodeIndex = 0; currentNodeIndex < currentLayerSize; currentNodeIndex++) {
                        int weightIndex = (currentNodeIndex * previousLayerSize) + previousNodeIndex;
                        double previousActivation;

                        // Set activation to be 1 if the previous node is a bias
                        if(previousNodeIndex == previousLayerSize - 1)
                            previousActivation = 1;
                        else
                            previousActivation = previousIsInput ? trainingExamples[trainingExampleIndex][previousNodeIndex] : predictions[deltaLayerIndex - 1][previousNodeIndex];

                        gradient[deltaLayerIndex][weightIndex] += deltas[deltaLayerIndex][currentNodeIndex] * previousActivation;
                    }
                }
            }
        }

        // Update weights
        for(int deltaLayerIndex = 0; deltaLayerIndex < gradient.length; deltaLayerIndex++) {
            boolean onInputLayer = (deltaLayerIndex == 0);
            int currentLayerSize = onInputLayer ? inputSize : hiddenLayerSizes[deltaLayerIndex - 1] + 1;
            boolean nextIsOutLayer = (deltaLayerIndex == gradient.length - 1);
            int nextLayerSize = nextIsOutLayer ? numClassifiers : hiddenLayerSizes[deltaLayerIndex];

            for(int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                for(int currentNodeIndex = 0; currentNodeIndex < currentLayerSize; currentNodeIndex++) {
                    int weightIndex = (nextNodeIndex * currentLayerSize) + currentNodeIndex;
                    gradient[deltaLayerIndex][weightIndex] /= trainingExamples.length;
                    weightsForAllLayers[deltaLayerIndex][weightIndex] *= 1 - learningRate * regularizationRate / trainingExamples.length;
                    weightsForAllLayers[deltaLayerIndex][weightIndex] -= gradient[deltaLayerIndex][weightIndex] * learningRate;
                }
            }
        }
    }

    // Generate a random matrix of weight values between min and mix based on the neural network's structure
    private double[][] generateRandomWeights() {
        // There will be a layer of weights between the layers of the neural network
        double[][] weights = new double[hiddenLayerSizes.length + 1][];

        // The range of weight randomization
        double min = -1.0;
        double max = 1.0;

        // Iterate through each layer of nodes
        for(int weightLayerIndex = 0; weightLayerIndex < weights.length; weightLayerIndex++) {
            boolean forHiddenLayer = (weightLayerIndex != weights.length - 1);
            int currentLayerSize = forHiddenLayer ? hiddenLayerSizes[weightLayerIndex] : numClassifiers;

            boolean previousIsInputLayer = (weightLayerIndex == 0);
            int previousLayerSize = previousIsInputLayer ? inputSize : hiddenLayerSizes[weightLayerIndex - 1] + 1;

            // If the previous layer is a hidden one, allocate room for the bias node that will be added later
            int weightLayerSize = currentLayerSize * previousLayerSize;
            weights[weightLayerIndex] = new double[weightLayerSize];

            // Set each weight's value to be between min and max
            for(int weightIndex = 0; weightIndex < weightLayerSize; weightIndex++) {
                Random random = new Random();
                double randomValue = min + (max - min) * random.nextDouble();
                weights[weightLayerIndex][weightIndex] = randomValue;
            }
        }

        return weights;
    }

    // Sigmoid function
    private double activate(double val) {
        return (1 / (1 + Math.exp(-val)));
    }

    // Derivative of the sigmoid function
    private double activatePrime(double val) {
        return val * (1 - val);
    }

    // Encode classes (i.e., 2 -> [0, 1, 0, .., n])
    private int[] oneHotEncode(int val) {
        int[] encodedVal = new int[numClassifiers];

        for(int i = 0; i < numClassifiers; i++)
            encodedVal[i] = (i == val) ? 1 : 0;

        return encodedVal;
    }

    // Decode class (i.e., [0, 1, 0, .., n] -> 2)
    private int translatePrediction(double[] encodedVal) {
        int predictedClass = 0;

        for(int classifierIndex = 0; classifierIndex < numClassifiers; classifierIndex++)
            if(encodedVal[classifierIndex] > encodedVal[predictedClass])
                predictedClass = classifierIndex;

        return predictedClass;
    }
}