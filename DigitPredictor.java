// File:  DigitPredictor.java
// Name:  Mo K. Eid (mohamedkeid@gmail.com)
// Desc:  Creates and trains a neural network to predict images (28x28 pixels) of hand-written digits
//        The accuracy based on its own training examples is then computed and printed to the log
//        Training example records in the mnist set consist of 29 elements separated by ","
//        Element 0 is the labeled class and the remaining 28 elements are pixel intensities for each image pixel
//        The default structure of the neural network consists of:
//              28 input nodes for each image pixel  (calculated)
//              1 hidden layer comprised of 30 nodes (hard-coded)
//              10 classes for hand-written digits   (calculated)

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

public class DigitPredictor {
    private static int[][] exampleSet;
    private static int[] labelSet;

    public static void main(String[] args) throws IOException {
        // Get example data from .csv file
        initTrainingExamples("mnist_train.csv");

        // Split the example and label set into training sets and testing sets
        double percentTrain = 0.66;
        int trainSetSize = (int) Math.round(exampleSet.length * percentTrain);
        int testSetSize = exampleSet.length - trainSetSize;
        int[][] trainingExampleSet = new int[trainSetSize][];
        int[] trainingLabelSet = new int[trainSetSize];
        int[][] testingExampleSet = new int[testSetSize][];
        int[] testingLabelSet = new int[testSetSize];
        int inputSize = exampleSet[0].length;

        // Initialize the training and testing sets
        for(int exampleIndex = 0; exampleIndex < (trainSetSize + testSetSize); exampleIndex++) {
            if(exampleIndex < trainSetSize) {
                trainingExampleSet[exampleIndex] = exampleSet[exampleIndex];
                trainingLabelSet[exampleIndex] = labelSet[exampleIndex];
            } else {
                int index = exampleIndex - trainSetSize;
                testingExampleSet[index] = exampleSet[index];
                testingLabelSet[index] = labelSet[index];
            }
        }

        // Initialize structure of neural network
        int[] hiddenLayerSizes = {30};
        int numClasses = getNumClasses();

        // Create instance of a neural network
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenLayerSizes, numClasses);

        // Train it and get the accuracy of the algorithm
        int iterationsOfTraining = 100;
        neuralNetwork.train(trainingExampleSet, trainingLabelSet, iterationsOfTraining);
        double accuracy = neuralNetwork.checkAccuracy(testingExampleSet, testingLabelSet);
        System.out.println("Accuracy: " + accuracy);
    }

    // Return the number of different classes from the training labels
    private static int getNumClasses() {
        int numClasses = 0;

        for(int label: labelSet)
            if(label > numClasses)
                numClasses = label;

        return numClasses + 1;
    }

    // Read the training examples from the .csv file and initialize them
    private static void initTrainingExamples(String filePath) throws IOException {
        // Read .csv file containing training examples
        String basePath = new File("").getAbsolutePath();
        File file = new File(basePath.concat("/" + filePath));
        List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
        lines.remove(0);

        // Randomize data
        long seed = System.nanoTime();
        Collections.shuffle(lines, new Random(seed));
        int lineCount = lines.size();

        // Initialize variables for training records
        exampleSet = new int[lineCount][];
        labelSet = new int[exampleSet.length];

        // Read through every line of the file
        int lineIndex = 0;
        for(String line : lines) {
            if(lineIndex != 0 &&lineIndex < lineCount) {
                // Parse training example and label into an array of characters
                String[] array = line.split(",");

                // First integer is the class (the written integer)
                labelSet[lineIndex - 1] = Integer.parseInt(array[0]);

                // The remaining integers are features (pixel brightness)
                exampleSet[lineIndex - 1] = new int[array.length - 1];
                for (int i = 1; i < array.length; i++)
                    exampleSet[lineIndex - 1][i - 1] = Integer.parseInt(array[i]);
            }

            lineIndex++;
        }
    }
}