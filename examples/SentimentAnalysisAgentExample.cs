using System;

public class SentimentAnalysisAgentExample
{
    static void Main()
    {
        // Define the parameters for the RNN agent
        int inputSize = 100; // Size of the input vector
        int hiddenSize = 64; // Size of the hidden layer
        int outputSize = 2;  // Size of the output vector (positive or negative sentiment)

        // Create the RNN agent
        RNNRLAgent agent = new RNNRLAgent(inputSize, hiddenSize, outputSize);

        // Train the agent
        TrainAgent(agent);

        // Evaluate the agent
        EvaluateAgent(agent);
    }

    static void TrainAgent(RNNRLAgent agent)
    {
        // Training data
        float[][] inputs = new float[][]
        {
            new float[] { /* input vector for review 1 */ },
            new float[] { /* input vector for review 2 */ },
            // ...
        };

        float[][] targets = new float[][]
        {
            new float[] { /* target vector for review 1 (e.g., [1, 0] for positive sentiment) */ },
            new float[] { /* target vector for review 2 (e.g., [0, 1] for negative sentiment) */ },
            // ...
        };

        // Train the agent with each training example
        for (int i = 0; i < inputs.Length; i++)
        {
            float[] input = inputs[i];
            float[] target = targets[i];

            // Perform forward propagation
            float[] output = agent.ForwardPropagation(input);

            // Perform backward propagation to update weights
            agent.BackwardPropagation(input, output, target);
        }

        // Save the trained weights to a file
        agent.SaveWeights("sentiment_analysis_weights.txt");
        Console.WriteLine("Agent trained and weights saved.");
    }

    static void EvaluateAgent(RNNRLAgent agent)
    {
        // Load the trained weights from a file
        agent.LoadWeights("sentiment_analysis_weights.txt");

        // Test data
        float[][] testInputs = new float[][]
        {
            new float[] { /* input vector for test review 1 */ },
            new float[] { /* input vector for test review 2 */ },
            // ...
        };

        // Evaluate the agent on each test example
        for (int i = 0; i < testInputs.Length; i++)
        {
            float[] testInput = testInputs[i];

            // Perform forward propagation
            float[] output = agent.ForwardPropagation(testInput);

            // Interpret the output
            string sentiment = GetSentiment(output);

            Console.WriteLine("Test Review {0}: Sentiment - {1}", i + 1, sentiment);
        }
    }

    static string GetSentiment(float[] output)
    {
        // Assuming output[0] corresponds to positive sentiment and output[1] corresponds to negative sentiment
        if (output[0] > output[1])
        {
            return "Positive";
        }
        else
        {
            return "Negative";
        }
    }
}
