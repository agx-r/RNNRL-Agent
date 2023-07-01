using System;
using System.IO;

public class RNNRLAgent
{
    private int inputSize; // Size of the input vector
    private int hiddenSize; // Size of the hidden layer
    private int outputSize; // Size of the output vector

    private float[,] inputHiddenWeights; // Weights between input and hidden layer
    private float[,] hiddenHiddenWeights; // Weights between hidden layer and itself
    private float[,] hiddenOutputWeights; // Weights between hidden layer and output

    private float[] hiddenLayer; // Hidden layer state

    private float learningRate = 0.1f; // Learning rate for updating weights

    public RNNRLAgent(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        inputHiddenWeights = new float[inputSize, hiddenSize];
        hiddenHiddenWeights = new float[hiddenSize, hiddenSize];
        hiddenOutputWeights = new float[hiddenSize, outputSize];

        hiddenLayer = new float[hiddenSize];

        // Initialize weights with random values
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Initialize input-hidden weights
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                inputHiddenWeights[i, j] = UnityEngine.Random.Range(-1f, 1f);
            }
        }

        // Initialize hidden-hidden weights
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                hiddenHiddenWeights[i, j] = UnityEngine.Random.Range(-1f, 1f);
            }
        }

        // Initialize hidden-output weights
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                hiddenOutputWeights[i, j] = UnityEngine.Random.Range(-1f, 1f);
            }
        }
    }

    public float[] ForwardPropagation(float[] inputs)
    {
        // Update hidden layer state
        for (int j = 0; j < hiddenSize; j++)
        {
            float sum = 0f;

            for (int i = 0; i < inputSize; i++)
            {
                sum += inputs[i] * inputHiddenWeights[i, j];
            }

            for (int k = 0; k < hiddenSize; k++)
            {
                sum += hiddenLayer[k] * hiddenHiddenWeights[k, j];
            }

            hiddenLayer[j] = (float)Math.Tanh(sum);
        }

        // Compute output
        float[] output = new float[outputSize];
        for (int j = 0; j < outputSize; j++)
        {
            float sum = 0f;

            for (int i = 0; i < hiddenSize; i++)
            {
                sum += hiddenLayer[i] * hiddenOutputWeights[i, j];
            }

            output[j] = (float)Math.Tanh(sum);
        }

        return output;
    }

    public void BackwardPropagation(float[] inputs, float[] outputs, float[] target)
    {
        float[] outputErrors = new float[outputSize];

        // Compute output errors
        for (int j = 0; j < outputSize; j++)
        {
            outputErrors[j] = target[j] - outputs[j];
        }

        // Update hidden-output weights
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                hiddenOutputWeights[i, j] += learningRate * outputErrors[j] * hiddenLayer[i];
            }
        }

        float[] hiddenErrors = new float[hiddenSize];

        // Compute hidden errors
        for (int i = 0; i < hiddenSize; i++)
        {
            float sum = 0f;

            for (int j = 0; j < outputSize; j++)
            {
                sum += outputErrors[j] * hiddenOutputWeights[i, j];
            }

            hiddenErrors[i] = sum * (1f - hiddenLayer[i] * hiddenLayer[i]);
        }

        // Update input-hidden weights
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                inputHiddenWeights[i, j] += learningRate * hiddenErrors[j] * inputs[i];
            }
        }

        // Update hidden-hidden weights
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                hiddenHiddenWeights[i, j] += learningRate * hiddenErrors[j] * hiddenLayer[i];
            }
        }
    }

    public void SaveWeights(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            // Write input-hidden weights
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    writer.WriteLine(inputHiddenWeights[i, j].ToString());
                }
            }

            // Write hidden-hidden weights
            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    writer.WriteLine(hiddenHiddenWeights[i, j].ToString());
                }
            }

            // Write hidden-output weights
            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    writer.WriteLine(hiddenOutputWeights[i, j].ToString());
                }
            }
        }
    }

    public void LoadWeights(string filePath)
    {
        using (StreamReader reader = new StreamReader(filePath))
        {
            // Read input-hidden weights
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    inputHiddenWeights[i, j] = float.Parse(reader.ReadLine());
                }
            }

            // Read hidden-hidden weights
            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    hiddenHiddenWeights[i, j] = float.Parse(reader.ReadLine());
                }
            }

            // Read hidden-output weights
            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    hiddenOutputWeights[i, j] = float.Parse(reader.ReadLine());
                }
            }
        }
    }
}
