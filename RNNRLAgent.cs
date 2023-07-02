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
        InitializeWeightMatrix(inputHiddenWeights, inputSize, hiddenSize);
        InitializeWeightMatrix(hiddenHiddenWeights, hiddenSize, hiddenSize);
        InitializeWeightMatrix(hiddenOutputWeights, hiddenSize, outputSize);
    }

    private void InitializeWeightMatrix(float[,] weights, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                weights[i, j] = UnityEngine.Random.Range(-1f, 1f);
            }
        }
    }

    public float[] ForwardPropagation(float[] inputs)
    {
        UpdateHiddenLayerState(inputs);

        float[] output = ComputeOutput();

        return output;
    }

    private void UpdateHiddenLayerState(float[] inputs)
    {
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
    }

    private float[] ComputeOutput()
    {
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
        float[] outputErrors = ComputeOutputErrors(outputs, target);

        UpdateHiddenOutputWeights(outputErrors);

        float[] hiddenErrors = ComputeHiddenErrors(outputErrors);

        UpdateInputHiddenWeights(inputs, hiddenErrors);

        UpdateHiddenHiddenWeights(hiddenErrors);
    }

    private float[] ComputeOutputErrors(float[] outputs, float[] target)
    {
        float[] outputErrors = new float[outputSize];

        for (int j = 0; j < outputSize; j++)
        {
            outputErrors[j] = target[j] - outputs[j];
        }

        return outputErrors;
    }

    private void UpdateHiddenOutputWeights(float[] outputErrors)
    {
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                hiddenOutputWeights[i, j] += learningRate * outputErrors[j] * hiddenLayer[i];
            }
        }
    }

    private float[] ComputeHiddenErrors(float[] outputErrors)
    {
        float[] hiddenErrors = new float[hiddenSize];

        for (int i = 0; i < hiddenSize; i++)
        {
            float sum = 0f;

            for (int j = 0; j < outputSize; j++)
            {
                sum += outputErrors[j] * hiddenOutputWeights[i, j];
            }

            hiddenErrors[i] = sum * (1f - hiddenLayer[i] * hiddenLayer[i]);
        }

        return hiddenErrors;
    }

    private void UpdateInputHiddenWeights(float[] inputs, float[] hiddenErrors)
    {
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                inputHiddenWeights[i, j] += learningRate * hiddenErrors[j] * inputs[i];
            }
        }
    }

    private void UpdateHiddenHiddenWeights(float[] hiddenErrors)
    {
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
            WriteWeightMatrix(inputHiddenWeights, writer);
            WriteWeightMatrix(hiddenHiddenWeights, writer);
            WriteWeightMatrix(hiddenOutputWeights, writer);
        }
    }

    private void WriteWeightMatrix(float[,] weights, StreamWriter writer)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                writer.WriteLine(weights[i, j].ToString());
            }
        }
    }

    public void LoadWeights(string filePath)
    {
        using (StreamReader reader = new StreamReader(filePath))
        {
            ReadWeightMatrix(inputHiddenWeights, reader);
            ReadWeightMatrix(hiddenHiddenWeights, reader);
            ReadWeightMatrix(hiddenOutputWeights, reader);
        }
    }

    private void ReadWeightMatrix(float[,] weights, StreamReader reader)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                weights[i, j] = float.Parse(reader.ReadLine());
            }
        }
    }
}
