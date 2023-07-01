# RNNRLAgent API Documentation

The `RNNRLAgent` class implements a recurrent neural network with reinforcement learning for computer games.

## Constructor

### `RNNRLAgent(int inputSize, int hiddenSize, int outputSize)`

Creates a new instance of the `RNNRLAgent` class.

- `inputSize`: Size of the input vector.
- `hiddenSize`: Size of the hidden layer.
- `outputSize`: Size of the output vector.

## Methods

### `float[] ForwardPropagation(float[] inputs)`

Performs forward propagation to compute the output based on the given input vector.

- `inputs`: Input vector for the agent.

Returns the computed output vector.

### `void BackwardPropagation(float[] inputs, float[] outputs, float[] target)`

Performs backward propagation to update the weights of the agent based on the given inputs, outputs, and target values.

- `inputs`: Input vector for the agent.
- `outputs`: Computed output vector.
- `target`: Target output vector.

### `void SaveWeights(string filePath)`

Saves the current weights of the agent to a text file.

- `filePath`: Path of the file to save the weights.

### `void LoadWeights(string filePath)`

Loads the weights of the agent from a text file.

- `filePath`: Path of the file to load the weights.

## Example Usage

```csharp
// Create a new instance of the RNNRLAgent class
RNNRLAgent agent = new RNNRLAgent(5, 10, 2);

// Perform forward propagation
float[] inputs = { 0.2f, 0.8f, 0.5f, 0.3f, 0.9f };
float[] outputs = agent.ForwardPropagation(inputs);

// Perform backward propagation with target values
float[] target = { 0.6f, 0.4f };
agent.BackwardPropagation(inputs, outputs, target);

// Save the weights to a file
agent.SaveWeights("weights.txt");

// Load the weights from a file
agent.LoadWeights("weights.txt");
```

## Notes

- This implementation is a simplified version and may require further modifications and optimizations based on specific requirements.
- It is recommended to customize the scoring logic in the `AddScore` function for reinforcement learning based on your game's requirements.
