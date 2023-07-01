# Recurrent Neural Network with Reinforcement Learning Agent for Unity

This is a script that implements a Recurrent Neural Network (RNN) with reinforcement learning in C# for computer games developed in Unity. The script allows you to configure inputs and outputs, and it stores the weights in text files.

The RNNRLAgent class provides the core functionality for the RNN-based agent. It includes methods for forward propagation, backward propagation, and saving/loading weights. The class is designed to be integrated into Unity projects for creating intelligent bots or agents that can learn and make decisions based on the game environment.

## API

[API Documentation](./RNNRLAgent_API.md)

## Features

- RNN architecture with configurable input size, hidden size, and output size.
- Forward propagation to compute output based on input vector.
- Backward propagation to update weights based on target output.
- Saving and loading weights from text files.
- Integration with Unity for game development.

## Usage

1. Copy the `RNNRLAgent.cs` script into your Unity project's scripts folder.

2. Create an instance of the `RNNRLAgent` class in your bot or agent script, and configure the input size, hidden size, and output size.

3. Use the `ForwardPropagation` method to compute the output of the agent based on the input vector.

4. Make decisions based on the output vector to control the behavior of your bot or agent.

5. Provide feedback to the agent using a scoring logic, and call the `BackwardPropagation` method to update the weights and reinforce learning.

6. Use the `SaveWeights` and `LoadWeights` methods to store and retrieve weights from text files, if needed.

## Example

```csharp
using UnityEngine;

public class Bot : MonoBehaviour
{
    private RNNRLAgent agent;

    private void Start()
    {
        // Initialize the RNNRL agent with input size, hidden size, and output size
        agent = new RNNRLAgent(inputSize, hiddenSize, outputSize);
    }

    private void Update()
    {
        // Gather inputs and compute output using ForwardPropagation
        float[] inputs = GatherInputs();
        float[] outputs = agent.ForwardPropagation(inputs);

        // Make decisions based on the outputs
        MakeDecisions(outputs);

        // Provide feedback and reinforce learning using the AddScore function
        AddScore();
    }

    private void AddScore()
    {
        // Your scoring logic here

        // Update the agent with the new target and perform backward propagation
        float[] target = ComputeTarget();
        agent.BackwardPropagation(inputs, outputs, target);
    }
}
```

## License

This script is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Authors

- [AGX](https://github.com/Aganow)

## Acknowledgments

This script is inspired by the concepts of Recurrent Neural Networks (RNN) and Reinforcement Learning (RL). Special thanks to the open-source community for their valuable contributions.
