# Player Movement with RNN

This example demonstrates how to use the `RNNRLAgent` class to control player movement in a Unity game. The script allows the player to move using a Recurrent Neural Network (RNN) trained by the `RNNRLAgent`.

## How to Use

1. Attach the `PlayerMovementWithRNN` script to the player object in your Unity scene.
2. Create a `RNNRLAgent` instance and assign it to the `agent` variable in the `PlayerMovementWithRNN` script using the Unity inspector.
3. Make sure the player object has a `Rigidbody` component attached to enable physics-based movement.

## Usage Details

- The `Start()` method initializes the necessary components and stores the initial position of the player.
- In the `FixedUpdate()` method, the player's input (horizontal and vertical axes) is obtained.
- The inputs are passed to the `ForwardPropagation()` method of the `RNNRLAgent` instance to get the corresponding outputs.
- The outputs control the player's movement by adding a force to the `Rigidbody` component.
- If the player falls off the platform (y position less than -2), it is reset to the initial position.

## Prerequisites

- Unity 3D
- `RNNRLAgent` class (implemented separately)
