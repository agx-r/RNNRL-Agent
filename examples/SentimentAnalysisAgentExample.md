# Sentiment Analysis Agent Example

This script demonstrates a simple Sentiment Analysis Agent implemented in C# using a Recurrent Neural Network (RNN). The agent is designed to perform sentiment analysis on text reviews, classifying them as either positive or negative.

## How It Works

The script consists of the following components:

1. **Main Method**: The `Main` method serves as the entry point of the program. It defines the parameters for the RNN agent, creates the agent, trains it, and evaluates its performance.

2. **Training the Agent**: The `TrainAgent` method is responsible for training the RNN agent. It defines training data, which includes input vectors (representing reviews) and target vectors (indicating sentiment labels). It iterates through the training data, performs forward and backward propagation to update the agent's weights, and saves the trained weights to a file ("sentiment_analysis_weights.txt").

3. **Evaluating the Agent**: The `EvaluateAgent` method loads the previously trained weights, defines test data (input vectors for test reviews), performs forward propagation, and interprets the output to determine sentiment. It then prints the sentiment of each test review to the console.

4. **GetSentiment Method**: This method compares the output vector of the RNN to determine sentiment. If the probability of positive sentiment (output[0]) is higher than the probability of negative sentiment (output[1]), it categorizes the sentiment as "Positive"; otherwise, it categorizes it as "Negative."

## Usage

To use this script for your own sentiment analysis tasks:

1. Implement the `RNNRLAgent` class with methods for forward and backward propagation, as well as weight saving and loading.

2. Provide actual input vectors for the reviews in the training and test data.

3. Adapt the code to your specific dataset and requirements, as needed.

## Notes

- This script provides a basic structure for sentiment analysis with a simple RNN, and it's essential to customize and expand it according to your project's needs.
- The sentiment classification is binary, i.e., it classifies reviews as either positive or negative. You can extend it for more complex sentiment analysis tasks.
- Make sure to replace the comment placeholders (e.g., `/* input vector for review 1 */`) with actual input data.

Feel free to modify and build upon this script to create your own sentiment analysis agent for text data.
