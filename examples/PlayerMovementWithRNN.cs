using UnityEngine;

public class PlayerMovementWithRNN : MonoBehaviour
{
    public RNNRLAgent agent;
    private Rigidbody rb;
    private Vector3 startPosition;

    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        startPosition = transform.position;
    }

    private void FixedUpdate()
    {
        // Get player input
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");

        // Create input vector
        float[] inputs = { horizontalInput, verticalInput };

        // Forward propagation
        float[] outputs = agent.ForwardPropagation(inputs);

        // Apply movement based on outputs
        float moveHorizontal = outputs[0];
        float moveVertical = outputs[1];
        Vector3 movement = new Vector3(moveHorizontal, 0f, moveVertical);
        rb.AddForce(movement * 10f);

        // Reset player position if it falls off the platform
        if (transform.position.y < -2f)
        {
            transform.position = startPosition;
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }
}
