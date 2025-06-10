import onnx
import onnxruntime as ort
import numpy as np
import time

# Function to load an ONNX model and run inference
def measure_inference_time(onnx_model_path, input_data):
    # Load the ONNX model using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_model_path)

    # Get the input name of the model (usually the first input in the model)
    input_name = ort_session.get_inputs()[0].name

    # Prepare the input (assuming input_data is a numpy array)
    # If input_data is not in the form of a numpy array, convert it
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

    # Start measuring inference time
    start_time = time.time()

    # Run inference on the model
    ort_outputs = ort_session.run(None, {input_name: input_data})

    # Measure the time it took
    end_time = time.time()
    inference_time = end_time - start_time

    # Return inference results and time
    return ort_outputs, inference_time

# Example usage
if __name__ == "__main__":
    # Path to the ONNX model (update this with the path to your ONNX model)
    onnx_model_path = 'models/unet_trained.onnx'
    
    # Input data generation
    input_data = np.random.random((1, 4, 512, 512)).astype(np.float32)

    # Run inference and measure time
    outputs, inference_time = measure_inference_time(onnx_model_path, input_data)

    # Print the results
    print(f"Inference Time: {inference_time:.6f} seconds")
    print(f"Model Output: {outputs[0]}")  # Assuming the model returns one output
