import torch
import onnxruntime as ort
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        
        # Encoding layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Decoding layers
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)  # Output 1-channel mask

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.upconv2(x)
        x = F.relu(self.conv4(x))
        x = self.upconv1(x)
        x = torch.sigmoid(x)  # Ensure output is in [0, 1] range

        return x

def export_and_compare(model, dummy_input, onnx_path):
    model.eval()

    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes=None,
        opset_version=13
    )
    print(f"Exported model to {onnx_path}")

    # PyTorch Inference
    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    # ONNX Inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {"input": dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compute Difference
    diff = np.abs(torch_output - ort_output)
    avg_diff = np.mean(diff)
    max_diff = np.max(diff)

    print(f"Average Difference: {avg_diff:.32f}")
    print(f"Maximum Difference: {max_diff:.32f}")

    # Check if outputs are numerically close
    assert np.allclose(torch_output, ort_output, atol=1e-5), "ONNX and PyTorch outputs differ significantly!"
    print("✅ ONNX and PyTorch outputs match!")

# Example usage:
model = SimpleSegmentationModel()  # Replace with your PyTorch or Lightning model
dummy_input = torch.randn(1, 4, 512, 512)  # filter + rgb, but adjust if needed
output_path = "./onnx_model.onnx"
export_and_compare(model, dummy_input, output_path)