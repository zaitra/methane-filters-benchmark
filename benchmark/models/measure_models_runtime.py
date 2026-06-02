import time
import numpy as np
import onnxruntime as ort
import csv

# Paths to your ONNX models
model_paths = ["linknet.onnx", "unet.onnx"]

# Number of iterations
num_runs = 100

# Input shape
input_shape = (1, 4, 512, 512)

# Initialize ONNX sessions
sessions = [ort.InferenceSession(path) for path in model_paths]

# Store runtimes for each model
runtimes = {f"model{i+1}": [] for i in range(len(model_paths))}

# Benchmark
for run in range(num_runs):
    x = np.random.rand(*input_shape).astype(np.float32)
    for i, session in enumerate(sessions):
        input_name = session.get_inputs()[0].name
        start = time.time()
        session.run(None, {input_name: x})
        end = time.time()
        runtimes[f"model{i+1}"].append(end - start)

# Compute statistics
stats = {}
for name, times in runtimes.items():
    times = np.array(times)
    stats[name] = {
        "min": times.min(),
        "max": times.max(),
        "mean": times.mean(),
        "median": np.median(times)
    }

# Save stats to CSV
csv_file = "onnx_benchmark.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Min(s)", "Max(s)", "Mean(s)", "Median(s)"])
    for model_name, s in stats.items():
        writer.writerow([model_name, s["min"], s["max"], s["mean"], s["median"]])

print(f"Benchmark saved to {csv_file}")