# Methane Filters Benchmark

<img src="resources/filters_visualization.png" alt="Filters Visualization" width="800"/>

Welcome to the **Methane Filters Benchmark** repository! This project provides a suite of methane filters designed to benchmark and compare their performance, with a focus on optimizing them for low-power edge devices. In addition to traditional filters we also explore use of machine learning models for filter output refinement.

## Citation [![ArXiv:2507.01472](https://img.shields.io/badge/arXiv-2507.01472-blue)](https://doi.org/10.48550/arXiv.2507.01472)
If you find our research useful, please cite our article:
```bibtex
@misc{herec2025optimizingmethanedetectionboard,
      title={Optimizing Methane Detection On Board Satellites: Speed, Accuracy, and Low-Power Solutions for Resource-Constrained Hardware}, 
      author={Jonáš Herec and Vít Růžička and Rado Pitoňák},
      year={2025},
      eprint={2507.01472},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.01472}, 
}
```

## Notebook Demos

You can try out our demos directly in Google Colab:

- <a href="https://colab.research.google.com/github/zaitra/methane-filters-benchmark/blob/main/ntbs/Models_demo.ipynb"> Models Demo <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a>  
  Demonstrates model inference.

- <a href="https://colab.research.google.com/github/zaitra/methane-filters-benchmark/blob/main/ntbs/Products_demo.ipynb"> Products Creation and Benchmarking Demo <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a>  
  Demonstrates generating products and measuring their runtime.

## Resources

- [🤗 STARCOP – All Bands Version](https://huggingface.co/collections/previtus/starcop-67f13cf30def71591f281a41)  
  Raw hyperspectral data, from which the filter products were computed.

- [🤗 Precomputed Filters](https://huggingface.co/datasets/onboard-coop/STARCOP-fast-products)  
  A selection of precomputed spectral filters and data products derived from the STARCOP dataset.

- [🤗 Trained Models](https://huggingface.co/onboard-coop/fast-methane-filters-models)  
  Methane detection models trained using the precomputed products above.

## Cloning the Repository

This repository includes submodules. To clone the repository along with all its submodules, use the following command:

```bash
git clone --recursive https://github.com/zaitra/methane-filters-benchmark.git
```

## Overview

The accelerated traditional filters are to be found in file `sped_up_filters.py`, whereas the **Mag1c-SAS**, a modified version of **Mag1c**, is included as a [submodule](https://github.com/zaitra/mag1c) in `benchmark/mag1c_fork`. You can test their runtime using the `test_sped_up_filters.py` script.

**Note**: Make sure to run the commands from the root directory of the methane-filters-benchmark repository. The paths are set to absolute, so you have to include the `benchmark/` prefix when running the `.py` commands.

## Runtime Measurement

To run the tests and benchmark the filters using the provided test tile, use the following command:

```bash
python benchmark/test_sped_up_filters.py --channels [N_of_channels]
```

You can also add the following options:
- `--compute-original-mag1c` to compute the original column-based Mag1c.
- `--compute-original-filters` to also run the original versions of the filters.

Be sure to run this on the target edge device, as the runtime on your host computer is not representative of edge device performance.

**Note**: The ML models runtime was measured by `benchmark/onnx_inference_time.py` script.

## Filter Generation

To generate the filters from the STARCOP data, use the `benchmark/create_filters_for_starcop.py` script.  
The script supports various tweaks and filter variants, so make sure to set the necessary parameters in the config file located at `benchmark/cfg/classic.yaml`, or create your own custom config.
```bash
python benchmark/create_filters_for_starcop.py --config <path/to/config.yaml>
```

## Assess the Metrics

After creating the products, you can assess the filter metrics using the `benchmark/metrics_runner/metrics_runner.py` script.  
Paths and threshold values are defined within the script, so ensure they are correctly set for your use case, or use the predefined values we found to perform best.

Our results are stored as CSV files inside the `csvs/` directory.

## Band Selection Strategies

The band selection logic is implemented in the `benchmark/utils.py` file.  
Based on our research, the `"highest-variance"` strategy generally yields the best performance.

To use it, provide a list of wavelengths, a list of CH₄ transmittance values per channel, and the number of channels `N` you want to select.  
You can call the function as follows:

```python
select_the_bands_by_transmittance(wavelengths, ch4_transmittance, N, strategy="highest-variance")
```

The function returns a tuple containing the selected N wavelengths and their corresponding transmittance values, ordered according to the chosen strategy.