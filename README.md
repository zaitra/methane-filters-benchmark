# Methane Filters Benchmark

Welcome to the **Methane Filters Benchmark** repository! This project provides a suite of filters designed to benchmark and compare their performance, with a focus on optimizing them for edge devices.

## Overview

This repository includes several filters, with an emphasis on those optimized for faster processing times on edge devices. The file `sped_up_filters.py` contains these optimized filters. You can test their runtime using the `test_sped_up_filters.py` script. Additionally, we incorporate **Mag1c-SAS**, a modified version of **Mag1c**, as a submodule.

## Cloning the Repository

This repository includes submodules. To clone the repository along with all its submodules, use the following command:

```bash
git clone --recursive <repository_url>
```

## Runtime Measurement

To run the tests and benchmark the filters using the provided test tile, use the following command:

```bash
python test_sped_up_filters.py --channels [N_of_channels]
```

You can also add the following options:
- `--compute-original-mag1c` to compute the original column-based Mag1c.
- `--compute-original-filters` to also run the original versions of the filters.

**Note**: Be sure to run this on the target edge device, as the runtime on your host computer is not representative of edge device performance.

## Filter Generation

To generate the filters from the STARCOP data, use the `create_filters_for_starcop.py` script. Make sure to set the necessary variables within the script before running it.
