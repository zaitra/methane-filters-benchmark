# Methane Filters Benchmark

Welcome to the **Methane Filters Benchmark** repository! This project provides various filters designed to benchmark and compare their performance, with a focus on optimizing them for edge devices.

## Overview

This repository includes several filters, with an emphasis on those optimized for faster processing times on edge devices. The file `sped_up_filters.py` contains these optimized filters, and you can test their throughput using the script `test_sped_up_filters.py`. Besides these we also utilize Mag1c-SAS, our adjustment of Mag1c, as a submodule.

## Cloning the Repository

This repository includes submodules. To clone the repository along with all its submodules, use the following command:

```bash
git clone --recursive <repository_url>
```

## Runtime measurement
To run the tests and benchmark the filters using the provided test tile, use the following command:

```bash
python test_sped_up_filters.py --hdr-path test_tile_512_512_125.hdr --channels [N_of_channels]
```
you can add --compute-original-mag1c to compute the original column based mag1c.
Remeber to run this on the target device, as the runtime on your host computer will is not relevant.

## Filter generation
To generate the filters from the STARCOP data, use `create_filters_for_starcop.py` and set all needed variables inside the script.