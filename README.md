# Methane Filters Benchmark

Welcome to the **Methane Filters Benchmark** repository! This project provides various filters designed to benchmark and compare their performance, with a focus on optimizing them for edge devices.

## Overview

This repository includes several filters, with an emphasis on those optimized for faster processing times on edge devices. The file `sped_up_filters.py` contains these optimized filters, and you can test their throughput using the script `test_sped_up_filters.py`.

## Cloning the Repository

This repository includes submodules. To clone the repository along with all its submodules, use the following command:

```bash
git clone --recursive <repository_url>
```

## Usage

To run the tests and benchmark the filters, use the following command:

```bash
python test_sped_up_filters.py --random 512 512 50 --precision 64
```

### Arguments:
- `--random` (default: `512 512 50`): If not specified, a random image with the specified shape will be generated. Provide it as Height (H), Width (W), and Channels (C).
- `--precision` (default: `32`): Specify the precision type for floating-point numbers. Options are `16`, `32`, or `64`.

## STARCOP Bands Information

The wavelength information for AVIRIS-NG is sourced from the `AVIRIS_WAVELENGTHS` and `AVIRIS_FWHM` variables, which were extracted from the `ang20191021t171902_rdn_v2x1/ang20191021t171902_rdn_v2x1_img.hdr` file. This data was downloaded in March 2025 from the following URL: [AVIRIS-NG Data](https://popo.jpl.nasa.gov/avng/y19/ang20191021t171902.tar.gz).

These wavelengths are also used in the STARCOP dataset and were processed through `mag1c` to generate files such as `aviris_mag1c_centers.npy` and `aviris_mag1c_spectrum.npy`.

## License

[Include your license details here]