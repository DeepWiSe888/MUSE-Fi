# MUSE-Fi: Contactless MUti-person SEnsing Exploiting Near-field Wi-Fi Channel Variation

Welcome to MUSE-Fi! This repository contains the code and resources to realize MUSE-Fi, the first Wi-Fi multi-person sensing system with physical separability.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [BFI and CSI Parsing and Comparison](#bfi-and-csi-parsing-and-comparison)
- [Sparse Recovery Algorithm (SRA)](#sparse-recovery-algorithm-sra)
- [Gesture Detection](#gesture-recognition)
- [Activity Recognition](#activity-recognition)
- [CSI Extraction on Smartphones](#csi-extraction-on-smartphones)
- [License](#license)

## Introduction

MUSE-Fi is the first Wi-Fi multi-person sensing system with physical separability. This repository contains the code and resources for the following five main parts of MUSE-Fi:

1. **BFI and CSI Parsing and Comparison**
2. **Sparse Recovery Algorithm (SRA)**
3. **Gesture Detection**
4. **Activity Recognition** 
5. **CSI Extraction on Smartphones**

Whether you're a researcher, developer, or enthusiast, MUSE-Fi opens up possibilities for advanced multi-person sensing applications. This repository provides you with the necessary resources to implement MUSE-Fi effectively.

## Getting Started

To get started with MUSE-Fi and use the code to realize the system, follow these steps:

1. **Clone** this repository to your local machine using `git clone https://github.com/uio789bnm/MUSE-Fi/.git`.
2. **Install Python 3.8** on your computer. You can download the latest version of Python from the official website: [python.org](https://www.python.org/).
3. **Set up a virtual environment** (recommended) using tools like `virtualenv` or `conda`. This ensures a clean and isolated environment for your MUSE-Fi implementation. Activate the virtual environment.
4. **Download the required datasets**. The specific datasets used with MUSE-Fi can be obtained from [Dataset Download](https://1drv.ms/u/s!AoeuYyk6v3AHe_ye4dP3ZitmG7o?e=fpPOom). Extract the contents of the `Data.zip` file and  make sure the extracted `Data` folder and the five code folders (gesture, activity, bfi-csi-comparison, SRA, and nexmon-csi) are placed in the same directory.
5. **Install the necessary packages** such as pytorch 2.0.0, tqdm 4.65.0, numpy 1.23.5, pandas 2.0.3

Now you're ready to implement and use MUSE-Fi!


## BFI and CSI Parsing and Comparison

The `bfi-csi-comparison` folder contains the code for parsing BFI and CSI data in MUSE-Fi. This module also provides functionality for comparing the BFI and CSI data. To utilize this module, follow these steps: 

1. Navigate to the `bfi-csi-comparison` folder: `cd bfi-csi-comparison`. 
2. Ensure your virtual environment is activated. 
3. Run the `run_Gen_BFICSI_CompareData.m` script using appropriate software (e.g., MATLAB).

## Sparse Recovery Algorithm (SRA)

The `SRA` folder contains the code and resources for Sparse Recovery Algorithm of MUSE-Fi. To run the code, follow these steps:

1. Navigate to the `SRA` folder: `cd SRA`.
2. Ensure your virtual environment is activated.
3. Run the `run_sra_train.sh` script: `bash run_sra_train.sh`.

## Gesture Detection

The `gesture` folder contains the code and resources for gesture recognition using MUSE-Fi. To run the gesture detection classifier, follow these steps:

1. Navigate to the `gesture` folder: `cd gesture`.
2. Ensure your virtual environment is activated.
3. Run the `main.py` script using Python 3.8: `python main.py`.

## Activity Recognition

The `activity` folder contains the code and resources for activity recognition using MUSE-Fi. To run the activity recognition classifier, follow these steps:

1. Navigate to the `activity` folder: `cd activity`.
2. Ensure your virtual environment is activated.
3. Run the `main.py` script using Python 3.8: `python main.py`.

## CSI Extraction on Smartphones

To perform CSI extraction on smartphones and extend MUSE-Fi's multi-person sensing capabilities to mobile devices, please follow the guidance provided in the external repository [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi). The Nexmon CSI repository offers valuable instructions and tools for extracting CSI data on smartphones. As part of the implementation, you may refer to the txt files provided in the `nexmon-csi` repository as an example.

## License

This repository is licensed under the MIT License. For more details, please refer to the `LICENSE` file.
