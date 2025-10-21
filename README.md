# vit-fi

ViT Dependability Analysis and Soft Error Hardening

This repository contains the complete experimental framework and scripts used for the systematic dependability analysis of the Vision Transformer (ViT-Base) architecture against transient soft errors in GPU environments. We employ a custom, high-fidelity fault injection methodology that simulates bit-flips directly in the IEEE 754 binary representation of intermediate tensors.

⚙️ Key Research Findings

The code reproduces the core results detailed in the accompanying publication, establishing architectural vulnerability and mitigation guidelines for ViT deployment:

Critical Vulnerability: The LayerNorm and Linear Layers outputs were the primary architectural focus for error amplification and failure under soft error conditions.   

FAT Limitation: Fault-Aware Training (FAT) with random bit-flips im the layer output, when applied globally or selectively, did not provide any measurable improvement in robustness against the highly random, transient nature of the injected bit-flips.  

Practical Mitigation: Runtime numerical filter (specifically, adaptive zero-replacement for NaN, Inf, and extreme outliers), which significantly stabilized inference performance by halting the propagation of numerical artifacts.  

🚀 Getting Started: Reproducing the Experiments

The entire experimental environment is packaged within a Docker image based on NVIDIA CUDA, ensuring a consistent setup across different GPU hardware.

Prerequisites

  - NVIDIA GPU with compatible drivers.

  - Docker installed.

  - NVIDIA Container Toolkit (formerly nvidia-docker) installed and configured to allow access to the GPU from within the container.

Installation and Setup (Using Docker)

**_1.Clone the repository_**

    git clone https://github.com/lesterfd95/vit-fi.git

The environment is built upon the official NVIDIA CUDA development image (nvidia/cuda:12.9.1-devel-ubuntu22.04) and installs project-specific Python dependencies from requirements.txt.

**_2.Build the Docker Image_** 

Use the provided Dockerfile and tag the resulting image as vit_fi:1.0, as required by the execution script.

    docker build -t vit_fi:1.0.

**_3.Launch the Container_**  

Execute the provided launch script (run_container.sh) to start the container with optimized resources.

    bash run_container.sh

The script executes: docker run --rm --runtime=nvidia --gpus all --shm-size=32gb -it --mount type=bind,src="$(pwd)"/scripts,target=/app vit_fi:1.0 bash

Resource Allocation: The script allocates all available GPUs (--gpus all) and sets the shared memory size to 32GB (--shm-size=32gb)

Volume Mount: A bind mount is established (src="$(pwd)"/scripts,target=/app) to map the local scripts directory into the container's working directory (/app). This allows for direct execution and modification of the research scripts from the host machine.

**_4.Running Core Experiments_**

The core functionality is accessed via wrapper scripts that utilize the main Python file (tt_inj.py) and pass configuration arguments:

Training and Testing (train_test.sh):

  Purpose: Launches a full training run and subsequent testing, potentially including Fault-Aware Training (FAT) if configured.
  
  Usage: Requires a log name and the model name to be created.
  
  Command Example (Inside Container):
  
    ./train_test.sh my_new_log_name ViT_FAT_Model


Testing Pre-Trained Model (test.sh):

  Purpose: Executes a set of tests (e.g., vulnerability analysis or mitigation evaluation) on an already trained model or checkpoint located in the checkpoints folder.
  
  Usage: Requires a log name for the test run and the name of the existing model/checkpoint.
  
  Command Example (Inside Container):
  
    ./test.sh baseline_ber_sweep ViT_Baseline_Model

Customization: Users can create their own custom launching scripts by examining train_test.sh and test.sh, which pass specific arguments to the principal Python execution file (tt_inj.py).
