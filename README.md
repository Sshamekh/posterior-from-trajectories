# Trajectory-based Parameter Inference with Normalizing Flows

This repository contains the code accompanying the paper on **trajectory-based parameter estimation in dynamical systems using conditional normalizing flows**. The goal is to learn posterior distributions over model parameters directly from time-ordered trajectories rather than from predefined summary statistics.

## Repository Structure

The main code for reproducing the experiments is organized into two directories:

- **L63** contains the experiments for the **Lorenz–63 system**.
- **L96** contains the experiments for the **Lorenz–96 system**.

Each directory includes the scripts and notebooks needed to generate data, train the inference model, and reproduce the results presented in the paper.

## Workflow

To reproduce the experiments, follow the steps below.

### 1. Generate simulation data

First generate the training and test datasets from the dynamical system.



This step simulates trajectories for different parameter values and stores the dataset used for training and evaluation.

### 2. Train the inference model

Once the dataset has been generated, train the inference model using the provided training notebook:

The notebook trains the encoder and conditional normalizing flow using simulated parameter–trajectory pairs.

### 3. Run inference and reproduce results

After training, the inference scripts can be used to evaluate the trained model and reproduce the results shown in the paper.

## Citation

If you use this code in your work, please cite the associated paper.

