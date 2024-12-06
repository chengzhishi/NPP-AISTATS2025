# Neural Point Processes (NPP)

This repository contains the implementation of Neural Point Processes (NPP) for various experiments.

## Getting Started

Run the script using the following command:

```
python npp --<arguments>
```
markdown
Copy code

## Command-Line Arguments

Below is a detailed list of the arguments you can pass to the script:

### Dataset and Hyperparameters
- `--dataset`: Dataset name (default: `"PinMNIST"`)
- `--modality`: Building dataset modality (default: `"PS-RGBNIR"`)
- `--feature`: Feature type, either `"AE"` or `"DDPM"` (default: `"AE"`)
- `--mode`: Mode, either `"mesh"` or `"random"` (default: `"mesh"`)
- `--d`: Value for distance of mesh grids `d` (default: `10`)
- `--n_pins`: Number of random points (default: `500`)
- `--r`: Value for radius of counting objects `r` (default: `3`)
- `--seed`: Seed for PyTorch and NumPy (default: `1`)

### Hyperparameters
- `--lr`: Initial learning rate (default: `0.001`)
- `--epochs`: Number of epochs (default: `1000`)
- `--batch_size`: Batch size (default: `64`)
- `--val_every_epoch`: Number of epochs between validations (default: `5`)
- `--num_runs`: Number of trainings per model and sigma (default: `3`)
- `--manual_lr`: Disable custom LR finder (default: `False`)

### Kernel Parameters
- `--kernel_param`: Kernel parameter, e.g., Q=1 (SMK) or sigma=1 (RBF) (default: `[1]`)

### Model Configuration
- `--num_encoder`: List of encoder kernel sizes (default: `[64, 32]`)
- `--num_decoder`: List of decoder kernel sizes (default: `[64]`)
- `--deeper`: Add an extra convolutional layer (default: `False`)
- `--kernel_mode`: Kernel mode, `"fixed"`, `"learned"`, or `"predicted"` (default: `"fixed"`)
- `--kernel`: Kernel type, `"RBF"` or `"SM"` (default: `"RBF"`)

### Experiment Settings
- `--experiment_name`: Save experiments in a specific folder (default: `None`)

## Example Usage
```
python npp --dataset PinMNIST --feature AE --mode mesh --d 10 --lr 0.001 --epochs 1000
```
