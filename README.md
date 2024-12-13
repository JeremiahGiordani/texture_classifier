# Concrete Texture Window Classification

This repository provides a PyTorch-based pipeline for training an image classifier on 3D printed concrete texture windows. The classifier is trained on labeled texture windows and can optionally use a pretrained model and apply data augmentations including a patch-and-shuffle approach.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/JeremiahGiordani/texture_classifier.git
    cd texture_classifier
    ```
   
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   
3. Ensure you have your dataset organized under `data/texture_windows` and `data/texture_windows-labels.csv`.

## Running the Training Script

The main training script is `aug_dataset_train.py`. It supports several command-line flags:

- `--pretrained`: If set, uses a pretrained ResNet-18 model as the backbone. If not set, uses `SimpleCNN`.
- `--epochs`: Number of epochs to train (default: `10`).
- `--test-split`: Fraction of data to hold out as test set (default: `0.2`).
- `--weight-decay`: L2 regularization factor (default: `1e-4`).
- `--patch`: If set, applies the patch-and-shuffle transform to the images before training.
- `--learning-rate`: The learning rate for the optimizer (default: `0.001`).
- `--log-file`: Optional file name to save the training logs. If not set, logs will be printed to stdout.
- `--focal-loss`: If set, uses focal loss instead of standard cross-entropy loss.

### Example Usage

```bash
python aug_dataset_train.py --pretrained --patch --epochs 20 --weight-decay 0.0005 --test-split 0.2 --learning-rate 0.0005
```

This command:

- Uses the pretrained model
- Applies patch-and-shuffle augmentation
- Trains for 20 epochs
- Uses a weight decay of 0.0005
- Splits 20% of the data for testing
- Uses a learning rate of 0.0005

The dataset in this repo comes from the following repo:

https://github.com/Sutadasuto/I3DCP