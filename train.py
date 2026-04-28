# --- Standard library ---
import os
import time

# --- PyTorch core ---
import torch                                     # Main PyTorch library
import torch.nn as nn                            # Neural network layers and loss functions
import torch.optim as optim                      # Optimisers (AdamW, SGD, etc.)
from torch.utils.data import DataLoader          # Batches and shuffles datasets for training
from torch.utils.data import Dataset             # Base class for custom dataset wrappers
from torch.utils.data import random_split        # Splits a dataset into non-overlapping subsets

# --- Torchvision ---
from torchvision import datasets                 # Provides OxfordIIITPet and other standard datasets
from torchvision import transforms               # Image preprocessing and augmentation pipelines

# ---------------------------------------------------------------------------
# Device
# Automatically use CUDA if available, otherwise fall back to CPU.
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---------------------------------------------------------------------------
# Hyperparameters
# These control the training process
# ---------------------------------------------------------------------------
IMG_SIZE     = 64       # Images are resized to IMG_SIZE x IMG_SIZE pixels
BATCH_SIZE   = 32       # Number of images processed per training step
NUM_EPOCHS   = 30       # Maximum allowed by the coursework spec
LR           = 1e-3     # Initial learning rate for the optimiser
WEIGHT_DECAY = 1e-4     # L2 regularisation — penalises large weights to reduce overfitting
VAL_SPLIT    = 0.1      # Fraction of trainval data held out for validation (10%)
NUM_CLASSES  = 37       # 25 dog breeds + 12 cat breeds
DATA_ROOT    = './data'  # Where the dataset will be downloaded to
MODEL_PATH   = 'model.pth'  # Where the best trained model will be saved
SEED         = 42       # Fixed seed for reproducibility

# Fix random seed so results are reproducible across runs
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Transforms
# Images need to be preprocessed before being fed into the network.
# ---------------------------------------------------------------------------

# Training transforms: augment to improve generalisation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)), # Resize slightly larger than target
    transforms.RandomCrop(IMG_SIZE),                 # Random crop back to IMG_SIZE — positional variation
    transforms.RandomHorizontalFlip(p=0.5),          # 50% chance of horizontal flip
    transforms.ColorJitter(                          # Randomly vary brightness, contrast,
        brightness=0.3,                              # saturation and hue — colour variation
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomRotation(degrees=15),           # Randomly rotate up to ±15 degrees
    transforms.ToTensor(),                           # Convert PIL image to a (C, H, W) float tensor
    transforms.Normalize(                            # Normalise each channel to zero mean, unit std
        mean=[0.485, 0.456, 0.406],                  # These are standard ImageNet channel means
        std=[0.229, 0.224, 0.225]                    # and stds — a good default for natural images
    ),  
])

# Validation transforms: deterministic — only resize and normalise
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset
# OxfordIIITPet is used to train
# ---------------------------------------------------------------------------

# Download and load the full trainval split
full_trainval = datasets.OxfordIIITPet(
    root=DATA_ROOT,
    split='trainval',
    download=True           # Downloads the dataset if not already present
)

print(f'Total trainval images: {len(full_trainval)}')

# Split into train and validation subsets using the fixed seed for reproducibility
val_size   = int(len(full_trainval) * VAL_SPLIT)
train_size = len(full_trainval) - val_size

train_subset, val_subset = random_split(
    full_trainval,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f'Training subset:   {train_size} images')
print(f'Validation subset: {val_size} images')


class TransformedSubset(Dataset):

    # Wraps a Subset so we can apply a different transform to each split.

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


# Apply the appropriate transforms to each split
train_dataset = TransformedSubset(train_subset, train_transforms)
val_dataset   = TransformedSubset(val_subset,   val_transforms)

# DataLoaders handle batching and shuffling during training
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # Shuffle training data each epoch
    num_workers=0,      # Set to 0 for CPU compatibility; increase on a server
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,      # No need to shuffle validation data
    num_workers=0,
    pin_memory=False
)

print(f'\nDataLoaders ready.')
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')